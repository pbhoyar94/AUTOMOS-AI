import logging
from abc import ABC
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import CAP_STYLE
from shapely.ops import unary_union

from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.scene_object import SceneObject
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.geometry.transform import transform
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import RoadBlockGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusData, TrafficLightStatusType
from nuplan.planning.metrics.utils.expert_comparisons import principal_value
from nuplan.planning.simulation.observation.idm.idm_policy import IDMPolicy
from nuplan.planning.simulation.observation.idm.idm_states import IDMAgentState, IDMLeadAgentState
from nuplan.planning.simulation.observation.idm.utils import path_to_linestring
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.occupancy_map.abstract_occupancy_map import OccupancyMap
from nuplan.planning.simulation.occupancy_map.strtree_occupancy_map import STRTreeOccupancyMapFactory
from nuplan.planning.simulation.path.path import AbstractPath
from nuplan.planning.simulation.path.utils import trim_path
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
import yaml
from openai import OpenAI
import os
import json
import math
import time
from zhipuai import ZhipuAI

UniqueObjects = Dict[str, SceneObject]

logger = logging.getLogger(__name__)


class AbstractIDMPlanner(AbstractPlanner, ABC):
    """
    An interface for IDM based planners. Inherit from this class to use IDM policy to control the longitudinal
    behaviour of the ego.
    """
    LOCAL_TIMER = 0
    DESIRED_SPEED = 30
    CURVE_SPEED_LIMIT = 30
    def __init__(
        self,
        target_velocity: float,
        min_gap_to_lead_agent: float,
        headway_time: float,
        accel_max: float,
        decel_max: float,
        planned_trajectory_samples: int,
        planned_trajectory_sample_interval: float,
        occupancy_map_radius: float,
    ):
        """
        Constructor for IDMPlanner
        :param target_velocity: [m/s] Desired velocity in free traffic.
        :param min_gap_to_lead_agent: [m] Minimum relative distance to lead vehicle.
        :param headway_time: [s] Desired time headway. The minimum possible time to the vehicle in front.
        :param accel_max: [m/s^2] maximum acceleration.
        :param decel_max: [m/s^2] maximum deceleration (positive value).
        :param planned_trajectory_samples: number of elements to sample for the planned trajectory.
        :param planned_trajectory_sample_interval: [s] time interval of sequence to sample from.
        :param occupancy_map_radius: [m] The range around the ego to add objects to be considered.
        """
        self._policy = IDMPolicy(target_velocity, min_gap_to_lead_agent, headway_time, accel_max, decel_max)
        self._planned_trajectory_samples = planned_trajectory_samples
        self._planned_trajectory_sample_interval = planned_trajectory_sample_interval
        self._planned_horizon = planned_trajectory_samples * planned_trajectory_sample_interval
        self._occupancy_map_radius = occupancy_map_radius
        self._max_path_length = self._policy.target_velocity * self._planned_horizon
        self._ego_token = "ego_token"
        self._red_light_token = "red_light"
        self._trajectory = None
        self.use_text_encoder = True
        # To be lazy loaded
        self._route_roadblocks: List[RoadBlockGraphEdgeMapObject] = []
        self._candidate_lane_edge_ids: Optional[List[str]] = None
        self._map_api: Optional[AbstractMap] = None

        # To be intialized by inherited classes
        self._ego_path: Optional[AbstractPath] = None
        self._ego_path_linestring: Optional[LineString] = None

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def _initialize_route_plan(self, route_roadblock_ids: List[str]) -> None:
        """
        Initializes the route plan with roadblocks.
        :param route_roadblock_ids: A list of roadblock ids that make up the ego's route
        """
        assert self._map_api, "_map_api has not yet been initialized. Please call the initialize() function first!"
        self._route_roadblocks = []
        for id_ in route_roadblock_ids:
            block = self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            block = block or self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK_CONNECTOR)
            self._route_roadblocks.append(block)

        self._candidate_lane_edge_ids = [
            edge.id for block in self._route_roadblocks if block for edge in block.interior_edges
        ]

        assert (
            self._route_roadblocks
        ), "Cannot create route plan. No roadblocks were extracted from the given route_roadblock_ids!"

    def _get_expanded_ego_path(self, ego_state: EgoState, ego_idm_state: IDMAgentState) -> Polygon:
        """
        Returns the ego's expanded path as a Polygon.
        :return: A polygon representing the ego's path.
        """
        assert self._ego_path, "_ego_path has not yet been initialized. Please call the initialize() function first!"
        ego_footprint = ego_state.car_footprint
        path_to_go = trim_path(
            self._ego_path,
            max(self._ego_path.get_start_progress(), min(ego_idm_state.progress, self._ego_path.get_end_progress())),
            max(
                self._ego_path.get_start_progress(),
                min(
                    ego_idm_state.progress + abs(self._policy.target_velocity) * self._planned_horizon,
                    self._ego_path.get_end_progress(),
                ),
            ),
        )
        expanded_path = path_to_linestring(path_to_go).buffer((ego_footprint.width / 2), cap_style=CAP_STYLE.square)
        return unary_union([expanded_path, ego_state.car_footprint.geometry])

    @staticmethod
    def _get_leading_idm_agent(ego_state: EgoState, agent: SceneObject, relative_distance: float) -> IDMLeadAgentState:
        """
        Returns a lead IDM agent state that represents another static and dynamic agent.
        :param agent: A scene object.
        :param relative_distance: [m] The relative distance from the scene object to the ego.
        :return: A IDM lead agents state
        """
        if isinstance(agent, Agent):
            # Dynamic object
            longitudinal_velocity = agent.velocity.magnitude()
            # Wrap angle to [-pi, pi]
            relative_heading = principal_value(agent.center.heading - ego_state.center.heading)
            projected_velocity = transform(
                StateSE2(longitudinal_velocity, 0, 0), StateSE2(0, 0, relative_heading).as_matrix()
            ).x
        else:
            # Static object
            projected_velocity = 0.0

        return IDMLeadAgentState(progress=relative_distance, velocity=projected_velocity, length_rear=0.0)

    def _get_free_road_leading_idm_state(self, ego_state: EgoState, ego_idm_state: IDMAgentState) -> IDMLeadAgentState:
        """
        Returns a lead IDM agent state when there is no leading agent.
        :return: A IDM lead agents state.
        """
        assert self._ego_path, "_ego_path has not yet been initialized. Please call the initialize() function first!"
        projected_velocity = 0.0
        relative_distance = self._ego_path.get_end_progress() - ego_idm_state.progress
        length_rear = ego_state.car_footprint.length / 2
        return IDMLeadAgentState(progress=relative_distance, velocity=projected_velocity, length_rear=length_rear)

    @staticmethod
    def _get_red_light_leading_idm_state(relative_distance: float) -> IDMLeadAgentState:
        """
        Returns a lead IDM agent state that represents a red light intersection.
        :param relative_distance: [m] The relative distance from the intersection to the ego.
        :return: A IDM lead agents state.
        """
        return IDMLeadAgentState(progress=relative_distance, velocity=0, length_rear=0)

    def _get_leading_object(
        self,
        ego_idm_state: IDMAgentState,
        ego_state: EgoState,
        occupancy_map: OccupancyMap,
        unique_observations: UniqueObjects,
    ) -> IDMLeadAgentState:
        """
        Get the most suitable leading object based on the occupancy map.
        :param ego_idm_state: The ego's IDM state at current iteration.
        :param ego_state: EgoState at current iteration.
        :param occupancy_map: OccupancyMap containing all objects in the scene.
        :param unique_observations: A mapping between the object token and the object itself.
        """
        intersecting_agents = occupancy_map.intersects(self._get_expanded_ego_path(ego_state, ego_idm_state))
        # Check if there are agents intersecting the ego's baseline
        if intersecting_agents.size > 0:

            # Extract closest object
            intersecting_agents.insert(self._ego_token, ego_state.car_footprint.geometry)
            nearest_id, nearest_agent_polygon, relative_distance = intersecting_agents.get_nearest_entry_to(
                self._ego_token
            )

            # Red light at intersection
            if self._red_light_token in nearest_id:
                return self._get_red_light_leading_idm_state(relative_distance)

            # An agent is the leading agent
            return self._get_leading_idm_agent(ego_state, unique_observations[nearest_id], relative_distance)

        else:
            # No leading agent
            return self._get_free_road_leading_idm_state(ego_state, ego_idm_state)

    def _construct_occupancy_map(
        self, ego_state: EgoState, observation: Observation
    ) -> Tuple[OccupancyMap, UniqueObjects]:
        """
        Constructs an OccupancyMap from Observations.
        :param ego_state: Current EgoState
        :param observation: Observations of other agents and static objects in the scene.
        :return:
            - OccupancyMap.
            - A mapping between the object token and the object itself.
        """
        if isinstance(observation, DetectionsTracks):
            unique_observations = {
                detection.track_token: detection
                for detection in observation.tracked_objects.tracked_objects
                if np.linalg.norm(ego_state.center.array - detection.center.array) < self._occupancy_map_radius
            }
            return (
                STRTreeOccupancyMapFactory.get_from_boxes(list(unique_observations.values())),
                unique_observations,
            )
        else:
            raise ValueError(f"IDM planner only supports DetectionsTracks. Got {observation.detection_type()}")

    def _propagate(self, ego: IDMAgentState, lead_agent: IDMLeadAgentState, tspan: float) -> None:
        """
        Propagate agent forward according to the IDM policy.
        :param ego: The ego's IDM state.
        :param lead_agent: The agent leading this agent.
        :param tspan: [s] The interval of time to propagate for.
        """
        # TODO: Set target velocity to speed limit
        solution = self._policy.solve_forward_euler_idm_policy(IDMAgentState(0, ego.velocity), lead_agent, tspan)
        ego.progress += solution.progress
        if solution.velocity > self.DESIRED_SPEED : solution.velocity = self.DESIRED_SPEED 
        if solution.velocity > self.CURVE_SPEED_LIMIT: solution.velocity = self.CURVE_SPEED_LIMIT
        
        ego.velocity = max(solution.velocity, 0)

    def _get_planned_trajectory(
        self, ego_state: EgoState, occupancy_map: OccupancyMap, unique_observations: UniqueObjects
    ) -> InterpolatedTrajectory:
        """
        Plan a trajectory w.r.t. the occupancy map.
        :param ego_state: EgoState at current iteration.
        :param occupancy_map: OccupancyMap containing all objects in the scene.
        :param unique_observations: A mapping between the object token and the object itself.
        :return: A trajectory representing the predicted ego's position in future.
        """
        
        assert (
            self._ego_path_linestring
        ), "_ego_path_linestring has not yet been initialized. Please call the initialize() function first!"
        # Extract ego IDM state
        ego_progress = self._ego_path_linestring.project(Point(*ego_state.center.point.array))
        ego_idm_state = IDMAgentState(progress=ego_progress, velocity=ego_state.dynamic_car_state.center_velocity_2d.x)
        vehicle_parameters = ego_state.car_footprint.vehicle_parameters

        # Initialize planned trajectory with current state
        current_time_point = ego_state.time_point
        projected_ego_state = self._idm_state_to_ego_state(ego_idm_state, current_time_point, vehicle_parameters)
        planned_trajectory: List[EgoState] = [projected_ego_state]
        
        config_file = "LLM.yml"
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
        self.use_text_encoder = config['use_text_encoder']
        use_llm = config['use_llm']
        scope = 100
        if use_llm and self.LOCAL_TIMER%10==0:
            ego_x = ego_state.center.x
            ego_y = ego_state.center.y
            ego_heading = ego_state.center.heading
            reference_path = [(state.x, state.y, state.heading) for state in self.discrete_path]
            ego_abs_speed = np.linalg.norm(np.array([ego_state.agent.velocity.x, ego_state.agent.velocity.y]))
            ego_frenet_x, ego_frenet_y, ego_frenet_heading = self.cartesian_to_frenet_with_heading(reference_path, (ego_x, ego_y, ego_heading))
            agents = []
            
            for Id, agent in unique_observations.items():
                agent_type = agent.tracked_object_type.name
                if agent_type != "GENERIC_OBJECT":
                    agent_x = agent.center.x
                    agent_y = agent.center.y
                    agent_heading = agent.center.heading
                    
                    agent_global_frenet_x, agent_global_frenet_y, agent_global_frenet_heading = self.cartesian_to_frenet_with_heading(reference_path, (agent_x, agent_y, agent_heading))
                    agent_local_frenet_x = agent_global_frenet_x - ego_frenet_x
                    agent_local_frenet_y = agent_global_frenet_y - ego_frenet_y
                    agent_local_frenet_heading = agent_global_frenet_heading - ego_frenet_heading
                    local_x, local_y, local_heading = self.global_to_local(agent_x, agent_y, agent_heading, ego_x, ego_y, ego_heading)
                    abs_speed = np.linalg.norm(np.array([agent.velocity.x, agent.velocity.y]))
                    agent_ID = agent_type + "_" + Id
                    local_agent = {'ID': agent_ID, 'Position': (round(local_x,1), round(local_y,1)), 'Size': {'Width': round(agent.box.width,1), 'Length': round(agent.box.length,1)}, 'Speed': round(abs_speed,2), 'Orientation': round(local_heading,2)}
                    frenet_agent = {'ID': agent_ID, 'Position': (round(agent_local_frenet_x,1), round(agent_local_frenet_y,1)), 'Size': {'Width': round(agent.box.width,1), 'Length': round(agent.box.length,1)}, 'Speed': round(abs_speed,2), 'Orientation': round(agent_local_frenet_heading,2)}
                    # create a scope of seeable region with a Chinese kite shape
                    frenet_x, frenet_y = agent_local_frenet_x, agent_local_frenet_y
                    if -scope/4 < frenet_y < scope/4 and frenet_y < -0.25*frenet_x + scope/4 and frenet_y > 0.25*frenet_x - scope/4 and frenet_y < 2.5*frenet_x + scope/4 and frenet_y > -2.5*frenet_x - scope/4:
                        if self.use_text_encoder: 
                            agents.append(frenet_agent)
                        else: 
                            agents.append(local_agent)

            # Generate descriptions for all agents
            descriptions = [self.describe_agent(agent) for agent in agents]

            # Output the descriptions
            descriptions_output = "\n\n".join(descriptions)
 
            # print(f"------------------------ At time step {self.LOCAL_TIMER} --------------------------")
            print(descriptions_output)
            use_open_ai = config['use_open_ai']


            if use_open_ai:
                os.environ["OPENAI_API_KEY"] = config['open_ai_key']
                client = OpenAI()
                model_choice = "gpt-4o-2024-08-06"
                time_to_sleep = 30
            else: 
                client = ZhipuAI(api_key=config['zhipu_ai_key']) 
                model_choice = "glm-4-flash"
                time_to_sleep = 7

            completion = client.chat.completions.create(
                model=model_choice,
                messages=[
                    {"role": "system", "content": f"""consider a driving scenario: The origin (0,0) is where the ego vehicle is located.
            ### Scenario:
            #### Ego Vehicle:
            - **Ego Vehicle**: Positioned at (0, 0) with an orientation of 0 degrees.
            - **Speed**: {ego_abs_speed} m/s

            ### Other Agents:
            """ + descriptions_output},
                    {
                        "role": "user",
                        "content": """Assume you are the driver of the ego vehicle, and you can adjust the speed (between 0 and 15). What speed do you suggest the ego vehicle should go at? You should not accelerate too hard, ensure comfort driving. Only Output a simple number, don't answer anything else.
            """
                    }
                ]
            )
            time.sleep(time_to_sleep)
            llm_ans_str = completion.choices[0].message.content
            if llm_ans_str.startswith("'t\n"):
                output_string = llm_ans_str.replace("'t\n", "", 1)
            else:
                output_string = llm_ans_str
            llm_ans = float(output_string)
            print(f"At time step {self.LOCAL_TIMER}, the desired speed is", llm_ans_str)
            self.DESIRED_SPEED = llm_ans

        use_curvature = False
        if self.LOCAL_TIMER != 0 and use_curvature:
            planned_path = [(point.center.x, point.center.y) for point in self._trajectory._trajectory]
            curve_radius = self.max_curvature_and_radius(planned_path)
            if curve_radius> 1000: curve_radius = 1000
            curve_speed = curve_radius/10+0.1
            self.CURVE_SPEED_LIMIT = curve_speed

        self.LOCAL_TIMER += 1

        # Propagate planned trajectory for set number of samples
        for _ in range(self._planned_trajectory_samples):

            # Propagate IDM state w.r.t. selected leading agent
            leading_agent = self._get_leading_object(ego_idm_state, ego_state, occupancy_map, unique_observations)
            self._propagate(ego_idm_state, leading_agent, self._planned_trajectory_sample_interval)

            # Convert IDM state back to EgoState
            current_time_point += TimePoint(int(self._planned_trajectory_sample_interval * 1e6))
            ego_state = self._idm_state_to_ego_state(ego_idm_state, current_time_point, vehicle_parameters)

            planned_trajectory.append(ego_state)
        self._trajectory = InterpolatedTrajectory(planned_trajectory)
        return self._trajectory


    def global_to_local(self, agent_x, agent_y, agent_heading, ego_x, ego_y, ego_heading):
        # Step 1: Create the translation vector (agent's position relative to the ego)
        translation_vector = np.array([agent_x - ego_x, agent_y - ego_y])
        
        # Step 2: Create the rotation matrix to rotate by -ego_heading (counterclockwise rotation)
        rotation_matrix = np.array([
            [np.cos(-ego_heading), -np.sin(-ego_heading)],
            [np.sin(-ego_heading), np.cos(-ego_heading)]
        ])
        
        # Step 3: Apply the rotation matrix to the translation vector
        local_position = np.dot(rotation_matrix, translation_vector)
        
        # Step 4: Calculate the local heading by subtracting ego_heading from agent_heading
        local_heading = agent_heading - ego_heading
        
        # Return the transformed local coordinates and heading
        return local_position[0], local_position[1], local_heading

    def describe_position(self, agent_position, ego_orientation):
        """
        Determine the relative position of an agent with respect to the ego vehicle.
        
        Parameters:
        - agent_position: tuple (x, y) relative to the ego vehicle.
        - ego_orientation: orientation of the ego vehicle in degrees (facing north is 0°).
        
        Returns:
        - A string description of the agent's relative position.
        """
        x, y = agent_position

        # Determine the relative position
        if x > 1:
            vertical_position = f"{x} meters ahead"
        elif x < -1:
            vertical_position = f"{abs(x)} meters behind"
        else:
            vertical_position = "parallel with the ego"

        if y > 1:
            horizontal_position = f"{y} meters left"
        elif y < -1:
            horizontal_position = f"{abs(y)} meters right"
        else:
            horizontal_position = "directly in line with the ego"

        return f"{vertical_position} and {horizontal_position}"

    def describe_agent(self, agent):
        """
        Create a textual description of the agent's relative position and orientation.
        
        Parameters:
        - agent: dictionary containing agent's data (ID, Position, Size, Speed, Orientation).
        
        Returns:
        - A string description of the agent's properties and relative position.
        """
        agent_id = agent['ID']
        position = agent['Position']
        size = agent['Size']
        speed = agent['Speed']
        orientation = agent['Orientation']
        x, y = position
        # Calculate the relative position description
        relative_position_desc = self.describe_position(position, ego_orientation=0)

        normalized_orientation = (orientation + math.pi) % (2 * math.pi) - math.pi
        if speed >= 0.01: 
            # Determine orientation relative to the ego vehicle
            if -0.06 <= normalized_orientation <= 0.06:
                orientation_desc = "moving in the same direction as the ego vehicle"
            elif normalized_orientation <= -3.08 or normalized_orientation >= 3.08:
                orientation_desc = "moving in the opposite direction of the ego vehicle"
            elif y >= 1 and -3.08 <= normalized_orientation <= -0.06:
                orientation_desc = "moving towards the ego vehicle's planned trajectory"
            elif y <= -1 and 3.08 >= normalized_orientation >= 0.06:
                orientation_desc = "moving towards the ego vehicle's planned trajectory"
            else:
                orientation_desc = "moving away from the ego vehicle's planned trajectory"
        else: 
            # Object is stationary, determine orientation relative to the ego vehicle
            if -0.06 <= normalized_orientation <= 0.06:
                orientation_desc = "facing the same direction as the ego vehicle"
            elif normalized_orientation <= -3.08 or normalized_orientation >= 3.08:
                orientation_desc = "facing the opposite direction of the ego vehicle"
            elif y >= 1 and -3.08 <= normalized_orientation <= -0.06:
                orientation_desc = "facing towards the ego vehicle's planned trajectory"
            elif y <= -1 and 3.08 >= normalized_orientation >= 0.06:
                orientation_desc = "facing towards the ego vehicle's planned trajectory"
            else:
                orientation_desc = "facing away from the ego vehicle's planned trajectory"


        # Generate the descriptive output
        if self.use_text_encoder: 
            description = (
                f"**{agent_id}**\n"
                f"   - **ID**: {agent_id}\n"
                f"   - **Position**: {position} meters ({relative_position_desc})\n"
                f"   - **Size**: Width: {size['Width']} meters, Length: {size['Length']} meters\n"
                f"   - **Speed**: {speed} m/s\n"
                f"   - **Orientation**: {orientation} rad ({orientation_desc})\n"
            )
        else: 
            description = (
                f"**{agent_id}**\n"
                f"   - **ID**: {agent_id}\n"
                f"   - **Position**: {position} meters \n"
                f"   - **Size**: Width: {size['Width']} meters, Length: {size['Length']} meters\n"
                f"   - **Speed**: {speed} m/s\n"
                f"   - **Orientation**: {orientation} rad\n"
            )
        
        return description

    def _idm_state_to_ego_state(
        self, idm_state: IDMAgentState, time_point: TimePoint, vehicle_parameters: VehicleParameters
    ) -> EgoState:
        """
        Convert IDMAgentState to EgoState
        :param idm_state: The IDMAgentState to be converted.
        :param time_point: The TimePoint corresponding to the state.
        :param vehicle_parameters: VehicleParameters of the ego.
        """
        assert self._ego_path, "_ego_path has not yet been initialized. Please call the initialize() function first!"

        new_ego_center = self._ego_path.get_state_at_progress(
            max(self._ego_path.get_start_progress(), min(idm_state.progress, self._ego_path.get_end_progress()))
        )
        return EgoState.build_from_center(
            center=StateSE2(new_ego_center.x, new_ego_center.y, new_ego_center.heading),
            center_velocity_2d=StateVector2D(idm_state.velocity, 0),
            center_acceleration_2d=StateVector2D(0, 0),
            tire_steering_angle=0.0,
            time_point=time_point,
            vehicle_parameters=vehicle_parameters,
        )

    def _annotate_occupancy_map(
        self, traffic_light_data: List[TrafficLightStatusData], occupancy_map: OccupancyMap
    ) -> None:
        """
        Add red light lane connectors on the route plan to the occupancy map. Note: the function works inline, hence,
        the occupancy map will be modified in this function.
        :param traffic_light_data: A list of all available traffic status data.
        :param occupancy_map: The occupancy map to be annotated.
        """
        assert self._map_api, "_map_api has not yet been initialized. Please call the initialize() function first!"
        assert (
            self._candidate_lane_edge_ids is not None
        ), "_candidate_lane_edge_ids has not yet been initialized. Please call the initialize() function first!"
        for data in traffic_light_data:
            if (
                data.status == TrafficLightStatusType.RED
                and str(data.lane_connector_id) in self._candidate_lane_edge_ids
            ):
                id_ = str(data.lane_connector_id)
                lane_conn = self._map_api.get_map_object(id_, SemanticMapLayer.LANE_CONNECTOR)
                occupancy_map.insert(f"{self._red_light_token}_{id_}", lane_conn.polygon)

    def euclidean_distance(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    # Helper function to calculate the heading between two points
    def calculate_heading(self, point1, point2):
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        return np.arctan2(dy, dx)

    # Helper function to find the closest point on the reference path considering heading
    def closest_point_on_path_with_heading(self, reference_path, point):
        closest_point = None
        closest_dist = float('inf')
        closest_index = None
        closest_heading = None
        
        for i in range(len(reference_path) - 1):
            # Start and end of the segment
            p1 = np.array(reference_path[i][:2])
            p2 = np.array(reference_path[i+1][:2])
            
            # Project point onto the line segment p1-p2
            v = p2 - p1
            w = np.array(point[:2]) - p1
            
            c1 = np.dot(w, v)
            c2 = np.dot(v, v)
            
            if c2 == 0:
                continue  # p1 and p2 are the same point
            
            t = max(0, min(1, c1 / c2))
            projection = p1 + t * v
            dist = self.euclidean_distance(projection, point[:2])
            
            if dist < closest_dist:
                closest_dist = dist
                closest_point = projection
                closest_index = i
                closest_heading = self.calculate_heading(p1, p2)
        
        return closest_point, closest_index, closest_heading

    # Function to convert Cartesian coordinates to Frenet coordinates considering heading
    def cartesian_to_frenet_with_heading(self, reference_path, point):
        closest_point, closest_index, path_heading = self.closest_point_on_path_with_heading(reference_path, point)
        
        # Calculate the s coordinate (arc length)
        s = sum(self.euclidean_distance(reference_path[i][:2], reference_path[i+1][:2]) for i in range(closest_index))
        s += self.euclidean_distance(reference_path[closest_index][:2], closest_point)
        
        # Calculate the d coordinate (perpendicular distance) with heading adjustment
        dx = point[0] - closest_point[0]
        dy = point[1] - closest_point[1]
        angle_diff = np.arctan2(dy, dx) - path_heading
        d = self.euclidean_distance(closest_point, point[:2]) * np.sign(np.sin(angle_diff))
        
        # Calculate the heading difference
        heading_diff = point[2] - path_heading  # Agent's heading minus path heading
        
        return s, d, heading_diff
    
    def calculate_curvature(self, x, y):
        # First derivatives
        dx = np.gradient(x)
        dy = np.gradient(y)
        
        # Second derivatives
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        
        # Curvature
        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
        
        return curvature

    def max_curvature_and_radius(self, coords):
        # Extract x and y coordinates
        x = np.array([p[0] for p in coords])
        y = np.array([p[1] for p in coords])
        
        # Calculate the curvature for each point
        curvature = self.calculate_curvature(x, y)
        
        # Find the maximum curvature
        max_curvature = np.max(curvature)
        
        # Calculate the corresponding radius of curvature
        max_radius = 1 / max_curvature if max_curvature != 0 else np.inf
        
        # Find the index of the maximum curvature
        max_index = np.argmax(curvature)
        
        return max_radius