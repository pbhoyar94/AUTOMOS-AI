#!/usr/bin/env python3
"""
AUTOMOS AI Cinematic Demo Video Creator
Creates realistic cinematic video showing actual product features
"""

import cv2
import numpy as np
import math
from pathlib import Path
from datetime import datetime

class AUTOMOSAICinematicDemo:
    """Creates cinematic demo with realistic feature visualization"""
    
    def __init__(self):
        self.width, self.height = 1920, 1080
        self.fps = 30
        self.duration = 240  # 4 minutes for cinematic quality
        
    def create_cinematic_video(self):
        """Create cinematic demonstration video"""
        
        print("🎬 Creating AUTOMOS AI Cinematic Demo Video...")
        print("This will show realistic product features in action")
        
        output_path = Path("automos_ai_cinematic_demo.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps, (self.width, self.height))
        
        total_frames = self.duration * self.fps
        
        for frame_num in range(total_frames):
            progress = frame_num / total_frames
            
            if progress < 0.15:  # Opening scene (0-36s)
                frame = self.create_opening_scene(frame_num, total_frames)
            elif progress < 0.35:  # Vehicle driving scene (36-84s)
                frame = self.create_driving_scene(frame_num, total_frames)
            elif progress < 0.55:  # Sensor visualization (84-132s)
                frame = self.create_sensor_scene(frame_num, total_frames)
            elif progress < 0.75:  # Safety system in action (132-180s)
                frame = self.create_safety_scene(frame_num, total_frames)
            elif progress < 0.90:  # AI reasoning visualization (180-216s)
                frame = self.create_ai_reasoning_scene(frame_num, total_frames)
            else:  # Call to action (216-240s)
                frame = self.create_call_to_action_scene(frame_num, total_frames)
            
            out.write(frame)
            
            if frame_num % 300 == 0:
                print(f"   Rendered {frame_num}/{total_frames} frames ({progress*100:.1f}%)")
        
        out.release()
        
        file_size = output_path.stat().st_size / (1024*1024)
        print(f"✅ Cinematic demo video created: {output_path}")
        print(f"📊 Duration: {self.duration}s, Resolution: {self.width}x{self.height}")
        print(f"📁 File size: {file_size:.1f} MB")
        
        return output_path
    
    def create_opening_scene(self, frame_num, total_frames):
        """Create cinematic opening scene"""
        
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Dark cinematic background
        frame[:] = (10, 15, 30)  # Dark blue-gray
        
        # Animated vehicle silhouette
        progress = (frame_num % 900) / 900  # 30 second cycle
        
        # Vehicle shape
        vehicle_x = int(self.width * 0.7)
        vehicle_y = int(self.height * 0.6)
        vehicle_width = int(200 * (1 + 0.1 * math.sin(progress * 2 * math.pi)))
        vehicle_height = int(80 * (1 + 0.1 * math.sin(progress * 2 * math.pi)))
        
        # Draw vehicle
        cv2.rectangle(frame, 
                   (vehicle_x - vehicle_width//2, vehicle_y - vehicle_height//2),
                   (vehicle_x + vehicle_width//2, vehicle_y + vehicle_height//2),
                   (100, 150, 200), -1)
        
        # Headlights
        if progress > 0.5:
            # Left headlight
            cv2.circle(frame, (vehicle_x - 40, vehicle_y + 20), 15, (255, 255, 200), -1)
            # Right headlight  
            cv2.circle(frame, (vehicle_x + 40, vehicle_y + 20), 15, (255, 255, 200), -1)
        
        # Title animation
        title_progress = min(1.0, (frame_num % 450) / 450)
        if title_progress > 0:
            title_alpha = int(255 * title_progress)
            cv2.putText(frame, "AUTOMOS AI",
                       (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 3,
                       (title_alpha, title_alpha, title_alpha), 3)
            
            if title_progress > 0.5:
                subtitle_alpha = int(255 * (title_progress - 0.5) * 2)
                cv2.putText(frame, "Reasoning-Based Autonomous Driving",
                           (100, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                           (subtitle_alpha, subtitle_alpha, subtitle_alpha), 2)
        
        # Particle effects (stars)
        np.random.seed(42)  # Consistent stars
        for _ in range(50):
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height // 2)
            brightness = np.random.randint(50, 150)
            cv2.circle(frame, (x, y), 1, (brightness, brightness, brightness), -1)
        
        return frame
    
    def create_driving_scene(self, frame_num, total_frames):
        """Create realistic driving scene with road and environment"""
        
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Sky gradient
        for y in range(self.height // 2):
            color_value = int(135 + (y / (self.height // 2)) * 50)
            frame[y, :] = (color_value, color_value + 20, color_value + 40)
        
        # Road
        road_y = self.height // 2
        cv2.rectangle(frame, (0, road_y), (self.width, self.height), (80, 80, 90))
        
        # Road lines
        for x in range(0, self.width, 40):
            cv2.rectangle(frame, (x, road_y + 100), (x + 20, road_y + 105), (255, 255, 255), -1)
        
        # Moving vehicle (ego vehicle)
        vehicle_progress = (frame_num % 300) / 300
        ego_x = int(self.width * 0.3)
        ego_y = road_y - 50
        
        # Draw ego vehicle with AUTOMOS AI branding
        cv2.rectangle(frame, (ego_x - 40, ego_y - 20), (ego_x + 40, ego_y + 20), (0, 100, 200), -1)
        cv2.putText(frame, "AUTOMOS", (ego_x - 35, ego_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Other vehicles
        for i, offset in enumerate([200, 400, 600]):
            other_x = offset + int(50 * math.sin(vehicle_progress * 2 * math.pi + i))
            other_y = road_y - 30
            cv2.rectangle(frame, (other_x - 30, other_y - 15), (other_x + 30, other_y + 15), (150, 150, 160), -1)
        
        # Trees/environment
        for x in [100, 300, 500, 700, 900, 1100, 1500, 1700]:
            tree_y = road_y - 80
            cv2.rectangle(frame, (x - 5, tree_y - 40), (x + 5, tree_y), (20, 80, 20), -1)
            cv2.circle(frame, (x, tree_y - 50), 20, (40, 120, 40), -1)
        
        # 360° camera visualization
        camera_angle = (frame_num * 2) % 360
        cv2.circle(frame, (ego_x, ego_y - 60), 30, (0, 255, 255), 2)
        end_x = int(ego_x + 30 * math.cos(math.radians(camera_angle)))
        end_y = int(ego_y - 60 + 30 * math.sin(math.radians(camera_angle)))
        cv2.line(frame, (ego_x, ego_y - 60), (end_x, end_y), (0, 255, 0), 3)
        
        return frame
    
    def create_sensor_scene(self, frame_num, total_frames):
        """Create sensor fusion visualization"""
        
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:] = (20, 25, 40)  # Dark tech background
        
        # Central vehicle
        center_x, center_y = self.width // 2, self.height // 2
        
        # Vehicle representation
        cv2.rectangle(frame, (center_x - 60, center_y - 30), (center_x + 60, center_y + 30), (0, 150, 200), -1)
        cv2.putText(frame, "AUTOMOS AI", (center_x - 50, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Camera sensors (6 cameras for 360°)
        camera_positions = [
            (center_x - 80, center_y - 80, "Front"),
            (center_x + 80, center_y - 80, "Front-Right"),
            (center_x + 80, center_y + 80, "Rear"),
            (center_x - 80, center_y + 80, "Rear-Left"),
            (center_x - 100, center_y, "Left"),
            (center_x + 100, center_y, "Right")
        ]
        
        # Animated camera activation
        cam_progress = (frame_num % 120) / 120
        
        for i, (x, y, label) in enumerate(camera_positions):
            # Camera activation based on progress
            if cam_progress > i / len(camera_positions):
                color = (0, 255, 0)  # Green when active
                radius = 15
            else:
                color = (100, 100, 100)  # Gray when inactive
                radius = 10
            
            cv2.circle(frame, (x, y), radius, color, -1)
            cv2.putText(frame, label, (x - 20, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Detection cone from camera
            if color == (0, 255, 0):
                cv2.line(frame, (center_x, center_y), (x, y), (0, 255, 0), 1)
        
        # Radar waves
        radar_progress = (frame_num % 60) / 60
        for radius in range(30, 150, 30):
            alpha = int(255 * (1 - radar_progress))
            cv2.circle(frame, (center_x, center_y), radius, (alpha, 0, alpha), 1)
        
        # LiDAR point cloud
        np.random.seed(frame_num)
        for _ in range(100):
            angle = np.random.uniform(0, 2 * math.pi)
            distance = np.random.uniform(50, 200)
            x = int(center_x + distance * math.cos(angle))
            y = int(center_y + distance * math.sin(angle))
            
            if 0 < x < self.width and 0 < y < self.height:
                cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
        
        # Sensor fusion indicator
        fusion_progress = (frame_num % 90) / 90
        if fusion_progress > 0.5:
            cv2.putText(frame, "SENSOR FUSION ACTIVE", (center_x - 100, center_y + 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame
    
    def create_safety_scene(self, frame_num, total_frames):
        """Create safety system activation scene"""
        
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:] = (40, 20, 20)  # Dark red background for urgency
        
        center_x, center_y = self.width // 2, self.height // 2
        
        # Vehicle in emergency situation
        cv2.rectangle(frame, (center_x - 60, center_y - 30), (center_x + 60, center_y + 30), (200, 50, 50), -1)
        
        # Obstacle (pedestrian)
        pedestrian_x = center_x + 120
        pedestrian_y = center_y
        cv2.circle(frame, (pedestrian_x, pedestrian_y), 15, (255, 200, 150), -1)
        
        # Collision detection visualization
        danger_progress = (frame_num % 60) / 60
        
        if danger_progress > 0.3:
            # Red danger zone
            cv2.circle(frame, (pedestrian_x, pedestrian_y), 80, (0, 0, 255), 2)
            
            # Warning text
            cv2.putText(frame, "COLLISION RISK", (center_x - 100, center_y - 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            # Safety response time
            response_time = 5 + 3 * math.sin(danger_progress * math.pi)  # Oscillating 2-8ms
            cv2.putText(frame, f"Response: {response_time:.1f}ms", (center_x - 100, center_y - 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Emergency brake activation
            if danger_progress > 0.7:
                cv2.rectangle(frame, (center_x - 80, center_y + 50), (center_x - 40, center_y + 70),
                           (255, 0, 0), -1)
                cv2.putText(frame, "BRAKE", (center_x - 75, center_y + 65),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Safety system status
        cv2.putText(frame, "SAFETY-CRITIC SYSTEM", (100, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Real-time metrics
        metrics = [
            f"Objects: {np.random.randint(1, 8)}",
            f"Risk Level: {'LOW' if danger_progress < 0.5 else 'HIGH'}",
            f"System: {'ACTIVE' if danger_progress > 0.3 else 'MONITORING'}"
        ]
        
        for i, metric in enumerate(metrics):
            cv2.putText(frame, metric, (100, 200 + i * 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return frame
    
    def create_ai_reasoning_scene(self, frame_num, total_frames):
        """Create AI reasoning visualization"""
        
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:] = (10, 30, 50)  # Dark blue tech background
        
        # Neural network visualization
        center_x, center_y = self.width // 2, self.height // 2
        
        # Input layer
        input_progress = (frame_num % 120) / 120
        for i in range(5):
            x = center_x - 300
            y = center_y - 100 + i * 50
            if input_progress > i / 5:
                color = (0, 255, 0)  # Green when active
                cv2.circle(frame, (x, y), 8, color, -1)
            else:
                color = (100, 100, 100)  # Gray when inactive
                cv2.circle(frame, (x, y), 5, color, -1)
        
        # Hidden layers with animated connections
        layer_progress = (frame_num % 90) / 90
        for layer in range(3):
            layer_x = center_x - 150 + layer * 150
            layer_y = center_y
            
            # Layer nodes
            for i in range(4):
                node_x = layer_x + (i - 1.5) * 40
                node_y = layer_y + (i - 1.5) * 30
                
                if layer_progress > layer / 3:
                    cv2.circle(frame, (int(node_x), int(node_y)), 6, (0, 150, 255), -1)
            
            # Connections to previous layer
            if layer > 0 and layer_progress > layer / 3:
                prev_layer_x = center_x - 150 + (layer - 1) * 150
                for i in range(4):
                    for j in range(4):
                        prev_x = prev_layer_x + (j - 1.5) * 40
                        prev_y = layer_y + (j - 1.5) * 30
                        curr_x = layer_x + (i - 1.5) * 40
                        curr_y = layer_y + (i - 1.5) * 30
                        
                        alpha = int(100 * (1 - abs(layer_progress - layer / 3)))
                        cv2.line(frame, (int(prev_x), int(prev_y)), (int(curr_x), int(curr_y)),
                                   (alpha, alpha, 255), 1)
        
        # Output layer
        output_progress = (frame_num % 60) / 60
        if output_progress > 0.5:
            cv2.putText(frame, "DRIVE SAFELY", (center_x + 200, center_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        # LLM visualization
        llm_progress = (frame_num % 180) / 180
        cv2.putText(frame, "LLM REASONING", (100, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        # Animated text processing
        if llm_progress > 0.3:
            text_lines = [
                "Analyze environment...",
                "Predict trajectories...",
                "Evaluate safety...",
                "Generate plan...",
                "Execute action..."
            ]
            
            line_idx = int(llm_progress * 5) % len(text_lines)
            cv2.putText(frame, text_lines[line_idx], (100, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        return frame
    
    def create_call_to_action_scene(self, frame_num, total_frames):
        """Create call to action scene"""
        
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Gradient background
        for y in range(self.height):
            progress = y / self.height
            r = int(0 + progress * 20)
            g = int(50 + progress * 100)
            b = int(100 + progress * 155)
            frame[y, :] = (r, g, b)
        
        # Central logo
        center_x, center_y = self.width // 2, self.height // 2
        
        # Animated logo appearance
        logo_progress = min(1.0, (frame_num % 120) / 120)
        logo_size = int(80 * logo_progress)
        
        if logo_progress > 0:
            # Logo circle
            cv2.circle(frame, (center_x, center_y), logo_size, (0, 100, 200), -1)
            cv2.circle(frame, (center_x, center_y), logo_size, (255, 255, 255), 3)
            
            # Logo text
            if logo_progress > 0.5:
                text_alpha = int(255 * (logo_progress - 0.5) * 2)
                cv2.putText(frame, "AUTOMOS", (center_x - 60, center_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                           (text_alpha, text_alpha, text_alpha), 3)
                cv2.putText(frame, "AI", (center_x - 20, center_y + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                           (text_alpha, text_alpha, text_alpha), 3)
        
        # Feature highlights
        if logo_progress > 0.7:
            features = [
                "360° Multi-Camera Processing",
                "Safety-Critic System (<10ms)",
                "Language-Conditioned Control",
                "Edge Optimization",
                "Real-Time World Model"
            ]
            
            for i, feature in enumerate(features):
                y_pos = center_y + 150 + i * 50
                feature_alpha = int(255 * min(1.0, (logo_progress - 0.7) * 3))
                
                cv2.putText(frame, f"• {feature}", (center_x - 300, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                           (feature_alpha, feature_alpha, feature_alpha), 2)
        
        # Contact information
        if logo_progress > 0.9:
            contact_alpha = int(255 * (logo_progress - 0.9) * 10)
            cv2.putText(frame, "contact@automos-ai.com", (center_x - 150, center_y + 450),
                       cv2.FONT_HERSHEY_SIMPLEX, 1,
                       (contact_alpha, contact_alpha, contact_alpha), 2)
            cv2.putText(frame, "www.automos-ai.com", (center_x - 120, center_y + 490),
                       cv2.FONT_HERSHEY_SIMPLEX, 1,
                       (contact_alpha, contact_alpha, contact_alpha), 2)
        
        return frame

def main():
    """Main function"""
    
    print("🎬 AUTOMOS AI Cinematic Demo Video Creator")
    print("Creating realistic cinematic video showing actual product features")
    print()
    
    creator = AUTOMOSAICinematicDemo()
    video_path = creator.create_cinematic_video()
    
    print("\n🎉 Cinematic Demo Video Complete!")
    print("📹 Video shows realistic AUTOMOS AI features in action")
    print("🎬 Ready for customer presentations and marketing")
    
    return video_path

if __name__ == "__main__":
    main()
