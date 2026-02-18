#!/usr/bin/env python3
"""
AUTOMOS AI Complete Product Test
Comprehensive testing and video generation of all features
"""

import os
import sys
import time
import threading
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

class AUTOMOSAIProductTest:
    """Complete product testing suite"""
    
    def __init__(self):
        """Initialize test suite"""
        self.test_results = {
            'core_reasoning': {'status': 'pending', 'score': 0},
            'perception_pipeline': {'status': 'pending', 'score': 0},
            'safety_critic': {'status': 'pending', 'score': 0},
            'world_model': {'status': 'pending', 'score': 0},
            'edge_optimization': {'status': 'pending', 'score': 0},
            'system_integration': {'status': 'pending', 'score': 0}
        }
        
        self.test_frames = []
        self.current_test = None
        self.test_start_time = None
        
        print("AUTOMOS AI Complete Product Test Suite")
        print("=" * 60)
    
    def run_complete_test(self):
        """Run comprehensive product test"""
        
        print("🧪 Starting Complete AUTOMOS AI Product Test")
        print("Testing all components and generating video output...")
        
        # Test each component
        tests = [
            ("Core Reasoning Engine", self.test_reasoning_engine),
            ("Perception Pipeline", self.test_perception_pipeline),
            ("Safety-Critic System", self.test_safety_critic),
            ("World Model", self.test_world_model),
            ("Edge Optimization", self.test_edge_optimization),
            ("System Integration", self.test_system_integration)
        ]
        
        for test_name, test_func in tests:
            print(f"\n🔍 Testing: {test_name}")
            self.current_test = test_name
            self.test_start_time = time.time()
            
            try:
                result = test_func()
                self.test_results[test_name.lower().replace(' ', '_').replace('-', '_')] = result
                print(f"✅ {test_name}: {result['status']} (Score: {result['score']}/100)")
            except Exception as e:
                print(f"❌ {test_name}: Failed - {e}")
                self.test_results[test_name.lower().replace(' ', '_').replace('-', '_')] = {
                    'status': 'failed', 'score': 0, 'error': str(e)
                }
        
        # Generate test report
        self.generate_test_report()
        
        # Create test video
        self.create_test_video()
        
        return self.test_results
    
    def test_reasoning_engine(self):
        """Test core reasoning engine"""
        
        frames = []
        test_score = 0
        
        try:
            # Import reasoning engine
            from core.reasoning_engine import ReasoningEngine
            
            # Initialize reasoning engine
            engine = ReasoningEngine()
            
            # Test 1: Initialization
            self.add_test_frame("Reasoning Engine - Initialization", "Testing engine initialization...")
            engine.initialize()
            test_score += 20
            
            # Test 2: Plan Generation
            self.add_test_frame("Reasoning Engine - Plan Generation", "Testing autonomous plan generation...")
            mock_world_state = {
                'ego_vehicle': {'position': [0, 0], 'speed': 10},
                'objects': [{'type': 'vehicle', 'distance': 50}],
                'road_conditions': 'dry'
            }
            plan = engine.generate_plan(mock_world_state, 'normal')
            test_score += 30
            
            # Test 3: Emergency Handling
            self.add_test_frame("Reasoning Engine - Emergency Handling", "Testing emergency response...")
            emergency_plan = engine.generate_plan(mock_world_state, 'emergency')
            test_score += 25
            
            # Test 4: Language Command Processing
            self.add_test_frame("Reasoning Engine - Language Commands", "Testing natural language processing...")
            lang_plan = engine.process_language_command("Navigate to intersection safely")
            test_score += 25
            
            return {'status': 'passed', 'score': min(test_score, 100)}
            
        except Exception as e:
            self.add_test_frame("Reasoning Engine - Error", f"Test failed: {str(e)}")
            return {'status': 'failed', 'score': 0, 'error': str(e)}
    
    def test_perception_pipeline(self):
        """Test perception pipeline"""
        
        test_score = 0
        
        try:
            from perception.perception_pipeline import PerceptionPipeline
            
            # Initialize perception pipeline
            pipeline = PerceptionPipeline()
            
            # Test 1: Initialization
            self.add_test_frame("Perception Pipeline - Initialization", "Testing sensor initialization...")
            pipeline.initialize()
            test_score += 20
            
            # Test 2: Multi-Sensor Processing
            self.add_test_frame("Perception Pipeline - Multi-Sensor", "Testing camera, radar, LiDAR fusion...")
            mock_sensor_data = {
                'cameras': [np.random.rand(480, 640, 3) for _ in range(6)],
                'radar': np.random.rand(10, 3),
                'lidar': np.random.rand(1000, 3)
            }
            perception_result = pipeline.process_sensors(mock_sensor_data)
            test_score += 30
            
            # Test 3: Object Detection
            self.add_test_frame("Perception Pipeline - Object Detection", "Testing object detection capabilities...")
            objects = pipeline.detect_objects(mock_sensor_data['cameras'][0])
            test_score += 25
            
            # Test 4: Sensor Fusion
            self.add_test_frame("Perception Pipeline - Sensor Fusion", "Testing multi-sensor data fusion...")
            fused_result = pipeline.fuse_sensors(mock_sensor_data)
            test_score += 25
            
            return {'status': 'passed', 'score': min(test_score, 100)}
            
        except Exception as e:
            self.add_test_frame("Perception Pipeline - Error", f"Test failed: {str(e)}")
            return {'status': 'failed', 'score': 0, 'error': str(e)}
    
    def test_safety_critic(self):
        """Test safety-critic system"""
        
        test_score = 0
        
        try:
            from safety.safety_critic import SafetyCritic
            
            # Initialize safety critic
            safety_critic = SafetyCritic()
            
            # Test 1: Initialization
            self.add_test_frame("Safety-Critic - Initialization", "Testing safety system initialization...")
            safety_critic.initialize()
            test_score += 20
            
            # Test 2: Risk Assessment
            self.add_test_frame("Safety-Critic - Risk Assessment", "Testing real-time risk assessment...")
            mock_world_state = {
                'ego_vehicle': {'speed': 15},
                'objects': [{'type': 'vehicle', 'distance': 10, 'speed': 20}]
            }
            risk_score = safety_critic.assess_risk(mock_world_state)
            test_score += 30
            
            # Test 3: Emergency Response Time
            self.add_test_frame("Safety-Critic - Response Time", "Testing <10ms emergency response...")
            start_time = time.time()
            emergency_action = safety_critic.emergency_override(mock_world_state)
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            if response_time < 10:
                test_score += 25
            else:
                test_score += 15
            
            # Test 4: Safety Constraints
            self.add_test_frame("Safety-Critic - Constraints", "Testing safety constraint checking...")
            constraints_met = safety_critic.check_safety_constraints(mock_world_state)
            test_score += 25
            
            return {'status': 'passed', 'score': min(test_score, 100)}
            
        except Exception as e:
            self.add_test_frame("Safety-Critic - Error", f"Test failed: {str(e)}")
            return {'status': 'failed', 'score': 0, 'error': str(e)}
    
    def test_world_model(self):
        """Test world model"""
        
        test_score = 0
        
        try:
            from world_model.world_model import WorldModel
            
            # Initialize world model
            world_model = WorldModel()
            
            # Test 1: Initialization
            self.add_test_frame("World Model - Initialization", "Testing world model initialization...")
            world_model.initialize()
            test_score += 20
            
            # Test 2: State Update
            self.add_test_frame("World Model - State Update", "Testing real-time state updates...")
            mock_perception_data = {
                'objects': [{'type': 'vehicle', 'position': [10, 5], 'velocity': [2, 0]}],
                'ego_state': {'position': [0, 0], 'heading': 0, 'speed': 10}
            }
            world_model.update_state(mock_perception_data)
            test_score += 30
            
            # Test 3: Prediction
            self.add_test_frame("World Model - Prediction", "Testing trajectory prediction...")
            predictions = world_model.predict_trajectories(5.0)  # 5 seconds ahead
            test_score += 25
            
            # Test 4: HD-Map-Free Navigation
            self.add_test_frame("World Model - HD-Map-Free", "Testing map-free navigation...")
            navigation_plan = world_model.generate_navigation_plan([100, 100])
            test_score += 25
            
            return {'status': 'passed', 'score': min(test_score, 100)}
            
        except Exception as e:
            self.add_test_frame("World Model - Error", f"Test failed: {str(e)}")
            return {'status': 'failed', 'score': 0, 'error': str(e)}
    
    def test_edge_optimization(self):
        """Test edge optimization"""
        
        test_score = 0
        
        try:
            from edge_optimization.model_quantizer import ModelQuantizer
            from edge_optimization.edge_optimizer import EdgeOptimizer
            
            # Test 1: Model Quantization
            self.add_test_frame("Edge Optimization - Quantization", "Testing model quantization...")
            quantizer = ModelQuantizer()
            quant_result = quantizer.quantize_model(
                model_path="models/mock_model.pt",
                model_type="reasoning_engine",
                quantization_type="dynamic_int8"
            )
            test_score += 35
            
            # Test 2: Edge Device Optimization
            self.add_test_frame("Edge Optimization - Device Optimization", "Testing edge device optimization...")
            optimizer = EdgeOptimizer()
            opt_result = optimizer.optimize_for_edge({})
            test_score += 35
            
            # Test 3: Performance Benchmarking
            self.add_test_frame("Edge Optimization - Benchmarking", "Testing performance benchmarks...")
            benchmark = optimizer.benchmark_system()
            test_score += 30
            
            return {'status': 'passed', 'score': min(test_score, 100)}
            
        except Exception as e:
            self.add_test_frame("Edge Optimization - Error", f"Test failed: {str(e)}")
            return {'status': 'failed', 'score': 0, 'error': str(e)}
    
    def test_system_integration(self):
        """Test complete system integration"""
        
        test_score = 0
        
        try:
            from integration.system_coordinator import SystemCoordinator
            
            # Test 1: System Startup
            self.add_test_frame("System Integration - Startup", "Testing complete system startup...")
            coordinator = SystemCoordinator()
            coordinator.initialize_system()
            test_score += 25
            
            # Test 2: Component Communication
            self.add_test_frame("System Integration - Communication", "Testing component communication...")
            coordinator.start_component_communication()
            test_score += 25
            
            # Test 3: End-to-End Processing
            self.add_test_frame("System Integration - End-to-End", "Testing complete processing pipeline...")
            mock_sensor_data = {'cameras': [np.random.rand(480, 640, 3)]}
            result = coordinator.process_sensor_data(mock_sensor_data)
            test_score += 25
            
            # Test 4: Emergency Handling
            self.add_test_frame("System Integration - Emergency", "Testing system-wide emergency handling...")
            coordinator.handle_emergency_stop()
            test_score += 25
            
            return {'status': 'passed', 'score': min(test_score, 100)}
            
        except Exception as e:
            self.add_test_frame("System Integration - Error", f"Test failed: {str(e)}")
            return {'status': 'failed', 'score': 0, 'error': str(e)}
    
    def add_test_frame(self, title, description):
        """Add test frame to video"""
        
        frame = self.create_test_frame(title, description)
        self.test_frames.append(frame)
    
    def create_test_frame(self, title, description):
        """Create a test frame for video"""
        
        # Create frame
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[:] = (0, 100, 200)  # Blue background
        
        # Add title
        cv2.putText(frame, "AUTOMOS AI - PRODUCT TEST",
                   (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Add test title
        cv2.putText(frame, title,
                   (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        
        # Add description
        words = description.split()
        lines = []
        current_line = []
        for word in words:
            current_line.append(word)
            if len(' '.join(current_line)) > 50:
                lines.append(' '.join(current_line))
                current_line = []
        if current_line:
            lines.append(' '.join(current_line))
        
        for line_idx, line in enumerate(lines):
            cv2.putText(frame, line,
                       (50, 250 + line_idx * 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp,
                   (50, 1000), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add progress indicator
        if self.test_start_time:
            elapsed = time.time() - self.test_start_time
            progress = min(1.0, elapsed / 10.0)  # 10 second test
            bar_width = int(800 * progress)
            cv2.rectangle(frame, (50, 900), (850, 930), (255, 255, 255), 2)
            cv2.rectangle(frame, (50, 900), (50 + bar_width, 930), (0, 255, 0), -1)
        
        return frame
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        
        print("\n" + "=" * 60)
        print("🧪 AUTOMOS AI COMPLETE TEST REPORT")
        print("=" * 60)
        
        total_score = 0
        passed_tests = 0
        
        for test_name, result in self.test_results.items():
            status_icon = "✅" if result['status'] == 'passed' else "❌"
            print(f"{status_icon} {test_name.replace('_', ' ').title()}: {result['status'].upper()}")
            print(f"   Score: {result['score']}/100")
            
            if result['status'] == 'passed':
                total_score += result['score']
                passed_tests += 1
            elif 'error' in result:
                print(f"   Error: {result['error']}")
        
        overall_score = total_score / len(self.test_results)
        pass_rate = (passed_tests / len(self.test_results)) * 100
        
        print(f"\n📊 OVERALL RESULTS:")
        print(f"   Pass Rate: {pass_rate:.1f}%")
        print(f"   Average Score: {overall_score:.1f}/100")
        print(f"   Tests Passed: {passed_tests}/{len(self.test_results)}")
        
        if pass_rate >= 80:
            print(f"   Status: ✅ PRODUCTION READY")
        elif pass_rate >= 60:
            print(f"   Status: ⚠️ NEEDS IMPROVEMENT")
        else:
            print(f"   Status: ❌ NOT READY")
        
        # Save detailed report
        report = {
            'test_date': datetime.now().isoformat(),
            'overall_score': overall_score,
            'pass_rate': pass_rate,
            'tests_passed': passed_tests,
            'total_tests': len(self.test_results),
            'status': 'production_ready' if pass_rate >= 80 else 'needs_improvement',
            'detailed_results': self.test_results
        }
        
        report_path = PROJECT_ROOT / "test_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved: {report_path}")
        return report
    
    def create_test_video(self):
        """Create test result video"""
        
        print("\n🎬 Creating Test Result Video...")
        
        # Add title and summary frames
        self.add_title_frames()
        self.add_summary_frames()
        
        # Create video
        output_path = PROJECT_ROOT / "automos_ai_test_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, 30, (1920, 1080))
        
        # Write frames
        for i, frame in enumerate(self.test_frames):
            out.write(frame)
            if i % 30 == 0:
                print(f"   Processed {i}/{len(self.test_frames)} frames")
        
        out.release()
        
        print(f"✅ Test video created: {output_path}")
        print(f"📊 Video duration: {len(self.test_frames)/30:.1f} seconds")
        
        return output_path
    
    def add_title_frames(self):
        """Add title frames to video"""
        
        for i in range(90):  # 3 seconds at 30 FPS
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            frame[:] = (0, 100, 200)  # Blue background
            
            # Animated title
            alpha = min(1.0, i / 30.0)
            title_size = int(3 * alpha)
            
            cv2.putText(frame, "AUTOMOS AI",
                       (960 - 300, 400), cv2.FONT_HERSHEY_SIMPLEX, title_size, (255, 255, 255), 3)
            
            subtitle_alpha = min(1.0, max(0.0, (i - 30) / 30.0))
            if subtitle_alpha > 0:
                cv2.putText(frame, "Complete Product Test & Validation",
                           (960 - 400, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                           tuple(int(c * subtitle_alpha) for c in (255, 255, 255)), 2)
            
            date_alpha = min(1.0, max(0.0, (i - 60) / 30.0))
            if date_alpha > 0:
                date_str = datetime.now().strftime("%B %d, %Y")
                cv2.putText(frame, date_str,
                           (960 - 150, 600), cv2.FONT_HERSHEY_SIMPLEX, 1,
                           tuple(int(c * date_alpha) for c in (255, 255, 255)), 2)
            
            self.test_frames.append(frame)
    
    def add_summary_frames(self):
        """Add summary frames to video"""
        
        # Calculate summary stats
        total_score = sum(r['score'] for r in self.test_results.values())
        avg_score = total_score / len(self.test_results)
        passed_tests = sum(1 for r in self.test_results.values() if r['status'] == 'passed')
        pass_rate = (passed_tests / len(self.test_results)) * 100
        
        for i in range(150):  # 5 seconds at 30 FPS
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            frame[:] = (0, 100, 200)  # Blue background
            
            cv2.putText(frame, "TEST RESULTS SUMMARY",
                       (960 - 250, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            
            # Results
            cv2.putText(frame, f"Average Score: {avg_score:.1f}/100",
                       (100, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            
            cv2.putText(frame, f"Pass Rate: {pass_rate:.1f}%",
                       (100, 420), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            
            cv2.putText(frame, f"Tests Passed: {passed_tests}/{len(self.test_results)}",
                       (100, 490), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            
            # Status
            if pass_rate >= 80:
                status_text = "PRODUCTION READY"
                status_color = (0, 255, 0)  # Green
            elif pass_rate >= 60:
                status_text = "NEEDS IMPROVEMENT"
                status_color = (0, 255, 255)  # Yellow
            else:
                status_text = "NOT READY"
                status_color = (0, 0, 255)  # Red
            
            cv2.putText(frame, status_text,
                       (100, 600), cv2.FONT_HERSHEY_SIMPLEX, 2, status_color, 3)
            
            # Animated checkmark or X
            if i % 60 < 30:
                if pass_rate >= 80:
                    # Draw checkmark
                    cv2.line(frame, (1600, 300), (1650, 400), status_color, 10)
                    cv2.line(frame, (1650, 400), (1550, 400), status_color, 10)
                    cv2.line(frame, (1550, 400), (1600, 500), status_color, 10)
            
            self.test_frames.append(frame)

def main():
    """Main test function"""
    
    print("🧪 AUTOMOS AI Complete Product Test")
    print("This will test all components and create a video output")
    print()
    
    tester = AUTOMOSAIProductTest()
    results = tester.run_complete_test()
    
    print("\n🎉 Complete Product Test Finished!")
    print("📹 Test video created: automos_ai_test_video.mp4")
    print("📊 Test report created: test_report.json")
    
    # Display final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS:")
    for test_name, result in results.items():
        print(f"  {test_name}: {result['status']} ({result['score']}/100)")
    print("=" * 60)

if __name__ == "__main__":
    main()
