#!/usr/bin/env python3
"""
AUTOMOS AI Demo Video Creator
Creates a demonstration video showing product capabilities
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

def create_automos_demo_video():
    """Create AUTOMOS AI demo video"""
    
    print("Creating AUTOMOS AI Demo Video...")
    
    # Video settings
    width, height = 1920, 1080
    fps = 30
    duration = 180  # 3 minutes
    
    # Create video writer
    output_path = Path("automos_ai_demo_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Generate frames
    total_frames = duration * fps
    
    for frame_num in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Blue background
        frame[:] = (0, 100, 200)
        
        # Calculate progress through video
        progress = frame_num / total_frames
        
        if progress < 0.2:  # Title section (0-36s)
            # Title animation
            alpha = progress * 5  # 0 to 1
            title_size = int(3 * alpha)
            
            cv2.putText(frame, "AUTOMOS AI",
                       (width//2 - 250, height//2 - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, title_size, (255, 255, 255), 3)
            
            subtitle_alpha = max(0, (progress - 0.1) * 10)
            if subtitle_alpha > 0:
                cv2.putText(frame, "World's First Reasoning-Based Autonomous Driving",
                           (width//2 - 450, height//2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 1,
                           tuple(int(c * subtitle_alpha) for c in (255, 255, 255)), 2)
        
        elif progress < 0.4:  # Features section (36-72s)
            # Feature showcase
            feature_idx = int((progress - 0.2) * 5) % 7
            features = [
                "360° Multi-Camera Processing",
                "Safety-Critic System (<10ms)",
                "Language-Conditioned Control", 
                "Real-Time World Model",
                "Edge Optimization",
                "HD-Map-Free Navigation",
                "Social Intent Recognition"
            ]
            
            cv2.putText(frame, "KEY FEATURES",
                       (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            
            cv2.putText(frame, features[feature_idx],
                       (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)
            
            # Feature description
            descriptions = [
                "Advanced vision backbone for comprehensive perception",
                "Industry-leading emergency response time",
                "Natural language control interface",
                "Predictive environmental modeling",
                "2.5x performance improvement on edge",
                "Navigation without pre-existing maps",
                "Human behavior understanding"
            ]
            
            desc = descriptions[feature_idx]
            words = desc.split()
            lines = []
            current_line = []
            for word in words:
                current_line.append(word)
                if len(' '.join(current_line)) > 40:
                    lines.append(' '.join(current_line))
                    current_line = []
            if current_line:
                lines.append(' '.join(current_line))
            
            for line_idx, line in enumerate(lines):
                cv2.putText(frame, line,
                           (100, 350 + line_idx * 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Progress bar
            bar_progress = (progress - 0.2) * 5
            bar_width = int(600 * bar_progress)
            cv2.rectangle(frame, (100, 700), (700, 730), (255, 255, 255), 2)
            cv2.rectangle(frame, (100, 700), (100 + bar_width, 730), (0, 255, 0), -1)
            
            cv2.putText(frame, f"Feature {feature_idx + 1}/7",
                       (100, 770), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        elif progress < 0.6:  # Technology section (72-108s)
            # Technology showcase
            cv2.putText(frame, "TECHNOLOGY STACK",
                       (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            
            tech_items = [
                "DualAD + OpenDriveVLA Integration",
                "Multi-Sensor Fusion (Camera/Radar/LiDAR)",
                "LLM-Powered Reasoning Engine",
                "Real-Time Safety Monitoring",
                "Edge-Optimized Models"
            ]
            
            for i, tech in enumerate(tech_items):
                y_pos = 250 + i * 60
                cv2.putText(frame, tech,
                           (100, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            # Animated tech icon
            angle = (frame_num * 3) % 360
            center_x, center_y = width - 200, height // 2
            radius = 80
            
            cv2.circle(frame, (center_x, center_y), radius, (255, 255, 255), 2)
            end_x = int(center_x + radius * np.cos(np.radians(angle)))
            end_y = int(center_y + radius * np.sin(np.radians(angle)))
            cv2.line(frame, (center_x, center_y), (end_x, end_y), (0, 255, 255), 3)
        
        elif progress < 0.8:  # Performance section (108-144s)
            # Performance metrics
            cv2.putText(frame, "PERFORMANCE METRICS",
                       (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            
            metrics = [
                ("Safety Response", "<10ms", "✅"),
                ("Processing Speed", "22-83 FPS", "✅"),
                ("Model Size", "70% Smaller", "✅"),
                ("Power Efficiency", "2.5x Better", "✅"),
                ("Multi-Device", "Jetson/PC/RPi", "✅")
            ]
            
            for i, (metric, value, status) in enumerate(metrics):
                y_pos = 250 + i * 80
                cv2.putText(frame, metric,
                           (100, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                cv2.putText(frame, value,
                           (100, y_pos + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, status,
                           (500, y_pos + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        else:  # Call to action (144-180s)
            # Final call to action
            cv2.putText(frame, "AUTOMOS AI",
                       (width//2 - 250, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
            
            cv2.putText(frame, "Ready for Customer Deployment",
                       (width//2 - 350, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            
            # Contact info
            contact_info = [
                "📧 contact@automos-ai.com",
                "🌐 www.automos-ai.com", 
                "📞 +1-800-AUTOMOS-AI"
            ]
            
            for i, info in enumerate(contact_info):
                y_pos = 450 + i * 50
                cv2.putText(frame, info,
                           (width//2 - 200, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Final branding
            cv2.putText(frame, "The Future of Autonomous Driving",
                       (width//2 - 300, 650), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp,
                   (50, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add progress indicator
        overall_progress = frame_num / total_frames
        bar_width = int(800 * overall_progress)
        cv2.rectangle(frame, (50, height - 100), (850, height - 70), (255, 255, 255), 2)
        cv2.rectangle(frame, (50, height - 100), (50 + bar_width, height - 70), (0, 255, 0), -1)
        
        # Write frame
        out.write(frame)
        
        # Progress update
        if frame_num % 300 == 0:
            print(f"Processed {frame_num}/{total_frames} frames ({overall_progress*100:.1f}%)")
    
    # Release video writer
    out.release()
    
    print(f"✅ Demo video created: {output_path}")
    print(f"📊 Duration: {duration} seconds")
    print(f"🎬 Resolution: {width}x{height} @ {fps} FPS")
    print(f"📁 File size: {output_path.stat().st_size / (1024*1024):.1f} MB")
    
    return output_path

if __name__ == "__main__":
    try:
        video_path = create_automos_demo_video()
        print("\n🎉 AUTOMOS AI Demo Video Complete!")
        print("📹 Video ready for customer presentation!")
    except Exception as e:
        print(f"❌ Error creating video: {e}")
