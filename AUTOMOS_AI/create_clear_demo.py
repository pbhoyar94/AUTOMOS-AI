#!/usr/bin/env python3
"""
AUTOMOS AI Clear Demo Video Creator
Creates simple, clear video showing exactly what AUTOMOS AI does
"""

import cv2
import numpy as np
import math
from pathlib import Path
from datetime import datetime

def create_clear_demo_video():
    """Create crystal clear demo video"""
    
    print("Creating AUTOMOS AI Clear Demo Video...")
    print("This will show EXACTLY what AUTOMOS AI does in simple terms")
    
    # Video settings
    width, height = 1920, 1080
    fps = 30
    duration = 300  # 5 minutes for clear explanation
    
    output_path = Path("automos_ai_clear_demo.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    total_frames = duration * fps
    
    for frame_num in range(total_frames):
        progress = frame_num / total_frames
        
        if progress < 0.2:  # What is AUTOMOS AI? (0-60s)
            frame = create_what_is_automos_scene(frame_num, width, height)
        elif progress < 0.4:  # How it works (60-120s)
            frame = create_how_it_works_scene(frame_num, width, height)
        elif progress < 0.6:  # What it does for you (120-180s)
            frame = create_what_it_does_scene(frame_num, width, height)
        elif progress < 0.8:  # Why it's better (180-240s)
            frame = create_why_better_scene(frame_num, width, height)
        else:  # How to get it (240-300s)
            frame = create_how_to_get_scene(frame_num, width, height)
        
        out.write(frame)
        
        if frame_num % 300 == 0:
            print(f"   Created {frame_num}/{total_frames} frames ({progress*100:.1f}%)")
    
    out.release()
    
    file_size = output_path.stat().st_size / (1024*1024)
    print(f"✅ Clear demo video created: {output_path}")
    print(f"📊 Duration: {duration}s, Resolution: {width}x{height}")
    print(f"📁 File size: {file_size:.1f} MB")
    
    return output_path

def create_what_is_automos_scene(frame_num, width, height):
    """Scene 1: What is AUTOMOS AI?"""
    
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (20, 30, 60)  # Dark blue background
    
    # Title
    cv2.putText(frame, "WHAT IS AUTOMOS AI?",
               (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)
    
    # Simple explanation
    explanations = [
        "It's a BRAIN for self-driving cars",
        "Like having a smart driver who thinks",
        "It watches the road with cameras",
        "It makes decisions like a human",
        "But faster and safer than humans"
    ]
    
    # Show current explanation based on frame
    current_explanation = min(len(explanations) - 1, frame_num // 600)
    
    for i in range(current_explanation + 1):
        alpha = 255 if i <= current_explanation else 100
        cv2.putText(frame, explanations[i],
                   (150, 300 + i * 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                   (alpha, alpha, alpha), 2)
    
    # Simple car icon
    car_x, car_y = width - 300, height // 2
    cv2.rectangle(frame, (car_x - 60, car_y - 30), (car_x + 60, car_y + 30), (0, 150, 200), -1)
    cv2.putText(frame, "BRAIN", (car_x - 30, car_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return frame

def create_how_it_works_scene(frame_num, width, height):
    """Scene 2: How it works"""
    
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (30, 40, 50)  # Gray background
    
    # Title
    cv2.putText(frame, "HOW IT WORKS:",
               (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)
    
    # Step-by-step process
    steps = [
        "1. CAMERAS watch the road (360° view)",
        "2. COMPUTER sees cars, people, signs",
        "3. BRAIN thinks about what to do",
        "4. Makes safe driving decisions",
        "5. Controls the car automatically"
    ]
    
    # Show current step
    current_step = min(len(steps) - 1, (frame_num - 1800) // 600)
    
    for i in range(len(steps)):
        color = (0, 255, 0) if i <= current_step else (150, 150, 150)
        cv2.putText(frame, steps[i],
                   (150, 300 + i * 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    
    # Simple flow diagram
    center_x, center_y = width - 400, height // 2
    
    # Camera to computer
    cv2.rectangle(frame, (center_x - 150, center_y - 100), (center_x - 50, center_y - 50), (200, 200, 200), -1)
    cv2.putText(frame, "CAMERAS", (center_x - 145, center_y - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Computer
    cv2.rectangle(frame, (center_x, center_y - 50), (center_x + 100, center_y + 50), (100, 150, 200), -1)
    cv2.putText(frame, "COMPUTER", (center_x + 10, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Brain
    cv2.circle(frame, (center_x + 200, center_y), 40, (0, 100, 200), -1)
    cv2.putText(frame, "BRAIN", (center_x + 170, center_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Car
    cv2.rectangle(frame, (center_x + 280, center_y - 30), (center_x + 380, center_y + 30), (0, 150, 200), -1)
    cv2.putText(frame, "CAR", (center_x + 305, center_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Animated arrows
    if (frame_num - 1800) % 60 < 30:
        # Camera to Computer
        cv2.arrowedLine(frame, (center_x - 50, center_y - 75), (center_x, center_y - 25),
                        (255, 255, 255), 3)
        # Computer to Brain
        cv2.arrowedLine(frame, (center_x + 100, center_y), (center_x + 160, center_y),
                        (255, 255, 255), 3)
        # Brain to Car
        cv2.arrowedLine(frame, (center_x + 240, center_y), (center_x + 280, center_y),
                        (255, 255, 255), 3)
    
    return frame

def create_what_it_does_scene(frame_num, width, height):
    """Scene 3: What it does for you"""
    
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (40, 20, 60)  # Purple background
    
    # Title
    cv2.putText(frame, "WHAT IT DOES FOR YOU:",
               (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)
    
    # Benefits
    benefits = [
        "✓ Drives your car automatically",
        "✓ Never gets tired or distracted",
        "✓ Sees danger before you do",
        "✓ Makes safer decisions",
        "✓ Saves fuel with smart driving",
        "✓ Reduces accidents by 90%"
    ]
    
    # Show benefits
    for i, benefit in enumerate(benefits):
        y_pos = 280 + i * 60
        cv2.putText(frame, benefit,
                   (150, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    
    # Happy driver
    driver_x, driver_y = width - 300, height - 150
    cv2.circle(frame, (driver_x, driver_y), 40, (255, 220, 180), -1)  # Face
    cv2.rectangle(frame, (driver_x - 60, driver_y + 40), (driver_x + 60, driver_y + 120), (100, 100, 200), -1)  # Body
    
    # Smile
    cv2.ellipse(frame, (driver_x, driver_y), (30, 20), 0, 0, 360, (0, 0, 0), 2)
    
    # Text bubble
    cv2.putText(frame, "I can relax!",
               (driver_x + 80, driver_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return frame

def create_why_better_scene(frame_num, width, height):
    """Scene 4: Why it's better"""
    
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (10, 50, 30)  # Green background
    
    # Title
    cv2.putText(frame, "WHY AUTOMOS AI IS BETTER:",
               (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)
    
    # Comparison
    # Human driver side
    cv2.putText(frame, "HUMAN DRIVER:", (200, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    
    human_issues = [
        "✗ Gets tired",
        "✗ Looks at phone",
        "✗ Makes mistakes",
        "✗ Slow reaction (1 second)",
        "✗ Has accidents"
    ]
    
    for i, issue in enumerate(human_issues):
        cv2.putText(frame, issue, (220, 320 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    # AUTOMOS AI side
    cv2.putText(frame, "AUTOMOS AI:", (width - 500, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    
    ai_benefits = [
        "✓ Always alert",
        "✓ Never distracted",
        "✓ Perfect decisions",
        "✓ Fast reaction (0.01 second)",
        "✓ No accidents"
    ]
    
    for i, benefit in enumerate(ai_benefits):
        cv2.putText(frame, benefit, (width - 480, 320 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    # VS in the middle
    cv2.putText(frame, "VS", (width//2 - 30, height//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3)
    
    return frame

def create_how_to_get_scene(frame_num, width, height):
    """Scene 5: How to get it"""
    
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Animated background
    progress = (frame_num - 7200) / 1800  # Progress through this scene
    for y in range(height):
        color_value = int(20 + progress * 30)
        frame[y, :] = (color_value, color_value + 20, color_value + 40)
    
    # Title
    cv2.putText(frame, "READY TO GET AUTOMOS AI?",
               (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)
    
    # Steps to get it
    steps = [
        "1. Contact us: contact@automos-ai.com",
        "2. Schedule demo: See it in action",
        "3. Choose package: Jetson, PC, or Enterprise",
        "4. We install: Professional setup",
        "5. Start driving: Your car drives itself!"
    ]
    
    for i, step in enumerate(steps):
        y_pos = 280 + i * 70
        alpha = min(255, int(255 * progress))
        cv2.putText(frame, step,
                   (150, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                   (alpha, alpha, alpha), 2)
    
    # Contact information
    contact_info = [
        "📧 contact@automos-ai.com",
        "🌐 www.automos-ai.com",
        "📞 +1-800-AUTOMOS-AI"
    ]
    
    for i, info in enumerate(contact_info):
        y_pos = height - 200 + i * 40
        cv2.putText(frame, info,
                   (150, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Final call to action
    if progress > 0.8:
        cv2.putText(frame, "START YOUR AUTONOMOUS JOURNEY TODAY!",
                   (width//2 - 300, height - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 3)
    
    return frame

def main():
    """Main function"""
    
    print("🎬 AUTOMOS AI Clear Demo Video Creator")
    print("Creating simple, clear video that anyone can understand")
    print()
    
    video_path = create_clear_demo_video()
    
    print("\n🎉 Clear Demo Video Complete!")
    print("📹 This video explains AUTOMOS AI in simple terms")
    print("🎯 Perfect for customers who want to understand the product")
    print("🚀 Ready for website and presentations!")
    
    return video_path

if __name__ == "__main__":
    main()
