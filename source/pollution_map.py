#!/usr/bin/env python
import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid
import cv2
import random

# 오염도 색상 매핑: 레벨별 RGB
POLLUTION_COLORS = {
    1: (255, 255, 180),  # 연한 누런색
    2: (255, 230, 150),
    3: (255, 200, 120),
    4: (220, 160, 90),
    5: (180, 100, 60),   # 갈색
}
# 회색 배경
BACKGROUND_COLOR = (200, 200, 200)
EMPTY_COLOR = (255, 255, 255)

# 레벨 확률 분포 (1~5)
POLLUTION_LEVELS = [1, 2, 3, 4, 5]
POLLUTION_PROBS = [0.35, 0.3, 0.2, 0.1, 0.05]  # 낮은 레벨일수록 더 자주 발생

def sample_pollution_level():
    return np.random.choice(POLLUTION_LEVELS, p=POLLUTION_PROBS)

def generate_pollution_image(occupancy_data, width, height):
    image = np.ones((height, width, 3), dtype=np.uint8)
    image[:, :] = BACKGROUND_COLOR  # 전체 회색 배경

    for y in range(height):
        for x in range(width):
            idx = y * width + x
            val = occupancy_data[idx]
            if val >= 0:
                if val != 0:
                    image[y, x] = EMPTY_COLOR
                else:
                    pollution_level = sample_pollution_level()  # ✅ 비균등 샘플링
                    rgb = POLLUTION_COLORS[pollution_level]
                    image[y, x] = (rgb[2], rgb[1], rgb[0])  # BGR로 변환
            # else: -1 → 그대로 회색

    image = cv2.flip(image, 0)  # ROS 좌표계 반전
    return image

def save_legend_image():
    import matplotlib.pyplot as plt

    levels = [1, 2, 3, 4, 5]
    colors_rgb = [tuple(c_i / 255.0 for c_i in POLLUTION_COLORS[lvl]) for lvl in levels]  # ⬅ RGB 그대로

    fig, ax = plt.subplots(figsize=(4, 2.5))
    for i, (level, color) in enumerate(zip(levels, colors_rgb)):
        ax.add_patch(plt.Rectangle((0, i), 1, 1, color=color))
        ax.text(1.1, i + 0.5, f"Level {level}", va='center', fontsize=12)

    ax.set_xlim(0, 2)
    ax.set_ylim(0, len(levels))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    plt.tight_layout()
    save_path = "/home/e2map/clean_map/pollution_legend.png"
    plt.savefig(save_path)
    plt.close()
    rospy.loginfo(f"🎨 Legend image saved to: {save_path}")

def occupancy_callback(msg):
    width = msg.info.width
    height = msg.info.height
    occupancy_data = msg.data

    rospy.loginfo("🖼️ Generating pollution map image...")

    pollution_img = generate_pollution_image(occupancy_data, width, height)
    save_path = "/home/e2map/clean_map/pollution_map.png"
    cv2.imwrite(save_path, pollution_img)
    rospy.loginfo(f"✅ Saved image to: {save_path}")
    save_legend_image()  # ← 여기에 추가!


def main():
    rospy.init_node("pollution_image_generator")
    rospy.Subscriber("/map", OccupancyGrid, occupancy_callback)
    rospy.loginfo("📡 Listening to /map and waiting for OccupancyGrid data...")
    rospy.spin()

if __name__ == '__main__':
    main()
