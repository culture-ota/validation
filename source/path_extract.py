#!/usr/bin/env python

import rospy
import numpy as np
import matplotlib.pyplot as plt
from nav_msgs.msg import OccupancyGrid

received = False  # 전역 변수로 한 번만 처리

def occupancy_grid_callback(msg):
    global received
    if received:
        return  # 한 번만 실행

    width = msg.info.width
    height = msg.info.height
    data = np.array(msg.data).reshape((height, width))

    image = np.zeros((height, width), dtype=np.uint8)
    image[data == -1] = 127   # unknown → 회색
    image[data == 0] = 255    # free → 흰색
    image[data > 0] = 0       # occupied → 검정색

    image = np.flipud(image)  # y축 뒤집기

    # 이미지 저장 (GUI 없이도 동작)
    plt.imsave("occupancy_z0_map.png", image, cmap='gray')

    rospy.loginfo("지도 저장 완료: occupancy_z0_map.png")
    received = True

def listener():
    rospy.init_node('map_visualizer', anonymous=True)
    rospy.Subscriber('/map', OccupancyGrid, occupancy_grid_callback)

    # ROS 이벤트 루프
    rospy.spin()

if __name__ == '__main__':
    listener()
