#!/usr/bin/env python
import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

def read_obstacles_with_size(file_path):
    obstacles = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 5:
                name = parts[0].strip()
                x = float(parts[1].strip())
                y = float(parts[2].strip())
                width = float(parts[3].strip())
                height = float(parts[4].strip())
                obstacles.append((name, x, y, width, height))
    return obstacles

def publish_rectangle_markers(obstacles):
    rospy.init_node('obstacle_rect_marker_publisher', anonymous=True)
    pub = rospy.Publisher('/obstacle_rectangles', Marker, queue_size=10)
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        for idx, (name, x, y, width, height) in enumerate(obstacles):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "obstacles"
            marker.id = idx
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.15  # 살짝 띄우기
            marker.pose.orientation.w = 1.0
            marker.scale.x = width
            marker.scale.y = height
            marker.scale.z = 0.3  # 고정 높이
            marker.color.r = 1.0
            marker.color.g = 0.4
            marker.color.b = 0.2
            marker.color.a = 0.9
            pub.publish(marker)
        rate.sleep()

if __name__ == '__main__':
    file_path = '/home/e2map/clean_map/obstacle_2.txt'  # ✏️ 경로 수정 가능
    obstacles = read_obstacles_with_size(file_path)
    publish_rectangle_markers(obstacles)
