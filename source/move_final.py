#!/usr/bin/env python
import rospy
import math
import time
import threading
from geometry_msgs.msg import Twist, Wrench, Point
from gazebo_msgs.srv import GetModelState, ApplyBodyWrench, ApplyBodyWrenchRequest
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
import sys
import select



# === 전역 설정 ===
model_name = 'clean_e2map'
cmd_vel_topic = '/cmd_vel'
goal_lock = threading.Lock()
goal_version = 0
traj_points = []

target_path_pub = None
total_force_count = 0  # ✅ stuck 해결을 위해 force 적용된 횟수

def get_yaw_from_orientation(orientation):
    import tf.transformations
    q = [orientation.x, orientation.y, orientation.z, orientation.w]
    _, _, yaw = tf.transformations.euler_from_quaternion(q)
    return yaw

def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

def publish_path_marker(points, publisher, ns, r, g, b, marker_id=0):
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()
    marker.ns = ns
    marker.id = marker_id     # ✅ 수정: id를 인자로 받도록
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.1
    marker.color = ColorRGBA(r, g, b, 1.0)
    marker.points = points
    marker.lifetime = rospy.Duration(0)  # 무제한 유지

    publisher.publish(marker)


def calculate_total_distance(path):
    total_distance = 0.0
    for i in range(1, len(path)):
        x1, y1 = path[i-1]
        x2, y2 = path[i]
        total_distance += math.hypot(x2 - x1, y2 - y1)
    return total_distance

def rotate_and_move(x_goal, y_goal, my_version):
    global robot_traj_pub, total_force_count

    rospy.wait_for_service('/gazebo/get_model_state')
    get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    apply_force = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)
    vel_pub = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)
    rate = rospy.Rate(1)

    stuck_counter = 0
    prev_pos = None
    is_turning = False
    start_time = time.time()  # ✅ 목표 도달 시간 측정 시작

    while not rospy.is_shutdown():
        # ⏱️ 목표 도달 시간 초과 시 skip
        if time.time() - start_time > 5.0:
            rospy.logwarn("⏰ 목표 도달 시간 초과! 다음 목표로 이동")
            vel_pub.publish(Twist())
            break

        with goal_lock:
            if my_version != goal_version:
                rospy.logwarn("🛑 이전 목표 취소됨 (새 좌표 우선)")
                vel_pub.publish(Twist())
                return

        resp = get_state(model_name, 'world')
        x_now = resp.pose.position.x
        y_now = resp.pose.position.y
        yaw_now = get_yaw_from_orientation(resp.pose.orientation)

        dx = x_goal - x_now
        dy = y_goal - y_now
        distance = math.hypot(dx, dy)
        angle_to_target = math.atan2(dy, dx)
        angle_diff = normalize_angle(angle_to_target - yaw_now)

        traj_points.append(Point(x_now, y_now, 0.0))
        publish_path_marker(traj_points, robot_traj_pub, "robot_path", 0.0, 0.0, 1.0, marker_id=1)

        if prev_pos is not None:
            moved = math.hypot(x_now - prev_pos[0], y_now - prev_pos[1])
            if moved < 0.01:
                stuck_counter += 1
            else:
                stuck_counter = 0
        prev_pos = (x_now, y_now)

        if distance < 0.1:
            vel_pub.publish(Twist())
            rospy.loginfo("✅ 목표 지점 도달!")
            break

        if stuck_counter > 5:
            rospy.logwarn("⚠️ Stuck 상태 감지됨. 강제 Force 적용 반복 시작...")
            force_magnitude = 50
            max_force = 500
            success = False

            while force_magnitude <= max_force:
                rospy.loginfo(f"💥 정방향 force {force_magnitude}N 적용 중...")
                resp_before = get_state(model_name, 'world')
                x_before, y_before = resp_before.pose.position.x, resp_before.pose.position.y

                wrench = Wrench()
                wrench.force.x = force_magnitude * math.cos(yaw_now)
                wrench.force.y = force_magnitude * math.sin(yaw_now)

                req = ApplyBodyWrenchRequest()
                req.body_name = model_name + "::dummy_base"
                req.reference_frame = "world"
                req.wrench = wrench
                req.duration = rospy.Duration(0.5)
                req.start_time = rospy.Time.now()

                apply_force(req)
                rospy.sleep(1.0)

                resp_after = get_state(model_name, 'world')
                x_after, y_after = resp_after.pose.position.x, resp_after.pose.position.y
                moved_distance = math.hypot(x_after - x_before, y_after - y_before)

                rospy.loginfo("📏 정방향 이동 거리: %.4f m", moved_distance)
                if moved_distance >= 0.1:
                    total_force_count += 1
                    success = True
                    break

                rospy.logwarn("↩️ 반대 방향 force 적용")
                reverse_wrench = Wrench()
                reverse_wrench.force.x = -force_magnitude * math.cos(yaw_now)
                reverse_wrench.force.y = -force_magnitude * math.sin(yaw_now)

                req.wrench = reverse_wrench
                apply_force(req)
                rospy.sleep(1.0)

                resp_after = get_state(model_name, 'world')
                x_after, y_after = resp_after.pose.position.x, resp_after.pose.position.y
                moved_distance = math.hypot(x_after - x_before, y_after - y_before)

                rospy.loginfo("📏 반대방향 이동 거리: %.4f m", moved_distance)
                if moved_distance >= 0.1:
                    total_force_count += 1
                    success = True
                    break

                force_magnitude += 50
                rospy.logwarn(f"⚠️ 이동 실패. force 증가 → {force_magnitude}N")

            if not success:
                rospy.logerr("❗ 최대 force까지 적용했지만 이동 실패")
            stuck_counter = 0

        vel = Twist()
        if is_turning:
            if abs(angle_diff) < 0.15:
                is_turning = False
                vel.linear.x = 0.3
                vel.angular.z = 0.0
                rospy.logwarn("🚗 직진중")
            else:
                vel.angular.z = 0.5 * angle_diff
                vel.linear.x = 0.0
                rospy.logwarn("🌀 회전중")
        else:
            if abs(angle_diff) > 0.3:
                is_turning = True
                vel.angular.z = 0.5 * angle_diff
                vel.linear.x = 0.0
                rospy.logwarn("🌀 회전중")
            else:
                vel.linear.x = 0.3
                vel.angular.z = 0.0
                rospy.logwarn("🚗 직진중")


        try:
            # 기존 코드
            vel_pub.publish(vel)
            rate.sleep()
        except KeyboardInterrupt:
            rospy.logwarn("🔴 이동 중 Ctrl+C 감지됨, 중단합니다.")
            break


def clear_marker(publisher, ns, marker_id):
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()
    marker.ns = ns
    marker.id = marker_id
    marker.action = Marker.DELETE
    publisher.publish(marker)


def read_path_from_txt(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    path = []
    for line in lines:
        x_str, y_str = line.strip().split(',')
        path.append((float(x_str), float(y_str)))
    return path

def save_trajectory_to_txt(points, filepath="robot_trajectory.txt"):
    with open(filepath, 'w') as f:
        for p in points:
            f.write(f"{p.x:.4f}, {p.y:.4f}\n")
    rospy.loginfo(f"✅ Trajectory 저장 완료: {filepath}")

def wait_for_key_or_timeout(timeout=0.5):
    """timeout 내에 키 입력을 기다림 (non-blocking 방식)"""
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    if rlist:
        return sys.stdin.readline()
    return None

def main():
    global robot_traj_pub, target_path_pub, total_force_count
    rospy.init_node('irobot_path_follow', anonymous=True)



    robot_traj_pub = rospy.Publisher('/robot_path_marker', Marker, queue_size=1)
    target_path_pub = rospy.Publisher('/target_path_marker', Marker, queue_size=1)
    rospy.sleep(1.0)
    # clear_marker(robot_traj_pub, "robot_path", 1)
    # clear_marker(target_path_pub, "target_path", 0)

    path = read_path_from_txt("/home/e2map/clean_map/ikea_1_path_1.txt")
    rospy.loginfo(f"🔁 총 {len(path)}개 포인트 경로 이동 시작")

    total_distance = calculate_total_distance(path)
    path_points = [Point(x, y, 0.0) for x, y in path]
    publish_path_marker(path_points, target_path_pub, "target_path", 135 / 255.0, 206 / 255.0, 250 / 255.0, marker_id=0)

    global goal_version
    start_time = time.time()
    i = 1

    while i < len(path):
        x, y = path[i]
        with goal_lock:
            goal_version += 1
            this_version = goal_version

        rospy.loginfo(f"📍 [{i+1}/{len(path)}] 이동 중: x={x:.2f}, y={y:.2f}")
        rotate_and_move(x, y, this_version)
        time.sleep(1.0)

        key = wait_for_key_or_timeout()
        if key is not None:
            i += 5
        else:
            i += 1

    end_time = time.time()
    elapsed_time = end_time - start_time

    rospy.loginfo("🎉 전체 경로 완료!")
    rospy.loginfo(f"🕒 총 소요 시간: {elapsed_time:.2f} 초")
    rospy.loginfo(f"📏 전체 이동 거리: {total_distance:.2f} m")
    rospy.loginfo(f"Stuck 횟수: {total_force_count} 회")

    save_trajectory_to_txt(traj_points)

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("🛑 ROS Interrupt 발생, 종료합니다.")
    except KeyboardInterrupt:
        rospy.loginfo("🔴 사용자 강제 종료(Ctrl+C) 감지됨, 안전 종료합니다.")
