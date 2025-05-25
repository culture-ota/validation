#!/usr/bin/env python
import rospy
import numpy as np
import cv2
from nav_msgs.msg import OccupancyGrid
from scipy.spatial.distance import cdist
from itertools import permutations
from collections import defaultdict
import faulthandler
faulthandler.enable()


import cv2
import numpy as np

def save_path_mp4_overlay(path, background_img, save_path='final_path.mp4',
                          points_per_second=10, fps=10, color=(255, 0, 0), flip_y=True):
    height, width, _ = background_img.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    total_points = len(path)
    step = max(1, int(fps / points_per_second))  # 예: 10fps, 10pt/s → step=1

    for i in range(2, total_points + 1, step):
        frame = background_img.copy()

        for j in range(1, i):
            pt1 = path[j - 1]
            pt2 = path[j]

            if flip_y:
                pt1 = (pt1[0], height - 1 - pt1[1])
                pt2 = (pt2[0], height - 1 - pt2[1])

            if all(0 <= v < width for v in [pt1[0], pt2[0]]) and all(0 <= v < height for v in [pt1[1], pt2[1]]):
                cv2.line(frame, pt1, pt2, color, thickness=1)

        video_writer.write(frame)

    # 마지막 장면 정지 1초
    for _ in range(fps):
        video_writer.write(frame)

    video_writer.release()
    print(f"🎥 MP4 저장 완료: {save_path}")




def generate_zigzag_path_from_mask(mask, resolution=0.05, interval_m=0.3, min_length_m=0.5, edge_margin_m=0.1):
    height, width = mask.shape
    interval_px = int(interval_m / resolution)
    min_length_px = int(min_length_m / resolution)
    margin_px = int(edge_margin_m / resolution)

    all_lines = []
    path = []
    bridges = []

    for y in range(margin_px, height - margin_px, interval_px):
        row = mask[y]
        x = 0
        while x < width:
            if row[x] == 255:
                x_start = x
                while x < width and row[x] == 255:
                    x += 1
                x_end = x - 1

                length = x_end - x_start + 1
                if length >= min_length_px and length > 2 * margin_px:
                    new_start = x_start + margin_px
                    new_end = x_end - margin_px
                    if new_end > new_start:
                        x_range = range(new_start, new_end + 1)
                        if (y // interval_px) % 2 == 1:
                            x_range = reversed(x_range)
                        line = [(xi, y) for xi in x_range]

                        valid = all(0 <= xi < width and 0 <= y < height and mask[y, xi] == 255 for xi, y in line)
                        if valid:
                            if len(all_lines) > 0:
                                prev_xs = set(x for x, _ in all_lines[-1])
                                curr_xs = set(x for x, _ in line)
                                prev_y = all_lines[-1][0][1]
                                if len(prev_xs & curr_xs) == 0 or abs(y - prev_y) > interval_px:
                                    # 👉 이전 줄과 연결 안 되면 다음 후보 line으로 계속 탐색
                                    x = x_end + 1  # 다시 x 시작
                                    continue

                            # ✅ 유효하고 이전 줄과도 연결되면 append하고 다음 y로 넘어감
                            all_lines.append(line)
                            break  # 다음 y로
            x += 1  # 흰색일 경우 다음 x로
    for i in range(len(all_lines)):
        path += all_lines[i]
        if i < len(all_lines) - 1:
            last_pt = all_lines[i][-1]
            next_pt = all_lines[i + 1][0]
            if abs(next_pt[1] - last_pt[1]) <= interval_px:
                bridge = get_line_within_mask(last_pt, next_pt, mask)
                if bridge:
                    bridges.append(bridge)
                    path += bridge

    return path, bridges


def get_line_within_mask(p1, p2, mask):
    x1, y1 = p1
    x2, y2 = p2
    candidate_points = []

    h, w = mask.shape

    def is_valid_path(path):
        for x, y in path:
            if not (0 <= x < w and 0 <= y < h) or mask[y, x] != 255:
                return False
        return True

    # 대각선 우선 시도
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    x, y = x1, y1
    sx = 1 if x2 >= x1 else -1
    sy = 1 if y2 >= y1 else -1
    diagonal_path = []

    if dx > dy:
        err = dx / 2.0
        while x != x2:
            diagonal_path.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y2:
            diagonal_path.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    diagonal_path.append((x2, y2))

    if is_valid_path(diagonal_path):
        return diagonal_path

    # ㄱ자 또는 ㄴ자 경로 시도 (x → y)
    corner_path_1 = [(x1, y1), (x2, y1), (x2, y2)]
    if is_valid_path(corner_path_1):
        return corner_path_1

    # ㄴ자 또는 ㄱ자 경로 시도 (y → x)
    corner_path_2 = [(x1, y1), (x1, y2), (x2, y2)]
    if is_valid_path(corner_path_2):
        return corner_path_2

    return []


def insert_vertical_if_needed(path):
    """줄 간 대각선을 수직+수평으로 보정"""
    if not path:
        return []
    fixed_path = [path[0]]
    for i in range(1, len(path)):
        x1, y1 = fixed_path[-1]
        x2, y2 = path[i]
        if x1 != x2 and y1 != y2:
            fixed_path.append((x1, y2))  # 수직 이동
        fixed_path.append((x2, y2))      # 수평 이동
    return fixed_path

def connect_zigzag_paths_custom(path1, path2, path3, mask):
    full_path = []
    bridges = []

    # ✅ [1] path1의 모든 점 vs path2의 양 끝점
    candidates_1 = path1
    candidates_2 = [path2[0], path2[-1]]
    min_dist = float('inf')
    best_p1, best_p2 = None, None
    reversed_2 = False

    for p1 in candidates_1:
        for p2 in candidates_2:
            dist = np.linalg.norm(np.array(p1) - np.array(p2))
            if dist < min_dist:
                min_dist = dist
                best_p1, best_p2 = p1, p2
                reversed_2 = (p2 == path2[-1])

    idx_p1 = path1.index(best_p1)
    trimmed_path1 = path1[:idx_p1 + 1]
    path1_rest = path1[idx_p1 + 1:]

    trimmed_path2 = path2 if not reversed_2 else path2[::-1]
    bridge_1_2 = get_line_within_mask(best_p1, best_p2, mask)

    full_path = trimmed_path1 + bridge_1_2 + trimmed_path2
    bridges.append(bridge_1_2)

    # ✅ [2] path3 연결: full_path의 모든 점 vs path3 양 끝
    candidates_12 = full_path
    candidates_3 = [path3[0], path3[-1]]
    min_dist = float('inf')
    best_p12, best_p3 = None, None
    reversed_3 = False

    for p1 in candidates_12:
        for p3 in candidates_3:
            dist = np.linalg.norm(np.array(p1) - np.array(p3))
            if dist < min_dist:
                min_dist = dist
                best_p12, best_p3 = p1, p3
                reversed_3 = (p3 == path3[-1])

    trimmed_path3 = path3 if not reversed_3 else path3[::-1]
    bridge_12_3 = get_line_within_mask(best_p12, best_p3, mask)

    full_path += bridge_12_3 + trimmed_path3
    bridges.append(bridge_12_3)

    # ✅ [3] path1_rest가 있다면 이어주기
    if len(path1_rest) >= 2:
        final_tail = full_path[-1]
        full_set = set(full_path)

        # 1. y축별로 path1_rest 그룹핑
        y_groups = defaultdict(list)
        for pt in path1_rest:
            y_groups[pt[1]].append(pt)

        # 2. y축 순서 정렬
        sorted_ys = sorted(y_groups.keys())
        filtered_path1_rest = []

        i = 0
        while i < len(sorted_ys):
            y = sorted_ys[i]
            row = y_groups[y]

            # 중복 제거
            unique_pts = [pt for pt in row if pt not in full_set]

            # x좌표 개수 체크
            if len(unique_pts) > 5:
                filtered_path1_rest.extend(unique_pts)

                # 👉 다음 줄(y가 다른)의 선도 유지 (자동됨)
                i += 1
            else:
                # 현재 줄 스킵 + 다음 줄로 이어지는 선도 삭제
                i += 2  # 두 줄 건너뜀

        # 3. 최종 path1_rest 다시 연결
        if len(path1_rest) >= 2:
            final_tail = full_path[-1]
            full_set = set(full_path)

            # 1. y 기준 그룹핑
            y_groups = defaultdict(list)
            for pt in path1_rest:
                y_groups[pt[1]].append(pt)

            sorted_ys = sorted(y_groups.keys())
            y_to_remove = set()
            next_removals = set()

            # 2. y별 판단
            for i in range(len(sorted_ys)):
                y = sorted_ys[i]
                row = y_groups[y]

                # 중복 제거
                unique_pts = [pt for pt in row if pt not in full_set]

                # 삭제 조건 판단
                if len(unique_pts) <= 5:
                    y_to_remove.add(y)

                    # 👉 다음 y줄에 있는 동일 x값 선 연결도 제거 대상
                    if i + 1 < len(sorted_ys):
                        next_y = sorted_ys[i + 1]
                        for pt in unique_pts:
                            next_removals.add((pt[0], next_y))  # 같은 x, 다음 y

            # 3. 최종 필터링
            filtered_path1_rest = []
            for pt in path1_rest:
                y = pt[1]
                if y in y_to_remove:
                    continue  # 현재 y줄 삭제
                if pt in next_removals:
                    continue  # 다음 y줄로 이어지는 x 선도 삭제
                filtered_path1_rest.append(pt)

            # 4. 연결 수행
            if len(filtered_path1_rest) >= 2:
                candidates_r = [filtered_path1_rest[0], filtered_path1_rest[-1]]
                min_dist = float('inf')
                best_rest = None
                reversed_rest = False

                for pt in candidates_r:
                    dist = np.linalg.norm(np.array(final_tail) - np.array(pt))
                    if dist < min_dist:
                        min_dist = dist
                        best_rest = pt
                        reversed_rest = (pt == filtered_path1_rest[-1])

                bridge_rest = get_line_within_mask(final_tail, best_rest, mask)
                aligned_rest = filtered_path1_rest if not reversed_rest else filtered_path1_rest[::-1]

                full_path += bridge_rest + aligned_rest
                bridges.append(bridge_rest)

    return full_path, bridges


def save_path_to_txt_ros(path, resolution, origin_x, origin_y, save_path='final_path_ros.txt'):
    with open(save_path, 'w') as f:
        for i in range(0, len(path)):  # ✅ 짝수 인덱스만
            x_px, y_px = path[i]
            x_m = x_px * resolution + origin_x
            y_m = y_px * resolution + origin_y
            f.write(f"{x_m:.4f}, {y_m:.4f}\n")
    print(f"✅ ROS 좌표계 경로 저장 완료 (짝수 인덱스만): {save_path}")


def occupancy_grid_callback(msg):
    width = msg.info.width
    height = msg.info.height
    resolution = msg.info.resolution
    data = np.array(msg.data).reshape((height, width))

    image = np.zeros((height, width), dtype=np.uint8)
    image[data == -1] = 127
    image[data == 0] = 255
    image[data == 100] = 0

    # 장애물 내부 추출
    _, obstacle_mask = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(obstacle_mask, kernel, iterations=1)
    cv2.imwrite("/home/e2map/clean_map/debug_dilated.jpg", cv2.flip(dilated, 0))
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)
    cv2.imwrite("/home/e2map/clean_map/debug_closed.jpg", cv2.flip(closed, 0))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    # filtered = np.zeros_like(closed)
    # for i in range(1, num_labels):
    #     if stats[i, cv2.CC_STAT_AREA] >= 100:
    #         filtered[labels == i] = 255
    # cv2.imwrite("/home/e2map/clean_map/debug_filtered.jpg", cv2.flip(filtered, 0))

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(closed)
    cv2.drawContours(filled, contours, -1, color=255, thickness=cv2.FILLED)
    cv2.imwrite("/home/e2map/clean_map/debug_filled.jpg", cv2.flip(filled, 0))
    interior = cv2.subtract(filled, closed)
    cv2.imwrite("/home/e2map/clean_map/debug_interior.jpg", cv2.flip(interior, 0))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(interior, connectivity=8)
    max_area = 0
    max_label = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > max_area:
            max_area = stats[i, cv2.CC_STAT_AREA]
            max_label = i
    largest_component = np.zeros_like(interior)
    largest_component[labels == max_label] = 255

    # 하늘색 만들기
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    color_image[largest_component == 255] = (230, 216, 173)
    lower_bound = np.array([220, 206, 163], dtype=np.uint8)
    upper_bound = np.array([240, 226, 183], dtype=np.uint8)
    sky_mask = cv2.inRange(color_image, lower_bound, upper_bound)
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    sky_mask[gray > 245] = 0
    sky_mask_closed = cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))




    # 완전 흰색 배경 이미지 생성
    comp_img2 = np.ones((height, width, 3), dtype=np.uint8) * 255  # 흰색 배경
    # largest_component 부분만 하늘색 칠하기
    comp_img2[largest_component == 255] = (230, 216, 173)
    # 이미지 뒤집어서 저장
    cv2.imwrite("/home/e2map/clean_map/zigzag_by_stage_1.jpg", cv2.flip(comp_img2, 0))

    comp_mask = cv2.inRange(comp_img2, np.array([230, 216, 173]), np.array([230, 216, 173]))
    cv2.imwrite("/home/e2map/clean_map/zigzag_by_stage_1_1.jpg", cv2.flip(comp_mask, 0))


    # 1차
    zigzag_path_1, bridges_1 = generate_zigzag_path_from_mask(comp_mask, resolution)
    path_mask_1 = np.zeros_like(comp_mask)
    for x, y in zigzag_path_1:
        cv2.circle(path_mask_1, (x, y), radius=3, color=255, thickness=cv2.FILLED)  # 3픽셀=15cm 반경

    # dilation으로 원형 커버리지 반영 (총 직경 30cm)
    radius_m = 0.3 / 2  # 15cm 반경
    resolution = 0.05  # 예: 5cm/pixel
    radius_px = int(radius_m / resolution)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius_px * 2 + 1, radius_px * 2 + 1))
    path_mask_1_dilated = cv2.dilate(path_mask_1, kernel, iterations=1)

    cv2.imwrite("/home/e2map/clean_map/zigzag_by_stage_2_0.jpg", cv2.flip(path_mask_1_dilated, 0))

    resolution = msg.info.resolution
    origin_x = msg.info.origin.position.x
    origin_y = msg.info.origin.position.y

    # stage_2_0 이미지에서 유효 영역 (흰색) 좌표만 txt로 저장
    ys, xs = np.where(comp_mask == 255)
    with open("/home/e2map/clean_map/valid_xy_stage_2_0.txt", "w") as f:
        for x_pix, y_pix in zip(xs, ys):
            x_m = x_pix * resolution + origin_x
            y_m = (comp_mask.shape[0] - y_pix - 1) * resolution + origin_y  # OpenCV → ROS 좌표계 변환
            f.write(f"{x_m:.4f}, {y_m:.4f}\n")
    print("✅ stage_2_0 유효 흰색 좌표 저장 완료: valid_xy_stage_2_0.txt")

    # 2차 경로 추출을 위한 마스크
    remaining_mask_2 = cv2.subtract(comp_mask, path_mask_1_dilated)
    zigzag_path_2, bridges_2 = generate_zigzag_path_from_mask(remaining_mask_2, resolution)
    cv2.imwrite("/home/e2map/clean_map/zigzag_by_stage_2_1.jpg", cv2.flip(remaining_mask_2, 0))

    # print("🔍 zigzag_path_2:")
    # for idx, (x, y) in enumerate(zigzag_path_2):
    #     print(f"{idx}: ({x}, {y})")
    # for idx, bridge in enumerate(bridges_2):
    #     print(f"Bridge {idx}:")
    #     for pt_idx, (x, y) in enumerate(bridge):
    #         print(f"  {pt_idx}: ({x}, {y})")

    path_mask_2 = np.zeros_like(comp_mask)
    for x, y in zigzag_path_2:
        path_mask_2[y, x] = 255

    # 3차
    remaining_mask_3 = cv2.subtract(remaining_mask_2, path_mask_2)
    zigzag_path_3, bridges_3 = generate_zigzag_path_from_mask(remaining_mask_3, resolution)

    # 시각화 1단계별
    stage_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    stage_img[largest_component == 255] = (230, 216, 173)
    video_img = stage_img.copy()
    for i in range(1, len(zigzag_path_1)):
        cv2.line(stage_img, zigzag_path_1[i - 1], zigzag_path_1[i], (255, 0, 0), 1)
    for i in range(1, len(zigzag_path_2)):
        cv2.line(stage_img, zigzag_path_2[i - 1], zigzag_path_2[i], (255, 100, 255), 1)
    for i in range(1, len(zigzag_path_3)):
        cv2.line(stage_img, zigzag_path_3[i - 1], zigzag_path_3[i], (0, 255, 0), 1)
    cv2.imwrite("/home/e2map/clean_map/zigzag_by_stage_2.jpg", cv2.flip(stage_img, 0))


    bridge_img = stage_img.copy()
    for x, y in zigzag_path_2:
        if 0 <= x < bridge_img.shape[1] and 0 <= y < bridge_img.shape[0]:
            bridge_img[y, x] = (238, 130, 238)  # 파란색 (BGR)

    # 저장 (y축 반전 포함)
    cv2.imwrite("/home/e2map/clean_map/zigzag_by_stage_2_2.jpg", cv2.flip(bridge_img, 0))
    rospy.loginfo("✅ 대각선 제거 + 최종 한붓그리기 경로 저장 완료")

    # 전체 path 리스트
    all_paths = [zigzag_path_1, zigzag_path_2, zigzag_path_3]
    all_paths = [p for p in all_paths if len(p) > 0]

    final_path, final_bridges = connect_zigzag_paths_custom(zigzag_path_1, zigzag_path_2, zigzag_path_3, comp_mask)

    # 시각화
    merged_img = stage_img.copy()
    for i in range(1, len(final_path)):
        cv2.line(merged_img, final_path[i - 1], final_path[i], (30, 144, 255), 1)

    # 🔴 연결된 bridge 경로 (빨간색)
    for bridge in final_bridges:
        for i in range(1, len(bridge)):
            cv2.line(merged_img, bridge[i - 1], bridge[i], (127, 255, 0), 1)  # Red

    cv2.imwrite("/home/e2map/clean_map/zigzag_by_stage_3.jpg", cv2.flip(merged_img, 0))
    rospy.loginfo("🔗 전체 zigzag 경로를 모든 점 기준으로 최단 연결 완료")

    # save_path_mp4_overlay(
    #     final_path,
    #     background_img=video_img,  # 기존 시각화 된 이미지
    #     save_path='/home/data/final_path.mp4',
    #     points_per_second=1000,
    #     fps=60,
    #     color=(255, 0, 0),  # BGR: 파란색
    #     flip_y=False
    # )

    resolution = msg.info.resolution
    origin_x = msg.info.origin.position.x
    origin_y = msg.info.origin.position.y
    save_path_to_txt_ros(final_path, resolution, origin_x, origin_y,save_path='/home/e2map/clean_map/ikea_1_path_1.txt')


def main():
    rospy.init_node('skyblue_zigzag_path_node')
    rospy.Subscriber('/map', OccupancyGrid, occupancy_grid_callback)
    rospy.spin()

if __name__ == '__main__':
    main()
