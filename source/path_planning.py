#!/usr/bin/env python
import rospy
import numpy as np
import cv2
from nav_msgs.msg import OccupancyGrid
from scipy.spatial.distance import cdist
from heapq import heappush, heappop


def find_greedy_rectangles(blue_mask, min_w=5, min_h=5, resolution=0.05):
    visited = np.zeros_like(blue_mask, dtype=bool)
    h, w = blue_mask.shape
    rectangles = []

    for y in range(h):
        for x in range(w):
            if blue_mask[y, x] != 255 or visited[y, x]:
                continue
            max_w = 0
            while x + max_w < w and blue_mask[y, x + max_w] == 255 and not visited[y, x + max_w]:
                max_w += 1
            max_h = 0
            is_valid = True
            while y + max_h < h and is_valid:
                for dx in range(max_w):
                    if blue_mask[y + max_h, x + dx] != 255 or visited[y + max_h, x + dx]:
                        is_valid = False
                        break
                if is_valid:
                    max_h += 1
            if max_w >= min_w and max_h >= min_h:
                x0, y0, w0, h0 = x, y, max_w, max_h

                # 상, 하, 좌, 우 순서로 0.1m씩 최대 0.5m까지 확장
                for direction in ['top', 'bottom', 'left', 'right']:
                    for step in np.arange(0.1, 0.6, 0.1):
                        pixels = int(step / resolution)
                        if direction == 'top':
                            y1 = max(y0 - pixels, 0)
                            band = blue_mask[y1:y0, x0:x0 + w0]
                            visited_band = visited[y1:y0, x0:x0 + w0]
                        elif direction == 'bottom':
                            y2 = min(y0 + h0 + pixels, h)
                            band = blue_mask[y0 + h0:y2, x0:x0 + w0]
                            visited_band = visited[y0 + h0:y2, x0:x0 + w0]
                        elif direction == 'left':
                            x1 = max(x0 - pixels, 0)
                            band = blue_mask[y0:y0 + h0, x1:x0]
                            visited_band = visited[y0:y0 + h0, x1:x0]
                        elif direction == 'right':
                            x2 = min(x0 + w0 + pixels, w)
                            band = blue_mask[y0:y0 + h0, x0 + w0:x2]
                            visited_band = visited[y0:y0 + h0, x0 + w0:x2]

                        if band.size == 0:
                            break

                        visited_ratio = np.sum(visited_band) / visited_band.size
                        if visited_ratio > 0.9:
                            break

                        ratio = np.sum(band == 255) / band.size
                        if ratio > 0.9:
                            if direction == 'top':
                                h0 += y0 - y1
                                y0 = y1
                            elif direction == 'bottom':
                                h0 = y2 - y0
                            elif direction == 'left':
                                w0 += x0 - x1
                                x0 = x1
                            elif direction == 'right':
                                w0 = x2 - x0
                        else:
                            break

                rectangles.append((x0, y0, w0, h0))
                visited[y0:y0 + h0, x0:x0 + w0] = True

    rospy.loginfo(f"🟦 생성된 사각형 수: {len(rectangles)}")
    return rectangles


def merge_rectangles(rectangles, occupancy_data, tolerance=10):
    merged = []
    for rect in rectangles:
        x, y, w, h = rect
        merged_flag = False
        for i in range(len(merged)):
            mx, my, mw, mh = merged[i]

            # 포함 여부 우선 검사
            if (x >= mx and y >= my and x + w <= mx + mw and y + h <= my + mh):
                merged_flag = True
                break
            if (mx >= x and my >= y and mx + mw <= x + w and my + mh <= y + h):
                merged[i] = (x, y, w, h)
                merged_flag = True
                break

            # 기존 병합 조건 유지 (수직, 수평)
            if abs(mx - x) <= tolerance and abs(mw - w) <= tolerance:
                if abs((my + mh) - y) <= tolerance or abs((y + h) - my) <= tolerance:
                    new_y = min(my, y)
                    new_h = max(my + mh, y + h) - new_y
                    new_x = mx
                    new_w = mw
                    block = occupancy_data[new_y:new_y+new_h, new_x:new_x+new_w]
                    if block.size == 0 or np.sum(block == -1) / block.size > 0.1:
                        continue
                    merged[i] = (new_x, new_y, new_w, new_h)
                    merged_flag = True
                    break
            if abs(my - y) <= tolerance and abs(mh - h) <= tolerance:
                if abs((mx + mw) - x) <= tolerance or abs((x + w) - mx) <= tolerance:
                    new_x = min(mx, x)
                    new_w = max(mx + mw, x + w) - new_x
                    new_y = my
                    new_h = mh
                    block = occupancy_data[new_y:new_y+new_h, new_x:new_x+new_w]
                    if block.size == 0 or np.sum(block == -1) / block.size > 0.1:
                        continue
                    merged[i] = (new_x, new_y, new_w, new_h)
                    merged_flag = True
                    break

        if not merged_flag:
            merged.append(rect)
    return merged

def tsp_order(points):
    n = len(points)
    dist = cdist(points, points)
    visited = [0]
    unvisited = list(range(1, n))
    while unvisited:
        last = visited[-1]
        next_idx = min(unvisited, key=lambda i: dist[last][i])
        visited.append(next_idx)
        unvisited.remove(next_idx)
    return visited

def generate_zigzag_from_corner(rect, occupancy_data, step, start_corner):
    x, y, w, h = rect
    height, width = occupancy_data.shape
    zigzag = []

    if start_corner in ['topleft', 'bottomleft']:
        reverse_x = False
    else:
        reverse_x = True

    if start_corner in ['topleft', 'topright']:
        row_range = range(y, y + h, step)
    else:
        row_range = range(y + h - step, y - 1, -step)

    for idx, i in enumerate(row_range):
        if reverse_x ^ (idx % 2 == 1):
            col_range = range(x + w - step, x - 1, -step)
        else:
            col_range = range(x, x + w, step)

        for j in col_range:
            cx = min(j + step // 2, x + w - 1)
            cy = min(i + step // 2, y + h - 1)
            if 0 <= cy < height and 0 <= cx < width:
                if occupancy_data[cy, cx] != -1:
                    zigzag.append((cx, cy))

    return zigzag

def best_corner_for_zigzag(rect, occupancy_data, step, from_point):
    corners = {
        'topleft':    (rect[0], rect[1]),
        'topright':   (rect[0] + rect[2] - 1, rect[1]),
        'bottomleft': (rect[0], rect[1] + rect[3] - 1),
        'bottomright':(rect[0] + rect[2] - 1, rect[1] + rect[3] - 1),
    }
    best_corner = min(corners.items(), key=lambda kv: np.linalg.norm(np.array(kv[1]) - np.array(from_point)))
    return best_corner[0]

def is_line_safe(p1, p2, occupancy_data):
    x1, y1 = map(int, p1)
    x2, y2 = map(int, p2)
    h, w = occupancy_data.shape
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    x, y = x1, y1
    sx = -1 if x1 > x2 else 1
    sy = -1 if y1 > y2 else 1
    if dx == 0 and dy == 0:
        return occupancy_data[y, x] != -1
    if dx > dy:
        err = dx / 2.0
        while x != x2:
            if 0 <= x < w and 0 <= y < h and occupancy_data[y, x] == -1:
                return False
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y2:
            if 0 <= x < w and 0 <= y < h and occupancy_data[y, x] == -1:
                return False
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    if 0 <= x2 < w and 0 <= y2 < h and occupancy_data[y2, x2] == -1:
        return False
    return True

def a_star(start, goal, occupancy_data):
    h, w = occupancy_data.shape
    start = tuple(start)
    goal = tuple(goal)
    def heuristic(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))
    def neighbors(p):
        x, y = p
        directions = [(-1, -1), (0, -1), (1, -1),
                      (-1,  0),          (1,  0),
                      (-1,  1), (0,  1), (1,  1)]
        return [(x+dx, y+dy) for dx, dy in directions if 0 <= x+dx < w and 0 <= y+dy < h and occupancy_data[y+dy, x+dx] >= 0]
    open_set = []
    heappush(open_set, (heuristic(start, goal), 0, start))
    came_from = {}
    cost_so_far = {start: 0}
    while open_set:
        _, cost, current = heappop(open_set)
        if current == goal:
            break
        for next_node in neighbors(current):
            new_cost = cost + heuristic(current, next_node)
            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                heappush(open_set, (new_cost + heuristic(goal, next_node), new_cost, next_node))
                came_from[next_node] = current
    path = []
    if goal in came_from or start == goal:
        curr = goal
        path.append(curr)
        while curr != start:
            curr = came_from[curr]
            path.append(curr)
        path.reverse()
    return path

def occupancy_grid_callback(msg):
    width = msg.info.width
    height = msg.info.height
    resolution = msg.info.resolution
    data = np.array(msg.data).reshape((height, width))
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[data == -1] = 127
    mask[data >= 0] = 255
    colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    block_pixel = int(0.5 / resolution)
    stride_pixel = int(0.1 / resolution)
    for y in range(0, height - block_pixel + 1, stride_pixel):
        for x in range(0, width - block_pixel + 1, stride_pixel):
            block = data[y:y+block_pixel, x:x+block_pixel]
            if block.size == 0:
                continue
            if np.sum(block != -1) / block.size > 0.9:
                obstacle_ratio = np.sum(block > 0) / block.size
                if obstacle_ratio < 0.15:
                    colored[y:y + block_pixel, x:x + block_pixel] = (255, 0, 0)
    blue_mask = np.all(colored == [255, 0, 0], axis=2).astype(np.uint8) * 255
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    min_pixels = int(0.5 / resolution)
    raw_rects = find_greedy_rectangles(blue_mask, min_pixels, min_pixels, resolution)
    merged_rects = raw_rects
    for _ in range(3):
        merged_rects = merge_rectangles(merged_rects, data, tolerance=10)
    rospy.loginfo(f"✅ 병합된 사각형 수: {len(merged_rects)}")
    order = tsp_order([((x + w // 2), (y + h // 2)) for (x, y, w, h) in merged_rects])
    ordered_rects = [merged_rects[i] for i in order]

    path_points = []
    a_star_segments = []
    prev_end = None
    for i, rect in enumerate(ordered_rects):
        if prev_end is not None:
            corner = best_corner_for_zigzag(rect, data, min_pixels, prev_end)
        else:
            corner = 'topleft'
        curr_zigzag = generate_zigzag_from_corner(rect, data, min_pixels, corner)
        if prev_end is not None and curr_zigzag:
            if is_line_safe(prev_end, curr_zigzag[0], data):
                path_points.append(prev_end)
                path_points.append(curr_zigzag[0])
            else:
                a_path = a_star(prev_end, curr_zigzag[0], data)
                if a_path:
                    a_star_segments.append(a_path)
                    path_points.extend(a_path)
        if curr_zigzag:
            if not path_points or path_points[-1] != curr_zigzag[0]:
                path_points.append(curr_zigzag[0])
            path_points.extend(curr_zigzag[1:])
            prev_end = curr_zigzag[-1]

    for i in range(1, len(path_points)):
        cv2.line(colored, path_points[i-1], path_points[i], (0, 0, 0), 1)
    for segment in a_star_segments:
        for j in range(1, len(segment)):
            if j % 2 == 0:
                cv2.line(colored, segment[j-1], segment[j], (0, 0, 255), 1)
    for (x, y, w, h) in ordered_rects:
        cv2.rectangle(colored, (x, y), (x + w, y + h), (0, 255, 255), 1)
    colored = cv2.flip(colored, 0)
    cv2.imwrite("ikea_2_2.png", colored)
    rospy.loginfo("🧭 정사각형 지그재그 + A* 경로 연결 (빨간 점선) 완료 → semantic_mask_tsp_path_safe.png")

def main():
    rospy.init_node('tsp_zigzag_path_safe')
    rospy.Subscriber('/map', OccupancyGrid, occupancy_grid_callback)
    rospy.spin()

if __name__ == '__main__':
    main()