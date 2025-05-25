import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

def extract_filtered_unique_model_positions(world_file_path):
    tree = ET.parse(world_file_path)
    root = tree.getroot()

    models = []
    seen_names = set()  # ✅ 중복 제거용

    for model in root.iter('model'):
        name = model.attrib.get('name')
        if name in seen_names:
            continue
        # ✅ 필터 조건 확장
        if (
            name.startswith('fence') or
            name.startswith('room_wall') or
            name.startswith('ground') or
            name.startswith('window') or
            name.startswith('door')
        ):
            continue

        for child in model:
            if child.tag == 'pose':
                pose_vals = list(map(float, child.text.strip().split()))
                x, y = pose_vals[0], pose_vals[1]
                models.append((name, x, y))
                seen_names.add(name)
                break  # pose 하나만 추출하고 종료

    return models

def save_models_to_txt(models, filename='obstacle_2.txt'):
    with open(filename, 'w') as f:
        for name, x, y in models:
            f.write(f"{name}, {x:.4f}, {y:.4f}\n")
    print(f"✅ 저장 완료: {filename}")

def plot_model_positions(models):
    plt.figure(figsize=(10, 8))
    for name, x, y in models:
        plt.plot(x, y, 'ro')
        plt.text(x + 0.1, y + 0.1, name, fontsize=9)

    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    plt.title('Gazebo World - Unique Model Positions (Filtered)')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# 사용 예시
world_file = '/home/e2map/clean_map/Ikea/ikea_2.world'  # 경로 수정 필요
filtered_unique_models = extract_filtered_unique_model_positions(world_file)

save_models_to_txt(filtered_unique_models, 'obstacle_2.txt')
plot_model_positions(filtered_unique_models)
