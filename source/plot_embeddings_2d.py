import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

# === cultural token embedding txt 파일 읽기 ===
def load_embeddings_from_txt(file_path):
    embeddings = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('==='):
                continue
            if line.strip() == '':
                continue
            values = list(map(float, line.strip().split()))
            embeddings.append(values)
    return np.array(embeddings)

# === 2D로 투영하기 (PCA 사용) ===
# def project_embeddings_to_2d(embeddings):
#     pca = PCA(n_components=2)
#     reduced = pca.fit_transform(embeddings)
#     return reduced

def project_embeddings_to_2d(embeddings):
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    reduced = tsne.fit_transform(embeddings)
    return reduced

# === 2D Plot 그리기 (복수 파일 지원) ===
def plot_multiple_embeddings_2d(all_reduced_embeddings, labels):
    plt.figure(figsize=(10, 8))

    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'pink', 'gray']

    for idx, (reduced_embeddings, label) in enumerate(zip(all_reduced_embeddings, labels)):
        color = colors[idx % len(colors)]
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.7, label=label, color=color)

        # 각 포인트에 round 번호 쓰기
        for point_idx, (x, y) in enumerate(reduced_embeddings):
            plt.text(x, y, str(point_idx + 1), fontsize=8, ha='center', va='center', color=color)

    plt.title('2D Projection of Multiple Cultural Token Embeddings')
    plt.xlabel('t-SNE Axis 1')
    plt.ylabel('t-SNE Axis 2')
    plt.legend()
    plt.grid(True)
    plt.show()

file_paths = [
    "/home/e2map/clean_map/Ikea/ikea_1_cultural_token.txt",
    "/home/e2map/clean_map/Ikea/ikea_2_cultural_token.txt",
    "/home/e2map/clean_map/Ikea/ikea_3_cultural_token.txt",
    "/home/e2map/clean_map/Ikea/ikea_4_cultural_token.txt",
    "/home/e2map/clean_map/Ikea/ikea_5_cultural_token.txt"
]

all_embeddings = []
labels = []


for file_path in file_paths:
    if os.path.exists(file_path):
        embeddings = load_embeddings_from_txt(file_path)
        reduced_embeddings = project_embeddings_to_2d(embeddings)

        # === Round 1 기준으로 이동 보정 ===
        origin = reduced_embeddings[0]  # Round 1 좌표
        reduced_embeddings = reduced_embeddings - origin

        all_embeddings.append(reduced_embeddings)

        # 파일명만 label로 사용
        label = os.path.basename(file_path)
        labels.append(label)
    else:
        print(f"❗ 파일 없음: {file_path}")

# 전체 plot
plot_multiple_embeddings_2d(all_embeddings, labels)