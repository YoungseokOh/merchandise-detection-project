import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from torchvision import models
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean, cityblock

# 유사도 계산 함수 (유클리드 거리 및 맨하탄 거리 지원)
def calculate_similarity(vec1, vec2, method='cosine'):
    if method == 'cosine':
        return cosine_similarity([vec1], [vec2])[0][0]
    elif method == 'euclidean':
        return -euclidean(vec1, vec2)  # 거리 기반이므로 -를 붙여 유사도를 높게 만듭니다.
    elif method == 'manhattan':
        return -cityblock(vec1, vec2)  # 거리 기반이므로 -를 붙여 유사도를 높게 만듭니다.
    else:
        raise ValueError("지원하지 않는 유사도 방법입니다.")


# 이미지 전처리 함수 정의
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)  # 배치 차원 추가

# 딥러닝 기반 특징 추출 함수
def extract_deep_features(image, model, device):
    image = preprocess_image(image).to(device)
    with torch.no_grad():
        features = model(image)
    return features.cpu().numpy().flatten()

# 상위 10개의 유사한 세그먼트 시각화 함수
def visualize_top_matches(source_image, top_matches, similarity_method):
    plt.figure(figsize=(15, 10))
    plt.subplot(3, 4, 1)
    plt.imshow(cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Source Image - Method: {similarity_method}")
    plt.axis('off')

    for idx, (path, similarity) in enumerate(top_matches[:10], start=2):
        segment = cv2.imread(path)
        plt.subplot(3, 4, idx)
        plt.imshow(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB))
        plt.title(f"Match {idx-1} - Score: {similarity:.2f}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# 테스트 실행
if __name__ == "__main__":
    # 모델 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # 마지막 FC 레이어 제외
    model = model.to(device)
    model.eval()

    # 데이터 경로 설정
    source_image_path = "E:/merchandise_dataset/feat_match/conch/con1.jpg"
    source_image = cv2.imread(source_image_path)

    results_folder = "E:/merchandise_dataset/scinario_2_results/frame_0482"
    segment_paths = [os.path.join(results_folder, seg) for seg in os.listdir(results_folder) if seg.endswith(".jpg")]

    # 소스 이미지 특성 추출
    source_features = extract_deep_features(source_image, model, device)

    # 모든 세그먼트에 대해 특성 추출 및 유사도 계산
    similarity_methods = ['cosine']
    for method in similarity_methods:
        similarities = []
        for seg_path in tqdm(segment_paths, desc=f"Matching with {method} similarity"):
            segment = cv2.imread(seg_path)
            segment_resized = cv2.resize(segment, (224, 224))  # ResNet 입력 크기로 조정

            # 딥러닝 기반 특성 추출
            segment_features = extract_deep_features(segment_resized, model, device)
            similarity = calculate_similarity(source_features, segment_features, method=method)
            similarities.append((seg_path, similarity))

        # 유사도 순으로 정렬하고 상위 10개 선택
        top_matches = sorted(similarities, key=lambda x: x[1], reverse=True)

        # 상위 10개의 유사한 세그먼트 시각화
        visualize_top_matches(source_image, top_matches, similarity_method=method)