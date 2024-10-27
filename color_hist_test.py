import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import models, transforms
from torch.nn.functional import normalize

# 사전 학습된 ResNet18 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True).to(device)
model.eval()

# 이미지 전처리 함수 정의
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 이미지 임베딩 추출 함수
def extract_embedding(img_path, model):
    image = cv2.imread(img_path)
    image = preprocess(image).unsqueeze(0).to(device)  # 전처리 후 배치 차원 추가
    with torch.no_grad():
        embedding = model(image).cpu().numpy().flatten()  # 임베딩 추출
    return normalize(torch.tensor(embedding).view(1, -1), p=2).numpy().flatten()  # 정규화

# 유사도 측정 실험 함수
def run_embedding_similarity_experiment(source_image_path, segments_folder, top_n=5):
    # 소스 이미지 임베딩 추출
    source_embedding = extract_embedding(source_image_path, model)

    # 세그먼트 이미지 임베딩 추출
    segment_filenames = [f for f in os.listdir(segments_folder) if f.endswith(".jpg")]
    similarities = []
    for filename in segment_filenames:
        segment_path = os.path.join(segments_folder, filename)
        segment_embedding = extract_embedding(segment_path, model)
        similarity = cosine_similarity([source_embedding], [segment_embedding])[0][0]
        similarities.append((filename, similarity))

    # 상위 유사도 순으로 정렬
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_similarities = similarities[:top_n]

    # 결과를 DataFrame으로 표시
    similarity_df = pd.DataFrame(top_similarities, columns=["Segment", "Similarity"])
    print("\nTop Embedding Similarities:")
    print(similarity_df)

    # 상위 N개 유사 세그먼트 시각화
    fig, axs = plt.subplots(1, top_n + 1, figsize=(15, 5))
    source_img = cv2.imread(source_image_path)
    axs[0].imshow(cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Source Image")
    axs[0].axis("off")

    for i, (segment_filename, similarity) in enumerate(top_similarities, start=1):
        segment_image = cv2.imread(os.path.join(segments_folder, segment_filename))
        axs[i].imshow(cv2.cvtColor(segment_image, cv2.COLOR_BGR2RGB))
        axs[i].set_title(f"{segment_filename}\nSim: {similarity:.2f}")
        axs[i].axis("off")

    plt.tight_layout()
    plt.show()

# 예시 실행
source_image_path = "E:/merchandise_dataset/feat_match/cham_2.jpg"  # 경로 업데이트
segments_folder = "E:/merchandise_dataset/results/frame_0267"       # 경로 업데이트
run_embedding_similarity_experiment(source_image_path, segments_folder, top_n=10)
