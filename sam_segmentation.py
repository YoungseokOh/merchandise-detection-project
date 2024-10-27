import os
import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import torchvision.transforms as transforms
from torchvision import models

# SAM 모델 설정
sam_checkpoint = "E:/merchandise_dataset/model/sam_vit_h_4b8939.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
sam.to(device=device)

# SAM 자동 마스크 생성기 설정
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.9,
    stability_score_thresh=0.9,
    min_mask_region_area=5000
)

# 딥러닝 임베딩을 위한 ResNet50 설정
def load_resnet50_model():
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # 마지막 FC 레이어 제외
    model.to(device)
    model.eval()
    return model

# 이미지 전처리 함수
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0).to(device)

# 임베딩 생성 함수
def extract_embedding(image, model):
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        embedding = model(image_tensor).cpu().numpy().flatten()
    return embedding

# Bounding Box 및 형태 필터링 함수
def filter_by_bounding_box(image, masks, min_size=50, max_area=100000, max_aspect_ratio=3.0):
    unique_masks = []
    cropped_segments = []

    for mask in masks:
        mask_image = mask['segmentation']
        y_indices, x_indices = np.where(mask_image)
        
        if len(x_indices) == 0 or len(y_indices) == 0:
            continue

        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        area = width * height
        aspect_ratio = max(width / height, height / width)

        if area > max_area or aspect_ratio > max_aspect_ratio or width < min_size or height < min_size:
            continue

        cropped_segment = cv2.bitwise_and(image, image, mask=mask_image.astype(np.uint8))
        cropped_segment = cropped_segment[y_min:y_max+1, x_min:x_max+1]
        
        cropped_segments.append(cropped_segment)
        unique_masks.append(mask)

    return unique_masks, cropped_segments

# 시각화 함수
def save_visualization_comparison(original_image, unique_masks, best_match_indices, save_path):
    final_image = original_image.copy()
    
    for i, mask in enumerate(unique_masks):
        mask_image = mask['segmentation']
        
        if i in best_match_indices:
            color = [0, 255, 0]  # 상위 유사 마스크는 파란색
        else:
            color = [0, 0, 255]  # 매칭되지 않은 마스크는 빨간색
        
        for j in range(3):
            final_image[:, :, j] = np.where(mask_image, original_image[:, :, j] * 0.5 + color[j] * 0.5, final_image[:, :, j])

    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    axs[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original Image")
    axs[0].axis('off')
    
    axs[1].imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Image with Top Matched Masks")
    axs[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 메인 실행 부분
if __name__ == "__main__":
    # ResNet50 모델 로드
    resnet_model = load_resnet50_model()

    # 소스 이미지 임베딩 생성
    source_folder_path = "E:/merchandise_dataset/feat_match/conch"
    source_images = [cv2.imread(os.path.join(source_folder_path, f)) for f in os.listdir(source_folder_path) if f.endswith(".jpg")]
    source_embeddings = [extract_embedding(img, resnet_model) for img in source_images]

    # 테스트 이미지 경로
    image_folder_path = "E:/merchandise_dataset/scinario_2"
    image_filenames = [f for f in os.listdir(image_folder_path) if f.endswith(".jpg")]

    for image_filename in tqdm(image_filenames, desc="Processing images"):
        image_path = os.path.join(image_folder_path, image_filename)
        image = cv2.imread(image_path)

        # SAM 마스크 생성
        masks_original = mask_generator.generate(image)

        # 필터링 및 마스크 세그먼트 추출
        unique_masks, cropped_segments = filter_by_bounding_box(image, masks_original, min_size=50, max_area=100000, max_aspect_ratio=3.0)
        
        # 세그먼트를 저장할 폴더 생성
        base_filename = os.path.splitext(image_filename)[0]
        result_folder = os.path.join("E:/merchandise_dataset/scinario_2_results", base_filename)
        os.makedirs(result_folder, exist_ok=True)
        
        # 각 세그먼트를 별도로 저장
        segment_embeddings = []
        for idx, segment in enumerate(cropped_segments):
            segment_path = os.path.join(result_folder, f"{base_filename}_segment_{idx}.jpg")
            cv2.imwrite(segment_path, segment)
            segment_embeddings.append(extract_embedding(segment, resnet_model))

        # 소스 이미지와 세그먼트 유사도 계산
        top_matched_indices = []
        max_similarity_scores = []
        similarity_threshold = 0.75  # 유사도 임계값 설정 (예: 0.8)

        for i, seg_embedding in enumerate(segment_embeddings):
            similarities = cosine_similarity([seg_embedding], source_embeddings)
            max_score = np.max(similarities)  # 해당 세그먼트의 소스 이미지들과의 최대 유사도
            max_similarity_scores.append(max_score)
            
            # 유사도가 임계값 이상인 경우 인덱스 추가
            if max_score >= similarity_threshold:
                top_matched_indices.append(i)

        # 시각화 및 결과 저장
        visualization_path = os.path.join("E:/merchandise_dataset/scinario_2_results", f"{base_filename}_results.jpg")
        save_visualization_comparison(image, unique_masks, top_matched_indices, visualization_path)

    print("Processing complete. All matched segments and visualizations saved.")
