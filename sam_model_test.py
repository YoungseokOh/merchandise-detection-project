import os
import torch
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
import time
from tqdm import tqdm  # tqdm 임포트

# SAM 모델 설정
sam_checkpoint = "E:/merchandise_dataset/model/sam_vit_h_4b8939.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
sam.to(device=device)

# SAM 자동 마스크 생성기 설정
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.8,
    stability_score_thresh=0.9,
    min_mask_region_area=0
)

# 원본 이미지 경로 설정
image_folder_path = "E:/merchandise_dataset/images"
image_filenames = [f for f in os.listdir(image_folder_path) if f.endswith(".jpg")]

# SSIM 기반 필터링 함수 (면적 및 길쭉한 마스크 제거 포함)
def filter_by_ssim(image, masks, ssim_threshold=0.3, min_size=50, max_area=5000, max_aspect_ratio=3.0):
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

        # 면적 및 길쭉한 형태 필터링
        if area > max_area or aspect_ratio > max_aspect_ratio:
            continue

        # 최소 크기 필터링
        if width < min_size or height < min_size:
            continue

        # 마스크 영역 크롭
        cropped_segment = cv2.bitwise_and(image, image, mask=mask_image.astype(np.uint8))
        cropped_segment = cropped_segment[y_min:y_max+1, x_min:x_max+1]
        
        # SSIM 기반 유사도 필터링
        is_unique = True
        for existing_segment in cropped_segments:
            if cropped_segment.shape == existing_segment.shape:
                ssim_value = ssim(cropped_segment, existing_segment, win_size=3, channel_axis=-1)
                if ssim_value >= ssim_threshold:
                    is_unique = False
                    break

        if is_unique:
            cropped_segments.append(cropped_segment)
            unique_masks.append(mask)
    
    return unique_masks

# 세그먼트 영역을 저장하는 함수
def save_segmented_regions(image, masks, segments_folder):
    for i, mask in enumerate(tqdm(masks, desc="Saving segments")):  # tqdm 추가
        mask_image = mask['segmentation']
        
        y_indices, x_indices = np.where(mask_image)
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        
        cropped_segment = cv2.bitwise_and(image, image, mask=mask_image.astype(np.uint8))
        cropped_segment = cropped_segment[y_min:y_max+1, x_min:x_max+1]

        segment_filename = os.path.join(segments_folder, f"segment_{i}.png")
        cv2.imwrite(segment_filename, cv2.cvtColor(cropped_segment, cv2.COLOR_RGB2BGR))

# 유니크 마스크 시각화 이미지를 저장하는 함수
def save_visualization_comparison(original_image, unique_masks, save_path):
    # 유니크 마스크가 적용된 이미지 생성 (배경은 검은색)
    final_image = np.zeros_like(original_image)
    
    for mask in unique_masks:
        color = [np.random.randint(0, 255) for _ in range(3)]
        mask_image = mask['segmentation']
        
        # 각 유니크 마스크에 대해 알파 블렌딩 적용
        for j in range(3):
            final_image[:, :, j] = np.where(mask_image, original_image[:, :, j] * 0.5 + color[j] * 0.5, final_image[:, :, j])

    # 원본과 유니크 마스크 이미지를 나란히 비교하는 이미지 생성
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    axs[0].imshow(original_image)
    axs[0].set_title("Original Image")
    axs[0].axis('off')
    
    axs[1].imshow(final_image)
    axs[1].set_title("Image with Unique Segmented Masks")
    axs[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 이미지별로 필터링, 세그먼트 저장 및 시각화
for image_filename in tqdm(image_filenames, desc="Processing images"):  # tqdm 추가
    # 이미지 불러오기
    image_path = os.path.join(image_folder_path, image_filename)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # SAM 마스크 생성
    masks_original = mask_generator.generate(image)

    # 필터링 실행
    unique_masks = filter_by_ssim(image, masks_original, ssim_threshold=0.3, min_size=25, max_area=10000, max_aspect_ratio=5.0)
    print(f"Image {image_filename}: Number of unique segments after filtering: {len(unique_masks)}")

    # 결과 저장 폴더 생성
    base_filename = os.path.splitext(image_filename)[0]
    result_folder = os.path.join("E:/merchandise_dataset/results", base_filename)
    segments_folder = os.path.join(result_folder, "segments")
    os.makedirs(segments_folder, exist_ok=True)

    # 세그먼트 저장
    save_segmented_regions(image, unique_masks, segments_folder)

    # 시각화 이미지 저장
    visualization_path = os.path.join("E:/merchandise_dataset/results", f"{base_filename}_results.jpg")
    save_visualization_comparison(image, unique_masks, visualization_path)

print("Processing complete. All segments and visualizations saved.")
