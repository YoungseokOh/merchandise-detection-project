import os
import torch
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
import time

# SAM 모델 설정
sam_checkpoint = "E:/merchandise_dataset/model/sam_vit_h_4b8939.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
sam.to(device=device)

# SAM 자동 마스크 생성기 설정
mask_generator = SamAutomaticMaskGenerator(sam)

# 원본 이미지 불러오기
image_path = "E:/merchandise_dataset/images/frame_0540.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 원본 이미지 처리 시간 측정
start_time = time.time()
masks_original = mask_generator.generate(image)
end_time = time.time()
print(f"Original image processing time: {end_time - start_time:.4f} seconds")

# SSIM 기반 필터링 함수
def filter_by_ssim(image, masks, ssim_threshold=0.3, min_size=50):
    unique_masks = []
    cropped_segments = []
    
    for mask in masks:
        mask_image = mask['segmentation']
        y_indices, x_indices = np.where(mask_image)
        
        if len(x_indices) == 0 or len(y_indices) == 0:
            continue

        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        
        if (x_max - x_min + 1) < min_size or (y_max - y_min + 1) < min_size:
            continue

        cropped_segment = cv2.bitwise_and(image, image, mask=mask_image.astype(np.uint8))
        cropped_segment = cropped_segment[y_min:y_max+1, x_min:x_max+1]
        
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

# 히스토그램 기반 필터링 함수
def filter_by_histogram(image, masks, hist_threshold=0.9):
    histograms = []
    unique_masks = []

    for mask in masks:
        mask_image = mask['segmentation'].astype(np.uint8)
        masked_img = cv2.bitwise_and(image, image, mask=mask_image)
        
        # RGB 채널별 히스토그램 계산
        hist_r = cv2.calcHist([masked_img], [0], mask_image, [256], [0, 256]).flatten()
        hist_g = cv2.calcHist([masked_img], [1], mask_image, [256], [0, 256]).flatten()
        hist_b = cv2.calcHist([masked_img], [2], mask_image, [256], [0, 256]).flatten()
        hist = np.concatenate([hist_r, hist_g, hist_b])

        is_unique = True
        for existing_hist, existing_mask in histograms:
            similarity = cv2.compareHist(hist.astype(np.float32), existing_hist.astype(np.float32), cv2.HISTCMP_CORREL)
            if similarity >= hist_threshold:
                is_unique = False
                break

        if is_unique:
            histograms.append((hist, mask))
            unique_masks.append(mask)
    
    return unique_masks

# SSIM + 히스토그램 필터링 결합 함수
def filter_by_ssim_then_histogram(image, masks, ssim_threshold=0.3, min_size=50, hist_threshold=0.9):
    # 1단계: SSIM 기반 필터링
    filtered_masks = filter_by_ssim(image, masks, ssim_threshold=ssim_threshold, min_size=min_size)
    # 2단계: 히스토그램 기반 필터링
    unique_masks = filter_by_histogram(image, filtered_masks, hist_threshold=hist_threshold)
    return unique_masks

# 필터링 실행
unique_masks = filter_by_ssim_then_histogram(image, masks_original, ssim_threshold=0.3, min_size=50, hist_threshold=0.9)
print(f"Number of unique segments after filtering: {len(unique_masks)}")

# 세그먼트 영역을 저장하는 함수
def save_segmented_regions(image, masks, save_folder=None):
    for i, mask in enumerate(masks):
        mask_image = mask['segmentation']
        
        y_indices, x_indices = np.where(mask_image)
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        
        cropped_segment = cv2.bitwise_and(image, image, mask=mask_image.astype(np.uint8))
        cropped_segment = cropped_segment[y_min:y_max+1, x_min:x_max+1]

        if save_folder:
            segment_filename = os.path.join(save_folder, f"segment_{i}.png")
            cv2.imwrite(segment_filename, cv2.cvtColor(cropped_segment, cv2.COLOR_RGB2BGR))

# 저장 폴더 생성 및 기존 파일 삭제
save_folder = os.path.join("E:/merchandise_dataset/results", os.path.basename(image_path).split('.')[0])
if os.path.exists(save_folder):
    for file in os.listdir(save_folder):
        file_path = os.path.join(save_folder, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
else:
    os.makedirs(save_folder, exist_ok=True)

# 고유 세그먼트 저장
save_segmented_regions(image, unique_masks, save_folder=save_folder)
print(f"Filtered segmented regions are saved in: {save_folder}")

# 전체 이미지에 마스크 적용 후 시각화
def visualize_segments(image, masks):
    final_image = image.copy()
    for mask in masks:
        color = [np.random.randint(0, 255) for _ in range(3)]
        colored_mask = np.zeros_like(image)
        mask_image = mask['segmentation']
        for j in range(3):
            colored_mask[:, :, j] = mask_image * color[j]
        final_image = cv2.addWeighted(final_image, 1, colored_mask, 0.5, 0)

    plt.figure(figsize=(10, 10))
    plt.imshow(final_image)
    plt.title("Original Image with Unique Segmented Masks")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# 시각화 함수 실행
visualize_segments(image, unique_masks)
