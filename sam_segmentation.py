import os
import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

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

# SIFT 피처 추출 함수
def extract_sift_features(image, nfeatures=1000):
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    keypoints, descriptors = sift.detectAndCompute(image, None)
    if descriptors is not None:
        descriptors = descriptors.astype(np.float32)
    return keypoints, descriptors

# RGB 히스토그램 유사도 계산 함수
def calculate_histogram_similarity(source_image, segment_image):
    source_hist = cv2.calcHist([source_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    segment_hist = cv2.calcHist([segment_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

    source_hist = cv2.normalize(source_hist, source_hist).flatten()
    segment_hist = cv2.normalize(segment_hist, segment_hist).flatten()
    similarity = cv2.compareHist(source_hist, segment_hist, cv2.HISTCMP_CORREL)
    return similarity

# BOW 히스토그램 생성 함수
def create_bow_histograms(images, k=100):
    all_descriptors = []
    sift = cv2.SIFT_create()

    for image in images:
        _, descriptors = extract_sift_features(image)
        if descriptors is not None and descriptors.shape[1] == 128:
            all_descriptors.append(descriptors)

    all_descriptors = np.vstack(all_descriptors)
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(all_descriptors)

    bow_histograms = []
    for image in images:
        _, descriptors = extract_sift_features(image)
        if descriptors is not None and descriptors.shape[1] == 128:
            visual_words = kmeans.predict(descriptors)
            histogram, _ = np.histogram(visual_words, bins=np.arange(k + 1))
            bow_histograms.append(histogram)
        else:
            bow_histograms.append(np.zeros(k))

    return np.array(bow_histograms), kmeans

# BoW 히스토그램으로 유사도 측정 함수
def find_most_similar_mask(source_hist, mask_histograms):
    similarities = cosine_similarity([source_hist], mask_histograms)
    best_match_idx = np.argmax(similarities)
    return best_match_idx, similarities[0, best_match_idx]

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
def save_visualization_comparison(original_image, unique_masks, matched_mask_indices, best_match_index, save_path):
    final_image = original_image.copy()
    
    for i, mask in enumerate(unique_masks):
        mask_image = mask['segmentation']
        y_indices, x_indices = np.where(mask_image)
        
        # Bounding Box 설정 및 표시
        if len(x_indices) > 0 and len(y_indices) > 0:
            x_min, x_max = x_indices.min(), x_indices.max()
            y_min, y_max = y_indices.min(), y_indices.max()
            
            # 초록색 바운딩 박스 추가
            if i == best_match_index or i in matched_mask_indices:
                color = (0, 255, 0)  # 초록색으로 표시
                thickness = 4 if i == best_match_index else 2  # 최상위 매칭은 두께 4, 나머지 매칭은 두께 2
                cv2.rectangle(final_image, (x_min, y_min), (x_max, y_max), color, thickness)

        # 마스크 색상 설정 (초록색은 매칭된 세그먼트, 빨간색은 매칭되지 않은 세그먼트)
        mask_color = [0, 255, 0] if i == best_match_index or i in matched_mask_indices else [0, 0, 255]
        
        # 마스크 색상 적용
        for j in range(3):
            final_image[:, :, j] = np.where(mask_image, original_image[:, :, j] * 0.5 + mask_color[j] * 0.5, final_image[:, :, j])

    # 시각화 및 저장
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    axs[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original Image")
    axs[0].axis('off')
    
    axs[1].imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Image with Bounding Boxes and Masks")
    axs[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 메인 실행 부분
if __name__ == "__main__":
    source_folder_path = "E:/merchandise_dataset/feat_match/shin"
    source_filenames = [f for f in os.listdir(source_folder_path) if f.endswith(".jpg")]
    source_images = [cv2.imread(os.path.join(source_folder_path, f)) for f in source_filenames]

    bow_histograms, kmeans_model = create_bow_histograms(source_images, k=100)

    image_folder_path = "E:/merchandise_dataset/scinario_3"
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
        result_folder = os.path.join("E:/merchandise_dataset/results", base_filename)
        os.makedirs(result_folder, exist_ok=True)
        
        # 각 세그먼트를 별도로 저장
        for idx, segment in enumerate(cropped_segments):
            segment_path = os.path.join(result_folder, f"{base_filename}_segment_{idx}.jpg")
            cv2.imwrite(segment_path, segment)
        
        # 각 unique_mask의 세그먼트에 대한 색상 유사도 및 히스토그램 생성
        segment_similarities = []
        mask_histograms = []
        for segment, mask in zip(cropped_segments, unique_masks):
            _, descriptors = extract_sift_features(segment)
            if descriptors is not None and descriptors.shape[1] == 128:
                visual_words = kmeans_model.predict(descriptors)
                histogram, _ = np.histogram(visual_words, bins=np.arange(101), density=True)
                mask_histograms.append(histogram)
            else:
                mask_histograms.append(np.zeros(100))

            color_similarities = [calculate_histogram_similarity(src, segment) for src in source_images]
            max_color_similarity = max(color_similarities)
            segment_similarities.append((mask, segment, max_color_similarity))
        
        # 색상 유사도 순으로 정렬하여 상위 세그먼트만 SIFT 매칭 수행
        segment_similarities.sort(key=lambda x: x[2], reverse=True)
        top_segments = [seg[1] for seg in segment_similarities[:10]]

        best_match_index = -1
        matched_mask_indices = []
        max_similarity = 0
        for i, segment in enumerate(top_segments):
            mask_hist = mask_histograms[i]
            best_source_index, similarity_score = find_most_similar_mask(mask_hist, bow_histograms)
            if similarity_score > 0.5:
                matched_mask_indices.append(i)
            if similarity_score > max_similarity:
                best_match_index = i
                max_similarity = similarity_score

        # 시각화 및 결과 저장
        visualization_path = os.path.join("E:/merchandise_dataset/scinario_3_results", f"{base_filename}_results.jpg")
        save_visualization_comparison(image, unique_masks, matched_mask_indices, best_match_index, visualization_path)

    print("Processing complete. All matched segments and visualizations saved.")
