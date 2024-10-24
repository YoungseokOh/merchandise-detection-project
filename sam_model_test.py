import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
import time  # 시간 측정을 위한 모듈

# SAM 모델 설정
sam_checkpoint = "E:/merchandise_dataset/model/sam_vit_h_4b8939.pth"  # 체크포인트 파일 경로
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
sam.to(device=device)

# SAM 자동 마스크 생성기 설정
mask_generator = SamAutomaticMaskGenerator(sam)

# 원본 이미지 불러오기
image_path = "E:/merchandise_dataset/test_image/frame_0000.jpg"  # 이미지 경로 입력
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 이미지 절반 크기로 줄이기
image_resized = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))

# 원본 이미지 처리 시간 측정
start_time = time.time()
masks_original = mask_generator.generate(image)
end_time = time.time()
print(f"Original image processing time: {end_time - start_time:.4f} seconds")

# 절반 크기 이미지 처리 시간 측정
start_time = time.time()
masks_resized = mask_generator.generate(image_resized)
end_time = time.time()
print(f"Resized image (1/2) processing time: {end_time - start_time:.4f} seconds")

# 마스크를 원본 이미지에 적용하여 시각화하는 함수
def apply_mask_on_image(image, masks):
    final_image = image.copy()
    for mask in masks:
        color = [np.random.randint(0, 255) for _ in range(3)]  # 랜덤 색상
        mask_image = mask['segmentation']  # 마스크 정보 가져오기
        colored_mask = np.zeros_like(image)
        for i in range(3):  # RGB 채널에 색상 적용
            colored_mask[:, :, i] = mask_image * color[i]
        final_image = cv2.addWeighted(final_image, 1, colored_mask, 0.5, 0)  # 마스크 적용
    return final_image

# 원본 이미지와 마스크 적용
final_image_original = apply_mask_on_image(image, masks_original)
final_image_resized = apply_mask_on_image(image_resized, masks_resized)

# 이미지 출력 (왼쪽: 원본, 오른쪽: 절반 크기)
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
fig.subplots_adjust(wspace=0.05, hspace=0)  # 여백 최소화

# 원본 이미지에 마스크 적용한 결과
axs[0].imshow(final_image_original)
axs[0].set_title('Original Image with Mask')
axs[0].axis('off')  # 축 제거

# 절반 크기로 줄인 이미지에 마스크 적용한 결과
axs[1].imshow(final_image_resized)
axs[1].set_title('Resized Image with Mask (1/2)')
axs[1].axis('off')  # 축 제거

plt.tight_layout()
plt.show()
