import os
import cv2
import torch
import requests
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# DeepSORT 관련 임포트
from deep_sort_realtime.deepsort_tracker import DeepSort

# Roboflow 모델 정보
API_KEY = "3eDw4J9aKw6I4hKjvGf1"  # 여기에 본인의 API 키를 입력하세요
MODEL_ENDPOINT = "merchandise-83wvm"  # 프로젝트 이름
MODEL_VERSION = "3"  # 모델 버전

# SAM 모델 설정
sam_checkpoint = "E:/merchandise_dataset/model/sam_vit_b_01ec64.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
sam.to(device=device)

# SAM 자동 마스크 생성기 설정
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.9,
    stability_score_thresh=0.9,
    min_mask_region_area=5000
)

# 색상 설정: 클래스마다 다르게
class_colors = {
    "cham_1": (255, 0, 0),  # 빨강
    "cham_2": (0, 255, 0),  # 초록
    "shin": (0, 0, 255),  # 파랑
    # 추가 클래스에 맞춰 색상 설정 가능
}

# DeepSORT 추적기 초기화
tracker = DeepSort(
    max_age=10,           # 객체가 5 프레임 동안 탐지되지 않으면 추적 종료
    n_init=5,            # 한 번의 탐지로 추적 확정
    nms_max_overlap=1, # 50% 이상 겹치는 바운딩 박스 제거
    max_cosine_distance=0.2,  # Appearance 매칭 임계값 조정
    nn_budget=None,
    override_track_class=None
)

# 이미지에서 객체 탐지 함수
def detect_objects(image_path, confidence=50, overlap=30):
    image = cv2.imread(image_path)
    _, img_encoded = cv2.imencode('.jpg', image)
    img_bytes = img_encoded.tobytes()

    url = f"https://detect.roboflow.com/{MODEL_ENDPOINT}/{MODEL_VERSION}"
    params = {
        "api_key": API_KEY,
        "confidence": confidence / 100.0,
        "overlap": overlap / 100.0,
    }

    response = requests.post(url, params=params, files={"file": img_bytes})
    response.raise_for_status()
    predictions = response.json()['predictions']

    detections = []
    for pred in predictions:
        x = pred['x']
        y = pred['y']
        width = pred['width']
        height = pred['height']
        confidence = pred['confidence']
        class_name = pred['class']
        # 클래스 ID를 정의하거나 필요하지 않다면 제거 가능
        # 여기서는 클래스 이름만 사용
        x1 = x - width / 2
        y1 = y - height / 2
        x2 = x + width / 2
        y2 = y + height / 2
        detections.append([x1, y1, x2, y2, confidence, class_name])

    return detections

# SAM을 사용해 바운딩 박스 내부의 마스크 생성
def apply_sam_within_bbox(image, bbox_coords):
    x1, y1, x2, y2 = bbox_coords
    # 이미지 경계 내로 좌표 클리핑
    h, w = image.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w - 1, x2)
    y2 = min(h - 1, y2)
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    cropped_image = image[y1:y2, x1:x2]

    # 만약 크롭된 이미지가 너무 작으면 마스크 생성을 생략
    if cropped_image.size == 0:
        return []

    # SAM을 이용해 마스크 생성
    masks = mask_generator.generate(cropped_image)
    return masks

# 바이너리 마스크 오버레이 함수
def overlay_binary_mask(image, bbox_coords, masks, color):
    x1, y1, x2, y2 = bbox_coords
    # 이미지 경계 내로 좌표 클리핑
    h, w = image.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w - 1, x2)
    y2 = min(h - 1, y2)
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    mask_overlay = image.copy()  # 이미지를 복사하여 덧씌울 마스크용 레이어 생성

    for mask_info in masks:
        mask = mask_info['segmentation']  # SAM 마스크의 바이너리 배열 사용
        mask = mask.astype(bool)

        # 마스크 영역에 색상을 적용하여 오버레이 생성
        mask_overlay[y1:y2, x1:x2][mask] = color  # mask가 True인 영역에 색상을 입힘

    # 마스크가 적용된 부분을 원본 이미지에 덧씌움
    image = cv2.addWeighted(mask_overlay, 0.5, image, 0.5, 0)  # 투명도 적용
    return image

# Detection 및 SAM 마스크 시각화 및 DeepSORT 추적
def visualize_and_save(image_path, detections, output_path):
    image = cv2.imread(image_path)

    # 추적기에 필요한 형식으로 탐지 준비
    detection_list = []
    for det in detections:
        x1, y1, x2, y2, confidence, class_name = det
        bbox = [x1, y1, x2 - x1, y2 - y1]  # [x, y, w, h] 형식으로 변환
        detection_list.append((bbox, confidence, class_name))

    # 추적기 업데이트
    tracks = tracker.update_tracks(detection_list, frame=image)  # Appearance 특징을 사용하려면 frame=image 전달

    # 추적 결과 시각화
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()  # 좌상단-우하단 형식의 바운딩 박스 얻기
        x1, y1, x2, y2 = map(int, ltrb)
        class_name = track.det_class  # 수정된 부분: 클래스 이름 얻기
        color = class_colors.get(class_name, (255, 255, 255))  # 클래스에 해당하는 색상 선택

        # 바운딩 박스 그리기
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{track_id} {class_name}"
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # SAM을 통해 마스크 생성 및 바이너리 마스크 오버레이
        track_bbox_coords = (x1, y1, x2, y2)
        masks = apply_sam_within_bbox(image, track_bbox_coords)
        image = overlay_binary_mask(image, track_bbox_coords, masks, color)

    # 결과 이미지 저장
    cv2.imwrite(output_path, image)
    print(f"Processed and saved: {output_path}")

# 폴더 내 모든 이미지 처리 함수
def process_images_in_folder(folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = sorted(os.listdir(folder_path))
    for filename in image_files:
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(folder_path, filename)
            output_path = os.path.join(output_folder, f"detection_{filename}")

            try:
                detections = detect_objects(image_path)
                visualize_and_save(image_path, detections, output_path)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# 사용 예시
process_images_in_folder("E:/merchandise_dataset/scinario_1", "E:/merchandise_dataset/detection_results")
