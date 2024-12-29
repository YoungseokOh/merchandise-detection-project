import os
import cv2
import torch
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# DeepSORT 관련 임포트
from deep_sort_realtime.deepsort_tracker import DeepSort

# Roboflow 모델 정보
API_KEY = "3eDw4J9aKw6I4hKjvGf1"  # 여기에 본인의 API 키를 입력하세요
MODEL_ENDPOINT = "merchandise-83wvm"  # 프로젝트 이름
MODEL_VERSION = "14"  # 모델 버전

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

# 색상 설정: 클래스마다 다르게, 같은 시리즈는 유사한 색상으로
class_colors = {
    "cham_1": (255, 100, 100),  # 밝은 빨강
    "cham_2": (200, 50, 50),    # 짙은 빨강
    "shinjjang_1": (125, 100, 255),  # 연한 파랑
    "shinjjang_2": (100, 50, 255),   # 짙은 파랑
    "shinjjang_3": (75, 0, 255),     # 더 짙은 파랑
    "cornchip_1": (0, 125, 225),     # 연한 파랑-초록
    "cornchip_2": (0, 100, 200),     # 짙은 파랑-초록
    "cornchip_3": (0, 75, 175),      # 더 짙은 파랑-초록
    # 추가 클래스에 맞춰 색상 설정 가능
}

# 클래스별 라벨 설정
label_map = {
    "cham": "참쌀",
    "shinjjang_1": "못말리는 신짱",
    "shinjjang_2": "못말리는 신짱(고구마)",
    "shinjjang_3": "못말리는 신짱(+15%)",
    "cornchip_1": "콘칩(군옥수수)",
    "cornchip_2": "콘칩(초당옥수수)"
}

# 한글 폰트 파일 경로 (예: 나눔고딕)
font_path = "C:/Windows/Fonts/NanumGothic.ttf"  # 시스템에 설치된 한글 폰트 경로 지정
font = ImageFont.truetype(font_path, 15)  # 폰트 크기는 상황에 맞게 조절


# DeepSORT 추적기 초기화
tracker = DeepSort(
    max_age=5,           # 객체가 5 프레임 동안 탐지되지 않으면 추적 종료
    n_init=3,            # 한 번의 탐지로 추적 확정
    nms_max_overlap=1, # 50% 이상 겹치는 바운딩 박스 제거
    max_cosine_distance=0.5,  # Appearance 매칭 임계값 조정
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
def apply_sam_within_bbox(image_pil, bbox_coords):
    x1, y1, x2, y2 = bbox_coords
    
    # 이미지 경계 내로 좌표 클리핑
    width, height = image_pil.size
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width - 1, x2)
    y2 = min(height - 1, y2)
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    # `PIL` 이미지에서 지정된 바운딩 박스 영역을 자름
    cropped_image_pil = image_pil.crop((x1, y1, x2, y2))

    # # 만약 크롭된 이미지가 너무 작으면 마스크 생성을 생략
    # if cropped_image_pil.size[0] == 0 or cropped_image_pil.size[1] == 0:
    #     return []

    # SAM이 `OpenCV` 이미지를 필요로 할 경우, PIL 이미지를 OpenCV로 변환
    cropped_image_cv = cv2.cvtColor(np.array(cropped_image_pil), cv2.COLOR_RGB2BGR)

    # SAM 모델을 사용하여 마스크 생성
    masks = mask_generator.generate(cropped_image_cv)
    return masks

def overlay_binary_mask(image_pil, bbox_coords, masks, color, alpha=0.5):
    x1, y1, x2, y2 = bbox_coords
    
    # 마스크 투명 레이어 생성
    for mask_info in masks:
        mask = mask_info['segmentation']
        
        # 마스크 크기를 bbox 크기로 조정하고, 알파 채널을 사용하여 반투명 적용
        mask = Image.fromarray(mask.astype('uint8') * 255).resize((x2 - x1, y2 - y1))
        
        # 마스크 색상 적용 (반투명 설정)
        mask_colored = Image.new("RGBA", mask.size, color + (int(255 * alpha),))  # 알파 값 조절
        
        # 원본 이미지에서 해당 영역 자르기
        region = image_pil.crop((x1, y1, x2, y2)).convert("RGBA")
        
        # 마스크와 원본 영역을 합성하여 블렌딩
        blended = Image.alpha_composite(region, mask_colored)
        
        # 합성된 이미지를 원본 이미지에 붙여넣기
        image_pil.paste(blended, (x1, y1), mask=mask)

    return image_pil

# Detection 및 SAM 마스크 시각화 및 DeepSORT 추적
def visualize_and_save(image_path, detections, output_path):
    # 이미지 읽기 (PIL로 직접 읽기)
    image_pil = Image.open(image_path).convert('RGB')
    
    # 한글 폰트 설정 (예: 나눔고딕)
    font_path = "C:/Windows/Fonts/NanumGothic.ttf"  # 한글 폰트 경로 지정
    font = ImageFont.truetype(font_path, 10)
    title_font = ImageFont.truetype(font_path, 10)  # 제목에 쓸 약간 큰 폰트
    
    # 탐지 목록 준비
    detection_list = []
    for det in detections:
        x1, y1, x2, y2, confidence, class_name = det
        bbox = [x1, y1, x2 - x1, y2 - y1]
        detection_list.append((bbox, confidence, class_name))
    
    # 추적기 업데이트 (OpenCV 이미지를 전달하기 위해 PIL 이미지를 OpenCV로 변환)
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    tracks = tracker.update_tracks(detection_list, frame=image_cv)
    
    # Draw 객체 생성
    draw = ImageDraw.Draw(image_pil)
    
    # 클래스별 카운트 딕셔너리 초기화
    class_counts = {}
    
    # 추적 결과 시각화
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        class_name = track.det_class
        color = class_colors.get(class_name, (255, 255, 255))
    
        # 클래스 이름 매핑 적용
        display_name = next((label_map[key] for key in label_map if key in class_name), class_name)
        
        # 클래스별 카운트 증가
        class_counts[display_name] = class_counts.get(display_name, 0) + 1
        
        label = f"ID:{track_id} {display_name}"
        
        # 바운딩 박스 그리기 (PIL에서 그리기)
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
        
        # 한글 라벨 표시
        draw.text((x1, y1 - 20), label, font=font, fill=color)
    
    # Legend 영역 크기와 위치 설정
    image_width, image_height = image_pil.size
    legend_x, legend_y = image_width - 170, 10  # 오른쪽 상단 위치
    legend_width, row_height = 160, 20  # 너비 및 각 행 높이 조정
    header_height = 30  # 열 제목 높이 조정
    legend_bg_color = (255, 255, 255)  # 흰색 배경
    outline_color = (0, 0, 0)  # 테두리 색상
    
    # 클래스별 카운트를 리스트로 변환 및 정렬
    legend_entries = sorted(class_counts.items(), key=lambda x: x[0])  # 이름순 정렬
    
    # 표시할 최대 항목 수 (최대 10개)
    max_entries = 10
    displayed_entries = legend_entries[:max_entries]  # 표시할 항목 제한
    
    # 둥근 직사각형 배경 그리기
    draw.rounded_rectangle(
        [legend_x - 5, legend_y - 5, legend_x + legend_width, legend_y + (row_height * len(displayed_entries)) + header_height],
        radius=10, fill=legend_bg_color, outline=outline_color, width=1
    )
    
    # 열 제목 추가
    draw.text((legend_x + 5, legend_y), "SNACK", font=title_font, fill=(0, 0, 0))
    draw.text((legend_x + 100, legend_y), "COUNT", font=title_font, fill=(0, 0, 0))
    draw.line([(legend_x, legend_y + header_height - 3), (legend_x + legend_width, legend_y + header_height - 3)], fill=outline_color, width=1)
    
    # Legend 테이블 내용 추가
    for idx, (display_name, count) in enumerate(displayed_entries):
        text_y = legend_y + header_height + (idx * row_height)  # 각 행의 y 위치 설정
        # SNACK과 COUNT 추가
        draw.text((legend_x + 5, text_y), display_name, font=font, fill=(0, 0, 0))
        draw.text((legend_x + 100, text_y), f"x {count}", font=font, fill=(0, 0, 0))
        
    # 결과 이미지 저장
    result_image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_image)
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
process_images_in_folder("E:/merchandise_dataset/emart_detection/resized_output", "E:/merchandise_dataset/detection_results")
