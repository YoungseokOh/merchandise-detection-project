import cv2
import os
import torch
import numpy as np
from YOLOX.yolox.data.data_augment import ValTransform
from YOLOX.yolox.utils import postprocess

# YOLOX 모델 로드
model = torch.hub.load('Megvii-BaseDetection/YOLOX', 'yolox_l', pretrained=True)
model.eval()

# 모델을 GPU로 이동 (GPU가 있는 경우)
if torch.cuda.is_available():
    model = model.cuda()

# 입력 영상 경로 및 출력 영상 경로 설정
input_video_path = 'C:/Users/seok436/Downloads/Seoul_Namsan_20241009.mp4'
output_video_path = 'C:/Users/seok436/Downloads/output_video.mp4'

# 영상 캡처 객체 생성
cap = cv2.VideoCapture(input_video_path)

# 영상 정보 가져오기
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 출력 영상을 위한 VideoWriter 객체 생성
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# 클래스별 색상 정의
np.random.seed(42)
colors = np.random.randint(0, 255, size=(80, 3), dtype='uint8')

# 클래스 이름 정의 (COCO 데이터셋 클래스 사용)
class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# 이미지 정규화 함수 정의 (ValTransform을 대체)
def preprocess(img):
    img = img.astype(np.float32)
    img = img.transpose(2, 0, 1)  # HWC에서 CHW로 변환
    return img

# 영상 프레임 단위로 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임 전처리 (스케일링 및 패딩 없이)
    img = preprocess(frame)
    img = torch.from_numpy(img).unsqueeze(0)

    # GPU 사용 여부에 따라 텐서를 GPU로 이동
    if torch.cuda.is_available():
        img = img.cuda()

    # YOLOX 모델로 객체 감지
    with torch.no_grad():
        outputs = model(img)
        outputs = postprocess(outputs, num_classes=80, conf_thre=0.25, nms_thre=0.45)

    # 감지 결과에서 박스 좌표와 레이블 가져오기
    if outputs[0] is not None:
        output = outputs[0].cpu().numpy()
        bboxes = output[:, 0:4]
        cls_conf = output[:, 4] * output[:, 5]
        cls_ids = output[:, 6].astype(int)

        # 바운딩 박스 좌표를 그대로 사용 (이미지 크기가 동일하므로)
        for bbox, score, cls_id in zip(bboxes, cls_conf, cls_ids):
            x1, y1, x2, y2 = bbox.astype(int)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            color = [int(c) for c in colors[cls_id % len(colors)]]
            label = f"{class_names[cls_id]} {score:.2f}"

            # 감지된 객체를 영상에 표시
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 처리된 프레임을 출력 영상에 기록
    out.write(frame)

    # 현재 프레임 화면에 표시 (옵션)
    cv2.imshow('YOLOX Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
out.release()
cv2.destroyAllWindows()