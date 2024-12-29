import cv2
import os
from tqdm import tqdm

# 이미지들이 저장된 폴더 경로와 출력 파일 경로 설정
image_folder = 'E:/merchandise_dataset/detection_results_after'
output_video = 'final_output_video_v3.mp4'
fps = 20  # 프레임 속도 설정

# 이미지 파일 리스트를 정렬된 순서로 불러옴
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]
images.sort()

# 첫 이미지의 크기 정보로 동영상 생성 준비
first_image_path = os.path.join(image_folder, images[0])
frame = cv2.imread(first_image_path)
height, width, layers = frame.shape

# VideoWriter 객체 생성
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4v 코덱 설정
video = cv2.VideoWriter(output_video, fourcc, fps, (360, 640))

# 이미지들을 순서대로 비디오에 추가
for image in tqdm(images):
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    video.write(frame)

# 비디오 객체 종료
video.release()
# cv2.destroyAllWindows()
