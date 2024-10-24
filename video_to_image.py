import cv2
import os

# 비디오 파일 경로 설정
video_path = 'E:/merchandise_dataset/test_video.mp4'  # MP4 비디오 파일 경로 입력
output_folder = 'E:\merchandise_dataset\images'  # 이미지가 저장될 폴더 이름

# 출력 폴더가 없으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_path)

# 프레임 수를 세기 위한 변수
frame_count = 0

# 비디오가 열렸는지 확인
while cap.isOpened():
    ret, frame = cap.read()
    
    # 프레임을 제대로 읽었다면 저장
    if ret:
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
    else:
        break

# 비디오 캡처 객체 해제
cap.release()

print(f"총 {frame_count}개의 이미지가 저장되었습니다.")