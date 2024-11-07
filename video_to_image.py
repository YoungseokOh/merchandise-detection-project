import cv2
import os
from tqdm import tqdm

# 비디오 파일들이 있는 폴더와 모든 프레임을 저장할 폴더 경로 설정
video_folder = 'E:/merchandise_dataset/emart'  # 비디오 파일들이 있는 폴더
output_folder = 'E:/merchandise_dataset/emart_images'  # 프레임이 저장될 폴더

# 출력 폴더가 없으면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 전체 프레임 수를 세기 위한 전역 변수
frame_count = 0

# 폴더 안에 있는 모든 MP4 파일을 순회
for video_file in tqdm(os.listdir(video_folder), desc="Processing Videos"):
    if video_file.endswith('.MP4') or video_file.endswith('.mp4'):
        video_path = os.path.join(video_folder, video_file)
        
        # 비디오 캡처 객체 생성
        cap = cv2.VideoCapture(video_path)
        
        # 비디오가 열렸는지 확인
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # tqdm으로 프레임 진행 상황을 표시
        for _ in tqdm(range(total_frames), desc=f"Extracting frames from {video_file}", leave=False):
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
