import cv2

# 입력 영상 경로 및 출력 영상 경로 설정
input_video_path = 'C:/Users/seok436/Downloads/Seoul_Namsan_20241009.mp4'
output_video_path = 'C:/Users/seok436/Downloads/output_video.mp4'

# 비디오 파일 읽기
cap = cv2.VideoCapture(input_video_path)

# 원본 비디오의 프레임 크기, FPS 정보 얻기
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 출력 비디오 코덱 및 포맷 설정 (MP4로 저장)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (960, 512))

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    
    # 더 이상 읽을 프레임이 없으면 루프 종료
    if not ret:
        break
    
    # 프레임 리사이즈 (960x512)
    resized_frame = cv2.resize(frame, (960, 512))
    
    # 리사이즈된 프레임을 출력 비디오에 쓰기
    out.write(resized_frame)

# 비디오 파일 객체 닫기
cap.release()
out.release()

print("비디오 리사이징 완료 및 저장됨!")
