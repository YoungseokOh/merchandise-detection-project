import requests
import os

# 다운로드할 SAM 'small' 모델의 URL
url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_s_0b3195.pth"

# 저장할 경로 설정
save_dir = "E:/merchandise_dataset/model/"
os.makedirs(save_dir, exist_ok=True)  # 디렉토리가 없으면 생성
save_path = os.path.join(save_dir, "sam_vit_s.pth")

# 파일 다운로드
response = requests.get(url, stream=True)
response.raise_for_status()  # 요청에 실패하면 예외 발생

# 파일을 바이너리 쓰기 모드로 열고 다운로드한 내용을 저장
with open(save_path, "wb") as file:
    for chunk in response.iter_content(chunk_size=8192):
        file.write(chunk)

print(f"SAM 'small' 모델이 '{save_path}'에 성공적으로 다운로드되었습니다.")
