from PIL import Image
import os
from tqdm import tqdm

# 경로 설정
input_folder = 'E:\merchandise_dataset\emart_detection'  # 이미지가 저장된 폴더 경로
output_folder = os.path.join(input_folder, 'resized_output')  # 리사이즈된 이미지 저장 폴더

# 리사이즈할 크기 설정 (예: 512x512)
new_size = (360, 640)

# 출력 폴더가 없는 경우 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 폴더 내 이미지 파일 불러오기 및 리사이즈
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]

for filename in tqdm(image_files, desc="Resizing images"):
    # 이미지 열기
    img_path = os.path.join(input_folder, filename)
    img = Image.open(img_path)

    # 리사이즈
    resized_img = img.resize(new_size)

    # 리사이즈된 이미지 저장
    output_path = os.path.join(output_folder, filename)
    resized_img.save(output_path)

print(f'모든 이미지가 {new_size} 크기로 리사이즈되어 {output_folder} 폴더에 저장되었습니다.')
