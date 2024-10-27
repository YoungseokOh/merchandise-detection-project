import os
import cv2
import numpy as np
from tqdm import tqdm

def extract_features(image, method='ORB', nfeatures=5000):
    if method == 'ORB':
        extractor = cv2.ORB_create()
    elif method == 'SIFT':
        extractor = cv2.SIFT_create(nfeatures=nfeatures)
    keypoints, descriptors = extractor.detectAndCompute(image, None)
    return keypoints, descriptors

# 컬러 유사도 계산 함수 (HSV 히스토그램 비교)
def color_similarity(img1, img2, threshold=0.7):
    img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    hist_img1 = cv2.calcHist([img1_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist_img2 = cv2.calcHist([img2_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])

    cv2.normalize(hist_img1, hist_img1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist_img2, hist_img2, 0, 1, cv2.NORM_MINMAX)

    similarity = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CORREL)
    return similarity >= threshold

# 피처 매칭 함수 (ORB 및 SIFT 선택 가능)
def feature_matching(source_image, segment_paths, method='ORB', nfeatures=5000):
    source_keypoints, source_descriptors = extract_features(source_image, method, nfeatures=nfeatures)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING if method == 'ORB' else cv2.NORM_L2, crossCheck=True)

    best_match = None
    best_match_score = 0

    for seg_path in tqdm(segment_paths, desc="Matching with segments"):
        mask = cv2.imread(seg_path)
        mask_resized = cv2.resize(mask, (source_image.shape[1], source_image.shape[0]))

        # 컬러 유사도 필터링
        if not color_similarity(source_image, mask_resized):
            continue

        mask_keypoints, mask_descriptors = extract_features(mask_resized, method, nfeatures=nfeatures)

        if mask_descriptors is not None and len(mask_descriptors) > 0:
            matches = bf.match(source_descriptors, mask_descriptors)
            matches = sorted(matches, key=lambda x: x.distance)

            good_matches = [m for m in matches if m.distance < 100]
            match_score = len(good_matches)

            if match_score > best_match_score:
                best_match_score = match_score
                best_match = (seg_path, good_matches)

    if best_match:
        print(f"Best match found in {best_match[0]} with {best_match_score} good matches using {method}.")
    else:
        print(f"No match found using {method}.")
    return best_match

# 시각화 함수
def visualize_match(source_image, mask, keypoints_source, keypoints_mask, matches):
    match_image = cv2.drawMatches(source_image, keypoints_source, mask, keypoints_mask, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Best Match", match_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 테스트 실행
if __name__ == "__main__":
    source_image_path = "E:/merchandise_dataset/feat_match/conch/con1.jpg"
    source_image = cv2.imread(source_image_path)

    results_folder = "E:/merchandise_dataset/scinario_2_results/frame_0477"
    segment_paths = [os.path.join(results_folder, seg) for seg in os.listdir(results_folder) if seg.endswith(".jpg")]

    # ORB로 피처 매칭
    print("ORB 매칭 결과:")
    best_match_orb = feature_matching(source_image, segment_paths, method='ORB')
    if best_match_orb:
        best_match_path, good_matches = best_match_orb
        matched_mask = cv2.imread(best_match_path)
        matched_mask_resized = cv2.resize(matched_mask, (source_image.shape[1], source_image.shape[0]))
        keypoints_source, _ = extract_features(source_image, method='ORB')
        keypoints_mask, _ = extract_features(matched_mask_resized, method='ORB')
        visualize_match(source_image, matched_mask_resized, keypoints_source, keypoints_mask, good_matches)

    # SIFT로 피처 매칭
    print("SIFT 매칭 결과:")
    best_match_sift = feature_matching(source_image, segment_paths, method='SIFT', nfeatures=5000)
    if best_match_sift:
        best_match_path, good_matches = best_match_sift
        matched_mask = cv2.imread(best_match_path)
        matched_mask_resized = cv2.resize(matched_mask, (source_image.shape[1], source_image.shape[0]))
        keypoints_source, _ = extract_features(source_image, method='SIFT', nfeatures=5000)
        keypoints_mask, _ = extract_features(matched_mask_resized, method='SIFT', nfeatures=5000)
        visualize_match(source_image, matched_mask_resized, keypoints_source, keypoints_mask, good_matches)
