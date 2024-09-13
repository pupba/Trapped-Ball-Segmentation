import numpy as np
import cv2
def thinning(fillmap: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """선 영역을 주변 채운 색상으로 채우는 함수.

    # Arguments
        fillmap: 채운 영역 ID가 표시된 이미지.
        max_iter: 최대 반복 횟수.

    # Returns
        채운 후의 이미지.
    """
    line_id = 0  # 기본 선 ID
    h, w = fillmap.shape[:2]  # 이미지 높이와 너비
    result = fillmap.copy()  # 결과를 저장할 배열 초기화

    for iterNum in range(max_iter):
        # 선의 포인트를 찾기. 포인트가 없으면 종료.
        line_points = np.where(result == line_id)
        if not len(line_points[0]) > 0:
            break

        # 선과 채운 영역 사이의 포인트를 찾기.
        line_mask = np.full((h, w), 255, np.uint8)  # 선 마스크 초기화
        line_mask[line_points] = 0  # 선의 포인트를 0으로 설정
        line_border_mask = cv2.morphologyEx(line_mask, cv2.MORPH_DILATE,
                                             cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), anchor=(-1, -1),
                                             iterations=1) - line_mask  # 선의 경계 포인트 찾기
        line_border_points = np.where(line_border_mask == 255)  # 경계 포인트 찾기

        result_tmp = result.copy()  # 임시 결과 배열 생성
        # 각 경계 포인트를 주변 채운 영역의 ID로 채움
        for i, _ in enumerate(line_border_points[0]):
            x, y = line_border_points[1][i], line_border_points[0][i]

            # 주변 픽셀을 확인하여 채움
            if x - 1 > 0 and result[y][x - 1] != line_id:
                result_tmp[y][x] = result[y][x - 1]
                continue

            if x - 1 > 0 and y - 1 > 0 and result[y - 1][x - 1] != line_id:
                result_tmp[y][x] = result[y - 1][x - 1]
                continue

            if y - 1 > 0 and result[y - 1][x] != line_id:
                result_tmp[y][x] = result[y - 1][x]
                continue

            if y - 1 > 0 and x + 1 < w and result[y - 1][x + 1] != line_id:
                result_tmp[y][x] = result[y - 1][x + 1]
                continue

            if x + 1 < w and result[y][x + 1] != line_id:
                result_tmp[y][x] = result[y][x + 1]
                continue

            if x + 1 < w and y + 1 < h and result[y + 1][x + 1] != line_id:
                result_tmp[y][x] = result[y + 1][x + 1]
                continue

            if y + 1 < h and result[y + 1][x] != line_id:
                result_tmp[y][x] = result[y + 1][x]
                continue

            if y + 1 < h and x - 1 > 0 and result[y + 1][x - 1] != line_id:
                result_tmp[y][x] = result[y + 1][x - 1]
                continue

        result = result_tmp.copy()  # 최종 결과 업데이트

    return result  # 채운 후의 이미지 반환

def show_fill_map(fillmap: np.ndarray) -> np.ndarray:
    """채운 영역을 색상으로 표시하여 시각화하는 함수.

    # Arguments
        fillmap: 채운 영역 ID가 포함된 이미지 배열.

    # Returns
        색상이 적용된 이미지 배열.
    """
    # 각 채운 영역에 대해 랜덤 색상 생성
    colors = np.random.randint(0, 255, (np.max(fillmap) + 1, 3))
    # 선의 ID는 0이며, 그 색상은 검정색으로 설정
    colors[0] = [0, 0, 0]

    return colors[fillmap]  # 색상이 적용된 이미지 반환

def show_fill_map_re(fillmap: np.ndarray, representative_image: np.ndarray) -> np.ndarray:
    """대표 이미지의 색상으로 채운 영역을 표시하는 함수.

    # Arguments
        fillmap: 채운 영역 ID가 포함된 2D 배열.
        representative_image: 색상 값을 추출할 대표 이미지.

    # Returns
        대표 색상으로 채운 이미지를 반환.
    """
    # 대표 이미지를 fillmap 크기에 맞게 리사이즈
    if representative_image.dtype != np.uint8:
        representative_image = representative_image.astype(np.uint8)
    representative_image = cv2.resize(representative_image, (fillmap.shape[1], fillmap.shape[0]))

    output_image = np.zeros_like(representative_image)  # 결과 이미지 초기화

    for fill_id in range(1, np.max(fillmap) + 1):
        mask = (fillmap == fill_id)  # 현재 fill_id에 대한 마스크 생성
        if np.any(mask):
            # 마스크가 적용된 영역에서 대표 이미지의 중앙값 색상 계산
            representative_color = np.median(representative_image[mask], axis=0)
            output_image[mask] = representative_color  # 결과 이미지에 색상 적용

    return output_image  # 결과 이미지 반환
