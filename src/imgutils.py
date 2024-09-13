import numpy as np
import cv2

def get_ball_structuring_element(radius:int)->np.ndarray:
    """특정 반지름을 가진 구형 구조 요소를 가져오는 함수.
    이 구조 요소는 형태학적 작업(morphology operation)에서 사용된다.
    구의 반지름은 일반적으로 (leaking_gap_size / 2)와 같다.

    # Arguments
        radius: 구형 구조 요소의 반지름 
             
    # Returns
        구형 구조 요소의 배열
    """
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))

def exclude_area(image:np.ndarray, radius)->np.ndarray:
    """이미지에서 경계에 가까운 포인트를 제외하기 위해 침식을 수행한다.
    팽창 후 시드 포인트를 사용하여 영역을 채우고자 한다.
    시드 포인트가 경계 근처에 있을 경우, 채우기 결과에 포함되지 않을 수 있으며,
    다음 채우기 작업을 위한 유효한 포인트가 되지 않을 수 있다. 따라서,
    이러한 포인트는 침식을 통해 무시한다.

    # Arguments
        image: 이미지
        radius: 구형 구조 요소의 반지름

    # Returns
        침식된 이미지
    """
    return cv2.morphologyEx(image, cv2.MORPH_ERODE, get_ball_structuring_element(radius), anchor=(-1, -1), iterations=1)

def get_unfilled_point(image: np.ndarray)->np.ndarray:
    """값이 255인 영역(빈 영역)에 해당하는 포인트를 가져온다.

    # 인자
        image: 처리할 이미지.

    # 반환 값
        빈 영역(선화가 없는 영역)에 대한 포인트 배열.
    """
    y, x = np.where(image == 255) # 이미지에서 값이 255인 픽셀의 x,y 좌표를 찾는다.

    return np.stack((x.astype(int), y.astype(int)), axis=-1) # x,y 좌표를 결합하여 포인트 배열을 생성한다.

def visualize_points(image: np.ndarray, points: np.ndarray):
    """주어진 이미지에 포인트를 시각화하는 함수.

    # 인자
        image: 원본 이미지.
        points: 시각화할 포인트 배열.
    """
    # 원본 이미지를 복사하여 표시용 이미지 생성
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # 그레이스케일 이미지를 컬러로 변환

    # 포인트를 빨간색으로 표시
    for point in points:
        cv2.circle(output_image, (point[0], point[1]), 2, (0, 0, 255), -1)  # 빨간색 원으로 포인트 표시

    # 결과 이미지 시각화
    cv2.imshow("test",output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def flood_fill_single(im:np.ndarray, seed_point:tuple[int,int]):
    """단일 Flood Fill 작업을 수행하는 함수.

    Flood Fill : 다차원 배열의 어떤 칸과 연결된 영역을 찾는 알고리즘.

    # Arguments
        im: 흰색(빈 영역) 배경의 검은(채워진) 선화를 가진 이미지
        seed_point: 갇힌 공 채우기용 시드 포인트, 튜플 형식 (정수, 정수)
    
    # Returns
        채운 후의 이미지.
    """
    pass1 = np.full(im.shape, 255, np.uint8)

    im_inv = cv2.bitwise_not(im)

    mask1 = cv2.copyMakeBorder(im_inv, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
    _, pass1, _, _ = cv2.floodFill(pass1, mask1, seed_point, 0, 0, 0, 4)

    return pass1

def flood_fill_multi(image: np.ndarray, max_iter: int = 20000) -> list:
    """다중 Flood Fill 작업을 수행하여 모든 유효한 영역을 채우는 함수.

    # Arguments
        image: 이미지. 흰색 배경, 검은색 선 및 채운 영역이 포함되어야 함.
               흰색 영역은 빈 영역을 나타내고, 검은색 영역은 채워진 영역을 나타낸다.
        max_iter: 최대 반복 횟수.

    # Returns
        채운 영역의 포인트 배열.
    """

    unfill_area = image  # 초기 빈 영역
    filled_area = []  # 채운 영역 저장 리스트

    for _ in range(max_iter):
        points = get_unfilled_point(unfill_area)  # 빈 영역의 포인트 찾기

        if not len(points) > 0:  # 빈 영역이 없으면 종료
            break

        # 단일 Flood Fill 수행
        fill = flood_fill_single(unfill_area, (points[0][0], points[0][1]))
        unfill_area = cv2.bitwise_and(unfill_area, fill)  # 현재 빈 영역 업데이트

        filled_area.append(np.where(fill == 0))  # 채운 영역의 포인트 저장

    return filled_area  # 채운 영역의 포인트 배열 반환

def build_fill_map(image: np.ndarray, fills: list) -> np.ndarray:
    """각 픽셀을 채운 영역의 ID로 표시하는 배열을 생성하는 함수.

    # Arguments
        image: 처리할 이미지.
        fills: 채운 영역의 포인트 배열. 각 포인트는 (y, x) 형태의 좌표를 포함해야 함.
    
    # Returns
        각 픽셀에 채운 영역의 ID가 표시된 배열.
    """
    result = np.zeros(image.shape[:2], np.int_)  # 결과 배열 초기화, 모든 값은 0

    for index, fill in enumerate(fills):
        result[fill] = index + 1  # 채운 영역의 ID로 표시

    return result  # 결과 배열 반환

def get_bounding_rect(points:np.ndarray)->tuple:
    """주어진 포인트 집합의 최소 및 최대 x,y 좌표를 기반으로 경계 사각형을 계산

    # Arguments
        points: (y,x) 형태의 배열
    # Returns
        사각형 형태의 경계 사각형 좌표
    """
    x1, y1, x2, y2 = np.min(points[1]), np.min(points[0]), np.max(points[1]), np.max(points[0])
    return (x1, y1, x2, y2)

def get_border_bounding_rect(h:int, w:int, p1:tuple, p2:tuple, r:int)->tuple:
    """ 주어진 경계 사각형에 특정 크기의 여백을 추가하여 유효한 경계 사각형을 계산 
    경계가 이미지의 크기를 초과하지 않도록 조정

    # Arguments
        h: 이미지의 최대 높이
        w: 이미지의 최대 너비
        p1: 경계 사각형의 시작 점
        p2: 경계 사각형의 끝 점
        r: 여백의 반경
    # Returns
        여백이 추가된 경계 사각형의 좌표를 반환
    """
    x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]

    x1 = x1 - r if 0 < x1 - r else 0
    y1 = y1 - r if 0 < y1 - r else 0
    x2 = x2 + r + 1 if x2 + r + 1 < w else w
    y2 = y2 + r + 1 if y2 + r + 1 < h else h

    return (x1, y1, x2, y2)

def get_border_point(points:list, rect:tuple, max_height:int, max_width:int)->tuple:
    """주어진 포인트의 경계 포인트와 그 형태를 계산 
    경계 사각형에 여백을 추가하고, 해당 영역에서 경계 픽셀을 추출

    # Arguments
        points: 채운 영역의 포인트
        rect: 채운 영역의 경계 사각형
        max_height: 이미지 최대 높이
        max_width: 이미지 최대 너비
    # Returns
        경계 픽셀 좌표와 포인트의 다각형 형태를 반환
    """
    # Get a local bounding rect.
    border_rect = get_border_bounding_rect(max_height, max_width, rect[:2], rect[2:], 2)

    # Get fill in rect.
    fill = np.zeros((border_rect[3] - border_rect[1], border_rect[2] - border_rect[0]), np.uint8)
    # Move points to the rect.
    fill[(points[0] - border_rect[1], points[1] - border_rect[0])] = 255

    # Get shape.
    contours, _ = cv2.findContours(fill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    approx_shape = cv2.approxPolyDP(contours[0], 0.02 * cv2.arcLength(contours[0], True), True)

    # Get border pixel.
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    border_pixel_mask = cv2.morphologyEx(fill, cv2.MORPH_DILATE, cross, anchor=(-1, -1), iterations=1) - fill
    border_pixel_points = np.where(border_pixel_mask == 255)

    # Transform points back to fillmap.
    border_pixel_points = (border_pixel_points[0] + border_rect[1], border_pixel_points[1] + border_rect[0])

    return border_pixel_points, approx_shape


def merge_fill(fillmap: np.ndarray, max_iter: int = 10) -> np.ndarray:
    """채운 영역 병합 함수.

    # Arguments
        fillmap: 채운 영역 ID가 표시된 이미지.
        max_iter: 최대 반복 횟수.
    
    # Returns
        병합된 채운 영역 이미지.
    """
    max_height, max_width = fillmap.shape[:2]
    result = fillmap.copy()  # 결과 배열 초기화

    for i in range(max_iter):

        result[np.where(fillmap == 0)] = 0  # 빈 영역은 0으로 설정

        fill_id = np.unique(result.flatten())  # 현재 채운 영역 ID 추출
        fills = []

        for j in fill_id:
            point = np.where(result == j)
            fills.append({
                'id': j,
                'point': point,
                'area': len(point[0]),
                'rect': get_bounding_rect(point)
            })

        for j, f in enumerate(fills):
            # 라인 ID는 무시
            if f['id'] == 0:
                continue

            border_points, approx_shape = get_border_point(f['point'], f['rect'], max_height, max_width)
            border_pixels = result[border_points]
            pixel_ids, counts = np.unique(border_pixels, return_counts=True)

            ids = pixel_ids[np.nonzero(pixel_ids)]
            new_id = f['id']
            if len(ids) == 0:
                # 주변에 색상이 변경된 라인으로 둘러싸인 경우
                if f['area'] < 5:
                    new_id = 0
            else:
                # 가장 접촉이 많은 영역의 ID로 설정
                new_id = ids[0]

            # 조건에 따라 ID 업데이트
            if len(approx_shape) == 1 or f['area'] == 1:
                result[f['point']] = new_id

            if len(approx_shape) in [2, 3, 4, 5] and f['area'] < 500:
                result[f['point']] = new_id

            if f['area'] < 250 and len(ids) == 1:
                result[f['point']] = new_id

            if f['area'] < 50:
                result[f['point']] = new_id

        # 변화가 없으면 종료
        if len(fill_id) == len(np.unique(result.flatten())):
            break

    return result  # 병합된 결과 반환


if __name__ == "__main__":
    # img = cv2.imread("./result/hemi/1.jpg",cv2.IMREAD_GRAYSCALE)
    img = cv2.imread("test.jpg",cv2.IMREAD_GRAYSCALE)
    # 1/2 resize 테스트를 위해
    h,w = img.shape
    # img = cv2.resize(img,(int(w//2),int(h//2)),interpolation=cv2.INTER_LANCZOS4)

    radius = 3
    points = get_unfilled_point(exclude_area(img, radius))
    visualize_points(img,points)
