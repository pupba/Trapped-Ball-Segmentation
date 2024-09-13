import cv2
import numpy as np
from imgutils import get_ball_structuring_element,get_unfilled_point,exclude_area,visualize_points

def trapped_ball_fill_single(image:np.ndarray, seed_point:tuple[int,int], radius:int)->np.ndarray:
    """단일 Trapped-ball 채우기 작업을 수행하는 함수
    
    Flood Fill : 다차원 배열의 어떤 칸과 연결된 영역을 찾는 알고리즘.

    # Arguments
        image: 흰색(빈 영역) 배경의 검은(채워진) 선화를 가진 이미지
        seed_point: 갇힌 공 채우기용 시드 포인트, 튜플 형식(int,int)
        radius: 구형 구조 요소의 반지름
    # Returns
        채운 이미지
    """
    ball = get_ball_structuring_element(radius) # radius로 구형 구조 요소 생성

    pass1 = np.full(image.shape, 255, np.uint8) # 전체가 흰색인 배열
    pass2 = np.full(image.shape, 255, np.uint8) # 전체가 흰색인 배열

    im_inv = cv2.bitwise_not(image) # 이미지 색상 반전

    # Flood fill 수행
    mask1 = cv2.copyMakeBorder(im_inv, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0) # 경계 추가
    _, pass1, _, _ = cv2.floodFill(pass1, mask1, seed_point, 0, 0, 0, 4) # flood fill

    # 이미지 팽창(dilation) 수행, 갭 사이의 영역이 분리됨.
    pass1 = cv2.morphologyEx(pass1, cv2.MORPH_DILATE, ball, anchor=(-1, -1), iterations=1)
    mask2 = cv2.copyMakeBorder(pass1, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0) # 경계 추가

    # 다시 시드 포인트로 Flood fill 수행하여 하나의 채움 영역 선택
    _, pass2, _, rect = cv2.floodFill(pass2, mask2, seed_point, 0, 0, 0, 4)
    # 채움 결과에 대하 침식(erosion) 수행하여 누수 방지 채움
    pass2 = cv2.morphologyEx(pass2, cv2.MORPH_ERODE, ball, anchor=(-1, -1), iterations=1)

    return pass2

def trapped_ball_fill_multi(image:np.ndarray, radius:int, method:str='mean', max_iter:int=1000)->np.ndarray:
    """여러번의 갇힌 공 채우기(Trapped-Ball Fill) 작업을 수행하여 모든 유효한 영역을 채운다.

    # Arguments
        image: 흰색(빈 영역) 배경의 검은(채워진) 선화를 가진 이미지
        radius: 구형 구조 요소의 반지름
        method: 채우기 결과 필터링 방법 
               'max' 는 일반적으로 큰 반지름을 사용하여 같은 큰 영역을 선택
        max_iter: 최대 반복 횟수
    # Returns
        채운 포인트 배열
    """

    unfill_area = image # 채우지 않은 영역을 우너본 이미지로 초기화
    filled_area, filled_area_size, result = [], [], []

    for _ in range(max_iter):
        # 경계에 가까운 포인트를 제외한 후 빈 영역의 포인트를 가졍옴
        points = get_unfilled_point(exclude_area(unfill_area, radius)) 

        # 빈 포인트가 없으면 종료
        if not len(points) > 0:
            break
        # 첫 번째 포인트를 기준으로 Trapped-Ball Fill 수행
        fill = trapped_ball_fill_single(unfill_area, (points[0][0], points[0][1]), radius)
        # 채운 영역과 원본 이미지의 비트 AND 연산 수행
        unfill_area = cv2.bitwise_and(unfill_area, fill)
        # 채운 영역의 포인트를 저장
        filled_area.append(np.where(fill == 0))
        filled_area_size.append(len(np.where(fill == 0)[0]))

    filled_area_size = np.asarray(filled_area_size) # 채운 영역의 크기를 배열로 받음


    # 필터링
    """
    ## max
    - 채운 영역의 크기 중 최대값을 기준으로 필터링, 가장 큰영역을 선택하여 결과에 포함, 그보다 작은 영역은 제외
    - 일반적으로 큰 반지름을 사용할 때 유용하며, 배경 같은 큰 영역을 선택할 때 적합
    - 이 방법은 노이즈가 작은 채움 결과를 무시하고, 주요 채운 영역만을 추출하고자 할 때 사용

    ## median
    - 채운 영역의 크기 중 중앙값을 기준으로 필터링, 중앙값은 배열을 정렬했을 때, 중간값에 위치한 값
    - 극단적인 값에 덜 영향을 받기 때문에, 이상치가 있는 경우에도 안정적인 결과를 제공
    - 다양한 크기의 채운 영역이 존재할 때, 중간 정도의 크기를 가진 영역을 선택하고자 할 때 사용
    - 크기가 매우 작은 영역과 매우 큰 영역이 혼재할 때, 중간 규모의 채운 영역을 선택하는 데 유용

    ## mean
    - 채운 영역의 크기 평균값을 기준으로 필터링
    - 모든 채운영역의 크기를 더한 후, 채운 영역의 개수로 나눈 값을 사용하여 필터링 기준을 설정
    - 평균적인 크기를 가진 채운 영역을 선택하고자 할 때 사용
    - 전체적인 채운 결과의 크기가 비슷할 때 적합하며, 크기가 크게 차이나지 않는 경우에 유용
    
    """
    if method == 'max':
        area_size_filter = np.max(filled_area_size)
    elif method == 'median':
        area_size_filter = np.median(filled_area_size)
    elif method == 'mean':
        area_size_filter = np.mean(filled_area_size)
    else:
        area_size_filter = 0

    # 필터링된 영역의 인덱스를 찾는다.
    result_idx = np.where(filled_area_size >= area_size_filter)[0]

    # 필터링 결과 리스트에 추가
    for i in result_idx:
        result.append(filled_area[i])

    return result # 채운 포인트의 배열 반환

def mark_fill(image:np.ndarray, fills:list)->np.ndarray:
    """채운 영역을 0으로 표시하는 역할

    # Arguments
        image: 타깃 이미지
        fills: 채운 영역의 포인트 배열, 각 포인트는 (y,x) 형태의 좌표를 포함해야 함.
    # Returns
        채운 영역이 0으로 표시된 이미지
    """
    result = image.copy()

    for fill in fills:
        result[fill] = 0

    return result

if __name__ == "__main__":
    # img = cv2.imread("./result/hemi/1.jpg",cv2.IMREAD_GRAYSCALE)
    img = cv2.imread("test.jpg",cv2.IMREAD_GRAYSCALE)
    # 1/2 resize 테스트를 위해
    h,w = img.shape
    # img = cv2.resize(img,(int(w//2),int(h//2)),interpolation=cv2.INTER_LANCZOS4)
    ret, binary = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)
    radius = 3
    cv2.imshow("origin",binary)
    max_points = trapped_ball_fill_multi(binary,3,"max")
    cv2.imshow("max",mark_fill(binary,max_points))
    med_points = trapped_ball_fill_multi(binary,3,"median")
    cv2.imshow("med",mark_fill(binary,max_points))
    mean_points = trapped_ball_fill_multi(binary,3,"mean")
    cv2.imshow("mean",mark_fill(binary,max_points))
    cv2.waitKey(0)
    cv2.destroyAllWindows()