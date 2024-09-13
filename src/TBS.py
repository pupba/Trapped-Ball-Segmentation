import cv2
import numpy as np
from trappedball_fill import trapped_ball_fill_multi,mark_fill
from imgutils import build_fill_map,merge_fill,flood_fill_multi
from flattening import thinning,show_fill_map

def pipeline(image:np.ndarray,radius:int=3,methods=["max",None,None])->np.ndarray:
    ret, binary = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY)
    unfilled = binary
    radius = radius

    fills = []
    for method in methods:
        fill = trapped_ball_fill_multi(image=unfilled,radius=3,method=method)
        fills+=fill
        unfilled = mark_fill(unfilled,fills)

    fill = flood_fill_multi(unfilled)
    fills += fill

    fillmap = build_fill_map(unfilled, fills)
    fillmap = merge_fill(fillmap)
    cv2.imshow("color",cv2.convertScaleAbs(show_fill_map(fillmap)))
    
    # visualization
    result = show_fill_map(thinning(fillmap))
    return result

if __name__ == "__main__":
    img = cv2.imread("./test.jpg",cv2.IMREAD_GRAYSCALE)
    cv2.imshow("origin",img)
    fill = pipeline(img)
    cv2.imshow("flatten",cv2.convertScaleAbs(fill))
    cv2.waitKey(0)
    cv2.destroyAllWindows()