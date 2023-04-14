import numpy as np
import cv2


def compute_crossing_number(values):
    return np.count_nonzero(values < np.roll(values, -1))

def minutiae_extraction(skeleton:np.ndarray) :
    """
    args : 
        skelteton : thinning 된 fingerprint image
    return : 
        minutiae : terminations, biufurcations의 좌표를 튜플로 묶고 두개의 구분은 True와 False로 한다. 이후 이것을 시각화 할때 활용함
        terminations : terminations의 이미지 skeleton과 크기가 같다.
        
    description : 
        crossing number 알고리즘을 활용하여 지문이미지의 특이점을 찾는다.
        Crossing Number 알고리즘 :
            1. 하나의 픽셀 선정 v라고 하겠음
            2.주변(8-neighborhood)에 대해서  cn(v)는 인접한 두 원소를 비교하여 왼쪽의 원소가 오른쪽의 원소보다 작은 경우의 수를 세는 함수입니다. cn(v) = sum(1 if v[i] < v[(i+1) % 8] else 0 for i in range(8))
            3.이렇게 구한 원소수가 1이면 terminations(지문 끝점) , 원소수가 3이면 biufurcations(분기점)이라 본다.
            4.이렇게 구하는 이유는 단순히 더 나아갈 방향이 없으면 원소수가 1이고 (선택한 픽셀 v에 대해서), 3이면 양방향으로 갈라지기 때문이다.
            4.위의 이미지로 궁금하다면 fingerprint_enhancement and Feature Extraction 부분을 참고하면된다.
       
            p[0] p[1] p[2]
            p[7]   v  p[3]
            p[6] p[5] p[4]

        cn_filter : 효율적인 계산을 위한 8방향 filter , 8bit를 통하여 lookup table을 만들꺼라 2^0~2^7까지의 숫자가 따로들어간다.
        all_8_neighborhoods : 8개의 비트의 모든 경우의 수다. ex)00000000,1000000....11111111
        cn_lut : all_8_neghborhoods로 만들어진 cn의 결과이다.
        thinning 된 이미지를 cn filter로 계산하고 픽셀의 값에 맞는 룩업테이블의 값을 전달해준다.
        걍 쉽게 이야기해서 filter를 통해서 나온 0~256의 결과에 대한 cn의 갯수를 lookuptable로 찾는거다.
        일일히 연산을 할필요없어서 속도가 빠르고 conv연산을 사용하기 때문에 더욱 더 효율적이다.
    reference : 
        https://www.semanticscholar.org/paper/Fingerprint-Recognition-Using-Minutia-Score-Ravi-Raja/5432561cbfc396051dae9f426346ff70c134124b
        https://colab.research.google.com/drive/1u5X8Vg9nXWPEDFFtUwbkdbQxBh4hba_M#scrollTo=RSGiRlE_zRqQ
    """
    # Create a filter that converts any 8-neighborhood into the corresponding byte value [0,255]
    cn_filter = np.array([[  1,  2,  4],
                        [128,  0,  8],
                        [ 64, 32, 16]
                        ])
    all_8_neighborhoods = [np.array([int(d) for d in f'{x:08b}'])[::-1] for x in range(256)]
    cn_lut = np.array([compute_crossing_number(x) for x in all_8_neighborhoods]).astype(np.uint8)

    # Skeleton: from 0/255 to 0/1 values
    skeleton01 = np.where(skeleton!=0, 1, 0).astype(np.uint8)
    # Apply the filter to encode the 8-neighborhood of each pixel into a byte [0,255]
    cn_values = cv2.filter2D(skeleton01, -1, cn_filter, borderType = cv2.BORDER_CONSTANT)
    # Apply the lookup table to obtain the crossing number of each pixel
    cn = cv2.LUT(cn_values, cn_lut)
    # Keep only crossing numbers on the skeleton
    cn[skeleton==0] = 0

    terminations = np.zeros_like(cn)
    terminations[cn == 1] = 255
    biufurcations = np.zeros_like(cn)
    biufurcations[cn == 3] = 255
    minutiae = [(x,y,cn[y,x]==1) for y, x in zip(*np.where(np.isin(cn, [1,3])))]
    
    return minutiae, terminations, biufurcations

def filtering_minutiae(minutiae:np.ndarray,mask:np.ndarray) :
    #cv2.distanceTransform 함수는 입력으로 주어진 이진 이미지에서 객체 경계선까지의 거리를 계산합니다. 이때 객체 경계선은 입력 이미지에서 객체와 배경을 구분하는 경계입니다. 객체 경계선은 이진화된 이미지에서 객체와 배경의 경계가 되는 픽셀들의 집합으로 정의됩니다.
    # 일반적으로 cv2.distanceTransform 함수는 입력 이미지에서 객체와 배경의 경계가 흰색(255)으로 표시되어 있다고 가정합니다. 이 경우, 입력 이미지를 1 픽셀만큼 확장한 후, 경계를 제외한 모든 픽셀을 검은색(0)으로 채워진 이미지를 생성합니다. 이후 cv2.distanceTransform 함수를 이 확장된 이미지에 적용하여, 각 픽셀으로부터 가장 가까운 객체 경계선까지의 거리를 계산합니다.
    # 결과적으로, cv2.distanceTransform 함수는 입력 이미지의 객체 경계선을 이용하여 객체와 배경 사이의 거리를 계산합니다. 이 때 객체와 배경을 구분하는 경계는 입력 이미지에서 미리 지정된 값(일반적으로 255)으로 표시되어 있어야 합니다.
    # 이걸 이 코드에 적용시키면 결국 mask_distance에 맞는 좌표의 minutiae가 10이상이 되어야 한다는건 배경과 10정도 떨어져 있어야됨을 의미합니다.
    # 지문 외각선에 대한 특징점을 찾으면 결국 좋지 않은결과가 나타나기 때문에 사용합니다.

    mask_distance = cv2.distanceTransform(cv2.copyMakeBorder(mask*255, 1, 1, 1, 1, cv2.BORDER_CONSTANT), cv2.DIST_C, 3)[1:-1,1:-1]
    filtered_minutiae = list(filter(lambda m: mask_distance[m[1], m[0]]>10, minutiae))
    return filtered_minutiae ,mask_distance
    