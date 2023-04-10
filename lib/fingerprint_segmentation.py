import cv2 
import numpy as np


""" 
- The presence of peaks in local histograms of ridge orientations 
    - Mehtre, B. M., Murthy, N. N., Kapoor, S., & Chatterjee, B. (1987). 
    Segmentation of fingerprint images using the directional image. Pattern Recognition, 20(4), 429–435.
    
    방향성 이미지를 추출해 히스토그램을 통하여 원본이미지를 Segmentation한다.
    1. 히스토그램에서의 피크 값이 일정 기준치를 초과하면 영역은 명확히 전경에 속한다고 볼 수 있습니다. 
    2. 히스토그램에서 최대값과 최소값의 차이가 일정 기준치보다 작으면 영역이 배경에 속한다는 것을 나타냅니다. 
    3. 방향의 분산(분산의 합으로 계산)이 일정 기준치보다 작으면 배경을 나타내고, 그렇지 않으면 전경을 나타냅니다.
      분산 기준은 개선된 이미지에는 유효하지 않으며, 저대비 이미지에는 적합하지 않은 기준입니다. 
      반면, 방향 기준은 이러한 문제에서 자유롭고, 개선된 이미지와 원본 이미지 모두에 대해 좋은 결과를 제공합니다. 
      사실, 이 방법은 주름 방향에만 의존하기 때문에 개선이 필요하지 않습니다. 
      따라서 저대비 및 희미한 이미지에서도 동일하게 좋은 결과를 제공한다는 것이 논문의 주장이다


    픽셀의 8개의 방면으로 값의 차이를 구하여 가장 작은 차이를 가진 쪽을 방향이라 여기고, 방향의 히스토그램을 통하여 세그멘테이션한다.
    이미지가 크면 느림 << make_directional_image를 할때 pixel단위로 계산하기 때문이다. caching을 통해 구현을 하면 더욱 빨라질 수 있지만 거의 pixel의 개수를 n이라고 했을때 n*n으로 늘어나는 급이라 구현하진 않았다.
    더러운 지문이미지 내에서 세그멘테이션을 할때 효과가 좋다. 애초에 깔끔한 지문이미지면 사용을 할 이유가 없다.
    args :
        image : ndarray
        block_size : int 
        n : int  방향성을 구할때 얼마나 멀리까지 구할 것인가
    
"""

def segmentation_local_histogram(image:np.ndarray,block_size:int = 10,n:int =1) :
    height, width = image.shape

    # 방향성 이미지 초기화
    directional_image = np.zeros((height, width)).astype(np.uint8)

    # 8 방향의 좌표 변화를 저장하는 배열
    directions = np.array([[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]])
    for i in range(1, height-1):
        for j in range(1, width-1):
            # 현재 픽셀 좌표
            current_pixel = np.array([i, j])
            direction = None

            # 8 방향에 대해서
            direction_value = []
            for d in range(8):
                # n번 만큼 방향으로 이동하며 찾는 픽셀 좌표
                pixels_to_check = []

                for count in range(1, n+1):
                    pixel_to_check = current_pixel + directions[d] * count
                    if (0 <= pixel_to_check[0] < height) and (0 <= pixel_to_check[1] < width):
                        pixels_to_check.append(pixel_to_check)
                    else:
                        break

                # n방향의 픽셀 값 - 현재위치의 픽셀값의 합 계산
                diff = 0
                for pixel in pixels_to_check:
                    diff += image[pixel[0],pixel[1]] - image[current_pixel[0],current_pixel[1]]
                    
                direction_value.append(diff)
            direction = min(range(len(direction_value)),key = lambda i:direction_value[i])   
            
            # 방향을 저장한다. 1에서부터 8까지 반시계방향이다.
            directional_image[i, j] = direction+1

    # 방향성 이미지를 통한 세그멘테이션
    threshold = 0.5
    # 각 블록 내부의 분산을 계산하여 threshold로 활용
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = directional_image[i:i+block_size, j:j+block_size]
            var = np.var(block)
            # 임의의 비율 값(0.4)으로 threshold 설정
            segmented_image[i:i+block_size, j:j+block_size] = (threshold< var).astype(int)
    segmented_image =segmented_image.astype(np.uint8)
    
        
        
    return segmented_image



