
import numpy as np
import cv2
import matplotlib as plt
from scipy.signal import convolve2d
from scipy import signal
from image_processing import segmentation_maksed_thresholding
from image_processing import normalize_with_mask
from image_processing import convolve





def estimate_orientation(image:np.ndarray, block_size:int=16, block_size_lp:int=3,smooth_flag:bool=True,interpolate:bool = True):
    """
    Args : 
        image : np.ndarray
        block_size: int, window (block) size 
        block_size_lp : lowpass filter window size 
        smooth_flag : block 사이사이 노이즈가 많을시 사용하면됨

    Return : 
        orientations :np.ndarray

    Description :
        주어진 이미지에 대한 block wise한 orientation 계산이다. 
        전체 이미지에 대한 x,y에 대한 gradient를 구한다. (굳이 sobel이 아닌 다른 방법도 상관없다)
        gradient를 얻은 이미지에 대해 블록 단위의 연산을 하여, orientation을 구한다. 
        만약 블록단위의 연산을 할때 갑자기 방향성이 튀는 경우가 있다. 지문만이 segmentaiton 되었다고 하였을때는, 연속된 방향성을 지니는 지문에서는 노이즈로 간주할 수 있다.
        이 노이즈를 없애주기 위한 block 단위의 lowpass filter를 사용하여, orientation을 구할 수 있다.
        수식에 대한 설명은 논문에 없었으나 (A. Ravishankar Rao (auth.) - A Taxonomy for Texture Description and Identification-Springer-Verlag New York (1990) 찾을 수 있다)
        
    Reference :
         reference : 
        Hong, l., wan, y., & jain, a. k. (1998). fingerprint image enhancement: algorithms and performance evaluation. ieee transactions on pattern analysis and machine intelligence, 20(8), 777–789.
        
    """
    #노이즈를 없애고 조금 더 ridge를 늘어뜨리기
    image = cv2.GaussianBlur(image, (5, 5), 1)
    # block_size만큼 이미지를 나누기
    n_rows, n_cols = image.shape
    n_blocks_row = n_rows // block_size
    n_blocks_col = n_cols // block_size
    
    #sobel filter conv 계산
    dx = convolve2d(image, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), mode='same')
    dy = convolve2d(image, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]), mode='same')

    # block 마다의 orientation 계산
    orientations = np.zeros((n_blocks_row, n_blocks_col))
    for i in range(n_blocks_row):
        for j in range(n_blocks_col):
            # 블락의 범위 설정
            block_x_start = i*block_size
            block_x_end = block_x_start + block_size
            block_y_start = j*block_size
            block_y_end = block_y_start + block_size

            # 블락의 범위에 맞는 sobel filter의 결과물을 block_dx와 block_dy에 넣음
            block_dx = dx[block_x_start:block_x_end, block_y_start:block_y_end]
            block_dy = dy[block_x_start:block_x_end, block_y_start:block_y_end]

            # 블락의 방향성을 구함 Vx ,Vy 계산 논문 수식임 
            #Hong, l., wan, y., & jain, a. k. (1998). fingerprint image enhancement: algorithms and performance evaluation. ieee transactions on pattern analysis and machine intelligence, 20(8), 777–789.
            #(5)(6)을 찾아보면 나옴
            Vx = np.sum(2 * block_dx * block_dy)
            Vy = np.sum(block_dx**2 - block_dy**2)

            # 라디안값을 도로 바꿔주고, VX와 Vy 둘다 0이라면 방향성이 없는걸로 간주하고 0을 넣는다.
            #Hong, l., wan, y., & jain, a. k. (1998). fingerprint image enhancement: algorithms and performance evaluation. ieee transactions on pattern analysis and machine intelligence, 20(8), 777–789.
            #(12)을 찾아보면 나옴
            if Vx or Vy :
                phi =0.5*np.arctan2(Vx, Vy)
                phi = phi + np.pi/2 if phi < 0 else phi - np.pi/2 
                #왼쪽으로 흐르는 지문이미지와 오른쪽으로 흐르는 지문이미지의 방향성을 조절합니다.
                #정확하게는 밑에서 arctan는 -파이에서 +파이까지의 값을 출력합니다.
                # 하지만 0~180도로 고정할때 np.mod를 사용하는데 np.mod의 나머지값은 음수 양수를 고려를 안해주기 때문에 이 작업을 해주지 않으면 문제가 발생할 수 있습니다. 
                #90도를 각도가 양수일때는 더해주고 음수일때는 90도를 뺍니다.
            else :
                phi = 0
            

            # Save the dominant orientation for the block
            orientations[i, j] = phi
            orientations = np.mod(orientations, np.pi) 

        

    if smooth_flag :
        #low pass filter
        phi_x = np.cos(2 * orientations)
        phi_y = np.sin(2 * orientations)
        
        #smoothing할 block size를 따로 w_f라한다.
        #block_size를 통하여 평균필터를 만든다.
        w_f = block_size_lp
        mean_filter = np.ones((w_f, w_f))
        mean_filter /= np.sum(mean_filter)

        
        # Apply the filter in a sliding window manner
        phi_x_smoothed = convolve2d(phi_x, mean_filter, mode='same')
        phi_y_smoothed = convolve2d(phi_y, mean_filter, mode='same')
        
        # 위의 2*orientation을 다시 원상복귀 시켜준다.
        orientations = np.arctan2(phi_y_smoothed, phi_x_smoothed) /2
    
    
    #frequency 계산을 위해 interpolate 한다.
    if interpolate:
        orientations = cv2.resize(orientations,image.shape,interpolation=cv2.INTER_NEAREST)
    
    
    return orientations



def show_orientations(image:np.ndarray, orientations:np.ndarray, label:str, w:int=16, vmin:float=0.0, vmax:float=255.0):
    """
    args : 
        image : 방향성을 그릴 원본 이미지
        orientation : 방향성 이미지
        label : plt의 label
        w : orientation을 그리기 위한 블락 사이즈
        vmin : 이미지 pixel 최솟값
        vmax : 이미지 pxiel 최댓값
    description :
    orientation 이미지를 시각화 하기 위한 코드.
    cos과 sin 함수는 각각 x, y축 방향의 이동 거리를 계산하는 데 사용됩니다. 
    이 함수들은 특정 각도에 대한 cos과 sin 값을 반환하는데, 이때의 각도는 라디안(radian) 값으로 입력되어야 합니다.
    따라서 위 코드에서는 주어진 방향(orientation) 값으로부터 cos과 sin 값을 계산하고, 이를 이용하여 시작점과 끝점을 결정합니다. 
    시작점(cx, cy)에서 cos과 sin 값에 비례하는 거리(w0.5)만큼 x, y 방향으로 이동한 끝점을 구한 후, 두 점을 선으로 이어주는 plot 함수를 사용하여 해당 방향을 시각화합니다. 
    선의 길이 또한 조절 가능하지만 인자로 굳이 만들지는 않았습니다.
    """

    #orientation을 그릴 이미지 띄우기
    # plt.figure().suptitle(label)
    plt.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)

    height, width = image.shape
    #블락만큼 돌면서 중심좌표를 구한 뒤 그에 맞는 방향성을 계산해 그린다.
    for y in range(0, height, w):
        for x in range(0, width, w):
            if np.any(orientations[y : y + w, x : x + w] == 0.0):
                continue

            cy = (y + min(y + w, height)) // 2
            cx = (x + min(x + w, width)) // 2
            if y+w//2 > height-1 or x+w//2 >width-1 :
                continue 
                
            orientation = orientations[y + w // 2, x + w // 2]
            plt.plot(
                [
                    cx - w * 0.5 * np.cos(orientation),
                    cx + w * 0.5 * np.cos(orientation),
                ],
                [
                    cy - w * 0.5 * np.sin(orientation),
                    cy + w * 0.5 * np.sin(orientation),
                ],
                "r-",
                lw=1.0,
            )
import scipy.ndimage as ndimage


def normalize(image):
    image = np.copy(image)
    image -= np.min(image)
    m = np.max(image)
    image = image.astype(np.float64)
    if m > 0.0:
        image *= 1.0 / m
    return image

def rotateAndCrop(image:np.ndarray, angle):
    """
    args :
        image : frequency를 구하기 위한 블락단위의 이미지 
        angle : orientation 이미지로 부터 얻은 블락단위의 gradient

    return :
        방향성에 따라 rotation된 블락 이미지 
    description :
        Fingerprint image enhancement: Algorithm and performance evaluation
        Hong, L., Wan, Y. & Jain, A. (1998) 논문의 구현을 따라 지문의 ridge 방향의 주파수를 얻기 위해 orientation의 방향의 반대로 
    
    
    Rotate an image and crop the result so that there are no black borders.
    This implementation is based on this stackoverflow answer:
        http://stackoverflow.com/a/16778797
    
    """

    h, w = image.shape

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(np.sin(angle)), abs(np.cos(angle))
    if side_short <= 2.0 * sin_a * cos_a * side_long:
        # half constrained case: two crop corners touch the longer side,
        # the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

    image = ndimage.interpolation.rotate(image, np.degrees(angle), reshape=False)

    hr, wr = int(hr), int(wr)
    y, x = (h - hr) // 2, (w - wr) // 2

    return image[y : y + hr, x : x + wr]

def estimate_frequencies(image:np.ndarray, orientations:np.ndarray, w:int=32):
    """
    args : 
        image : normalization and segmenation 된 이미지
        orientaiton : 방향성 이미지
        w : window 크기
    return np.ndarray frequency field

    reference : 
        based code :
            https://github.com/tommythorsen/fingerprints/blob/master/utils.py

        the paper:
        Fingerprint image enhancement: Algorithm and performance evaluation
        Hong, L., Wan, Y. & Jain, A. (1998)

    Description :
        1. window 사이즈만큼 블락 단위로 만든다.
        2. rotateAndCrop 함수로 orientation의 방향성으로 rotate 한 뒤 필요없는 검은 부분을 crop한다.
            2.1 crop을 할때 릿지 모양이 꽤나 많이 보일 수록 더 좋은 결과를 띄기 때문에 적절한 w설정이 중요해 보인다.
        3. rotation crop한 블록의 colums의 peacks를 구하여 주파수를 계산한다. 
        
    """
    #1.
    height, width = image.shape
    yblocks, xblocks = height // w, width // w
    F = np.empty((yblocks, xblocks))
    #2.
    for y in range(yblocks):
        for x in range(xblocks):
            orientation = orientations[y * w + w // 2, x * w + w // 2]
            block = image[y * w : (y + 1) * w, x * w : (x + 1) * w]
            block = rotateAndCrop(block, np.pi * 0.5 + orientation)
            if block.size == 0:
                F[y, x] = -1
                continue

            columns = np.sum(block, (0,))#block의 열의 값을 다 더해준다. 이 block은 논문 fig8에 나오는 Oriented Window다.
            columns = normalize(columns)
            peaks = signal.find_peaks_cwt(columns, np.array([3]))#peaks를 구해준다. 이부분은 수식 공부가 필요 
            if len(peaks) < 2: #peaks의 길이가 2개 이하면 block이 지문이 아니라고 판단
                F[y, x] = -1
            else: #peaks를 통하여 지문의 freq를구한다. 
                #논문과 다른 기준치로 구분을 하지만 실제로 논문에서도 300x300 resoltion에서 1/3,1/25를 freq의 기준치로 잡아서 -1를 넣어 구분한다. 이유는 공부해야함
                f = (peaks[-1] - peaks[0]) / (len(peaks) - 1)
                if f < 5 or f > 15:
                    F[y, x] = -1
                else:
                    F[y, x] = 1 / f
    frequencies = np.full(image.shape, -1.0)
    F = np.pad(F, 1, mode="edge") #가장자리 값을 그대로 padding
    for y in range(yblocks):
        for x in range(xblocks):
            surrounding = F[y : y + 3, x : x + 3]
            surrounding = surrounding[np.where(surrounding >= 0.0)]
            if surrounding.size == 0:
                frequencies[y * w : (y + 1) * w, x * w : (x + 1) * w] = -1
            else:
                frequencies[y * w : (y + 1) * w, x * w : (x + 1) * w] = np.median(
                    surrounding
                )

    return frequencies





def gaborKernel(size, angle, frequency):
    """
    args :
        size : 커널 사이즈
        angle : 블락 이미지의 각도 

    description :
        블락마다의 가버커널을 만들어주기 위해 사용한다.
        hyper parametar로 sigma로 sigma가 더 작아질수록 고주파성분을 끌어내지만 세부내용이 사라질 수 있어 유의해야한다.

    reference : 
        take code : https://github.com/rtshadow/biometrics.git

    """

    angle += np.pi * 0.5
    cos = np.cos(angle)
    sin = -np.sin(angle)

    yangle = lambda x, y: x * cos + y * sin
    xangle = lambda x, y: -x * sin + y * cos

    xsigma = ysigma = 5
    f = lambda x,y : np.exp(-((xangle(x, y) ** 2) / (xsigma ** 2) +(yangle(x, y) ** 2) / (ysigma ** 2)) / 2) *np.cos(2 * np.pi * frequency * xangle(x, y))
    kernel = np.empty((size, size))
    for i in range(0, size):
        for j in range(0, size):
            kernel[i, j] = f(i - size / 2, j - size / 2)

    return kernel


def gaborFilter(image:np.ndarray, orientations:np.ndarray, frequencies:np.ndarray, w:int=32):
    """
    args :
        image : normalization된 지문 이미지 
        orientations : normalization된 지문이미지의 방향성
        frequencies : 지문이미지의 frequency (고주파 성분)
        w : ori,freq를 구한것과 같은 block size

    return : 
        result : enhancement된 이미지 np.ndarray

    description : 

    """
    result = np.empty(image.shape)

    height, width = image.shape
    for y in range(0, height - w, w):
        for x in range(0, width - w, w):
            
            orientation = orientations[y:y+w, x:x+w]
            frequency = frequencies[y:y+w, x:x+w]

            orientation = orientation[np.where(orientation >= 0.0)]
            frequency = frequency[np.where(frequency >= 0.0)]

            if orientation.size ==0 :
                orientation = -1
            else :
                orientation = np.mean(orientation [np.where(orientation >= 0.0)])
            
            if frequency.size == 0 :
                frequency =-1 
            else :
                frequency = np.mean(frequency)
    
            if frequency < 0.0:
                result[y:y+w, x:x+w] = image[y:y+w, x:x+w]
                continue

            kernel = gaborKernel(16, orientation, frequency)
            result[y:y+w, x:x+w] = convolve(image, kernel, (y, x), (w, w))
    return result

def binarize(image:np.ndarray, w:int=16,max_val:float =255.0, min_val:float =0.0):
    """
    args:
        image : 이진화할 이미지
        w : block size
        mav_val : 이진화된 이미지의 최대 값 float
        min_val : 이진화된 이미지의 최솟 값 float
    return : 
        image : 이진화된 이미지(default 255,0)
    description:   
        지역 이진화 블락의 평균값을 활용하여 threshold를 지정한다.
        기본적으로 255,0인 값인 이진화 이미지를 return한다. (이후의 있을 thinning을 위해서)
    """

    image = np.copy(image)
    height, width = image.shape
    for y in range(0, height, w):
        for x in range(0, width, w):
            block = image[y : y + w, x : x + w]
            threshold = np.average(block)
            image[y : y + w, x : x + w] = np.where(block >= threshold, max_val, min_val)

    return image


#thinning 알고리즘 내에 - 연산이 있기때문에 0~255로 변환을 해야 본코드에선 적용가능 다른 thinning알고리즘은 0~1로해도 상관없다.

def thinning_image(img:np.ndarray,mask) :
    """  
    args : 
        img : 이진화된 이미지(0~255) , (0~1)인 이진화 이미지를 넣는다면 ridge정보가 다 사라짐

    return :
        skel : thinning image 

    description :
        전처리로 이진화된 이미지를 가우시안필터로 처리한 이후 ZHANGSUEN Thinning 알고리즘을 적용한다.

    """
    arr1 = np.array([1, -2, 3, -4, 5])
    arr2 = np.array([-1, -2, -3, -4, -5])
    condition = arr1 > 0
    condition = mask >0 
    img = np.where(condition,img,0)
    img= cv2.GaussianBlur(img, (5, 5), 0).astype(np.uint8)
    skel = np.zeros(img.shape, np.uint8)
    # element = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
    # done = False
    # while not done:
    #     eroded = cv2.erode(img, element)
    #     temp = cv2.dilate(eroded, element)
    #     temp = img-temp 
    #     temp = temp.astype(np.uint8)
    #     skel = np.bitwise_or(skel, temp)
    #     img = eroded.copy()
    #     if np.all(img==0):
    #         done = True
    skel = cv2.ximgproc.thinning(img, cv2.ximgproc.THINNING_ZHANGSUEN)
    

    return skel
    


def thinning_image2(img:np.ndarray,kernel_size=(3,3)) :
    """  
    args : 
        img : 이진화된 이미지(0~255) , (0~1)인 이진화 이미지를 넣는다면 ridge정보가 다 사라짐

    return :
        thinning 이미지

    description :
        선처리 :
            1.3x3 크기의 cross-shaped(십자) 커널을 생성하여 이미지를 erode합니다.
            2.모폴로지 Closing 연산을 한 이후 temp에 저장
            3.skel에 temp와 bitwise_or 연산을 수행하여 저장.
            4.erode된 이미지를 다시 원래 이미지로 복사합니다.
            위의 과정을 반복하면서, Closing을 통해 img가 모두 0이 되면 끝.
        후처리 :
            skel 이미지에 대하여 다시 thinning을 수행합니다. 이때는 Zhang-Suen 알고리즘을 사용합니다.
            thinning한 결과물을 반환합니다.
    """

    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
    done = False
    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = img-temp 
        temp = temp.astype(np.uint8)
        skel = np.bitwise_or(skel, temp)
        img = eroded.copy()
        if np.all(img==0):
            done = True
    skel = cv2.ximgproc.thinning(skel, cv2.ximgproc.THINNING_ZHANGSUEN)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (2,2))
    skel = cv2.dilate(skel,element)

    return skel
    

def fingerprint_pipeline(image:np.ndarray,block:int =32) :
    seg_img,mask = segmentation_maksed_thresholding(image,block=block)
    norm_img = normalize_with_mask(seg_img,mask=mask)
    ori_img = estimate_orientation(norm_img)
    freq_img =estimate_frequencies(norm_img,ori_img,w=block)
    ehance_img= gaborFilter(image,ori_img,freq_img,w=block)
    bi_img = binarize(ehance_img,max_val=255,min_val=0) 
    th_img = thinning_image(bi_img,mask)
    return th_img, mask

# 논문 구현 실패
# import numpy as np
# from scipy.signal import convolve2d

# def estimate_local_ridge_frequency(image, orientation_map, block_size=(16,16), ori_window_size=(32,16), freq_range=(1/25, 1/3)):

#     height, width = image.shape
#     block_h, block_w = block_size
#     ori_window_h, ori_window_w = ori_window_size
#     freq_min, freq_max = freq_range
#     freq_map = np.zeros((height, width))
#     ot_len = len(orientation_map)
    
#     for i,i_o in zip(range(0, height - (height % block_h), block_h),range(ot_len)):
#         for j,j_o in zip(range(0,  width - (width % block_w), block_w),range(ot_len)):
            
#             i_center = i + block_h // 2
#             j_center = j + block_w // 2
#             orientation = orientation_map[i_o,j_o]
            
#             ori_window = np.zeros((ori_window_h, ori_window_w))
           
#             for k in range(ori_window_h):
#                 for l in range(ori_window_w):
#                     u = i_center + (l - ori_window_w//2) * np.sin(orientation) + (k - ori_window_h//2) * np.cos(orientation)
#                     v = j_center + (l - ori_window_w//2) * np.cos(orientation) - (k - ori_window_h//2) * np.sin(orientation)
#                     if 0 <= u < height and 0 <= v < width: #좌표저장
#                         ori_window[k, l] = image[int(u), int(v)]
           
#             # Step 3: Compute x-signature
#             x_signature = np.zeros(ori_window_w)
#             for k in range(ori_window_w):
#                 sum_g = 0
#                 for d in range(ori_window_h):
#                     if 0 <= u < height and 0 <= v < width:
#                         sum_g += ori_window[int(d), int(k)]
#                 x_signature[k] = sum_g / ori_window_h
            
          
#             # Check for consecutive peaks to estimate frequency
#             peak_indices = np.where((x_signature[1:-1] > x_signature[:-2]) & (x_signature[1:-1] > x_signature[2:]))[0] + 1
        
#             if len(peak_indices) > 1:
#                 avg_distance = np.mean(np.diff(peak_indices))
#                 freq = 1 / avg_distance
#                 if freq_min <= freq <= freq_max:
#                     freq_map[i-block_h//2:i+block_h//2, j-block_w//2:j+block_w//2] = freq
#             else:
#                 freq_map[i-block_h//2:i+block_h//2, j-block_w//2:j+block_w//2] = -1

#     # Step 4: Assign -1 to invalid frequency values
#     freq_map[np.where((freq_map < freq_min) | (freq_map > freq_max))] = -1
    
#     # Step 5: Interpolate frequency values for invalid blocks
#     kernel = np.ones((7,7)) / 49
#     freq_map_convolved = convolve2d(freq_map, kernel, mode='same', boundary='symm')
#     freq_map[np.where(freq_map == -1)] = freq_map_convolved[np.where(freq_map == -1)]
    
#     # Step 6: Apply low-pass filter
#     filter_kernel = np.ones((7,7)) / 49
#     filtered_freq_map = convolve2d(freq_map, filter_kernel, mode='same', boundary='symm')
#     return filtered_freq_map

# freq_img = estimate_local_ridge_frequency(img,ori_img)
# print(freq_img)
# plt.imshow(freq_img)