
import numpy as np
import cv2
import math
import matplotlib as plt
from scipy.signal import convolve2d
from scipy import signal


def normalization_with_mask(img:np.ndarray,desire_var_ratio:float =0.7,mask:np.ndarray=None):
    """ 
    Args :
        img: 입력 이미지 
        desire_var_ratio: 목표 분산 비율
        mask : segmentation mask 
    Return : normalized_img : np.ndarray 
    
    Description :
        Desired mean과 desired variance는 이미지의 특성에 따라 다르게 설정될 수 있습니다. 따라서 이 값들을 추출하기 위해서는 해당 이미지를 미리 분석하여 적절한 값을 설정하는 것이 좋습니다.
        일반적으로, desired mean은 이미지에서 밝기의 중간값(median)을 사용하는 것이 일반적입니다. 이는 이미지 전체의 밝기 대략적인 중심값을 나타내므로, 이미지의 대부분의 부분에서 너무 밝거나 어두운 영역이 발생하지 않도록 하는 데 도움이 됩니다.
        반면에, desired variance는 이미지의 밝기 변화의 정도에 따라 다르게 설정될 수 있습니다. 예를 들어, 밝기 변화가 큰 이미지의 경우 작은 값으로 설정하여 이미지의 대비를 높일 수 있습니다. 하지만 밝기 변화가 작은 이미지의 경우 큰 값으로 설정하여 이미지를 부드럽게 만들 수 있습니다.
        desired mean과 desired variance를 추출하는 것은 해당 이미지의 특성을 고려하여 적절한 값으로 설정하는 것이 중요합니다.

        Customize
        *mask를 통하여 ROI부분만 Nomalize합니다. 
        *이미지의 var과 median을 통하여 desire mean과 desire var를 결정합니다. 이 때 desire_var_ratio를 통하여 이미지에 맞게 목표분산을 바꿔주면 됩니다. 
    reference : 
        Hong, l., wan, y., & jain, a. k. (1998). fingerprint image enhancement: algorithms and performance evaluation. ieee transactions on pattern analysis and machine intelligence, 20(8), 777–789.
    """
    
    
    
    normalized_img = np.zeros_like(img) # 정규화된 이미지

    if mask is None :
        m = np.mean(img) # 입력 이미지의 평균\
        var = np.var(img) # 입력 이미지의 분산
        desire_mean = np.median(img)
        desire_var = var*desire_var_ratio
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i,j] > m:
                    normalized_img[i,j] = desire_mean + np.sqrt((desire_var * (img[i,j]-m)**2) / var)
                else:
                    normalized_img[i,j] = desire_mean - np.sqrt((desire_var * (img[i,j]-m)**2) / var)

    else :
        #마스크에서 0이 아닌부분의 값만 평균과 분산을 낸다.
        mask_indices = np.argwhere(mask != 0)
        # 각 위치의 원본 이미지의 픽셀 값을 가져옵니다.
        pixel_value = img[mask_indices[:, 0], mask_indices[:, 1]] #0이 아닌부분의 픽셀값
        m = np.mean(pixel_value)
        var = np.var(pixel_value)

        desire_mean = np.median(pixel_value)
        desire_var = var*desire_var_ratio
        for i,j in zip(mask_indices[:, 0],mask_indices[:, 1]):
                if img[i,j] > m:
                    normalized_img[i,j] = desire_mean + np.sqrt((desire_var * (img[i,j]-m)**2) / var)
                else:
                    normalized_img[i,j] = desire_mean - np.sqrt((desire_var * (img[i,j]-m)**2) / var)
    return normalized_img



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
    #Reduce noise
    image = cv2.GaussianBlur(image, (5, 5), 1)
    # Divide image into blocks of size block_size x block_size
    n_rows, n_cols = image.shape
    n_blocks_row = n_rows // block_size
    n_blocks_col = n_cols // block_size
    
    # Compute gradients using Sobel operator

    dx = convolve2d(image, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), mode='same')
    dy = convolve2d(image, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]), mode='same')

    # Estimate local orientation for each block
    orientations = np.zeros((n_blocks_row, n_blocks_col))
    for i in range(n_blocks_row):
        for j in range(n_blocks_col):
            # Define the boundaries of the block
            block_x_start = i*block_size
            block_x_end = block_x_start + block_size
            block_y_start = j*block_size
            block_y_end = block_y_start + block_size

            # Compute the Fourier transform of the block
            block_dx = dx[block_x_start:block_x_end, block_y_start:block_y_end]
            block_dy = dy[block_x_start:block_x_end, block_y_start:block_y_end]

            # Compute Vx and Vy
            Vx = np.sum(2 * block_dx * block_dy)
            Vy = np.sum(block_dx**2 - block_dy**2)

            # Compute the dominant orientation
            if Vx or Vy :
                phi = 0.5 * np.arctan2(Vx, Vy)
                phi = phi + np.pi/2 if phi < 0 else phi - np.pi/2
            else :
                phi = 0
            

            # Save the dominant orientation for the block
            orientations[i, j] = phi
            orientations = np.mod(orientations, np.pi)

        

    if smooth_flag :
        #Low-pass filter the orientation image
        phi_x = np.cos(2 * orientations)
        phi_y = np.sin(2 * orientations)
        
        # Define the low-pass filter
        w_f = block_size_lp
        block_size = np.ones((w_f, w_f))
        block_size /= np.sum(block_size) # Ensure unit integral
        
        # Apply the filter in a sliding window manner
        phi_x_smoothed = convolve2d(phi_x, block_size, mode='same')
        phi_y_smoothed = convolve2d(phi_y, block_size, mode='same')
        
        # Convert the smoothed vector field back to an orientation field
        orientations = np.arctan2(phi_y_smoothed, phi_x_smoothed) / 2
    
    

    if interpolate:
        orientations = cv2.resize(orientations, (500,500),interpolation=cv2.INTER_NEAREST)
    
    
    return orientations


def showOrientations(image:np.ndarray, orientations:np.ndarray, label:str, w:int=16, vmin:float=0.0, vmax:float=1.0):
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
    plt.figure().suptitle(label)
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

def estimateFrequencies(image:np.ndarray, orientations:np.ndarray, w:int=32):
    """
    args : 
        image : normalization and segmenation 된 이미지
        orientaiton : 방향성 이미지
        w : window 크기
    return np.ndarray frequency field

    reference : 
        code :
            https://github.com/tommythorsen/fingerprints/blob/master/utils.py

        the paper:
        Fingerprint image enhancement: Algorithm and performance evaluation
        Hong, L., Wan, Y. & Jain, A. (1998)

    Description :
        1. window 사이즈만큼 블락 단위로 만든다.
        2. rotateAndCrop 함수로 orientation의 방향성으로 rotate 한 뒤 필요없는 검은 부분을 crop한다.
            2.1 crop을 할때 릿지 모양이 꽤나 많이 보일 수록 더 좋은 결과를 띄기 때문에 적절한 w설정이 중요해 보인다.
        3. rotation crop한 블록의 colums의 peacks를 구하여 주파수를 계산한다. 
            3.1 대략적인 이해는 되었으나 이부분은 위의 코드들을 대부분 인용한 것이라 이해하기 어려움 논문의 내용과 달라보인다.
        
    """
    height, width = image.shape
    yblocks, xblocks = height // w, width // w
    F = np.empty((yblocks, xblocks))
    for y in range(yblocks):
        for x in range(xblocks):
            count+=1
            orientation = orientations[y * w + w // 2, x * w + w // 2]

            block = image[y * w : (y + 1) * w, x * w : (x + 1) * w]
            block = rotateAndCrop(block, np.pi * 0.5 + orientation)
            if block.size == 0:
                F[y, x] = -1
                continue

            columns = np.sum(block, (0,))
            columns = normalize(columns)
            peaks = signal.find_peaks_cwt(columns, np.array([3]))
            if len(peaks) < 2:
                F[y, x] = -1
            else:
                f = (peaks[-1] - peaks[0]) / (len(peaks) - 1)
                if f < 5 or f > 15:
                    F[y, x] = -1
                else:
                    F[y, x] = 1 / f
    frequencies = np.full(image.shape, -1.0)
    F = np.pad(F, 1, mode="edge")
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


def convolve(image, kernel, origin=None, shape=None, pad=True):
    """
    Apply a kernel to an image or to a part of an image.
    :param image:   The source image.
    :param kernel:  The kernel (an ndarray of black and white, or grayvalues).
    :param origin:  The origin of the part of the image to be convolved.
                    Defaults to (0, 0).
    :param shape:   The shape of the part of the image that is to be convolved.
                    Defaults to the shape of the image.
    :param pad:     Whether the image should be padded before applying the
                    kernel. Passing False here will cause indexing errors if
                    the kernel is applied at the edge of the image.
    :returns:       The resulting image.
    """
    if not origin:
        origin = (0, 0)

    if not shape:
        shape = (image.shape[0] - origin[0], image.shape[1] - origin[1])

    result = np.empty(shape)

    if callable(kernel):
        k = kernel(0, 0)
    else:
        k = kernel

    kernelOrigin = (-k.shape[0] // 2, -k.shape[1] // 2)
    kernelShape = k.shape

    topPadding = 0
    leftPadding = 0

    if pad:
        topPadding = max(0, -(origin[0] + kernelOrigin[0]))
        leftPadding = max(0, -(origin[1] + kernelOrigin[1]))
        bottomPadding = max(
            0,
            (origin[0] + shape[0] + kernelOrigin[0] + kernelShape[0]) - image.shape[0],
        )
        rightPadding = max(
            0,
            (origin[1] + shape[1] + kernelOrigin[1] + kernelShape[1]) - image.shape[1],
        )

        padding = (topPadding, bottomPadding), (leftPadding, rightPadding)

        if np.max(padding) > 0.0:
            image = np.pad(image, padding, mode="edge")

    for y in range(shape[0]):
        for x in range(shape[1]):
            iy = topPadding + origin[0] + y + kernelOrigin[0]
            ix = leftPadding + origin[1] + x + kernelOrigin[1]

            block = image[iy : iy + kernelShape[0], ix : ix + kernelShape[1]]
            if callable(kernel):
                result[y, x] = np.sum(block * kernel(y, x))
            else:
                result[y, x] = np.sum(block * kernel)

    return result


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


def gaborFilter(image, orientations, frequencies, w=32):
    """
    
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

def binarize(image, w=16):
    """
    Perform a local binarization of an image. For each cell of the given size
    w, the average value is calculated. Every pixel that is below this value,
    is set to 0, every pixel above, is set to 1.

    :param image: The image to be binarized.
    :param w:     The size of the cell.
    :returns:     The binarized image.
    """

    image = np.copy(image)
    height, width = image.shape
    for y in range(0, height, w):
        for x in range(0, width, w):
            block = image[y : y + w, x : x + w]
            threshold = np.average(block)
            image[y : y + w, x : x + w] = np.where(block >= threshold, 1.0, 0.0)

    return image

"""  
이미지를 이진화합니다. (흑백 이미지로 변환)
3x3 크기의 cross-shaped 커널을 생성하여 이미지를 erode합니다.
erode한 결과물을 다시 dilate하고, 이를 원래 이미지에서 뺀 결과물을 생성합니다. 이를 temp 변수에 저장합니다.
skel 변수에 temp와 bitwise_or 연산을 수행하여 추가합니다.
erode된 이미지를 다시 원래 이미지로 복사합니다.
위의 과정을 반복하면서, img에서 0이 아닌 픽셀의 개수가 0이 될 때까지 반복합니다.
skel 이미지에 대하여 다시 thinning을 수행합니다. 이때는 Zhang-Suen 알고리즘을 사용합니다.
thinning한 결과물을 반환합니다.
코드에서 사용된 함수들은 다음과 같습니다.

cv2.threshold() : 이미지 이진화 함수
cv2.getStructuringElement() : 커널 생성 함수
cv2.erode() : 이미지 erode 함수
cv2.dilate() : 이미지 dilate 함수
cv2.subtract() : 이미지 간 차이 구하기 함수
cv2.countNonZero() : 이미지에서 0이 아닌 픽셀 개수 구하기 함수
cv2.bitwise_or() : 이미지 OR 연산 함수
cv2.ximgproc.thinning() : 이미지 thinning 함수 (다양한 알고리즘 제공)
"""

def thinning_image(img) :
    #thresh = thin_image(img)
    #thresh = cv2.ximgproc.thinning(img,cv2.ximgproc.THINNING_GUOHALL)
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    ret, img = cv2.threshold(img, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    done = False

    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        temp = temp.astype(np.uint8)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True

    skel = cv2.ximgproc.thinning(skel, cv2.ximgproc.THINNING_ZHANGSUEN)

    return skel


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