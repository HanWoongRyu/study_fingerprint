
import numpy as np
import cv2
from skimage.filters import (threshold_niblack,threshold_sauvola)
# class Smoothing 

# class Sharping 

# class Frequencing 

#binarizaiton
def apply_binarization_otsu(img:np.ndarray,min_val:int = 0,max_val:int=255,type:int = cv2.THRESH_BINARY|cv2.THRESH_OTSU) :
    th, dst = cv2.threshold(img, min_val, max_val,type)
    return th,dst

def apply_binarization_sauvola(img:np.ndarray,window_size:int, k:float, r:int) :
    """    
    Args :
        k :  contrast(대비)에 대한 가중치입니다. k 값이 높을수록 이미지 전체에 대한 threshold 값이 높아지므로, 이진화 결과에서 흰색 픽셀의 개수가 적어집니다. 일반적으로 0.2 ~ 0.5 사이의 값이 사용됩니다.
        r : 표준편차를 정규화하기 위한 값으로, 이미지 밝기에 따라 달라질 수 있습니다. r 값이 클수록, 이미지의 밝기 범위가 넓어지므로 threshold 값이 커지고, 이진화 결과에서 흰색 픽셀의 개수가 감소합니다. 일반적으로 128의 값을 사용합니다.
    Return : 
        0 or 1 ndarray
    Description : 
        Sauvola Thresholding은다른 부분에서 특정 상수값인 k와 R 값을 사용합니다. k는 평균값에 더할 상수값이며, R은 표준 편차를 조절하는 상수값입니다. 
        이러한 상수값을 조정함으로써, 이미지의 밝기와 대조가 큰 영역에서도 임계값을 정확하게 계산할 수 있습니다.
        niblack과 유사하지만 조금 더 강건하고 이미지가 크면 속도가 살짝 더 느립니다.
    Reference :
        https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_sauvola
        T = m(x,y) * (1 + k * ((s(x,y) / R) - 1))    
    """
    thresh_sauvola = threshold_sauvola(img, window_size=window_size)
    binary_sauvola = img > thresh_sauvola

    return binary_sauvola 
    
def apply_binarization_niblack(img:np.ndarray, window_size :int ,k:float):
    """
    Args : 
        img : ndarray
        window_size : int window size
        k : float constant for bright of blocks
    Return : 
        0 or 1 ndarray
    Description  :
        k는 상수이며, 일반적으로 -0.2에서 0.2 사이의 값을 사용합니다. 이 값은 평균과 표준편차를 고려하여 픽셀의 밝기값이 어두운지, 밝은지에 따라 임계값을 조정합니다. 만약 k 값이 양수라면, 해당 픽셀보다 밝은 픽셀을 더 많이 남기게 됩니다. 반대로, k 값이 음수라면, 해당 픽셀보다 어두운 픽셀을 더 많이 남기게 됩니다.
        Niblack Thresholding은 이미지에서 전경과 배경이 대조적인 경우에 효과적입니다. 예를 들어, 조명이 일정하지 않은 이미지나, 색조가 일정하지 않은 이미지, 혹은 문서 이미지와 같이 글자와 배경이 대조적으로 나타나는 이미지 등이 있습니다.
        장점은 Niblack Thresholding은 매우 빠르고 간단하게 구현할 수 있으며, 광범위한 이미지 분야에서 적용할 수 있습니다.
        하지만, Niblack Thresholding은 이미지의 밝기 분포가 일정하지 않은 경우에는 제대로 작동하지 않을 수 있습니다. 또한, 노이즈가 많은 이미지나, 전경과 배경이 명확히 구분되지 않는 이미지에서도 정확도가 떨어지는 경향이 있습니다. 이러한 경우에는 다른 이진화 기술을 사용하는 것이 더 적합할 수 있습니다.
    
    Reference :
        https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_niblack
        T = m(x,y) - k * s(x,y)
        T(x,y) = μ(x,y) + k * σ(x,y)

    """
    thresh_niblack = threshold_niblack(img, window_size=window_size)
    binary_niblack = img > thresh_niblack

    return binary_niblack
    
#gernerally use
def apply_nomalize(img:np.array) :
    return (img - np.mean(img))/np.std(img)
#segmentation

def segmentation_maksed_thresholding(img:np.ndarray,block=10,threshold=0.2) :
    """
    block wise segmentation   
    args :
        img  : numpy.ndarray 
        block : block_size 
        threshold : 0.2 defualt 
        return :Return Segmentation Image , Nomalise_img,ROI mask 
    description : 
        block의 std와 전체 이미지의 std*threshold를 통하여 mask와 세그멘테이션 이미지를 return한다.
        전경과 배경이 확실이 구분되어지는 이미지 ex) 지문이미지, 세포이미지에서 사용을 권장한다. 
        std가 높으면 object가 있다는 것으로 간주하는 코드이다. threshold를 높일수록 정확도가 높아진다.
        
    """
    (y, x) = img.shape
    threshold = np.std(img)*threshold

    image_variance = np.zeros(img.shape)
    segmented_image = img.copy()
    mask = np.ones_like(img)

    for i in range(0, x, block):
        for j in range(0, y, block):
            box = [i, j, min(i + block, x), min(j + block, y)]
            block_stddev = np.std(img[box[1]:box[3], box[0]:box[2]])
            image_variance[box[1]:box[3], box[0]:box[2]] = block_stddev

    # apply threshold
    mask[image_variance < threshold] = 0

    # smooth mask with a open/close morphological filter
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(block*2, block*2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # normalize segmented image
    segmented_image *= mask
    
    return segmented_image, mask

def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
    """
    Contrast Limited Adaptive Histogram Equalization
    Args :
    image : np.ndarraay
    clipLimit: float 평활화된 이미지에서 픽셀 값의 최대 값이 될 수 있는 제한 값입니다. 
    tileGridSize: tuple 이미지를 처리할 작은 영역의 크기입니다. 
    
    decription : 
    clipLimit : 이 값이 작을수록 평활화된 이미지가 더욱 어두워지고, 값이 크면 더 밝아집니다. 보통 2.0에서 4.0 사이의 값을 사용합니다.
    tileGridSize 값이 작을수록 더 많은 작은 영역이 생성되어 이미지의 세부 사항이 더욱 잘 보존됩니다. 하지만 이 값이 너무 작으면 연산 속도가 느려지고, 이미지가 잘못 처리될 수 있습니다. 대부분의 경우 (8, 8) 또는 (16, 16)의 값을 사용합니다.
    clipLimit tileGridSi와ze는 이미지의 특성에 따라 조정되어야 합니다. 예를 들어, 이미지가 어둡고 대조가 낮은 경우 clipLimit를 작게 설정하여 밝기를 높일 수 있습니다. 이미지의 크기와 특성에 따라 tileGridSize를 조정할 수 있습니다.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    return clahe.apply(image)


#get orientaton


# def get_orientation(img:np.ndarray, )
# """
# Reference Handbook of Fingerprint Recognition(2022)
# The simplest and most natural approach for extracting local ridge orientation is based on the computation of gradients in the fingerprint image. The gradient ∇(x, y) at point [x,
# y] of I is a two-dimensional vector [∇x(x, y),∇y(x, y)], where ∇x and ∇y components are the derivatives of I at [x, y] with respect to the x- and y-directions, respectively. It is well known that the gradient phase angle denotes the direction of the maximum intensity
# change. Therefore, the direction θ of a hypothetical edge that crosses the region centered at [x, y] is orthogonal to the gradient phase angle at [x, y]. This method, although simple and efficient, has some drawbacks. First, using the classical Prewitt or Sobel convolution masks (Gonzales & Woods, 2007) to determine ∇x and ∇y components of the gradient
# and computing θ according to the arctangent of the ∇y/∇x ratio present problems due to the non-linearity and discontinuity around 90°

# """
