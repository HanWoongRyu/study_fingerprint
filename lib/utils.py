import matplotlib as plt 
import numpy as np
from PIL import Image
import cv2 
import matplotlib.pyplot as plt


def display_image(image: np.array, title: str = None, cmap: str = None, figsize: tuple = None):
    """
    matplotlib을 통하여 이미지를 띄운다.
    

    Args:
        image (nd.array): Image that should be visualised.
        title      (str): Displayed graph title.
        cmap       (str): Cmap type.
        figsize  (tuple): Size of the displayed figure. 

    """

    if figsize:
        plt.figure(figsize=figsize)

    plt.imshow(image, cmap=cmap)

    if (len(image.shape) == 2) or (image.shape[-1] == 1):
        plt.gray()

    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title(title)
    plt.show()

def resize_image(image: np.array, size: tuple):
    return cv2.resize(image, size)

def resize_images(images : list , size :tuple ):
    return [cv2.resize(img,size) for img in images]

def cal_samepadding_size(image:np.array,window_size:int = 10) :
    """
        calculate samepadding size for conv.. etc
         Args:
            image (nd.array): Image that should be visualised.
            window_size(int) : kerner size 
        SamePadding = ((W - 1) * S + K - W) / 2
        W = width, S =Stride , K = Kenrl size 
        
    """
    height, width = image.shape
    pad_w = int(((width- 1) * 1 + window_size - width) / 2)
    pad_h = int(((height- 1) * 1 + window_size - height) / 2)
    return pad_w,pad_h

def array_to_image(image):
    """
    Returns a PIL Image object

    """
    return Image.fromarray(image)

def image_to_array(image: np.array, mode: str = 'LA'):
    """
    Returns an image 2D array.

    """

    image_array = np.fromiter(iter(image.getdata()), np.uint8)
    image_array.resize(image.height, image.width)

    return image_array

#SamePadding = ((W - 1) * S + F - W) / 2
#same padding  ((출력 크기(output size) - 1) x 스트라이드(stride) - 입력 크기(input size) + 필터 크기(filter size)) / 2
   #1.윈도우 사이즈로 이미지 width와 height의 나머지를 구한다(지금 내 경우는 이미지사이즈가 고정이기 때문에 이 계산을 굳이 안해줘도 됨으로 상수로 넣기로함)
    #2.양옆밑을 반반씩 넣어주기로함(손실을 최소화 하기 위해서)
    # height, width = gray.shape
    # pad_w = int(((width- 1) * 1 + window_size - width) / 2)
    # pad_h = int(((height- 1) * 1 + window_size - height) / 2)
def cal_samepadding_size(image:np.array,window_size=10) :
    height, width = image.shape
    pad_w = int(((width- 1) * 1 + window_size - width) / 2)
    pad_h = int(((height- 1) * 1 + window_size - height) / 2)
    return pad_w,pad_h
