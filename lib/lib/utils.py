import matplotlib as plt 
import numpy as np
from PIL import Image
import cv2 
import matplotlib.pyplot as plt
import math
import urllib.request
import IPython
import base64
import html

def draw_minutiae(fingerprint, minutiae, termination_color = (255,0,0), bifurcation_color = (0,0,255)):

    res = cv2.cvtColor(fingerprint, cv2.COLOR_GRAY2BGR)
    
    for x, y, t, *d in minutiae:
        color = termination_color if t else bifurcation_color
        if len(d)==0:
            cv2.drawMarker(res, (x,y), color, cv2.MARKER_CROSS, 8)
        else:
            d = d[0]
            ox = int(round(math.cos(d) * 7))
            oy = int(round(math.sin(d) * 7))
            cv2.circle(res, (x,y), 3, color, 1, cv2.LINE_AA)
            cv2.line(res, (x,y), (x+ox,y-oy), color, 1, cv2.LINE_AA)        
    return res


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

############ 분석이후 코드 변경 필요!
def show(*images, enlarge_small_images = True, max_per_row = -1, font_size = 0):
  if len(images) == 2 and type(images[1])==str:
      images = [(images[0], images[1])]

  def convert_for_display(img):
      if img.dtype!=np.uint8:
          a, b = img.min(), img.max()
          if a==b:
              offset, mult, d = 0, 0, 1
          elif a<0:
              offset, mult, d = 128, 127, max(abs(a), abs(b))
          else:
              offset, mult, d = 0, 255, b
          img = np.clip(offset + mult*(img.astype(float))/d, 0, 255).astype(np.uint8)
      return img

  def convert(imgOrTuple):
      try:
          img, title = imgOrTuple
          if type(title)!=str:
              img, title = imgOrTuple, ''
      except ValueError:
          img, title = imgOrTuple, ''        
      if type(img)==str:
          data = img
      else:
          img = convert_for_display(img)
          if enlarge_small_images:
              REF_SCALE = 100
              h, w = img.shape[:2]
              if h<REF_SCALE or w<REF_SCALE:
                  scale = max(1, min(REF_SCALE//h, REF_SCALE//w))
                  img = cv2.resize(img,(w*scale,h*scale), interpolation=cv2.INTER_NEAREST)
          data = 'data:image/png;base64,' + base64.b64encode(cv2.imencode('.png', img)[1]).decode('utf8')
      return data, title
    
  if max_per_row == -1:
      max_per_row = len(images)

  rows = [images[x:x+max_per_row] for x in range(0, len(images), max_per_row)]
  font = f"font-size: {font_size}px;" if font_size else ""

  html_content = ""
  for r in rows:
      l = [convert(t) for t in r]
      html_content += "".join(["<table><tr>"] 
              + [f"<td style='text-align:center;{font}'>{html.escape(t)}</td>" for _,t in l]    
              + ["</tr><tr>"] 
              + [f"<td style='text-align:center;'><img src='{d}'></td>" for d,_ in l]
              + ["</tr></table>"])
  IPython.display.display(IPython.display.HTML(html_content))


# Utility function to show an image
def show(*images, enlarge_small_images = True, max_per_row = -1, font_size = 0):
  if len(images) == 2 and type(images[1])==str:
      images = [(images[0], images[1])]

  def convert_for_display(img):
      if img.dtype!=np.uint8:
          a, b = img.min(), img.max()
          if a==b:
              offset, mult, d = 0, 0, 1
          elif a<0:
              offset, mult, d = 128, 127, max(abs(a), abs(b))
          else:
              offset, mult, d = 0, 255, b
          img = np.clip(offset + mult*(img.astype(float))/d, 0, 255).astype(np.uint8)
      return img

  def convert(imgOrTuple):
      try:
          img, title = imgOrTuple
          if type(title)!=str:
              img, title = imgOrTuple, ''
      except ValueError:
          img, title = imgOrTuple, ''        
      if type(img)==str:
          data = img
      else:
          img = convert_for_display(img)
          if enlarge_small_images:
              REF_SCALE = 100
              h, w = img.shape[:2]
              if h<REF_SCALE or w<REF_SCALE:
                  scale = max(1, min(REF_SCALE//h, REF_SCALE//w))
                  img = cv2.resize(img,(w*scale,h*scale), interpolation=cv2.INTER_NEAREST)
          data = 'data:image/png;base64,' + base64.b64encode(cv2.imencode('.png', img)[1]).decode('utf8')
      return data, title
    
  if max_per_row == -1:
      max_per_row = len(images)

  rows = [images[x:x+max_per_row] for x in range(0, len(images), max_per_row)]
  font = f"font-size: {font_size}px;" if font_size else ""

  html_content = ""
  for r in rows:
      l = [convert(t) for t in r]
      html_content += "".join(["<table><tr>"] 
              + [f"<td style='text-align:center;{font}'>{html.escape(t)}</td>" for _,t in l]    
              + ["</tr><tr>"] 
              + [f"<td style='text-align:center;'><img src='{d}'></td>" for d,_ in l]
              + ["</tr></table>"])
  IPython.display.display(IPython.display.HTML(html_content))

# Utility function to load an image from an URL
def load_from_url(url):
  resp = urllib.request.urlopen(url)
  image = np.asarray(bytearray(resp.read()), dtype=np.uint8)
  return cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

# Utility function to draw orientations over an image
def draw_orientations(fingerprint, orientations, strengths, mask, scale = 3, step = 8, border = 0):
    if strengths is None:
        strengths = np.ones_like(orientations)
    h, w = fingerprint.shape
    sf = cv2.resize(fingerprint, (w*scale, h*scale), interpolation = cv2.INTER_NEAREST)
    res = cv2.cvtColor(sf, cv2.COLOR_GRAY2BGR)
    d = (scale // 2) + 1
    sd = (step+1)//2
    c = np.round(np.cos(orientations) * strengths * d * sd).astype(int)
    s = np.round(-np.sin(orientations) * strengths * d * sd).astype(int) # minus for the direction of the y axis
    thickness = 1 + scale // 5
    for y in range(border, h-border, step):
        for x in range(border, w-border, step):
            if mask is None or mask[y, x] != 0:
                ox, oy = c[y, x], s[y, x]
                cv2.line(res, (d+x*scale-ox,d+y*scale-oy), (d+x*scale+ox,d+y*scale+oy), (255,0,0), thickness, cv.LINE_AA)
    return res

# Utility function to draw a set of minutiae over an image
def draw_minutiae(fingerprint, minutiae, termination_color = (255,0,0), bifurcation_color = (0,0,255)):
    res = cv2.cvtColor(fingerprint, cv2.COLOR_GRAY2BGR)
    
    for x, y, t, *d in minutiae:
        color = termination_color if t else bifurcation_color
        if len(d)==0:
            cv2.drawMarker(res, (x,y), color, cv2.MARKER_CROSS, 8)
        else:
            d = d[0]
            ox = int(round(math.cos(d) * 7))
            oy = int(round(math.sin(d) * 7))
            cv2.circle(res, (x,y), 3, color, 1, cv2.LINE_AA)
            cv2.line(res, (x,y), (x+ox,y-oy), color, 1, cv2.LINE_AA)        
    return res

# Utility function to generate gabor filter kernels

_sigma_conv = (3.0/2.0)/((6*math.log(10))**0.5)
# sigma is adjusted according to the ridge period, so that the filter does not contain more than three effective peaks 
def _gabor_sigma(ridge_period):
    return _sigma_conv * ridge_period

def _gabor_size(ridge_period):
    p = int(round(ridge_period * 2 + 1))
    if p % 2 == 0:
        p += 1
    return (p, p)

def gabor_kernel(period, orientation):
    f = cv2.getGaborKernel(_gabor_size(period), _gabor_sigma(period), np.pi/2 - orientation, period, gamma = 1, psi = 0)
    f /= f.sum()
    f -= f.mean()
    return f


# Utility functions for minutiae
def angle_abs_difference(a, b):
    return math.pi - abs(abs(a - b) - math.pi)

def angle_mean(a, b):
    return math.atan2((math.sin(a)+math.sin(b))/2, ((math.cos(a)+math.cos(b))/2))

# Utility functions for MCC
def draw_minutiae_and_cylinder(fingerprint, origin_cell_coords, minutiae, values, i, show_cylinder = True):

    def _compute_actual_cylinder_coordinates(x, y, t, d):
        c, s = math.cos(d), math.sin(d)
        rot = np.array([[c, s],[-s, c]])    
        return (rot@origin_cell_coords.T + np.array([x,y])[:,np.newaxis]).T
    
    res = draw_minutiae(fingerprint, minutiae)    
    if show_cylinder:
        for v, (cx, cy) in zip(values[i], _compute_actual_cylinder_coordinates(*minutiae[i])):
            cv2.circle(res, (int(round(cx)), int(round(cy))), 3, (0,int(round(v*255)),0), 1, cv2.LINE_AA)
    return res

def draw_match_pairs(f1, m1, v1, f2, m2, v2, cells_coords, pairs, i, show_cylinders = True):
    #nd = _current_parameters.ND
    h1, w1 = f1.shape
    h2, w2 = f2.shape
    p1, p2 = pairs
    res = np.full((max(h1,h2), w1+w2, 3), 255, np.uint8)
    res[:h1,:w1] = draw_minutiae_and_cylinder(f1, cells_coords, m1, v1, p1[i], show_cylinders)
    res[:h2,w1:w1+w2] = draw_minutiae_and_cylinder(f2, cells_coords, m2, v2, p2[i], show_cylinders)
    for k, (i1, i2) in enumerate(zip(p1, p2)):
        (x1, y1, *_), (x2, y2, *_) = m1[i1], m2[i2]
        cv2.line(res, (int(x1), int(y1)), (w1+int(x2), int(y2)), (0,0,255) if k!=i else (0,255,255), 1, cv2.LINE_AA)
    return res