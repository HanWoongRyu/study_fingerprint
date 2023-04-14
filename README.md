# FingerPrint Study

## Goal
- Fingerprint image의 Feature Extraction과 Matching
- opencv에 익숙해지고, vison task를 공부한다.

## Dataset 
- socofing dataset
- https://www.kaggle.com/datasets/ruizgara/socofing?datasetId=38300&sortBy=voteCount

## 기존의 Minutiae 기반 매칭방법 



fingerprint 분야에서 가장 많이 쓰이는 feature matching 방법은 minutiae 기반 매칭입니다. 이 방법은 이미지에서 추출된 미세한 지문 특징점인 minutiae를 이용하여 지문 간의 매칭을 수행합니다. 이 방법은 다른 feature matching 방법에 비해 정확도가 높고, 연산 속도가 빠르다는 장점이 있습니다. 또한, minutiae 기반 매칭 방법은 지문 매칭 분야에서 오랫동안 사용되어 왔고, 많은 연구가 이 방법에 집중되어 왔습니다. 최근에는 딥러닝을 이용한 지문 매칭 기술도 많이 연구되고 있지만, 여전히 minutiae 기반 매칭이 가장 대표적인 방법 중 하나입니다.


<p align="center">
<img src="https://i.postimg.cc/jS7mBNtx/fingerprint-feture.png" width="400" height="300"/>
</p>

Minutiae란? 
> 지문상에서 끝점, 분기점, 중심점 등과 같은 지문의 지역적인 특징점을 의미합니다. 이러한 Minutiae는 fingerprint 인식에서 가장 기본적이면서도 중요한 요소 중 하나로, 지문을 특정 짓는 정보로 사용됩니다. Minutiae는 지문의 상세한 형태, 방향, 굵기, 위치 등을 표현할 수 있어, fingerprint 매칭 알고리즘에서 핵심적인 역할을 합니다. Minutiae의 특징 중 하나는 지문의 크기나 회전, 이동 등에도 영향을 받지 않는다는 것입니다. 이러한 특성으로 인해, fingerprint 인식 분야에서는 Minutiae 기반 매칭 방법이 가장 대중적으로 사용되고 있습니다.


> 본 코드에서는 지문 분기점(Bifurcation Point)과 지문 끝점 (ridge ending or terminations)를 찾아냅니다.


## Preprocessing

- Feature Extraction을 하기 위한 *전형적인* 지문이미지 전처리 
- 전체적인 설명은 이름과 같은 함수 수식과 레퍼런스를 모두 적어놓았습니다! (tutorial 혹은 lib의 소스에서 확인할 수 있습니다.)
- Tutorial을 통하여 더욱 자세한 시각화를 볼 수있습니다.

<p align="center">
<img src="https://i.postimg.cc/XJP0xgkn/preprocess-steps-typical-minutiae-extraction-pipeline.png" width="400" height="300"/>
</p>

reference : Davide Maltoni, Dario Maio, Anil K. Jain, Jianjiang Feng - Handbook of Fingerprint Recognition-Springer (2022)


### Segmentation

- segmentation_maksed_thresholding 
- normalize_with_mask
### Orientation
- estimate_orientation
### Frequency
- estimate_frequencies
### Enhancement
- gaborFilter
### Thinning
- binarize
- thinning_image



## Feature Extraction

### Crossing number
- minutiae_extraction

    - algorithm
![crossing_number_algorithm](https://i.postimg.cc/q73q0y91/crossnumber-algorithm.png)
    -  쉽게 이야기해서 픽셀주변을 둘러보고 1이면 끝점 3이면 분기점이란 이야기이다.

$$ cn(v) = \sum_{i=0}^7 \begin{cases}
    1 & \text{if } v[i] < v[(i+1) \mod 8] \\
    0 & \text{otherwise}
\end{cases} $$

![non_filtered_minutiae](https://i.postimg.cc/SNJ8vTy9/nonfilter-minutiae.png)

## minutiae filtering
- filtering_minutiea 
    - cv2.distanceTransform 함수를 segmenation mask와 함께 사용해서 전경과 배경의 거리를 가지는 거리 마스크를 만들어냅니다.
    - 이 거리 마스크를 통하여 배경에서 어느정도 떨어진 지점에서의 minutiae만 구하게된다. 위의 사진들을 보면 지문외각선의 안좋은 특징들을 볼 수 있는데 밑의 처리된 사진을 보면 깔끔해진것을 볼 수 있습니다.

![mask](https://i.postimg.cc/0jd8Wqzv/image.png)


![filtered_minutiae](https://i.postimg.cc/gjcyjGFt/filtered-minutiae.png)

-------------------------
이 부분 부터는 또 새로운 논문과 다른 방식의 접근으로 이론적으로 이해가 되지 않은 부분이 많지만, 좋은 reference를 구하여 ipynb에 구현은 되었습니다. 
아직 정리가 더 필요합니다 ㅜ
## find minutiae directions

## make mcc  

## mathing fingerprint use mcc 


## Extra
- fingerprint_enhacement.py :지문이미지 thinning까지 과정에 필요한 함수들 모음   
- fingerprint_feature_extraction.py : minutiae를 뽑아내고 filtering 하는 함수들 모음  
- image_processing.py : 지문이미지 뿐만 아니라 다른데서도 사용가능할거 같은 함수들을 따로 뽑아놓았습니다.  
- utils.py : 이미지를 읽어오거나 출력하는 util 함수들  
- fingerprint_segmenation.py : 지문이미지 segmentation 논문을 통하여 만들었지만 실상 효과는 별로없음 그래서 사용하지 않습니다..

test.ipynb는 그냥 여러 실험을 한 곳입니다...
