# Fingerprint Segmentation Reference 

- Reference Handbook of fingerprint recognition 


## Block WISE Solution Refernce

이 방법들은 특징이 추출된 후, 전역 이진화(global thresholding)는 일반적으로 분할(segmentation)에 매우 효과적입니다. 이 방법들은 배경이 균일하거나, 노이즈가 많이 없을때 적용하면 좋습니다 :) , 속도가 장점이기 때문에 특정 환경에서 필요하다면 사용가능 할것입니다!

- The presence of peaks in local histograms of ridge orientations 
    - Mehtre, B. M., Murthy, N. N., Kapoor, S., & Chatterjee, B. (1987). 
    Segmentation of fingerprint images using the directional image. Pattern Recognition, 20(4), 429–435.
    

- The variance of gray levels in the orthogonal direction to the ridge orientation 
    - Ratha, N. K., Chen, S. Y., & Jain, A. K. (1995). Adaptive flow orientation-based feature extractio in fingerprint images. Pattern Recognition, 28(11), 1657–1672. 
    


- The average magnitude of the gradient in each image block 
    - Maio, D., & Maltoni, D. (1997). Direct gray-scale minutiae detection in fingerprints. IEEE Transactions on Pattern Analysis and Machine Intelligence, 19(1).


- The variance of the Gabor filter responses  
    - Wang, L., Suo, H., & Dai, M. (2005). Fingerprint image segmentation based on Gaussian–Hermite moments. In Proceedings of International Conference on Advanced Data Mining and Applications.  
    - Shen, L., Kot, A., & Koo, W. M. (2001). Quality measures of fingerprint images. In Proceedings of 3rd International Conference on Audio- and Video-Based Biometric Person Authentication (pp. 266–271).   
    - Alonso-Fernandez, F., Fierrez-Aguilar, J., & Ortega-Garcia, J. (2005). An enhanced Gabor filter based segmentation algorithm for fingerprint recognition systems. In Proceedings of International Symposium on Image and Signal Processing and Analysis.  


- The local energy in the Fourier spectrum 
    - Pais Barreto Marques, A. C., & Gay Thome, A. C. (2005). A neural network fingerprint segmentation method. In Proceedings of International Conference on Hybrid Intelligent Systems.
    - Chikkerur, S., Cartwright, A. N., & Govindaraju, V. (2007). Fingerprint enhancement using STFT analysis. Pattern Recognition, 40(1), 198–211.


## 학습기반 Thresholding 

