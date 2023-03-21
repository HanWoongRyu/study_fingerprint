from preprocess_gui import Ui_MainWindow #UI import
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QFileDialog
import copy
import time
import numpy as np
import cv2
import os


class ImageViewer(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super(ImageViewer, self).__init__(parent)
        #self.setScene(QtWidgets.QGraphicsView(self))
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        
    def wheelEvent(self, event): 
        # 마우스 휠 이벤트를 처리하여 이미지 확대 / 축소
        zoomInFactor = 1.25
        zoomOutFactor = 1 / zoomInFactor

        # 현재 뷰의 중심점을 계산
        oldPos = self.mapToScene(event.pos())
        zoom = zoomInFactor if event.angleDelta().y() > 0 else zoomOutFactor
        self.scale(zoom, zoom)
        newPos = self.mapToScene(event.pos())

        # 스크롤 처리
        delta = newPos - oldPos
        self.scrollContentsBy(delta.x(), delta.y())

#UI의 동작기능을 구현하는 class
class Preprocessor(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self,l_image_list=None,r_image_list=None):
        super().__init__()
        self.setupUi(self)
    
    # variable
        #image list
        self.l_image_list = copy.deepcopy(l_image_list) if l_image_list is not None else []
        self.r_image_list = copy.deepcopy(r_image_list) if r_image_list is not None else []
        self.l_origin_image_list =copy.deepcopy(self.l_image_list) if r_image_list is not None else []
        self.r_origin_image_list = copy.deepcopy(self.r_image_list) if r_image_list is not None else []
        #image path
        self.L_path =None
        self.R_path = None  
        #idx
        self.left_idx = 0
        self.right_idx = 0
        
    #Connect to UI
        #Image viewer 초기화 
        self.L_graphicsView = ImageViewer(self)
        self.L_graphicsView.setGeometry(QtCore.QRect(60, 60, 511, 351))
        self.L_graphicsView.setObjectName("L_graphicsView")
        self.L_graphicsScene = QtWidgets.QGraphicsScene(self.L_graphicsView)
        self.L_graphicsView.setScene(self.L_graphicsScene)
        
        self.R_graphicsView = ImageViewer(self)
        #self.R_graphicsView = QtWidgets.QGraphicsView(self)
        self.R_graphicsView.setGeometry(QtCore.QRect(610, 60, 511, 351))
        self.R_graphicsView.setObjectName("R_graphicsView")
        self.R_graphicsScene = QtWidgets.QGraphicsScene(self.R_graphicsView)
        self.R_graphicsView.setScene(self.R_graphicsScene)
        
        # Output Label
        self.L_Time = QtWidgets.QLabel()
        self.R_Time = QtWidgets.QLabel()

        # Input index
        self.L_SpinBox.valueChanged.connect(self.L_spinbox_changed)
        self.R_SpinBox.valueChanged.connect(self.R_spinbox_changed)

        #Path Buttons 
        self.L_path_button.clicked.connect(self.showDialog)
        self.R_path_button.clicked.connect(self.showDialog)

        #scrollbar
        self.L_horizontalScrollBar.valueChanged.connect(self.L_scrollbar_changed)
        self.R_horizontalScrollBar.valueChanged.connect(self.R_scrollbar_changed)

        # Filter Buttons
        #Left
        self.L_Median.clicked.connect(lambda: self.median_filter())
        # self.L_Gabor.clicked.connect(lambda : self.gabor_filter())
        self.L_Gaussian.clicked.connect(self.gaussian_filter)
        # self.L_Morphology_Dilation.clicked.connect(self.morphology_dilation)
        # self.L_Morphology_Erode.clicked.connect(self.morphology_erode)

        #Right
        self.R_Median.clicked.connect(lambda: self.median_filter())
        # self.R_Gabor.clicked.connect(self.gabor_filter)
        self.R_Gaussian.clicked.connect(self.gaussian_filter)
        # self.R_Morphology_Dilation.clicked.connect(self.morphology_dilation)
        # self.R_Morphology_Erode.clicked.connect(self.morphology_erode)

        self.show_images()


    def show_images(self):
       
        #왼쪽 이미지 뷰어에 이미지 출력
        if len(self.l_image_list) > 0:
            left_pixmap = self.nparray_to_pixmap(self.l_image_list[self.left_idx])
            self.L_graphicsView.setScene(QtWidgets.QGraphicsScene(self))
            self.L_graphicsView.scene().addPixmap(left_pixmap)

        # 오른쪽 이미지 뷰어에 이미지 출력
        if len(self.r_image_list) > 0:
            right_pixmap = self.nparray_to_pixmap(self.r_image_list[self.right_idx])
            self.R_graphicsView.setScene(QtWidgets.QGraphicsScene(self))
            self.R_graphicsView.scene().addPixmap(right_pixmap)   


# move fucntion
    def L_spinbox_changed(self, value):
        self.left_idx = value - 1
        self.show_images()

    def R_spinbox_changed(self, value):
        self.right_idx = value - 1
        self.show_images()


    def L_scrollbar_changed(self, value):
        self.left_idx = value
        self.show_images()

    def R_scrollbar_changed(self, value):
        self.right_idx = value
        self.show_images()
    

    def update_image(self, idx):
        # 인덱스에 해당하는 이미지를 이미지 뷰어에 출력합니다.
        pixmap = self.nparray_to_pixmap(self.image_list[idx])
        self.graphicsView.setScene(QtWidgets.QGraphicsScene(self))
        self.graphicsView.scene().addPixmap(pixmap)
        
#read image
    def read_images_from_folder(self, folder_path):
        images_list = []
        for filename in os.listdir(folder_path):
            if filename.endswith(('.png', '.jpg', '.jpeg', '.BMP', '.tif')):
                img = cv2.imread(os.path.join(folder_path, filename))
                images_list.append(img)        
        return images_list
            
    def showDialog(self):
        # 폴더 선택 다이얼로그 생성
        fname = QFileDialog.getExistingDirectory(self, '폴더 선택')
        # 선택된 경로를 변수에 저장
        if self.sender() == self.L_path_button:
            self.L_path = fname
            self.l_image_list=self.read_images_from_folder(fname)
            self.show_images()

        elif self.sender() == self.R_path_button:
            self.R_path = fname
            self.r_image_list=self.read_images_from_folder(fname)
            self.show_images()
    
    def nparray_to_pixmap(self, image_np):
        # 넘파이 배열로부터 QImage 객체 생성
        image_qt = QtGui.QImage(image_np.data, image_np.shape[1], image_np.shape[0], QtGui.QImage.Format_RGB888)
        # QImage 객체로부터 QPixmap 객체 생성
        pixmap = QtGui.QPixmap.fromImage(image_qt)
        return pixmap
    
    #filter  function
    def median_filter(self):
        sender = self.sender()
        if sender == self.L_Median:
            if len(self.l_image_list) > 0:
                # Median 필터링을 적용하고 결과 이미지를 img에 저장
                start_time = time.time()
                img = cv2.medianBlur(self.l_image_list[self.left_idx], 3)
                end_time = time.time()
                elapsed_time = end_time - start_time
                self.l_image_list[self.left_idx] = img
                # 결과 이미지를 이미지 뷰어에 출력
                self.show_images()
        elif sender ==self.R_Median :
            if len(self.r_image_list) > 0:
                # Median 필터링을 적용하고 결과 이미지를 img에 저장
                start_time = time.time()
                img = cv2.medianBlur(self.r_image_list[self.right_idx], 3)
                end_time = time.time()
                elapsed_time = end_time - start_time
                self.r_image_list[self.right_idx] = img
                # 결과 이미지를 이미지 뷰어에 출력
                self.show_images()
    def gaussian_filter(self) :
        sender = self.sender()
        if sender == self.L_Gaussian:
            if len(self.l_image_list) > 0:
                # Median 필터링을 적용하고 결과 이미지를 img에 저장
                start_time = time.time()
                img = cv2.medianBlur(self.l_image_list[self.left_idx], 3)
                end_time = time.time()
                elapsed_time = end_time - start_time
                self.l_image_list[self.left_idx] = img
                # 결과 이미지를 이미지 뷰어에 출력
                self.show_images()
        elif sender ==self.R_Gaussian:
            if len(self.r_image_list) > 0:
                # Median 필터링을 적용하고 결과 이미지를 img에 저장
                start_time = time.time()
                img = cv2.medianBlur(self.r_image_list[self.right_idx], 3)
                end_time = time.time()
                elapsed_time = end_time - start_time
                self.r_image_list[self.right_idx] = img
                # 결과 이미지를 이미지 뷰어에 출력
                self.show_images()
           
    # def gabor_filter(self):
    #     if self.sender()== self.L_Gabor :
    #         if len(self.l_image_list) > 0:
    #             image = self.l_image_list[self.left_idx]
    #             kernel = cv2.getGaborKernel((31, 31), 4, np.pi/4, 10, 0.5, 0, ktype=cv2.CV_32F)
    #             filtered = cv2.filter2D(image, -1, kernel)
    #             self.l_image_list[self.left_idx] = filtered
    #             self.show_images()
    #     elif self.sender() == self.R_Gabor :
  
    
if __name__ == '__main__':
    print('start')
    import sys
    app = QtWidgets.QApplication(sys.argv)
    os.listdir()
    ui = Preprocessor()
    ui.show()
    sys.exit(app.exec_())


