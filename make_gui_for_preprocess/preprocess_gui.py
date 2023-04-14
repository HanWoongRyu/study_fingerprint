# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\preprocess_gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1224, 963)
        MainWindow.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(40, 480, 181, 231))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.L_gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.L_gridLayout.setContentsMargins(0, 0, 0, 0)
        self.L_gridLayout.setObjectName("L_gridLayout")
        self.L_Morphology_Erode = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.L_Morphology_Erode.setObjectName("L_Morphology_Erode")
        self.L_gridLayout.addWidget(self.L_Morphology_Erode, 4, 0, 1, 1)
        self.L_Median = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.L_Median.setObjectName("L_Median")
        self.L_gridLayout.addWidget(self.L_Median, 1, 0, 1, 1)
        self.L_Gaussian = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.L_Gaussian.setObjectName("L_Gaussian")
        self.L_gridLayout.addWidget(self.L_Gaussian, 0, 0, 1, 1)
        self.L_Wiener = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.L_Wiener.setObjectName("L_Wiener")
        self.L_gridLayout.addWidget(self.L_Wiener, 9, 0, 1, 1)
        self.L_Gabor = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.L_Gabor.setObjectName("L_Gabor")
        self.L_gridLayout.addWidget(self.L_Gabor, 8, 0, 1, 1)
        self.L_Morphology_Dilation = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.L_Morphology_Dilation.setObjectName("L_Morphology_Dilation")
        self.L_gridLayout.addWidget(self.L_Morphology_Dilation, 5, 0, 1, 1)
        self.gridLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(610, 470, 181, 247))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.R_gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        self.R_gridLayout.setContentsMargins(0, 0, 0, 0)
        self.R_gridLayout.setObjectName("R_gridLayout")
        self.R_Median = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.R_Median.setObjectName("R_Median")
        self.R_gridLayout.addWidget(self.R_Median, 1, 0, 1, 1)
        self.R_Morphology_Dilation = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.R_Morphology_Dilation.setObjectName("R_Morphology_Dilation")
        self.R_gridLayout.addWidget(self.R_Morphology_Dilation, 6, 0, 1, 1)
        self.R_Gabor = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.R_Gabor.setObjectName("R_Gabor")
        self.R_gridLayout.addWidget(self.R_Gabor, 7, 0, 1, 1)
        self.R_Morpholog_Erode = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.R_Morpholog_Erode.setObjectName("R_Morpholog_Erode")
        self.R_gridLayout.addWidget(self.R_Morpholog_Erode, 4, 0, 1, 1)
        self.R_Gaussian = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.R_Gaussian.setObjectName("R_Gaussian")
        self.R_gridLayout.addWidget(self.R_Gaussian, 0, 0, 1, 1)
        self.R_Wiener = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.R_Wiener.setObjectName("R_Wiener")
        self.R_gridLayout.addWidget(self.R_Wiener, 8, 0, 1, 1)
        self.L_SpinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.L_SpinBox.setGeometry(QtCore.QRect(550, 460, 42, 22))
        self.L_SpinBox.setObjectName("L_SpinBox")
        self.R_SpinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.R_SpinBox.setGeometry(QtCore.QRect(1120, 460, 42, 22))
        self.R_SpinBox.setObjectName("R_SpinBox")
        self.Prev_Show_Image = QtWidgets.QPushButton(self.centralwidget)
        self.Prev_Show_Image.setGeometry(QtCore.QRect(500, 740, 75, 23))
        self.Prev_Show_Image.setObjectName("Prev_Show_Image")
        self.Next_Show_Image = QtWidgets.QPushButton(self.centralwidget)
        self.Next_Show_Image.setGeometry(QtCore.QRect(620, 740, 75, 23))
        self.Next_Show_Image.setObjectName("Next_Show_Image")
        self.L_Reset = QtWidgets.QPushButton(self.centralwidget)
        self.L_Reset.setGeometry(QtCore.QRect(270, 480, 75, 23))
        self.L_Reset.setObjectName("L_Reset")
        self.R_Reset = QtWidgets.QPushButton(self.centralwidget)
        self.R_Reset.setGeometry(QtCore.QRect(870, 480, 75, 23))
        self.R_Reset.setObjectName("R_Reset")
        self.L_Time = QtWidgets.QLabel(self.centralwidget)
        self.L_Time.setGeometry(QtCore.QRect(520, 30, 56, 12))
        self.L_Time.setObjectName("L_Time")
        self.R_Time = QtWidgets.QLabel(self.centralwidget)
        self.R_Time.setGeometry(QtCore.QRect(1100, 20, 56, 12))
        self.R_Time.setObjectName("R_Time")
        self.L_path_button = QtWidgets.QPushButton(self.centralwidget)
        self.L_path_button.setGeometry(QtCore.QRect(40, 10, 75, 23))
        self.L_path_button.setObjectName("L_path_button")
        self.R_path_button = QtWidgets.QPushButton(self.centralwidget)
        self.R_path_button.setGeometry(QtCore.QRect(610, 10, 75, 23))
        self.R_path_button.setObjectName("R_path_button")
        self.L_horizontalScrollBar = QtWidgets.QScrollBar(self.centralwidget)
        self.L_horizontalScrollBar.setGeometry(QtCore.QRect(40, 440, 549, 17))
        self.L_horizontalScrollBar.setOrientation(QtCore.Qt.Horizontal)
        self.L_horizontalScrollBar.setObjectName("L_horizontalScrollBar")
        self.R_horizontalScrollBar = QtWidgets.QScrollBar(self.centralwidget)
        self.R_horizontalScrollBar.setGeometry(QtCore.QRect(610, 440, 549, 17))
        self.R_horizontalScrollBar.setOrientation(QtCore.Qt.Horizontal)
        self.R_horizontalScrollBar.setObjectName("R_horizontalScrollBar")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1224, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Preprocess_Tool"))
        self.L_Morphology_Erode.setText(_translate("MainWindow", "Morphology(erode)"))
        self.L_Median.setText(_translate("MainWindow", "Median"))
        self.L_Gaussian.setText(_translate("MainWindow", "Gaussian"))
        self.L_Wiener.setText(_translate("MainWindow", "Wiener"))
        self.L_Gabor.setText(_translate("MainWindow", "Gabor"))
        self.L_Morphology_Dilation.setText(_translate("MainWindow", "Morphology(dilatation)"))
        self.R_Median.setText(_translate("MainWindow", "Median"))
        self.R_Morphology_Dilation.setText(_translate("MainWindow", "Morphology(dilatation)"))
        self.R_Gabor.setText(_translate("MainWindow", "Gabor"))
        self.R_Morpholog_Erode.setText(_translate("MainWindow", "Morphology(erode)"))
        self.R_Gaussian.setText(_translate("MainWindow", "Gaussian"))
        self.R_Wiener.setText(_translate("MainWindow", "Wiener"))
        self.Prev_Show_Image.setText(_translate("MainWindow", "Prev"))
        self.Next_Show_Image.setText(_translate("MainWindow", "Next"))
        self.L_Reset.setText(_translate("MainWindow", "Reset"))
        self.R_Reset.setText(_translate("MainWindow", "Reset"))
        self.L_Time.setText(_translate("MainWindow", "TextLabel"))
        self.R_Time.setText(_translate("MainWindow", "TextLabel"))
        self.L_path_button.setText(_translate("MainWindow", "path"))
        self.R_path_button.setText(_translate("MainWindow", "path"))
