import sys
from PyQt5 import QtWidgets, uic
import os



class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        # Load the .ui file
        uic.loadUi('E:\study_opencv\\fingerprint\make_gui_for_preprocess\preprocess_gui.ui', self)

    #UI connection 
        


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MyWindow()
    window.show()
    app.exec_()


'''

uic.loadUi('output.ui', self)는 디자인된 ui파일을 가져옵니다.

app = QtWidgets.QApplication([])은 PyQt5에서 실행하는 모든 GUI 응용 프로그램의 시작점입니다.

[]는 실행 시 프로그램에 전달되는 인수의 목록을 나타내며, 여기서는 빈 목록을 전달합니다. 이 인수는 명령 줄 인수를 포함하여 응용 프로그램에서 사용 가능한 인수를 정의할 때 사용됩니다.

이 메소드는 QApplication 클래스의 인스턴스를 만들고, 이벤트 루프를 시작하여 PyQt5 응용 프로그램의 실행을 시작합니다. 이벤트 루프는 애플리케이션에 대한 모든 이벤트(예: 마우스 클릭, 키 입력 등)를 처리합니다.

따라서 이 코드는 빈 QApplication 객체를 생성하고 이벤트 루프를 시작하여 PyQt5 기반 애플리케이션을 실행합니다.

'''