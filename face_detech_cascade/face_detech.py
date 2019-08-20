import cv2
import numpy as np
import os

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from libs.utils import *

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

class ResultDialog(QMainWindow):
    def __init__(self,parent=None):
        super(ResultDialog,self).__init__(parent)
        self.setGeometry(QRect(0,0,300,300))
        self.setWindowTitle("Result Dialog")

        self.initVar()
        self.initUI()

    def initUI(self):

        self.statusBar = QStatusBar(self)
        self.labelCoordinates = QLabel(self)
        self.statusBar.addPermanentWidget(self.labelCoordinates)
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Face detech using CascadeClassifier",3000)

        layout = QVBoxLayout()

        hlayout = QHBoxLayout()

        but_import = newButton("Retest",icon="saveImage.png",slot=self.reTest)
        self.lb_importName = QLineEdit(self)
        but_load = newButton("Load Image",icon="saveImage.png",slot=self.loadImage)

        addLayouts(hlayout,[but_load,but_import,self.lb_importName])
        

        hlayout1 = QHBoxLayout()

        lb1 = QLabel("scaleFactor",self)
        self.ln_scale = QLineEdit("1.1",self)

        lb2 = QLabel("minNeighbors",self)
        self.ln_n = QLineEdit("5",self)

        lb3 = QLabel("minSize",self)
        self.ln_size = QLineEdit("30",self)

        addLayouts(hlayout1,[lb1,self.ln_scale,lb2,self.ln_n,lb3,self.ln_size])

        self.frame = QLabel(self)
        self.frame.setStyleSheet("QLabel{background-color:black}")
        self.frame.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        
        addLayouts(layout,[hlayout,hlayout1,self.frame])
  
        widget = QWidget(self)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def initVar(self):
        self.mat = None
        self.scale = 1.0
        self.n = 4
        self.size = 30
        # self.loadParams()
        pass

    def loadParams(self):
        self.scale = str2float(self.ln_scale.text())
        self.n = str2int(self.ln_n.text())
        self.size = str2int(self.ln_size.text())
        pass

    def reTest(self):
        if self.mat is None:
            return
        self.showImage(self.importData())

    def importData(self):
        # if self.lb_importName.text() == "":
        #     QMessageBox.warning(self,"Warning","Enter the name of the object!?")
        #     return
        # save_folder = QFileDialog.getExistingDirectory(self,'Select a folder:',os.getcwd())
        # if save_folder == "":
        #     return
        self.loadParams()
        gray = cv2.cvtColor(self.mat, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,scaleFactor = self.scale,minNeighbors = self.n 
                            , minSize = (self.size,self.size))

        mat = self.mat.copy()
        for (x,y,w,h) in faces:
            cv2.rectangle(mat,(x,y),(x+w,y+h),(255,0,0),2)

        return mat

    def showImage(self,mat):
        # self.mat = mat
        showImage(mat,self.frame,fitwindow=True)

    def saveImage(self):
        mkdir("Result")
        cv2.imwrite("Result/%s.jpg"%self.cbb_result.currentText(),self.mat)

    def loadImage(self):
        fileName, _ = QFileDialog.getOpenFileName(self,"",os.getcwd(),"All Files (*);;Python Files (*.py)")
        if fileName == "":
            return
        self.statusBar.showMessage(fileName,5000)
        self.mat = cv2.imread(fileName)
        self.labelCoordinates.setText("%dx%d"%(self.mat.shape[1],self.mat.shape[0]))
        
        faces = self.importData()

        self.showImage(faces)


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    ui = ResultDialog()
    ui.show()
    app.exec_()