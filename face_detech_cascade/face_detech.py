import cv2
import numpy as np
import os

import tensorflow as tf

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from libs.utils import *
# from align_mtcnn import AlignMTCNN
from mtcnn.mtcnn import MTCNN

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_mtcnn = MTCNN()

class ResultDialog(QMainWindow):
    def __init__(self,parent=None):
        super(ResultDialog,self).__init__(parent)
        self.setGeometry(QRect(0,0,300,300))
        self.setWindowTitle("Result Dialog")

        self.initVar()
        self.initUI()

        # if self.ch_useCascade.isChecked():
        #     self.detection = face

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
        self.ch_useCascade = QCheckBox("useCascade",self)
        but_load = newButton("Load Image",icon="saveImage.png",slot=self.loadImage)

        addLayouts(hlayout,[but_load,but_import,self.lb_importName,self.ch_useCascade])
        

        hlayout1 = QHBoxLayout()

        lb1 = QLabel("scaleFactor",self)
        self.ln_scale = QLineEdit("1.1",self)

        lb2 = QLabel("minNeighbors",self)
        self.ln_n = QLineEdit("5",self)

        lb3 = QLabel("minSize",self)
        self.ln_size = QLineEdit("30",self)

        lb4 = QLabel("threshold",self)
        self.ln_threshold = QLineEdit("0.8",self)

        addLayouts(hlayout1,[lb1,self.ln_scale,lb2,self.ln_n,lb3,self.ln_size,lb4,self.ln_threshold])

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
        self.steps_threshold = 0.7
        pass

    def loadParams(self):
        self.scale = str2float(self.ln_scale.text())
        self.n = str2int(self.ln_n.text())
        self.size = str2int(self.ln_size.text())
        self.steps_threshold = str2float(self.ln_threshold.text())
        pass

    def reTest(self):
        if self.mat is None:
            return
        self.showImage(self.importData())

    def importData(self):
        self.loadParams()
        
        mat = self.mat.copy()

        if self.ch_useCascade.isChecked():
            gray = cv2.cvtColor(self.mat, cv2.COLOR_BGR2GRAY)
            if self.scale < 1.0:
                self.scale = 1.1
            faces = face_cascade.detectMultiScale(gray,scaleFactor = self.scale,minNeighbors = self.n 
                                , minSize = (self.size,self.size))

            for (x,y,w,h) in faces:
                cv2.rectangle(mat,(x,y),(x+w,y+h),(255,0,0),2)
        else:
            if self.steps_threshold <= 0.:
                face_mtcnn.steps_threshold = None
            else:
                face_mtcnn.steps_threshold = [self.steps_threshold]*3
            face_mtcnn.scale_factor = self.scale
            face_mtcnn.min_face_size = self.size

            print(face_mtcnn.steps_threshold,face_mtcnn.scale_factor,face_mtcnn.min_face_size)

            result = face_mtcnn.detect_faces(mat)
            for per in result:
                box,score,points = self.get_bounding_boxes(per)
                x,y,w,h = list(map(int,box))
                cv2.rectangle(mat,(x,y),(x+w,y+h),(255,0,0),2)
                cv2.putText(mat,"%.2f"%score,(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
        
        return mat

    def get_bounding_boxes(self,mtcnn_detech):
        box = mtcnn_detech["box"]
        score = mtcnn_detech["confidence"]
        kp = mtcnn_detech["keypoints"]
        points = [kp["left_eye"],kp["right_eye"],kp["nose"],kp["mouth_left"],kp["mouth_right"]]
        return box,score,points

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