
#source https://github.com/mfitzp/15-minute-apps/blob/master/camera/camera.py
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtPrintSupport import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *

from libs.utils import newAction,addActions,struct

import os
import sys
import time
from functools import partial

fileLabel = "save_path.txt"
listLabel = []
if os.path.exists(fileLabel):
    with open(fileLabel,"r") as inFile:
        save_path = inFile.readline()

class cameraDialog(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(cameraDialog, self).__init__(*args, **kwargs)

        self.available_cameras = QCameraInfo.availableCameras()
        if not self.available_cameras:
            pass #quit

        self.status = QStatusBar()
        self.setStatusBar(self.status)


        self.save_path = save_path

        self.viewfinder = QCameraViewfinder()
        self.viewfinder.show()
        self.setCentralWidget(self.viewfinder)

        self.encoderSettings = QImageEncoderSettings()

        # Setup tools
        camera_toolbar = QToolBar("Camera")
        # camera_toolbar.setIconSize(QSize(14, 14))
        camera_toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.addToolBar(Qt.LeftToolBarArea,camera_toolbar)
        #D:/GitHub/PyQt_Form/LabelImg/

        camera_selector = QComboBox()
        camera_selector.setToolTip("Change your Camera")
        camera_selector.setStatusTip("Change your Camera which you want.")
        camera_selector.addItems([c.description() for c in self.available_cameras])
        camera_selector.currentIndexChanged.connect( self.select_camera )

        camera_toolbar.addWidget(camera_selector)

        action = partial(newAction,self)
        start_action = action("Start Camera",self.start_camera,"","res/camera2.png","Start Camera")
        stop_action = action("Stop Camera",self.stop_camera,"","res/stop.png","Stop Camera")
        photo_action = action("Take Photo",self.take_photo,"","res/capture.png","Take Photo")
        change_folder_action = action("Save Folder",self.change_folder,"","res/openDir.png","Change save folder")

        self.actions = struct(start_action=start_action
                            ,stop_action=stop_action
                            ,photo_action=photo_action
                            ,change_folder_action=change_folder_action)
        
        addActions(camera_toolbar,[start_action,stop_action,photo_action,change_folder_action])

        self.setWindowTitle("Camera %s"%self.save_path)
        # self.show()
        # Set the default camera.
        self.select_camera(0)

    def select_camera(self, i):
        self.camera = QCamera(self.available_cameras[i])

        # self.camera = QCamera("Integrated Webcam")
        self.camera.setViewfinder(self.viewfinder)
        self.camera.setCaptureMode(QCamera.CaptureStillImage)
        self.camera.error.connect(lambda: self.alert(self.camera.errorString()))

        # self.encoderSettings.setCodec("image/jpg");
        # self.encoderSettings.setResolution(1280, 720);
        
        self.start_camera()

        self.capture = QCameraImageCapture(self.camera)
        # self.capture.setEncodingSettings(self.encoderSettings)
        self.capture.error.connect(lambda i, e, s: self.alert(s))
        self.capture.imageCaptured.connect(lambda d, i: self.status.showMessage("Image %04d captured" % self.save_seq))

        self.current_camera_name = self.available_cameras[i].description()
        self.save_seq = 0

        self.camera.unlock()
        # self.show()

    def take_photo(self):
        # print(type(self.viewfinder.grab()))
        timestamp = time.strftime("%d-%b-%Y-%H_%M_%S")
        self.capture.capture(os.path.join(self.save_path, "%s-%04d-%s.jpg" % (
            self.current_camera_name,
            self.save_seq,
            timestamp
        )))
        self.save_seq += 1

    def start_camera(self):
        self.camera.start()
        if self.camera.state() == QCamera.ActiveState:
        # print(self.camera.errorString())
            self.actions.start_action.setEnabled(False)
            self.actions.stop_action.setEnabled(True)
        else:
            self.actions.start_action.setEnabled(True)
            self.actions.stop_action.setEnabled(False)

    def stop_camera(self):
        self.camera.stop()
        self.actions.start_action.setEnabled(True)
        self.actions.stop_action.setEnabled(False)

    def change_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Snapshot save location", "")
        if path:
            self.save_path = path
            self.save_seq = 0

    def alert(self, s):
        """
        Handle errors coming from QCamera dn QCameraImageCapture by displaying alerts.
        """
        # self.actions.start_action.setEnabled(True)
        # self.actions.stop_action.setEnabled(False)
        err = QErrorMessage(self)
        err.showMessage(s)


    def closeEvent(self, event):
        print("Close camera")
        self.stop_camera()

# if __name__ == '__main__':

#     app = QApplication(sys.argv)
#     app.setApplicationName("NSAViewer")

#     window = cameraDialog()
#     app.exec_()
