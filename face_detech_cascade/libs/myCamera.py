from pypylon import pylon,genicam
import cv2

def getBaslerDevices():
    lstDev = []
    devices = pylon.TlFactory.GetInstance().EnumerateDevices()
    for dev in devices:
        camera = createDevice(dev)
        camera.StartGrabbing()
        lstDev.append([camera,camera.IsGrabbing(),dev.GetSerialNumber()])
    return lstDev

def createDevice(dev):
    return pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(dev))

import wmi
import re
 
def getAllDeviceUSB(listName):
    A = []
    devs = []
    wql = "Select * From Win32_USBControllerDevice"
    for item in wmi.WMI().query(wql):
        q = item.Dependent.Caption
        for dev in listName:
            if re.findall(dev,q):
                A.append(q)
    device_name = removed(A)
    for i in range(len(device_name)):
        cap = cv2.VideoCapture(i)
        devs.append([cap,cap.isOpened(),device_name[i]])
    return devs
def removed(lst):
    hdPro = []
    dino = []
    i = 0
    for a in lst:
        if "HD Pro Webcam" in a:
            hdPro.append(a+" %d"%i)
            i+=1
        else:
            dino.append(a)         
    return hdPro[::2] + dino


from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtPrintSupport import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *

import os
import sys
import time


class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.available_cameras = QCameraInfo.availableCameras()
        if not self.available_cameras:
            pass #quit

        self.status = QStatusBar()
        self.setStatusBar(self.status)


        self.save_path = ""

        self.viewfinder = QCameraViewfinder()
        self.viewfinder.show()
        self.setCentralWidget(self.viewfinder)

        # Set the default camera.
        self.select_camera(0)

        # Setup tools
        camera_toolbar = QToolBar("Camera")
        camera_toolbar.setIconSize(QSize(14, 14))
        self.addToolBar(camera_toolbar)

        photo_action = QAction(QIcon(os.path.join('images', 'camera-black.png')), "Take photo...", self)
        photo_action.setStatusTip("Take photo of current view")
        photo_action.triggered.connect(self.take_photo)
        camera_toolbar.addAction(photo_action)

        change_folder_action = QAction(QIcon(os.path.join('images', 'blue-folder-horizontal-open.png')), "Change save location...", self)
        change_folder_action.setStatusTip("Change folder where photos are saved.")
        change_folder_action.triggered.connect(self.change_folder)
        camera_toolbar.addAction(change_folder_action)


        camera_selector = QComboBox()
        camera_selector.addItems([c.description() for c in self.available_cameras])
        camera_selector.currentIndexChanged.connect( self.select_camera )

        camera_toolbar.addWidget(camera_selector)


        self.setWindowTitle("NSAViewer")
        self.show()

    def select_camera(self, i):
        self.camera = QCamera(self.available_cameras[i])
        self.camera.setViewfinder(self.viewfinder)
        self.camera.setCaptureMode(QCamera.CaptureStillImage)
        self.camera.error.connect(lambda: self.alert(self.camera.errorString()))
        self.camera.start()

        self.capture = QCameraImageCapture(self.camera)

        # self.encoderSettings = QImageEncoderSettings()
        # self.encoderSettings.setCodec("image/jpg");
        # self.encoderSettings.setResolution(800, 600);
        # print(self.capture.supportedResolutions())
        # self.capture.setEncodingSettings(self.encoderSettings)

        self.capture.error.connect(lambda i, e, s: self.alert(s))
        self.capture.imageCaptured.connect(lambda d, i: self.status.showMessage("Image %04d captured" % self.save_seq))

        self.current_camera_name = self.available_cameras[i].description()
        self.save_seq = 0

    def take_photo(self):
        timestamp = time.strftime("%d-%b-%Y-%H_%M_%S")
        self.capture.capture(os.path.join(self.save_path, "%s-%04d-%s.jpg" % (
            self.current_camera_name,
            self.save_seq,
            timestamp
        )))
        self.save_seq += 1

    def change_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Snapshot save location", "")
        if path:
            self.save_path = path
            self.save_seq = 0

    def alert(self, s):
        """
        Handle errors coming from QCamera dn QCameraImageCapture by displaying alerts.
        """
        err = QErrorMessage(self)
        err.showMessage(s)


if __name__ == '__main__':

    app = QApplication(sys.argv)
    app.setApplicationName("NSAViewer")

    window = MainWindow()
    app.exec_()





# print(getAllDeviceUSB())

# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     _,img = cap.read()
#     cv2.imshow("",img)
#     if cv2.waitKey(22) == ord("q"):
#         cv2.imwrite("data/1.jpg",img)
#         break
# cap.release()
# cv2.destroyAllWindows()

# Pypylon get camera by serial number
# serial_number = '21043274'
# info = None
# for i in pylon.TlFactory.GetInstance().EnumerateDevices():
#     if i.GetSerialNumber() == serial_number:
#         info = i
#         break
# else:
#     print('Camera with {} serial number not found'.format(serial_number))

# # VERY IMPORTANT STEP! To use Basler PyPylon OpenCV viewer you have to call .Open() method on you camera
# if info is not None:
#     camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(info))
#     camera.StartGrabbing()
#     print(camera.IsGrabbing())
#     while (camera.IsGrabbing()):
#         grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
#         if grabResult.GrabSucceeded():
#             img = grabResult.Array
#             cv2.imshow("",img)
#             if cv2.waitKey(22) == ord("q"):
#                 break
#     cv2.destroyAllWindows()
#     camera.StopGrabbing()