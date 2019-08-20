try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

import time,os

from libs.utils import *
from libs.constants import *

absPath = os.path.abspath("parameter")

styleDialog = "QDialog{    border-style: solid; \
                            border-color : darkblue; \
                                        border-width: 2px; \
                                        border-radius: 10px; \
                                        font: bold 10px; \
                                        padding: 6px;}"

class DataDialog(QDialog):
    def __init__(self,parent=None,listModel=[]):
        super(DataDialog,self).__init__(parent)
        self.setStyleSheet(styleDialog)
        # self.setWindowTitle("Visualize data")
        # self.setGeometry(QRect(100,100,640,480))
  
        self.initUI(listModel)
        self.initVar()

    def initUI(self,listModel):
        #=======
        # self.centraldWidget = QWidget(self)
        # self.setCentralWidget(self.centralWidget)
        
        vlayout = QVBoxLayout()

        hlayout0 = QHBoxLayout()

        label = QLabel("Model",self)
        self.cbb_model = QComboBox(self)
        self.cbb_model.setStyleSheet("QComboBox{min-width:10em};")
        for model in listModel:
            self.cbb_model.addItem(model)

        label2 = QLabel("Find",self)
        self.ln_model = QLineEdit("Search model",self)
        

        button = newButton("","Search.png",self.search)
        button.setToolTip("Searching...")
        button2 = newButton("Image","load_image.png",self.loadImage)
        
        self.cbb_model.currentTextChanged.connect(self.changed)
        # frame
        hlayout0.addWidget(label)
        hlayout0.addWidget(self.cbb_model)
        hlayout0.addWidget(label2)
        hlayout0.addWidget(self.ln_model)
        hlayout0.addWidget(button)
        hlayout0.addWidget(button2)

        vlayout.addLayout(hlayout0)

        hlayout1 = QVBoxLayout()

        self.table = QTableWidget(self)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)

        buttons.button(QDialogButtonBox.Ok).setIcon(newIcon("ok.png"))
        buttons.button(QDialogButtonBox.Cancel).setIcon(newIcon("cancel.png"))

        hlayout1.addWidget(self.table)
        hlayout1.addWidget(buttons)

        # buttons.accepted.connect(self.accept)
        # buttons.rejected.connect(self.reject)

        vlayout.addLayout(hlayout1)

        self.ln_model.setFocus(Qt.PopupFocusReason)
        self.setLayout(vlayout)

        palette = self.table.palette()
        role = self.table.backgroundRole()
        palette.setColor(role, QColor(100, 232, 232, 255))
        self.table.setPalette(palette)
        
        # self.centralWidget().setLayout(vlayout)

    def initVar(self):
        self.pixmap = QPixmap()
        self.models = []
        self.labelToFrame = {}
        pass

    def changed(self,dtype):
        pass

    def loadImage(self):
        pass

    def search(self):
        pass

class ModelDialog(QDialog):
    def __init__(self,parent=None):
        super(ModelDialog,self).__init__(parent)
        self.setStyleSheet(styleDialog)
        self.setWindowTitle("Choose your model?")
        self.setGeometry(QRect(1,1,200,50))
  
        self.initUI()
        self.initVar()

    def initUI(self):
        vlayout = QVBoxLayout()

        hlayout0 = QHBoxLayout()
        label = QLabel("Model",self)
        self.cbb_model = QComboBox(self)
        
        self.cbb_model.currentTextChanged.connect(self.changed)

        hlayout0.addWidget(label)
        hlayout0.addWidget(self.cbb_model)

        vlayout.addLayout(hlayout0)


        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)

        buttons.button(QDialogButtonBox.Ok).setIcon(newIcon("ok.png"))
        buttons.button(QDialogButtonBox.Cancel).setIcon(newIcon("cancel.png"))

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        vlayout.addWidget(buttons)

        self.setLayout(vlayout)
        
    def initVar(self):
        self.models = []
        pass

    def changed(self,dtype):
        pass

    def popUp(self,models=[]):
        self.models = models
        self.cbb_model.clear()
        for key in self.models:
            self.cbb_model.addItem(key)
        self.move(QCursor.pos())
        return self.cbb_model.currentText() if self.exec_() else None

class CameraDialog(QDialog):
    def __init__(self,parent=None):
        super(CameraDialog,self).__init__(parent)
        self.setStyleSheet(styleDialog)
        self.setWindowTitle("Choose Your Camera?")
        self.setGeometry(QRect(1,1,200,50))
  
        self.initUI()
        self.initVar()

    def initUI(self):
        vlayout = QVBoxLayout()

        hlayout0 = QHBoxLayout()
        label = QLabel("Type",self)
        self.cbb_dtype = QComboBox(self)
        
        self.cbb_dtype.currentTextChanged.connect(self.changed)

        hlayout0.addWidget(label)
        hlayout0.addWidget(self.cbb_dtype)

        vlayout.addLayout(hlayout0)

        hlayout1 = QHBoxLayout()
        label = QLabel("ID",self)
        self.cbb_id = QComboBox(self)
        # self.cbb_id.addItem("0")
        # self.cbb_id.addItem("1")
        hlayout1.addWidget(label)
        hlayout1.addWidget(self.cbb_id)

        # self.runProcess = QCheckBox("Run Process",self)
        # self.runProcess.stateChanged.connect(self.stateChanged)

        vlayout.addLayout(hlayout1)
        # vlayout.addWidget(self.runProcess)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)

        buttons.button(QDialogButtonBox.Ok).setIcon(newIcon("ok.png"))
        buttons.button(QDialogButtonBox.Cancel).setIcon(newIcon("cancel.png"))

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        self.bDraw = QCheckBox("Drawing",self) 
        vlayout.addWidget(self.bDraw)
        vlayout.addWidget(buttons)

        self.setLayout(vlayout)
    
    def initVar(self):
        self.dictId = {}
        pass

    def stateChanged(self):
        # self.parent().window().bStart = self.runProcess.isChecked()
        pass

    def changed(self,dtype):
        if dtype not in list(self.dictId.keys()):
            return
        self.cbb_id.clear()
        listId = self.dictId[dtype]
        for idc in listId:
            self.cbb_id.addItem(idc)

    def popUp(self,dictId={}):
        self.dictId = dictId
        self.cbb_dtype.clear()
        for key in self.dictId.keys():
                self.cbb_dtype.addItem(key)
        self.move(QCursor.pos())
        return [self.cbb_dtype.currentText(),self.cbb_id.currentText()] if self.exec_() else None

class PasswordDialog(QDialog):
    def __init__(self,parent=None):
        super(PasswordDialog,self).__init__(parent)
  
        self.initUI()
        self.initVar()

    def initUI(self):
        vlayout = QVBoxLayout()

        hlayout = QHBoxLayout()
        label = QLabel("Password",self)
        self.ln_password = QLineEdit("Password",self)
        self.ln_password.setFocus()
        hlayout.addWidget(label)
        hlayout.addWidget(self.ln_password)

        vlayout.addLayout(hlayout)

        # hlayout = QHBoxLayout(self)
        # but_ok = newButton("OK",slot=self.ok_)
        # but_cancel = newButton("Cancel",slot=self.cancel_)
        # hlayout.addWidget(but_ok)
        # hlayout.addWidget(but_cancel)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)

        buttons.button(QDialogButtonBox.Ok).setIcon(newIcon("ok.png"))
        buttons.button(QDialogButtonBox.Cancel).setIcon(newIcon("cancel.png"))

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        vlayout.addWidget(buttons)

        vlayout.addLayout(hlayout)

        self.setLayout(vlayout)

    def initVar(self):

        pass

    # def keyPressEvent(self, qKeyEvent):
    #     if qKeyEvent.key() == Qt.Key_Return: 
    #         self.accept()

    @staticmethod
    def getPass(parent=None):
        dialog = PasswordDialog(parent)
        result = dialog.exec_()
        password = dialog.ln_password.text()
        return (password,result == QDialog.Accepted)

    def popUp(self):
        self.move(QCursor.pos())
        return [self.ln_password.text(),self.exec_()] if self.exec_() else None
        
class LogfileDialog(QDialog):
    def __init__(self,parent=None):
        super(LogfileDialog,self).__init__(parent)
        self.setStyleSheet(styleDialog)
        self.initUI()
        self.initVar()

    def initUI(self):
        layout = QVBoxLayout()

        self.listLog = QListWidget(self)
        layout.addWidget(self.listLog)

        self.setLayout(layout)

    def initVar(self):
        self.addLog("logfile")
        pass

    def addLog(self,obj):
        txt = time.strftime("%H:%M:%S ")+str(obj)
        self.listLog.addItem(txt)


class ResultDialog(QDialog):
    def __init__(self,parent=None):
        super(ResultDialog,self).__init__(parent)
        self.setGeometry(QRect(0,0,300,300))
        self.setWindowTitle("Result Dialog")
        self.setStyleSheet(styleDialog)
        self.initVar()
        self.initUI()

    def initUI(self):

        layout = QVBoxLayout()

        hlayout = QHBoxLayout()

        lb = QLabel("Result",self)

        # model = QStringListModel()
        # model.setStringList(lbOfShapes)

        self.cbb_result = CheckableComboBox(self)
        but_save = newButton("Save Image",icon="saveImage.png",slot=self.saveImage)

        addLayouts(hlayout,[lb,self.cbb_result,but_save])
        layout.addLayout(hlayout)



        hlayout1 = QHBoxLayout()
        lb1 = QLabel("Barcode",self)
        self.lb_barcode = QLineEdit(self)
        # self.lb_barcode.setStyleSheet("{background-color:black};")
        addLayouts(hlayout1,[lb1,self.lb_barcode])

        hlayout2 = QHBoxLayout()
        lb2 = QLabel("OCR",self)
        self.lb_OCR = QLineEdit(self)
        # self.lb_OCR.setStyleSheet("{background-color:black};")
        addLayouts(hlayout2,[lb2,self.lb_OCR])

        addLayouts(layout,[hlayout1,hlayout2])

        self.frame = QLabel(self)
        self.frame.setStyleSheet("QLabel{background-color:black}")
        self.frame.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        layout.addWidget(self.frame)

        self.setLayout(layout)

    def initVar(self):
        self.mat = None
        pass

    def resetState(self):
        self.lb_OCR.setText("")
        self.lb_barcode.setText("")
        showImage(QPixmap(),self.frame)

    def setItem(self,items):
        self.cbb_result.clear()
        self.cbb_result.setItem(items)

    def showImage(self,mat):
        self.mat = mat
        showImage(mat,self.frame,fitwindow=True)

    def saveImage(self):
        mkdir("Result")
        cv2.imwrite("Result/%s.jpg"%self.cbb_result.currentText(),self.mat)


        


class ParamerterDialog(QDialog):
    def __init__(self,parent=None):
        super(ParamerterDialog,self).__init__(parent)
        self.setStyleSheet(styleDialog)
        self.initVar()
        self.initUI()
        
    def initUI(self):
        myLayout = QVBoxLayout()

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)

        buttons.button(QDialogButtonBox.Ok).setIcon(newIcon("ok.png"))
        buttons.button(QDialogButtonBox.Cancel).setIcon(newIcon("cancel.png"))

        buttons.accepted.connect(self.changeParams)
        buttons.rejected.connect(self.cancel)

        
        # layout1
        layout1 = QVBoxLayout()

        advance_action = newAction(self,"Advance Options",self.advance,ADVANCE_PARAMS,"advance_tools.png")
        advance_button = QToolButton(self)
        advance_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        advance_button.setDefaultAction(advance_action)
        layout1.addWidget(advance_button)

        hlayout = QHBoxLayout()
        lb = QLabel("modeThresh",self)
        self.useOtsu = QCheckBox("useOtsu",self)
        self.cbb_modeThresh = QComboBox(self)
        addLayouts(hlayout,[lb,self.useOtsu,self.cbb_modeThresh])
        layout1.addLayout(hlayout)

        hlayout = QHBoxLayout()
        lb1 = QLabel("findContours",self)
        self.cbb_modeFindcontours = QComboBox(self)
        self.cbb_methodFindcontours = QComboBox(self)
        addLayouts(hlayout,[lb1,self.cbb_modeFindcontours,self.cbb_methodFindcontours])
        layout1.addLayout(hlayout)

        addItem(self.cbb_modeThresh, items=["cv2.THRESH_BINARY","cv2.THRESH_BINARY_INV"])
        addItem(self.cbb_modeFindcontours, items=["cv2.RETR_EXTERNAL","cv2.RETR_LIST"])
        addItem(self.cbb_methodFindcontours, items=["cv2.CHAIN_APPROX_NONE","cv2.CHAIN_APPROX_SIMPLE"])

        label1 = QLabel("threshold",self)
        label2 = QLabel("block size",self)
        label3 = QLabel("C",self)

        self.ln_threshold = spinBox(self,100)
        self.ln_blockSize =  spinBox(self,31,range=(0,5000))
        self.ln_C =  spinBox(self,range=(-255,255))

        # self.widget = QWidget()
        
        for lb,ln in zip([label1,label2,label3],[self.ln_threshold,self.ln_blockSize,self.ln_C]):
            layout = QHBoxLayout()
            layout.addWidget(lb)
            layout.addWidget(ln)
            layout1.addLayout(layout)



        # myLayout.addLayout(layout1)

        # layout2
        label4 = QLabel("iter",self)
        label5 = QLabel("size morphology",self)
        label51 = QLabel("size blur",self)

        self.ln_iter = spinBox(self,1)
        self.ln_sizeMorphology =  spinBox(self,3)
        self.ln_sizeBlur =  spinBox(self,3)

        # layout2 = QVBoxLayout(self)
        for lb,ln in zip([label4,label5,label51],[self.ln_iter,self.ln_sizeMorphology,self.ln_sizeBlur]):
            layout = QHBoxLayout()
            layout.addWidget(lb)
            layout.addWidget(ln)
            layout1.addLayout(layout)

        # myLayout.addLayout(layout2)

        # layout3
        label7 = QLabel("range width",self)
        label8 = QLabel("range height",self)
        label9 = QLabel("range area",self)

        self.ln_rangeWidth =  QLineEdit("-1,-1",self)
        self.ln_rangeHeight =  QLineEdit("-1,-1",self)
        self.ln_rangeArea =  QLineEdit("-1,-1",self)

        # layout3 = QVBoxLayout(self)
        for lb,ln in zip([label7,label8,label9],[self.ln_rangeWidth,self.ln_rangeHeight,self.ln_rangeArea]):
            layout = QHBoxLayout()
            layout.addWidget(lb)
            layout.addWidget(ln)
            layout1.addLayout(layout)

        layout1.addWidget(buttons)

        myLayout.addLayout(layout1)

        self.setLayout(myLayout)
        self.setEnabled(False)
    def initVar(self):
        self.advanceDialog = AdvanceParamsDialog(self)
        pass

    def advance(self):
        self.advanceDialog.show()
        self.advanceDialog.move(QCursor.pos())
        pass

    def keyEvent(self,ev):
        print(ev)
        if ev.key() == Qt.Key_Return:
            self.save()

    def releaseState(self):
        self.ln_threshold.setText("100")
        self.ln_blockSize.setText("11")
        self.ln_C.setText("3")
        self.ln_iter.setText("1")
        self.ln_sizeBlur.setText("3")
        self.ln_sizeMorphology.setText("3")
        self.ln_rangeArea.setText("-1,-1")
        self.ln_rangeHeight.setText("-1,-1")
        self.ln_rangeWidth.setText("-1,-1")

    def activeState(self):
        pass

    def edit(self):
        pass
        
    def changeParams(self):
        window = self.parent().window()
        if window.canvas.selectedShape is None:
            return       
        msg = QMessageBox.question(self,"Save file?","Do you want to set shape params?",QMessageBox.Yes|QMessageBox.No)
        if msg == QMessageBox.Yes:
            self.changeShapeParams()
            window.setStatusTip("Done changed shape params")
            self.setEnabled(False)
        else:
            pass

    def cancel(self):
        self.setEnabled(False)

    def getParams(self):
        """
        return dict params vision.
        """
        advance = self.advanceDialog

        if advance.rad_sobel.isChecked():
            modeFilter = "sobel"
        elif advance.rad_laplace.isChecked():
            modeFilter = "laplace"
        else:
            modeFilter = "scharr"

        if advance.rad_normalBlur.isChecked():
            modeBlur = "normal"
        elif advance.rad_gaussBlur.isChecked():
            modeBlur = "median"
        else:
            modeBlur = "gauss"

        if advance.rad_dilate.isChecked():
            modeMorphology = "dilate"
        elif advance.rad_erode.isChecked():
            modeMorphology = "erode"
        elif advance.rad_open.isChecked():
            modeMorphology = "open"
        else:
            modeMorphology = "close"

        if advance.rad_horizontal.isChecked() :
            modeExtract = "horizontal" 
        else :
            modeExtract = "vertical"

        config={
            
            "findContours" :    {
                                    "mode" : self.cbb_modeFindcontours.currentText(),
                                    "method" : self.cbb_methodFindcontours.currentText(),
                                },

            "adaptive" :    {
                                "blockSize":self.ln_blockSize.text(),
                                "C":self.ln_C.text()
                            },
            
            "threshold":    {   
                                "mode" : self.cbb_modeThresh.currentText(),
                                "kThresh" : self.ln_threshold.text(),
                                "useOtsu" : self.useOtsu.isChecked()
                            },
            
            "morphology" :  {   
                                "mode" : modeMorphology,
                                "element" : advance.cbb_modeElement.currentText(),
                                "iter" : self.ln_iter.text(),
                                "size" : self.ln_sizeMorphology.text(),
                            },
            
            "removeBlobs" : {
                                "rangeWidth":self.ln_rangeWidth.text(),
                                "rangeHeight":self.ln_rangeHeight.text(),
                                "rangeArea":self.ln_rangeArea.text()
                            },
            
            "extractLines" : { 
                                "mode":modeExtract,
                                "hSize" : advance.ln_sizeHorizontal.text(),
                                "vSize" : advance.ln_sizeVertical.text()
                             },

            "filter" :  {
                            "mode" : modeFilter,
                            "size" : advance.ln_sizeFilter.text(),
                            "delta" : advance.ln_deltaFilter.text(),
                            "scale" : advance.ln_scaleFilter.text(),
                            "xAxis" : advance.ch_xAxis.isChecked(),
                            "yAxis" : advance.ch_yAxis.isChecked()
                        },
            
            "blur" :    {   
                            "mode" : modeBlur,
                            "size":self.ln_sizeBlur.text()
                        },

            "tesseract" :   {
                                "oem" : advance.cbb_oem.currentIndex(),
                                "psm" : advance.cbb_psm.currentIndex(),
                                "lang" : advance.cbb_lang.currentText()
                            }
            
        }
        return config

    def changeShapeParams(self):
        # filename = os.path.join(absPath,"params.json")
        # params = load_from_json(filename)
        # if params is None:
            # params = {}
        window = self.parent().window()
        shape = window.canvas.selectedShape
        i,label,qRect = window.formatShape(shape)

        shape.paramsVision = self.getParams()

    def loadPara(self):
        pass

class AdvanceParamsDialog(QDialog):
    def __init__(self,parent=None):
        super(AdvanceParamsDialog,self).__init__(parent)
        self.setWindowTitle("Advance Options")
        self.initVar()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        #=================
        extractLayout = QGridLayout()

        self.ln_sizeHorizontal = spinBox(self,30,range=(0,100))
        self.ln_sizeVertical = spinBox(self,30,range=(0,100))
        self.rad_horizontal = QRadioButton("horizontal")
        self.rad_horizontal.setChecked(True)
        self.rad_vertical = QRadioButton("vertical")

        extractLayout.addWidget(self.rad_horizontal,0,0)
        extractLayout.addWidget(self.ln_sizeHorizontal,0,1)
        extractLayout.addWidget(self.rad_vertical,0,2)
        extractLayout.addWidget(self.ln_sizeVertical,0,3)

        filterLayout = QGridLayout()

        self.rad_sobel =  QRadioButton("Sobel",self)
        self.rad_sobel.setChecked(True)
        self.rad_laplace =  QRadioButton("Laplace",self)
        self.rad_scharr =  QRadioButton("Scharr",self)
        

        self.ch_xAxis = QCheckBox("x-Axis",self)
        self.ch_yAxis = QCheckBox("y-Axis",self)
        self.ch_xAxis.setChecked(True)
        self.ch_yAxis.setChecked(True)

        lbKernel = QLabel("Kernel")
        lbScale = QLabel("Scale")
        lbDelta = QLabel("Delta")
        self.ln_sizeFilter = QLineEdit("3",self)
        self.ln_scaleFilter = QLineEdit("1",self)
        self.ln_deltaFilter = QLineEdit("0",self)

        filterLayout.addWidget(self.rad_sobel,0,0)
        filterLayout.addWidget(self.rad_laplace,0,1)
        filterLayout.addWidget(self.rad_scharr,0,2)
        filterLayout.addWidget(self.ch_xAxis,1,0)
        filterLayout.addWidget(self.ch_yAxis,1,1)
        filterLayout.addWidget(lbKernel,1,2)
        filterLayout.addWidget(self.ln_sizeFilter,1,3)
        filterLayout.addWidget(lbScale,2,0)
        filterLayout.addWidget(self.ln_scaleFilter,2,1)
        filterLayout.addWidget(lbDelta,2,2)
        filterLayout.addWidget(self.ln_deltaFilter,2,3)
        #=================
        morphologyLayout = QGridLayout()

        self.rad_dilate =  QRadioButton("Dilate",self)
        self.rad_dilate.setChecked(True)
        self.rad_erode =  QRadioButton("Erode",self)
        self.rad_open =  QRadioButton("Open",self)
        self.rad_close =  QRadioButton("Close",self)

        lbModeElement = QLabel("Element",self)
        self.cbb_modeElement = QComboBox(self)
        addItem(self.cbb_modeElement,["cv2.MORPH_RECT","cv2.MORPH_ELLIPSE","cv2.MORPH_CROSS"])
        
        morphologyLayout.addWidget(lbModeElement,0,0)
        morphologyLayout.addWidget(self.cbb_modeElement,0,1)
        morphologyLayout.addWidget(self.rad_dilate,1,0)
        morphologyLayout.addWidget(self.rad_erode,1,1)
        morphologyLayout.addWidget(self.rad_open,1,2)
        morphologyLayout.addWidget(self.rad_close,1,3)
        #=================
        blurLayout = QGridLayout()

        self.rad_normalBlur =  QRadioButton("Normal",self)
        self.rad_normalBlur.setChecked(True)
        self.rad_medianBlur =  QRadioButton("Median",self)
        self.rad_gaussBlur =  QRadioButton("Gaussian",self)
        
        blurLayout.addWidget(self.rad_normalBlur,0,0)
        blurLayout.addWidget(self.rad_medianBlur,0,1)
        blurLayout.addWidget(self.rad_gaussBlur,0,2)
        #========

        tessLayout = QGridLayout()
        lbOem = QLabel("oem",self)
        lbPsm = QLabel("psm",self)
        lbLang = QLabel("lang",self)

        self.cbb_oem = QComboBox(self)
        self.cbb_psm = QComboBox(self)
        self.cbb_lang = QComboBox(self)

        
        addItem(self.cbb_oem,self.oems)
        addItem(self.cbb_psm,self.psms)
        addItem(self.cbb_lang,self.langs)

        tessLayout.addWidget(lbOem,0,0)
        tessLayout.addWidget(self.cbb_oem,0,1)
        tessLayout.addWidget(lbPsm,1,0)
        tessLayout.addWidget(self.cbb_psm,1,1)
        tessLayout.addWidget(lbLang,2,0)
        tessLayout.addWidget(self.cbb_lang,2,1)
        #=================
        extractGroup = QGroupBox("Extract Horizontal-Vertical",self)
        extractGroup.setLayout(extractLayout)

        filterGroup = QGroupBox("Filter",self)
        filterGroup.setLayout(filterLayout)

        morphologyGroup = QGroupBox("Morphology",self)
        morphologyGroup.setLayout(morphologyLayout)

        blurGroup = QGroupBox("Bluring",self)
        blurGroup.setLayout(blurLayout)

        tessGroup = QGroupBox("Tesseract",self)
        tessGroup.setLayout(tessLayout)

        layout.addWidget(filterGroup)
        layout.addWidget(morphologyGroup)
        layout.addWidget(blurGroup)
        layout.addWidget(extractGroup)
        layout.addWidget(tessGroup)

        self.setLayout(layout)
        # self.move(QCursor.pos())

    def initVar(self):
        self.psms = ["0    Orientation and script detection (OSD) only."
                  ,"1    Automatic page segmentation with OSD."
                  ,"2    Automatic page segmentation, but no OSD, or OCR."
                  ,"3    Fully automatic page segmentation, but no OSD. (Default)"
                  ,"4    Assume a single column of text of variable sizes."
                  ,"5    Assume a single uniform block of vertically aligned text."
                  ,"6    Assume a single uniform block of text."
                  ,"7    Treat the image as a single text line."
                  ,"8    Treat the image as a single word."
                  ,"9    Treat the image as a single word in a circle."
                  ,"10    Treat the image as a single character."
                  ,"11    Sparse text. Find as much text as possible in no particular order."
                  ,"12    Sparse text with OSD."
                  ,"13    Raw line. Treat the image as a single text line"]

        self.oems = ["0    Legacy engine only."
                ,"1    Neural nets LSTM engine only."
                ,"2    Legacy + LSTM engines."
                ,"3    Default, based on what is available."]

        self.langs = ["eng","chi_sim","jpn","kor","vie"]
        pass
       
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    parent = QMainWindow()
    w = ParamerterDialog(parent)
    w.show()
    sys.exit(app.exec_())

