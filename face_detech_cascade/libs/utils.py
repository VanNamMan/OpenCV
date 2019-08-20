from math import sqrt
from libs.ustr import ustr
import hashlib
import re
import sys

from PIL import ImageQt
from scipy import misc
import cv2,time,os,json,threading
from configparser import ConfigParser

try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

absPath = os.path.abspath("res")
print("res : ",absPath)


class spinBox(QSpinBox):
    def __init__(self,parent,value=0,range=(0,255),singleStep=1):
        super(spinBox,self).__init__(parent)
        self.setRange(range[0],range[1])
        self.setValue(value)
        self.setSingleStep(singleStep)

    def setText(self,text):
        i = str2int(text)
        self.setValue(i)

class CheckableComboBox(QComboBox):
    def __init__(self,parent=None,listItem=[]):
        super(CheckableComboBox, self).__init__(parent)
        self.view().pressed.connect(self.handleItemPressed)
        # model.setStringList(listItem)


    def handleItemPressed(self, index):
        self.setOnlyItemCheckState(index.row())
        self.parent().window().resetState()
        # item = self.model().itemFromIndex(index)
        # item.setCheckState(not item.checkState())

    def setItem(self,items):
        model = QStringListModel()
        model.setStringList(items)
        for i in range(len(items)):
            self.addItem(items[i])
            item = self.model().item(i, 0)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(not Qt.Checked)

    def setOnlyItemCheckState(self,i):
        for j in range(self.count()):
            item = self.model().item(j,0)
            if j == i :
                item.setCheckState(True)
                self.setCurrentText(item.text())
            else:
                item.setCheckState(False)

def newIcon(icon):
    # return QIcon(':/' + icon)
    return QIcon(os.path.join(absPath,icon))


def newButton(text, icon=None, slot=None):
    b = QPushButton(text)
    if icon is not None:
        b.setIcon(newIcon(icon))
    if slot is not None:
        b.clicked.connect(slot)
    return b


def newAction(parent, text, slot=None, shortcut=None, icon=None,
              tip=None, checkable=False, enabled=True):
    """Create a new action and assign callbacks, shortcuts, etc."""
    a = QAction(text, parent)
    if icon is not None:
        a.setIcon(newIcon(icon))
    if shortcut is not None:
        if isinstance(shortcut, (list, tuple)):
            a.setShortcuts(shortcut)
        else:
            a.setShortcut(shortcut)
    if tip is not None:
        a.setToolTip(tip)
        a.setStatusTip(tip)
    if slot is not None:
        a.triggered.connect(slot)
    if checkable:
        a.setCheckable(True)
    a.setEnabled(enabled)
    return a

def addLayouts(layout,widgets):
    for w in widgets:
        if isinstance(w,QLayout):
            layout.addLayout(w)
        else :
            layout.addWidget(w)

def addActions(widget, actions):
    for action in actions:
        if action is None:
            widget.addSeparator()
        elif isinstance(action, QMenu):
            widget.addMenu(action)
        else:
            widget.addAction(action)

def addItem(cbb,items):
    for item in items:
        cbb.addItem(item)

def labelValidator():
    return QRegExpValidator(QRegExp(r'^[^ \t].+'), None)


class struct(object):

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def distance(p):
    return sqrt(p.x() * p.x() + p.y() * p.y())


def fmtShortcut(text):
    mod, key = text.split('+', 1)
    return '<b>%s</b>+<b>%s</b>' % (mod, key)

def string2QFont(string):
    family,pointSize,weight,italic = string.split(",")
    return QFont(family,int(pointSize),int(weight),italic=="True")

def qFont2String(font):
    return "%s,%s,%s,%s"%(font.family(),font.pointSize(),font.weight(),font.italic())

def generateColorByText(text):
    s = ustr(text)
    hashCode = int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16)
    r = int((hashCode / 255) % 255)
    g = int((hashCode / 65025)  % 255)
    b = int((hashCode / 16581375)  % 255)
    return QColor(r, g, b, 100)

def have_qstring():
    '''p3/qt5 get rid of QString wrapper as py3 has native unicode str type'''
    return not (sys.version_info.major >= 3 or QT_VERSION_STR.startswith('5.'))

def util_qt_strlistclass():
    return QStringList if have_qstring() else list

def natural_sort(list, key=lambda s:s):
    """
    Sort the list into natural alphanumeric order.
    """
    def get_alphanum_key_func(key):
        convert = lambda text: int(text) if text.isdigit() else text
        return lambda s: [convert(c) for c in re.split('([0-9]+)', key(s))]
    sort_key = get_alphanum_key_func(key)
    list.sort(key=sort_key)

#byME
def runThread(target=None,args=()):
    if target :
        thread = threading.Thread(target=target,args=args)
        thread.start()

def sendCmd(cmd):
    os.system(cmd)

def str2ListInt(mstr,sep=","):
    lst = mstr.split(sep)
    return [str2int(t) for t in lst]

def str2int(str):
    try:
        return int(str)
    except:
        return 0

def str2float(str):
    try:
        return float(str)
    except:
        return 0.0

def save_to_json(filename,data):
    try:
        with open(filename,"w") as ff:
            json.dump(data,ff)
            return True
    except:
        return False

def load_from_json(filename):
    try:
        with open(filename,"r") as ff:
            data = json.load(ff)
            return data
    except:
        print("Load file json error.")
        return None

def load_from_cfg(filename):
    try:
        config = ConfigParser()
        config.read(filename)
        return config
    except:
        print("Load file cfg error.")
        return None

def save_to_cfg(filename,config):
    try:
        with open(filename, 'w') as configfile:
            config.write(configfile)
            return True
    except:
        return False

def getStrDateTime():
    return time.strftime("%d%m%y_%H%M%S")\

def mkdir(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)

def qRect2cvRect(qRect):
    return qRect.x(),qRect.y(),qRect.width(),qRect.height()

def showImage(image,frame,fitwindow=False):
    if image is None:
        return
    if isinstance(image,QPixmap):
        frame.setPixmap(image)
    elif isinstance(image,QImage):
        qpix = QPixmap()
        frame.setPixmap(ndArray2Qpixmap(qpix.fromImage(image)))
    else:
        size = None
        w,h = frame.width(),frame.height()
        if fitwindow:
            h0,w0 = image.shape[:2]

            tx = w/w0
            ty = h/h0

            if ty < tx :
                h_new = h - 2
                w_new = int(min((w0*h_new/h0),w-2))
            else:
                w_new = w - 2
                h_new = int(min((h0*w_new/w0),h-2))
            size = (w_new,h_new) 
        else:
            size = (w,h)
            
        frame.setPixmap(ndArray2Qpixmap(image,size=size))

def ndArray2Qpixmap(img,size=None):
    if size is not None:
        copy = cv2.resize(img,size)
    else:
        copy = img.copy()

    h,w = copy.shape[:2]

    if len(img.shape) == 2:
        qPix = QPixmap.fromImage(ImageQt.ImageQt(misc.toimage(copy)))
    else:
        ch = 3
        copy = cv2.cvtColor(copy,cv2.COLOR_BGR2RGB)
        qImg = QImage(copy.data,w,h,ch*w,QImage.Format_RGB888)
        qPix = QPixmap(qImg)

    return qPix
#============
