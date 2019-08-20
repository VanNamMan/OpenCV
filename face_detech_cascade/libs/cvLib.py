import cv2
import numpy as np
import pytesseract
from pyzbar.pyzbar import decode
from pylibdmtx.pylibdmtx import decode as D

from PyQt5.QtGui import QImage,qRgb,QPixmap
# import qimage2ndarray

def flag(string):
	if string == "cv2.THRESH_BINARY":
		return cv2.THRESH_BINARY
	elif string == "cv2.THRESH_BINARY_INV":
		return cv2.THRESH_BINARY_INV
	elif string == "cv2.RETR_EXTERNAL":
		return cv2.RETR_EXTERNAL
	elif string == "cv2.RETR_LIST":
		return cv2.RETR_LIST
	elif string == "cv2.CHAIN_APPROX_NONE":
		return cv2.CHAIN_APPROX_NONE
	elif string == "cv2.CHAIN_APPROX_SIMPLE":
		return cv2.CHAIN_APPROX_SIMPLE
	elif string == "cv2.MORPH_RECT":
		return cv2.MORPH_RECT
	elif string == "cv2.MORPH_ELLIPSE":
		return cv2.MORPH_ELLIPSE
	elif string == "cv2.MORPH_CROSS":
		return cv2.MORPH_CROSS

def bgr2gray(bgr):
	return cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)

def rgb2gray(rgb):
	return cv2.cvtColor(bgr,cv2.COLOR_RGB2GRAY)

def gray2bgr(gray):
	return cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)

def gray2rgb(gray):
	return cv2.cvtColor(bgr,cv2.COLOR_GRAY2RGB)

def threshold(gray,thresh,mode=cv2.THRESH_BINARY):
    return cv2.threshold(gray,thresh,255,mode)[1]

def adaptive(gray,method=cv2.ADAPTIVE_THRESH_MEAN_C,mode=cv2.THRESH_BINARY,blockSize=31,C=2):
	return cv2.adaptiveThreshold(gray,255,method,mode,blockSize,C)

def invert(img):
	return 255-img
def get_meanStd(img,rois=-1):
	if rois == -1:
		return cv2.meanStdDev(img)
	else:
		return [cv2.meanStdDev(img[y1:y2,x1:x2]) for x1,y1,x2,y2 in rois]

# If you don't have tesseract executable in your PATH, include the following:
# pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_your_tesseract_executable>'

#Add the following config, if you have tessdata error like: "Error opening data file..."
# Example config: r'--tessdata-dir "C:\Program Files (x86)\Tesseract-OCR\tessdata"'
# It's important to add double quotes around the dir path.
# tessdata_dir_config = r'--tessdata-dir "<replace_with_your_tessdata_dir_path>"'
# pytesseract.image_to_string(image, lang='chi_sim', config=tessdata_dir_config)

def get_text(img,lang="eng",config = ('--oem 1 --psm 3')):
    return pytesseract.image_to_string(img,lang=lang)
# barcode
def getMatrixCode(img):
	try:
		codes = D(img)
		return [[code.data.decode("utf-8"),code.type]for code in codes]
	except:
		return None
def getBarcode(img):
	try:
		codes = decode(img)
		return [[code.data.decode("utf-8"),code.type]for code in codes]
	except:
		return None

def extractHorizontal(img,horizontal_size):
	"""
	extractHorizontal binary image , return extract iamge
	ex : horizontal_size = img.shape[1]//30
	"""
    # Create structure element for extracting horizontal lines through morphology operations
	horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
	# Apply morphology operations
	horizontal = cv2.erode(img, horizontalStructure)
	horizontal = cv2.dilate(horizontal, horizontalStructure)

	return horizontal

def extractVertical(img,vertical_size):
	"""
	extractHorizontal binary image , return extract iamge
	ex : vertical_size = img.shape[0]//30
	"""
    # Create structure element for extracting horizontal lines through morphology operations
	verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1,vertical_size))
	# Apply morphology operations
	vertical = cv2.erode(img, verticalStructure)
	vertical = cv2.dilate(vertical, verticalStructure)

	return vertical



