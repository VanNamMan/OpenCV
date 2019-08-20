try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

class ParamerterDialog(QDialog):
	def __init__(self,parent):
		super(ParamerterDialog,self).__init__(parent)
		self.initUI()
		self.initVar()
	def initUI(self):
		myLayout = QVBoxLayout(self)
		# layout1
		label1 = QLabel("threshold",self)
		label2 = QLabel("block size",self)
		label3 = QLabel("C",self)

		self.ln_threshold =  QLineEdit(self)
		self.ln_blockSize =  QLineEdit(self)
		self.ln_C =  QLineEdit(self)

		layout1 = QVBoxLayout(self)
		for lb,ln in zip([label1,label2,label3],[self.ln_threshold,self.ln_blockSize,self.ln_C]):
			layout = QHBoxLayout(self)
			layout.addWidget(lb)
			layout.addWidget(ln)
			layout1.addLayout(layout)

		myLayout.addLayout(layout1)

		# layout2
		label4 = QLabel("iter",self)
		label5 = QLabel("size morphology",self)
		label51 = QLabel("size blur",self)

		self.ln_iter =  QLineEdit(self)
		self.ln_sizeMorphology =  QLineEdit(self)
		self.ln_sizeBlur =  QLineEdit(self)

		layout2 = QVBoxLayout(self)
		for lb,ln in zip([label4,label5,label51],[self.ln_iter,self.ln_sizeMorphology,self.ln_sizeBlur]):
			layout = QHBoxLayout(self)
			layout.addWidget(lb)
			layout.addWidget(ln)
			layout2.addLayout(layout)

		myLayout.addLayout(layout2)

		# layout3
		label7 = QLabel("range width",self)
		label8 = QLabel("range height",self)
		label9 = QLabel("range area",self)

		self.ln_rangeWidth =  QLineEdit(self)
		self.ln_rangeHeight =  QLineEdit(self)
		self.ln_rangeArea =  QLineEdit(self)

		layout3 = QVBoxLayout(self)
		for lb,ln in zip([label7,label8,label9],[self.ln_rangeWidth,self.ln_rangeHeight,self.ln_rangeArea]):
			layout = QHBoxLayout(self)
			layout.addWidget(lb)
			layout.addWidget(ln)
			layout3.addLayout(layout)

		myLayout.addLayout(layout3)

		# layout4
		label10 = QLabel("oem",self)
		label11 = QLabel("psm",self)
		label12 = QLabel("language",self)

		self.cbb_oem =  QComboBox(self)
		self.cbb_psm =  QComboBox(self)
		self.cbb_language =  QComboBox(self)


		layout3 = QVBoxLayout(self)
		for lb,ln in zip([label10,label11,label12],[self.cbb_oem,self.cbb_psm,self.cbb_language]):
			layout = QHBoxLayout(self)
			layout.addWidget(lb)
			layout.addWidget(ln)
			layout3.addLayout(layout)

		myLayout.addLayout(layout3)

		#set layout
		self.setLayout(myLayout)
	def initVar(self):
		self.ln_threshold.setText("100")
		self.ln_blockSize.setText("11")
		self.ln_C.setText("2")

		self.ln_iter.setText("1")
		self.ln_sizeMorphology.setText("3")
		self.ln_sizeBlur.setText("3")

		self.ln_rangeWidth.setText("-1,-1")
		self.ln_rangeHeight.setText("-1,-1")
		self.ln_rangeArea.setText("-1,-1")

		  # 0    Legacy engine only.
		  # 1    Neural nets LSTM engine only.
		  # 2    Legacy + LSTM engines.
		  # 3    Default, based on what is available.

		for item in ["0    Legacy engine only."
		,"1    Neural nets LSTM engine only."
		,"2    Legacy + LSTM engines."
		,"3    Default, based on what is available."]:
			self.cbb_oem.addItem(item)
		for item in ["0    Orientation and script detection (OSD) only."
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
					,"13    Raw line. Treat the image as a single text line"]:#,\nbypassing hacks that are Tesseract-specific.
			self.cbb_psm.addItem(item)
		for item in ["eng","chi_sim","jpn","kor","vie"]:
			self.cbb_language.addItem(item)

		for cbb in [self.cbb_language,self.cbb_oem,self.cbb_psm]:
			cbb.setToolTip(cbb.currentText())

if __name__ == "__main__":
	import sys
	app = QApplication(sys.argv)
	parent = QMainWindow()
	w = ParamerterDialog(parent)
	w.show()
	sys.exit(app.exec_())

