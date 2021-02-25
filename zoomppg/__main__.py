#### —————————————————————————————————————————— ####
#### ——————————————— Zoom-PPG ————————————————— ####
#### —————————————————————————————————————————— ####
####  /\      /\      /\    /\        /\  /\    ####
#### /  \/\  /  \  /\/  \  /  \  /\  /  \/  \   ####
####       \/    \/      \/    \/  \/        \  ####
#### —————————————————————————————————————————— ####

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import sys
import os
#### —————————————————————————————————
import numpy as np
import cv2
from statistics import mean, stdev
from scipy.signal import butter, find_peaks, lfilter, lfilter_zi
import time
import dlib


class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.setWindowTitle("ZoomPPG")

        # data
        self.time = [0] * 240
        self.data = [0] * 240
        self.peaktimes = []
        self.peakpoints = []
        self.bpm = 0

        # toolbar
        toolbar = self.headbar()

        # graphs
        graph1 = self.graphppg1()
        graph2 = self.graphppg2()

        # infogroup
        self.infogroup = QGroupBox("Statistics")
        vbox = QVBoxLayout()
        self.bpmlabel = QLabel("")
        self.specbpmlabel = QLabel("")
        vbox.addWidget(self.bpmlabel)
        vbox.addWidget(self.specbpmlabel)
        self.infogroup.setLayout(vbox)
    
        # webcam
        self.feedlabel = QLabel()
        self.feedlabel.setAlignment(Qt.AlignTop)
        self.thread = VideoThread()
        gray = QPixmap(250, 250)
        gray.fill(QColor('k'))
        self.feedlabel.setPixmap(gray)

        # layout
        layout_head = QVBoxLayout()
        layout_head.setContentsMargins(0, 0, 0, 0)
        layout_head.setSpacing(0)
        layout_head.addWidget(toolbar)

        container = QHBoxLayout()

        settlayout = QVBoxLayout()
        settlayout.setContentsMargins(10, 10, 0, 0)
        settlayout.addWidget(self.feedlabel)
        settlayout.addWidget(self.infogroup)

        graphlayout = QVBoxLayout()
        graphlayout.setContentsMargins(10, 0, 0, 0)
        graphlayout.setSpacing(0)
        graphlayout.addWidget(graph1)
        graphlayout.addWidget(graph2)
        
        widget = QWidget()
        widget.setLayout(layout_head)
        layout_head.addLayout(container)
        container.addLayout(settlayout)
        container.addLayout(graphlayout)
        self.setCentralWidget(widget)
        self.setStatusBar(QStatusBar())


    def headbar(self):
        toolbar = QToolBar("Toolbar")

        cameraBTN = QAction("Camera", self)
        cameraBTN.setStatusTip("Activate webcam")
        cameraBTN.triggered.connect(self.toolbarClick1)
        cameraBTN.setCheckable(True)
        toolbar.addAction(cameraBTN)
        toolbar.addSeparator()

        return toolbar
    

    def graphppg1(self, data=([0], [0])):
        graph = PlotWidget()
        graph.setBackground('w')
        graph.setLabel("left", "Absorbance")
        graph.setLabel("bottom", "Time (s)")
        pen = pg.mkPen(color=QColor("darkred"), width=2)
        # pen1 = pg.mkPen(color=QColor("darkred"), width=2)
        self.dataline1 = graph.plot(self.time, self.data, pen=pen)
        # self.dataline1a = graph.plot(self.time, self.data, pen=pen1)
        self.dataline1b = graph.plot(self.peaktimes, self.peakpoints, pen=None, symbol='t1', symbolBrush=(128,0,0),
                                     symbolSize=5)

        return graph


    def graphppg2(self, data=([0], [0])):
        graph = PlotWidget()
        graph.setBackground('w')
        graph.setLabel("left", "Spectral Density")
        graph.setLabel("bottom", "Freq (Hz)")
        pen = pg.mkPen(color=QColor("darkred"), width=2)
        self.dataline2 = graph.plot(self.time, self.data, pen=pen)

        return graph
    

    def toolbarClick1(self, s):
        if s == True:
            self.thread.start()
            self.thread.update.connect(self.imgupdateSlot)
        else:
            self.thread.stop()
            self.thread.update.disconnect(self.imgupdateSlot)
            gray = QPixmap(250,250)
            gray.fill(QColor('k'))
            self.feedlabel.setPixmap(gray)
    

    def imgupdateSlot(self, Image, data, time):
        self.feedlabel.setPixmap(QPixmap.fromImage(Image))
        self.time = self.time[1:] + [time]
        self.data = self.data[1:] + [data]
        
        # normalize signal
        std_ = stdev(self.data)
        if std_ == 0:
            std = 1
        else:
            std = std_
        ppg = np.subtract(np.array(self.data), mean(self.data)) / std

        # apply FFT to get PSD
        b, a = butter(N=5, Wn=[0.6, 4], fs=30, btype='bandpass')
        zi = lfilter_zi(b, a)
        filtered, zo = lfilter(b, a, ppg, zi=zi*ppg[0])
        fmag = np.abs(np.fft.rfft(filtered))
        freq = np.fft.rfftfreq(filtered.size, d=(1/30))
        self.dataline2.setData(freq, fmag)
        self.specbpmlabel.setText(f"Spectral BPM: {int(60*freq[np.where(fmag == fmag.max())])}")

        # find peaks
        cut = 50
        peaks = find_peaks(filtered[cut:], distance=8, width=(4, None))[0]
        self.peaktimes = [self.time[cut:][i] for i in peaks]
        self.peakpoints = [filtered[cut:][i] for i in peaks]
        IBIs = np.diff(np.array(self.peaktimes))
        if IBIs.size == 0:
            bpm = 0
        else:
            bpm = 60 / IBIs.mean()

        self.dataline1.setData(self.time[cut:], filtered[cut:])
        # self.dataline1a.setData(self.time, filtered)
        self.dataline1b.setData(self.peaktimes, self.peakpoints)
        self.bpmlabel.setText(f"BPM: {int(bpm)}")


class VideoThread(QThread):
    update = pyqtSignal(QImage, float, float)

    def run(self):
        #initialize
        self.active = True
        self.delaunay = True
        self.tick = 0
        self.counter = 0
        self.t = time.time()
        cap = cv2.VideoCapture(0)
        intensity = (0,0,0,0)
        
        # initialize dlib
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("models/shape_predictor_81_face_landmarks.dat")
        self.lpoints = []
        
        while self.active:
            # update ticker/counter
            self.counter += 1
            self.tick += time.time() - self.t
            self.fps = 1 / (time.time() - self.t)
            print(self.fps)
            self.t = time.time()
            ret, frame = cap.read()
            if ret:
                # reshape
                shape = frame.shape
                left = int((shape[1]/2)-(shape[0]/2))
                right = int((shape[1]/2)+(shape[0]/2))
                img = frame
                img[:,:,0] = 0
                img[:,:,2] = 0
                img = img[:,left:right,:]
                green = cv2.resize(img, (380,380), interpolation=cv2.INTER_AREA)

                # extract landmarks
                if self.counter % 15 == 0:
                    faces = self.detector(green)
                    for face in faces:
                        landmarks = self.predictor(green, box=face)
                        self.lpoints = [(i.x, i.y) for i in landmarks.parts()]
                        fhead = [self.lpoints[i] for i in [19, 24, 71]]

                if self.lpoints != []:
                    # mask
                    mask = np.zeros(green.shape[:2], dtype='uint8')
                    convex = cv2.convexHull(np.array(fhead))
                    cv2.fillConvexPoly(mask, convex, 255)
                    intensity = cv2.mean(green, mask)
                # delaunay
                if (self.lpoints != []) & (self.delaunay == True):
                    rect = cv2.boundingRect(np.array(self.lpoints))
                    subdiv = cv2.Subdiv2D(rect)
                    subdiv.insert(self.lpoints)
                    triangles = subdiv.getTriangleList()
                    triangles = np.array(triangles, dtype=np.int32)
                    for t in triangles:
                        pt1 = (t[0], t[1])
                        pt2 = (t[2], t[3])
                        pt3 = (t[4], t[5])
                        cv2.line(green, pt1, pt2, (255, 255, 255), 1)
                        cv2.line(green, pt2, pt3, (255, 255, 255), 1)
                        cv2.line(green, pt1, pt3, (255, 255, 255), 1)

                # convert to Q and emit
                ConvertToQtFormat = QImage(green.data.tobytes(), green.shape[1], green.shape[0], QImage.Format_RGB888)
                p = ConvertToQtFormat.scaled(250, 250, Qt.KeepAspectRatio)
                self.update.emit(p, intensity[1], self.tick)
    
    def stop(self):
        self.active = False
        self.quit()
    
    def maskshow(self, bool_):
        if bool_:
            self.delaunay = True
        else:
            self.delaunay = False


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()

if __name__ == '__main__':
    main()

