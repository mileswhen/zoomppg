import numpy as np
import warnings; warnings.filterwarnings("ignore")
import cv2
import mss
import matplotlib.pyplot as plt
import matplotlib as mplib
import scipy.signal as signal
import time


# initialize ppg data
six = 0
hcount = 100
tick = 0
ppg_green = [1] * hcount
times = [0] * hcount
t_ = time.time()

# mplib graph
mplib.use('TkAgg')
fig = plt.figure()
ax = fig.add_subplot(111)

with mss.mss() as sct:
    monitor = {"top": 0, "left": 0, "width": 2560, "height": 1600}
    while "Screen capturing":
        img = np.array(sct.grab(monitor))
        r = cv2.selectROI("crop", img, False, False)
        left, top, width, height = r
        cv2.destroyWindow("crop")
        break     

x, y, w, h = [0, 0, width, height]

with mss.mss() as sct:
    monitor = {"top": top, "left": left, "width": width, "height": height}
    while "Screen capturing":
        # fps
        six += 1
        fps = 1 / (time.time() - t_)
        tick += time.time() - t_
        t_ = time.time()

        # grab frame
        img = np.array(sct.grab(monitor))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Display the picture
        cv2.imshow("crop", gray[y:y + h, x:x + w])

        # isolate green channel
        green = np.copy(img)
        green[:,:,0] = 0
        green[:,:,2] = 0
        ppg_green = ppg_green[1:] + [np.average(green[y:y + h, x:x + w])]
        times = times[1:] + [tick]

        if six%6 == 0:
            # detect local maxima
            peaks = signal.find_peaks(ppg_green, distance=5)[0]
            bpm = 60 * len(peaks) / (times[-1] - times[0])

            # plot ppg
            ax.cla()
            ax.plot(times, ppg_green, color='red')
            ax.plot([times[i] for i in peaks], [ppg_green[i] for i in peaks], '.', color='blue')
            ax.text(0.8, 0.9, f'BPM:{round(bpm)}', transform=ax.transAxes)
            ax.grid()
            ax.set_ylabel('Intensity')
            ax.set_xlabel('Time (s)')

            # show graph
            fig.canvas.draw()
            fig.canvas.tostring_rgb()
            graph = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            graph = graph.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            cv2.imshow('PPG', graph)

        # Press "q" to quit
        if cv2.waitKey(5) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
        elif cv2.waitKey(33) & 0xFF == ord('t'):
            r = cv2.selectROI("crop", img, False, False)
            x, y, w, h = r
            cv2.destroyWindow("crop")
