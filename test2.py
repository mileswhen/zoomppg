import numpy as np
import warnings; warnings.filterwarnings("ignore")
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mplib
import scipy.signal as signal
from statistics import mean, stdev
import time

cap = cv2.VideoCapture(0)
x, y, w, h = 540, 180, 100, 100

# initialize ppg data
ten = 0
hcount = 180
tick = 0
ppg_green = [1] * hcount
times = [0] * hcount
t_ = time.time()

# mplib graph
mplib.use('TkAgg')
fig = plt.figure()
ax = fig.add_subplot(211)
ax1 = fig.add_subplot(212)
plt.tight_layout()

while(True):
    # fps
    tick += time.time() - t_
    fps = 1 / (time.time() - t_)
    ten += 1
    t_ = time.time()

    # frame capture
    ret, frame = cap.read()

    # isolate green channel
    green = frame
    green[:,:,0] = 0
    green[:,:,2] = 0
    ppg_green = ppg_green[1:] + [np.average(green[y:y + h, x:x + w])]
    times = times[1:] + [tick]

    if ten%10 == 0:
        # detect local maxima
        peaks = signal.find_peaks(ppg_green, distance=4, width=(1, None))[0]
        peaks_times = [times[i] for i in peaks]
        peaks = [ppg_green[i] for i in peaks]
        IBIs = []
        for i in range(len(peaks)-1):
            IBIs.append(peaks_times[i+1] - peaks_times[i])
        IBIs = [i for i in IBIs if (0.25 < i < 2.5)]
        if IBIs != []:
            bpm = 60 / mean(IBIs)
        else:
            bpm = 0

        # filter and compute power spectra
        b, a = signal.butter(N=16, Wn=[0.7, 3.5], fs=fps, btype='bandpass')
        filtered1 = signal.lfilter(b, a, ppg_green)
        filtered = np.copy(ppg_green)
        freq, pxx = signal.welch(filtered1, round(fps,1), nperseg=hcount)
        maxfreq = freq[np.where(pxx == pxx.max())]

        # plot ppg
        ax.cla()
        ax.plot(times, [(i-mean(ppg_green))/stdev(ppg_green) for i in ppg_green], color='darkblue')
        ax.plot(peaks_times, [0]*len(peaks_times), '|', color='darkblue')
        ax.text(0.87, 0.1, f'BPM: {round(bpm)}', transform=ax.transAxes)
        ax.set_ylabel('Intensity')
        ax.set_xlabel('Time (s)')

        # plot ps
        ax1.cla()
        ax1.plot(freq, pxx, color='tab:orange')
        ax1.axvline(maxfreq, color='black')
        ax1.text(0.78, 0.1, f'spectral BPM: {int(maxfreq[0]*60)}', transform=ax1.transAxes)
        ax1.set_ylabel('Density')
        ax1.set_xlabel('Freq (Hz)')

        # graph_all
        fig.canvas.draw()
        fig.canvas.tostring_rgb()
        graph = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        graph = graph.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        cv2.imshow('PPG', graph)

    if cv2.waitKey(33) & 0xFF == ord('q'):
        break
    
    elif cv2.waitKey(33) & 0xFF == ord('t'):
        r = cv2.selectROI("crop", green, False, False)
        x, y, w, h = r
        cv2.destroyWindow("crop")

# # When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

