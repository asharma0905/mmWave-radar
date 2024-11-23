# from matplotlib.pylab import randn
import glob
import os
import numpy as np
import csv
from sklearn.cluster import DBSCAN
import cv2
import pandas as pd

parent_dir = 'C:/Users/Ashwin Adarsh/Desktop/TestRad/fall_new_fall2.csv/data/'
sub_dirs=['Fall', 'Walk']
extract_path = 'C:/Users/Ashwin Adarsh/Desktop/TestRad/fall_new_fall2.csv/Pre-Processed/'

class KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)


    def predict(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        x, y = int(predicted[0]), int(predicted[1])
        return x, y

kf = {}
fm = {}

def pre_process_data(path):
    with open(path) as csvfile:
        data_p1 = np.array(list(csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)))

    data_p1[:,1] -= np.amin(data_p1[:,1]) 
    data = data_p1

    n_frames = int(np.amax(data[:,1]))

    pipeline_1 = np.zeros(shape=(1, 9))
    pipeline_2 = np.zeros(shape=(1, 10))

    def euclid_weighted(p1, p2):
        x1,y1,z1 = p1
        x2,y2,z2 = p2
        dist = (x2-x1)**2 + (y2-y1)**2 + 0.25*(z2-z1)**2
        return np.sqrt(dist)


    for i in range(n_frames):
        try:
            data_s = data[data[:, 1] == i]
            model = DBSCAN(eps=0.0625, min_samples=15)
            cluster = model.fit(data_s[:,3:6])
            pred = cluster.labels_
            frame_intensity = data_s[:, -1]

            weight = ((frame_intensity - np.amin(frame_intensity)) / (np.amax(frame_intensity) - np.amin(frame_intensity)))

            # print(len(set(pred)), set(pred))
            data_s = np.hstack((data_s, (pred*weight).reshape(-1, 1)))
            data_filt = data_s[data_s[:, -1] > 0]
            pipeline_1 = np.concatenate((pipeline_1, data_filt), axis=0)
        except:
            pass

    pipeline_1 = pipeline_1[1:,:]

    for i in range(n_frames):
        try:
            data_filt = pipeline_1[pipeline_1[:,1] == i]
            model = DBSCAN(eps=2, min_samples=20)
            cluster = model.fit(data_filt[:,[3,4,5]])
            pred = cluster.labels_

            data_filt = np.hstack((data_filt, (pred).reshape(-1, 1)))

            data_filt = data_filt[data_filt[:, -1] > -1]
            pipeline_2 = np.concatenate((pipeline_2, data_filt), axis=0)
        except:
            pass

    pipeline_2 = pipeline_2[1:,:]

    for i in range(n_frames):
        try:
            frame = pipeline_2[pipeline_2[:,1] == i]
            if frame.shape[0] > 0:
                kf[i] = KalmanFilter()
            for j in range(frame.shape[0]):
                fm[i] = kf[i].predict(frame[j,3], frame[j,4])
        except:
            pass
    
    return pipeline_2

def save_pre_process_data(data, path):
    headers = ['elapsed_time','frame_number','nPoint','x','y','z','velocity','intensity', 'xy_weight', 'xz_weight']
    df = pd.DataFrame(data)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, header=headers, index=False)


def parse_RF_files(parent_dir, sub_dirs, file_ext='*.csv'):
    print(sub_dirs)
    labels = []

    for sub_dir in sub_dirs:
        labels.append(sub_dir)
        files=sorted(glob.glob(os.path.join(parent_dir,sub_dir, file_ext)))
        # print(files)
        for fn in files:
            x = fn.replace('\\', "/").split('/')
            pre_processed_data = pre_process_data(fn)
            save_path = extract_path+sub_dir+'/'+x[-1]
            save_pre_process_data(pre_processed_data, save_path)

parse_RF_files(parent_dir, sub_dirs)
