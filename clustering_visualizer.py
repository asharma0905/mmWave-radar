# from matplotlib.pylab import randn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import csv
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
from visual_helper import create_skeleton, plot_2D_box, plot_3D_box
import cv2
import numpy as np

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
with open('C:/Users/Ashwin Adarsh/Desktop/TestRad/fall_new_fall2.csv/data/Fall/r_1.csv') as csvfile:
    data_p1 = np.array(list(csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)))

data_p1[:,1] -= np.amin(data_p1[:,1]) 
data = data_p1

max_time = np.amax(data[:,0])
n_frames = int(np.amax(data[:,1]))

print(n_frames)

x_max, x_min = max(data[:,3]), min(data[:,3])
y_max, y_min = max(data[:,4]), min(data[:,4])
z_max, z_min = max(data[:,5]), min(data[:,5])

pipeline_1 = np.zeros(shape=(1, 9))
pipeline_2 = np.zeros(shape=(1, 10))

def euclid_weighted(p1, p2):
    x1,y1,z1 = p1
    x2,y2,z2 = p2
    dist = (x2-x1)**2 + (y2-y1)**2 + 0.25*(z2-z1)**2
    return np.sqrt(dist)


for i in range(n_frames):
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

pipeline_1 = pipeline_1[1:,:]

for i in range(n_frames):
    try:
        data_filt = pipeline_1[pipeline_1[:,1] == i]
        model = DBSCAN(eps=2, min_samples=20)
        cluster = model.fit(data_filt[:,[3,4,5]])
        pred = cluster.labels_

        occurrences = Counter(pred)
        # Find the most common value
        most_common_value = occurrences.most_common(1)[0][0]
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

# pred = []
# kff = KalmanFilter()

# for i in fm.values():
#     pred = kff.predict(i[0], i[1])

pred = np.array(pred)

fig2D = plt.figure(figsize = (10,6))
xy2D = fig2D.add_subplot(1,2,1)
xz2D = fig2D.add_subplot(1,2,2)

#TODO: How to find human???
#Clustering ki MKB
def update2D(i):
    frame = data[data[:, 1] == i]
    # model = DBSCAN(eps=0.04, min_samples=4)
    # cluster = model.fit(data_s[:,3:6])
    # pred = cluster.labels_
    # frame_intensity = data_s[:, -1]

    # weight = ((frame_intensity - np.amin(frame_intensity)) / (np.amax(frame_intensity) - np.amin(frame_intensity)))

    xy2D.clear()
    xz2D.clear()

    xy2D.set_xlim(x_min, x_max)
    xy2D.set_ylim(y_min, y_max)
    xz2D.set_xlim(x_min, x_max)
    xz2D.set_ylim(z_min, z_max)

    xy2D.scatter(frame[:,3], frame[:,4], c=frame[:,-1],s=5)
    xz2D.scatter(frame[:,3], frame[:,5], c=frame[:,-1],s=5)

ani2D = anim.FuncAnimation(fig2D, update2D, frames=n_frames, interval=200)
plt.show()

fig2D_cluster = plt.figure(figsize = (10,6))
xy2D = fig2D_cluster.add_subplot(1,2,1)
xz2D = fig2D_cluster.add_subplot(1,2,2)

#TODO: How to find human???
#Clustering ki MKB
def update2D_cluster(i):
    try:
        data_s = pipeline_1[pipeline_1[:,1] == i]
        data_filt = pipeline_2[pipeline_2[:,1] == i]

        xy2D.clear()
        xz2D.clear()

        xy2D.set_xlim(x_min, x_max)
        xy2D.set_ylim(y_min, y_max)
        xz2D.set_xlim(x_min, x_max)
        xz2D.set_ylim(z_min, z_max)

        xy2D.scatter(data_s[:,3], data_s[:,4], c=data_s[:,-1], s=10)
        # xz2D.scatter(fm[i][0], fm[i][1], c="black", s=10)
        xz2D.scatter(data_filt[:,3], data_filt[:,5], c=data_filt[:,-1], s=10)

        border_x_max = max(data_filt[:,3])
        border_y_max = max(data_filt[:,4])
        border_z_max = max(data_filt[:,5])
        border_x_min = min(data_filt[:,3])
        border_y_min = min(data_filt[:,4])
        border_z_min = min(data_filt[:,5])
        plot_2D_box(border_x_min, border_z_min, border_x_max-border_x_min, border_z_max-border_z_min,xz2D)
    except:
        pass
    # fig2D.savefig("C:/Users/Ashwin Adarsh/Desktop/TestRad/fall_new_fall2.csv/fall_new_fall2.csv/data/temp_%04d.jpg" % i)

ani2D = anim.FuncAnimation(fig2D_cluster, update2D_cluster, frames=n_frames, interval=20)
plt.show()

fig3D = plt.figure(figsize = (10,6))
xyz3D_noise = fig3D.add_subplot(1,2,1, projection='3d')
xyz3D_filtered = fig3D.add_subplot(1,2,2, projection='3d')

#TODO: Check on removing noise
def update3D(i):
    try:
        data_s = data[data[:, 1] == i]
        data_filt = pipeline_2[pipeline_2[:,1] == i]
        
# -1] == most_common_value]
        # print(data_filt.shape)
    
        xyz3D_noise.clear()
        xyz3D_filtered.clear()

        xyz3D_noise.set_xlim3d(x_min, x_max)
        xyz3D_noise.set_ylim3d(y_min, y_max)
        xyz3D_noise.set_zlim3d(z_min, z_max)
        xyz3D_filtered.set_xlim3d(x_min, x_max)
        xyz3D_filtered.set_ylim3d(y_min, y_max)
        xyz3D_filtered.set_zlim3d(z_min, z_max)

        border_x_max = max(data_filt[:,3])
        border_y_max = max(data_filt[:,4])
        border_z_max = max(data_filt[:,5])
        border_x_min = min(data_filt[:,3])
        border_y_min = min(data_filt[:,4])
        border_z_min = min(data_filt[:,5])


        height=1.5
        widthX = widthY = 0.5
        # if(border_y_max-border_y_min < 1):
        #     widthY = border_y_max-border_y_min
        # else:
        #     border_y_min = 0.5
        # if(border_x_max-border_x_min < 1):
        #     widthX = border_y_max-border_y_min
        # else:
        #     widthX = 0.5
        #     border_x_min = 0.5

        # print(height, widthY, widthX)
        # create_skeleton(border_x_max,border_y_max,border_z_min, height, xyz3D_filtered)

        plot_3D_box(fm[i][0], fm[i][1], border_z_min, widthX, height, widthY, xyz3D_filtered)


        sc_noise = xyz3D_noise.scatter(data_s[:,3], data_s[:,4], data_s[:,5], s=10)
        sc_filt = xyz3D_filtered.scatter(data_filt[:,3], data_filt[:,4], data_filt[:,5], s=10)
        xyz3D_filtered.scatter(fm[i][0], fm[i][1], s=10)

        sc_filt.set_array(data_filt[:,-1])

        if i == n_frames:
            plt.close('all')
    except:
        pass


ani3D = anim.FuncAnimation(fig3D, update3D, frames=n_frames, interval=20)
plt.show()