import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_skeleton(x, y, z, height, ax):
    # Define the joints of the skeleton
    joints = {
        'head': np.array([x, y, z + height]),
        'torso_top': np.array([x, y, z + height * 0.8]),
        'torso_bottom': np.array([x, y, z + height * 0.2]),
        'left_arm_top': np.array([x - height * 0.3, y, z + height * 0.7]),
        'right_arm_top': np.array([x + height * 0.3, y, z + height * 0.7]),
        'left_leg_bottom': np.array([x - height * 0.15, y, z]),
        'right_leg_bottom': np.array([x + height * 0.15, y, z]),
    }

    # Define connections between joints
    connections = [
        ('head', 'torso_top'),
        ('torso_top', 'torso_bottom'),
        ('torso_top', 'left_arm_top'),
        ('torso_top', 'right_arm_top'),
        ('torso_bottom', 'left_leg_bottom'),
        ('torso_bottom', 'right_leg_bottom')
    ]

    # Plot connections
    for j1, j2 in connections:
        ax.plot3D([joints[j1][0], joints[j2][0]], [joints[j1][1], joints[j2][1]], [joints[j1][2], joints[j2][2]], 'b-')

    # Plot joints as dots
    for joint in joints:
        if joint == 'head':
            ax.scatter(joints[joint][0], joints[joint][1], joints[joint][2], color='b', marker='o', s=100)
        else:
            ax.scatter(joints[joint][0], joints[joint][1], joints[joint][2], color='b', marker='o')

def plot_3D_box(x, y, z, width, height, depth, ax):
    # Define box vertices
    vertices = np.array([[x, y, z],
                        [x + width, y, z],
                        [x + width, y , z+height],
                        [x, y , z+height],
                        [x, y + depth, z ],
                        [x + width, y + depth, z ],
                        [x + width, y + depth, z + height],
                        [x, y + depth, z + height]])

    # Define edges of the box
    edges = [[0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]]

    # Plot box edges
    for edge in edges:
        ax.plot3D(*vertices[edge].T, color='blue')

def plot_2D_box(x, y, width, height, ax):
    # Define box vertices
    vertices = np.array([[x, y],
                        [x + width, y],
                        [x, y + height],
                        [x + width, y + height ]])

    # Define edges of the box
    edges = [[0, 1], [1, 3], [2, 3], [2, 0]]

    # Plot box edges
    for edge in edges:
        ax.plot(*vertices[edge].T, color='blue')
