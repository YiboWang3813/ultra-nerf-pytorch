import open3d as od
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.rcParams['figure.dpi'] = 300

def visualize_trajectory(trajectory: np.ndarray):
    origins = trajectory[::, :3, -1]
    rotation_matrixs = trajectory[::, :3, :3]
    things_to_draw = []
    # things_to_draw.append(od.geometry.TriangleMesh.create_coordinate_frame(0.1))
    for origin, rotation_matrix in zip(origins, rotation_matrixs):
        things_to_draw.append(od.geometry.TriangleMesh.create_coordinate_frame(0.001, origin).rotate(rotation_matrix, origin))
    od.visualization.draw_geometries(things_to_draw)

def plot_trajectories_phantom(trajectory_1: np.ndarray, trajectory_2: np.ndarray):
    origins_1 = trajectory_1[:, :3, -1] * 1e3
    origins_2 = trajectory_2[:, :3, -1] * 1e3

    f, (ax1, ax2) = plt.subplots(2, 1, sharex='col')
    ax1.set_aspect('equal')
    ax1.scatter(origins_1[:, 1], origins_1[:, 0], c='r', s=0.1, marker='o')
    ax1.scatter(origins_2[:, 1], origins_2[:, 0], c='b', s=0.1, marker='x')
    ax1.set_xlabel('y')
    ax1.set_ylim([504, 508])
    ax1.set_ylabel('x')
    ax1.legend(['Ground truth', 'Estimated'])    
    ax2.set_aspect('equal')
    ax2.scatter(origins_1[:, 1], origins_1[:, 2], c='r', s=0.1, marker='o')
    ax2.scatter(origins_2[:, 1], origins_2[:, 2], c='b', s=0.1, marker='x')
    ax2.set_ylabel('z')
    ax2.set_ylim([348, 352])
    f.suptitle('Estimated trajectory against the ground truth. Unit is mm.')
    f.savefig('trajectories_phantom.png')

def plot_trajectories_synthetic(trajectory_1: np.ndarray, trajectory_2: np.ndarray):
    origins_1 = trajectory_1[30:-30, :3, -1] * 1e3
    origins_2 = trajectory_2[30:-30, :3, -1] * 1e3

    f, (ax1, ax2) = plt.subplots(2, 1, sharex='col')
    ax1.set_aspect('equal')
    ax1.scatter(origins_1[:, 2], origins_1[:, 0], c='r', s=0.1, marker='o')
    ax1.scatter(origins_2[:, 2], origins_2[:, 0], c='b', s=0.1, marker='x')
    ax1.set_xlabel('z')
    ax1.set_ylim([60, 80])
    ax1.set_ylabel('x')
    ax1.legend(['Ground truth', 'Estimated'])    
    ax2.set_aspect('equal')
    ax2.scatter(origins_1[:, 2], origins_1[:, 1], c='r', s=0.1, marker='o')
    ax2.scatter(origins_2[:, 2], origins_2[:, 1], c='b', s=0.1, marker='x')
    ax2.set_ylabel('y')
    ax2.set_ylim([-120, -105])
    f.suptitle('Estimated trajectory against the ground truth. Unit is mm.')
    f.savefig('trajectory_synthetic.png')

def calculate_errors(trajectory_true, trajectory_estimated):  
    trajectory_true = trajectory_true[:, :3, -1]
    trajectory_estimated = trajectory_estimated[:, :3, -1]
    total_drift = (trajectory_estimated[-1] - trajectory_estimated[0]) - (trajectory_true[-1] - trajectory_true[0])
    d1 = trajectory_true[1:] - trajectory_true[:-1]
    d2 = trajectory_estimated[1:] - trajectory_estimated[:-1]
    absolute_error = np.abs(d2 - d1)
    mean_absolute_error = np.mean(absolute_error, axis=0)
    mean_relative_error = np.mean( np.linalg.norm(d2 - d1, axis=-1) / np.linalg.norm(d1, axis=-1) ) 
    return total_drift, mean_absolute_error, mean_relative_error

def main():
    trajectory_true = np.load('/home/mirmi/Documents/UltraAssistant/refactor/trajectory_true_1.npy')
    trajectory_estimated = np.load('/home/mirmi/Documents/UltraAssistant/refactor/trajectory_estimated_1.npy')

    # visualize_trajectory(np.concatenate((trajectory_true, trajectory_estimated)))
    # plot_trajectories_synthetic(trajectory_true, trajectory_estimated)
    plot_trajectories_phantom(trajectory_true, trajectory_estimated)
    # print(calculate_errors(trajectory_true, trajectory_estimated))
if __name__ == '__main__':
    main()