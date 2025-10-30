import numpy as np
import matplotlib.pyplot as plt

def plot_points(points, filename='robot_path.png'):
    points = np.array(points) # keep 1/3 of the points for clarity
    points = points[::3]  # downsample to 1/3
    plt.figure()
    plt.plot(points[:, 0], points[:, 1], marker='o')
    plt.title('Robot Path')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.axis('equal')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

# Sum the total distance traveled
def total_distance(points):
    points = np.array(points)[:, :2]  # only x,y
    diffs = np.diff(points, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    return np.sum(dists)

def total_rotation(yaws):
    yaws = np.array(yaws)
    diffs = np.diff(yaws)
    diffs = (diffs + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
    return np.sum(np.abs(diffs))


    
if __name__ == '__main__':
    # Load points from npy file
    points = np.load('robot_path.npy')
    yaws = points[:, 2]  # extract yaws
    plot_points(points)
    print(f'Total distance traveled: {total_distance(points):.2f} m')
    print(f'Total rotation (radians): {total_rotation(yaws):.2f} rad')