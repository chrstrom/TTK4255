#
# This script uses example localization results to show
# what the figure should look like. You need to modify
# this script to work with your data.
#

import matplotlib.pyplot as plt
import numpy as np
from localize_camera import LocalizeCamera

def draw_frame(ax, T, scale):
    X0 = T @ np.array((0, 0, 0, 1))
    X1 = T @ np.array((1, 0, 0, 1))
    X2 = T @ np.array((0, 1, 0, 1))
    X3 = T @ np.array((0, 0, 1, 1))
    ax.plot([X0[0], X1[0]], [X0[2], X1[2]], [X0[1], X1[1]], color="#FF7F0E")
    ax.plot([X0[0], X2[0]], [X0[2], X2[2]], [X0[1], X2[1]], color="#2CA02C")
    ax.plot([X0[0], X3[0]], [X0[2], X3[2]], [X0[1], X3[1]], color="#1F77B4")


def draw_point_cloud(X, T_m2q, xlim, ylim, zlim, colors, marker_size, frame_size):
    ax = plt.axes(projection="3d")
    ax.set_box_aspect((1, 1, 1))
    if colors.max() > 1.1:
        colors = colors.copy() / 255
    ax.scatter(
        X[0, :], X[2, :], X[1, :], c=colors, marker=".", s=marker_size, depthshade=False
    )
    draw_frame(ax, np.linalg.inv(T_m2q), scale=frame_size)
    ax.grid(False)
    ax.set_xlim(xlim)
    ax.set_ylim(zlim)
    ax.set_zlim([ylim[1], ylim[0]])
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    plt.title("[Click, hold and drag with the mouse to rotate the view]")


if __name__ == "__main__":

    img = 8223
    query = f"../data/undistorted/IMG_{img}.jpg"

    scaling = 4.55 / 2.25 # 4.55 is the actual length of the doors, 2.25 is the unscaled point cloud distance

    X = np.loadtxt("../localization/X_ivan.out")
    desc = np.loadtxt("../localization/desc_ivan.out").astype("float32")
    model = (X, desc)

    run_monte_carlo = False
    localize = LocalizeCamera(query, model)

    if run_monte_carlo:
        #          nf, ncx, ncy
        sigmas1 = [50, 0.1, 0.1]
        sigmas2 = [0.1, 50, 0.1]
        sigmas3 = [0.1, 0.1, 50]

        sigmas = sigmas3
        COV = localize.run_monte_carlo(500, sigmas) 

        print("Sigmas:")
        print(f"  nf: {sigmas[0]}")
        print(f"  ncx: {sigmas[1]}")
        print(f"  ncy: {sigmas[2]}")

    else:
        _, J, T_m2q = localize.run()

        COV = np.linalg.inv(J.T @ J)
        
    STD = np.sqrt(np.diag(COV))
    STD[:3] *= 180 / np.pi
    STD[3:] *= (1000 * scaling)


    print(f"\nImage: {img}")
    print("Standard deviation for rpy in deg")
    print(STD[:3])
    print("Standard deviation for xyz in mm")
    print(STD[3:])

    colors = localize.get_colors()

    plt.figure("3D point cloud", figsize=(6, 6))
    draw_point_cloud(
        X,
        T_m2q,
        xlim=[-10, +10],
        ylim=[-10, +10],
        zlim=[0, +20],
        colors=colors,
        marker_size=5,
        frame_size=1,
    )
    plt.tight_layout()
    plt.show()
