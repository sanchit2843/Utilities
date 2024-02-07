import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


## TODO: this code expects images to be undistorted and rectified as of now, add option to undistort and rectify the images
left = cv2.imread(
    "/home/sanchit/Workspace/clutterbot/habitat_small_baseline/bot_data/left/camera_0_frame_1707214782.0682263.png"
)
right = cv2.imread(
    "/home/sanchit/Workspace/clutterbot/habitat_small_baseline/bot_data/right/camera_1_frame_1707214782.0682263.png"
)

## draw epipolar lines on the left image


def compute_fundamental_matrix(K1, K2, R, t):
    E = np.cross(t, R)
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    return F


# Function to draw epipolar lines on the second image
# def draw_epipolar_lines(img1, img2, F, pts1):
#     h, w = img1.shape[:2]
#     lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
#     lines2 = lines2.reshape(-1, 3)

#     img2_with_lines = img2.copy()
#     for line in lines2:
#         color = tuple(np.random.randint(0, 255, 3).tolist())
#         x0, y0, x1, y1 = map(
#             int, [0, -line[2] / line[1], w, -(line[2] + line[0] * w) / line[1]]
#         )
#         img2_with_lines = cv2.line(img2_with_lines, (x0, y0), (x1, y1), color, 1)
#     cv2.circle(img1, (pts1[0], pts1[1]), 3, (0, 0, 255), -1)
#     plt.imshow(np.hstack([img1[:, :, ::-1], img2_with_lines[:, :, ::-1]]))
#     plt.show()
#     return img2_with_lines


def draw_epipolar_lines(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        new_K_left, new_K_right, F = param
        pts1 = np.array([[x, y]])

        # Draw epipolar lines on the right image
        img2_with_lines = draw_epipolar_lines_on_right(right, F, pts1)
        img1 = left.copy()
        cv2.circle(img1, (pts1[0, 0], pts1[0, 1]), 3, (0, 0, 255), -1)
        cv2.imshow("images", np.hstack([img1, img2_with_lines]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Display the images side by side


def draw_epipolar_lines_on_right(img2, F, pts1_undistorted):
    h, w = img2.shape[:2]
    lines2 = cv2.computeCorrespondEpilines(pts1_undistorted.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)

    img2_with_lines = img2.copy()
    for line in lines2:
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0, x1, y1 = map(
            int, [0, -line[2] / line[1], w, -(line[2] + line[0] * w) / line[1]]
        )
        img2_with_lines = cv2.line(img2_with_lines, (x0, y0), (x1, y1), color, 2)

    return img2_with_lines


if __name__ == "__main__":
    K_left = np.asarray(
        [
            [415.81468891, 0.0, 414.94026509],
            [0.0, 416.00958101, 346.26848797],
            [0.0, 0.0, 1.0],
        ]
    ).reshape(3, 3)
    D_left = np.asarray([-0.05114882, 0.03373574, -0.03893983, 0.01327103]).reshape(
        4, 1
    )
    new_K_left = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K_left, D_left, (852, 640), np.eye(3), balance=1.0
    )
    map1_left, map2_left = cv2.fisheye.initUndistortRectifyMap(
        K_left, D_left, np.eye(3), new_K_left, (852, 640), cv2.CV_16SC2
    )
    K_right = np.asarray(
        [
            [410.71753775, 0.0, 399.23367331],
            [0.0, 410.8432021, 297.35806696],
            [0.0, 0.0, 1.0],
        ]
    ).reshape(3, 3)
    D_right = np.asarray([-0.04996972, 0.03487752, -0.037981, 0.01261172]).reshape(4, 1)
    new_K_right = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K_right, D_right, (852, 640), np.eye(3), balance=1.0
    )
    map1_right, map2_right = cv2.fisheye.initUndistortRectifyMap(
        K_right, D_right, np.eye(3), new_K_right, (852, 640), cv2.CV_16SC2
    )

    T = np.array([-0.12621615, -0.0043145, -0.00415989])
    R = np.array(
        [
            [0.9986176, 0.0297274, 0.0433487],
            [-0.0317493, 0.9984031, 0.0467245],
            [-0.0418905, -0.0480362, 0.9979668],
        ]
    )
    F = compute_fundamental_matrix(new_K_left, new_K_right, R, T)
    ## mouse callback for the left image to draw epipolar lines on the right image
    while True:
        cv2.namedWindow("Left Image")
        cv2.setMouseCallback("Left Image", draw_epipolar_lines, (K_left, K_right, F))

        # Display the left image
        cv2.imshow("Left Image", left)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# draw_epipolar_lines(left, right, F, np.array([248, 311]))
