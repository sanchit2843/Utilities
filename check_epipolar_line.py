import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


## TODO: this code expects images to be undistorted and rectified as of now, add option to undistort and rectify the images
## draw epipolar lines on the left image


def draw_epipolar_lines(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        left, right = params
        pts1 = np.array([[x, y]])

        # Draw epipolar lines on the right image
        img2_with_lines = draw_epipolar_lines_on_right(right, pts1)
        img1 = left.copy()
        cv2.circle(img1, (pts1[0, 0], pts1[0, 1]), 3, (0, 0, 255), -1)
        cv2.imshow("images", np.hstack([img1, img2_with_lines]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Display the images side by side


def draw_epipolar_lines_on_right(img2, pts1_undistorted):

    # draw epipolar lines in rectified image, this will be along the same row as the point in the left image
    img2_with_lines = img2.copy()
    for pt in pts1_undistorted:
        cv2.line(
            img2_with_lines,
            (0, pt[1]),
            (img2_with_lines.shape[1], pt[1]),
            (0, 255, 0),
            1,
        )
    return img2_with_lines

    # h, w = img2.shape[:2]
    # lines2 = cv2.computeCorrespondEpilines(pts1_undistorted.reshape(-1, 1, 2), 1, F)
    # lines2 = lines2.reshape(-1, 3)

    # img2_with_lines = img2.copy()
    # for line in lines2:
    #     color = tuple(np.random.randint(0, 255, 3).tolist())
    #     x0, y0, x1, y1 = map(
    #         int, [0, -line[2] / line[1], w, -(line[2] + line[0] * w) / line[1]]
    #     )
    #     img2_with_lines = cv2.line(img2_with_lines, (x0, y0), (x1, y1), color, 2)

    # return img2_with_lines


def rectify_images(
    left_image, right_image, K_left, D_left, K_right, D_right, R, T, image_size
):
    left_R, right_R, left_P, right_P, Q = cv2.fisheye.stereoRectify(
        K_left,
        D_left,
        K_right,
        D_right,
        image_size[::-1],
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        fov_scale=1,
        balance=0.0,
    )

    left_map_x, left_map_y = cv2.fisheye.initUndistortRectifyMap(
        K=K_left,
        D=D_left,
        R=left_R,
        P=left_P,
        size=image_size[::-1],
        m1type=cv2.CV_32FC1,
    )

    right_map_x, right_map_y = cv2.fisheye.initUndistortRectifyMap(
        K=K_right,
        D=D_right,
        R=right_R,
        P=right_P,
        size=image_size[::-1],
        m1type=cv2.CV_32FC1,
    )
    left_rect = cv2.remap(left_image, left_map_x, left_map_y, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_image, right_map_x, right_map_y, cv2.INTER_LINEAR)
    return left_rect, right_rect


if __name__ == "__main__":
    left = cv2.imread(
    )
    right = cv2.imread(
    )

    K_left = np.asarray(
        [
            [407.84422822, 0.0, 415.60713165],
            [0.0, 407.69391138, 344.0302208],
            [0.0, 0.0, 1.0],
        ]
    ).reshape(3, 3)
    d_left = np.asarray([-0.02332444, -0.01839451, 0.00901099, -0.00230693]).reshape(
        4, 1
    )

    K_right = np.asarray(
        [
            [407.03636678, 0.0, 399.33632963],
            [0.0, 406.55630645, 295.28552185],
            [0.0, 0.0, 1.0],
        ]
    ).reshape(3, 3)
    d_right = np.asarray([-0.02898533, -0.00643433, -0.00255016, 0.0013462]).reshape(
        4, 1
    )
    R = np.asarray(
        [
            [0.9979929, 0.0242555, 0.0584955],
            [-0.0270840, 0.9984774, 0.0480562],
            [
                -0.0572408,
                -0.0495440,
                0.9971303,
            ],
        ]
    ).reshape(3, 3)
    T = np.asarray([-0.11881959, -0.00348413, -0.00431191])

    ## invert in this case
    R_t = np.hstack([R, T.reshape(3, 1)])
    R_t = np.vstack([R_t, [0, 0, 0, 1]])

    R_t_ = np.linalg.inv(R_t)
    R = R_t_[:3, :3]
    T = R_t_[:3, 3]

    image_size = [640, 852]
    left_rect, right_rect = rectify_images(
        left, right, K_left, d_left, K_right, d_right, R, T, image_size
    )
    plt.imshow(np.hstack([left_rect[:, :, ::-1], right_rect[:, :, ::-1]]))
    plt.show()
    ## mouse callback for the left image to draw epipolar lines on the right image
    while True:
        cv2.namedWindow("Left Image")
        cv2.setMouseCallback("Left Image", draw_epipolar_lines, (left_rect, right_rect))

        # Display the left image
        cv2.imshow("Left Image", left_rect)
        if cv2.waitKey(0) == ord("q"):
            cv2.destroyAllWindows()
            break

# draw_epipolar_lines(left, right, F, np.array([248, 311]))
