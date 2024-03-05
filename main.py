import cv2
import numpy as np


def undistort_image(image_path, camera_matrix, distortion_coeffs):
    """
    Undistorts an image using camera calibration parameters.

    Args:
        image_path (str): The path to the image file.
        camera_matrix (numpy.ndarray): The camera matrix.
        distortion_coeffs (numpy.ndarray): The distortion coefficients.

    Returns:
        numpy.ndarray: The undistorted image.

    Raises:
        Exception: If an error occurs while undistorting the image.

    """
    try:
        # Load the image
        image = cv2.imread(image_path)

        # Get the image size
        h, w = image.shape[:2]

        # Generate the new camera matrix from the distortion coefficients
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            camera_matrix, distortion_coeffs, (w, h), 1, (w, h)
        )

        # Undistort the image
        undistorted_image = cv2.undistort(
            image, camera_matrix, distortion_coeffs, None, new_camera_matrix
        )

        return undistorted_image
    except Exception as e:
        print(f"Error occurred while undistorting image: {str(e)}")
        return None


def undistort_fisheye_image(image_path, camera_matrix, distortion_coeffs):
    """
    Undistorts a fisheye image using camera calibration parameters.

    Args:
        image_path (str): The path to the image file.
        camera_matrix (numpy.ndarray): The camera matrix.
        distortion_coeffs (numpy.ndarray): The distortion coefficients.

    Returns:
        numpy.ndarray: The undistorted image.

    Raises:
        Exception: If an error occurs while undistorting the image.

    """
    try:
        # Load the image
        image = cv2.imread(image_path)

        # Get the image size
        h, w = image.shape[:2]

        # Generate the new camera matrix from the distortion coefficients
        new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            camera_matrix, distortion_coeffs, (w, h), np.eye(3)
        )

        # Undistort the image
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix,
            distortion_coeffs,
            np.eye(3),
            new_camera_matrix,
            (w, h),
            cv2.CV_16SC2,
        )
        undistorted_image = cv2.remap(
            image,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        return undistorted_image
    except Exception as e:
        print(f"Error occurred while undistorting fisheye image: {str(e)}")
        return None


# Example usage
image_path = "frame223.png"
camera_matrix = np.array(
    [
        [1171.10, 0, 959.97],
        [0, 1171.1, 546.6],
        [0, 0, 1],
    ]
)  # Replace with your camera matrix
# TODO: test with samples
distortion_coeffs = np.array(
    [
        0.34282590887394071,
        -0.043989126788329475,
        0.00025097091996533855,
        -0.00012177940476213143,
        # 0.0054380368453423701,
    ]
)  # Replace with your distortion coefficients

undistorted_image = undistort_fisheye_image(
    image_path, camera_matrix, distortion_coeffs
)

# Display the undistorted image
cv2.imshow("Undistorted Image", undistorted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
