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


# Example usage
image_path = "frame223.png"
camera_matrix = np.array(
    [
        [2152.8, 0, 971.3],
        [0, 2155.5, 605.9],
        [0, 0, 1],
    ]
)  # Replace with your camera matrix
#TODO: test with samples
distortion_coeffs = np.array(
    [
        0.32140973983357052, 0.03911369921165378, 0.0046883190539842362, 0.71061685361298299, 0
    ]
)  # Replace with your distortion coefficients

undistorted_image = undistort_image(image_path, camera_matrix, distortion_coeffs)

# Display the undistorted image
cv2.imshow("Undistorted Image", undistorted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
