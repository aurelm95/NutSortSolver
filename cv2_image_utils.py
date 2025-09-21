import numpy as np
import cv2

def resize(img: np.ndarray, scale: float) -> np.ndarray:
    """
    Resizes the input image by a given scale factor.

    Args:
        img (np.ndarray): The input image to be resized.
        scale (float): The scaling factor. Values > 1 will enlarge the image, values < 1 will reduce it.
    
    Returns:
        np.ndarray: The resized image.
    
    """

    return cv2.resize(img, (0, 0), fx=scale, fy=scale)