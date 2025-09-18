import cv2
import numpy as np
from typing import Optional


def get_visible_width(image: np.ndarray, debug: bool = False) -> int:
    """
    Calculates the visible (non-transparent) width of an RGBA image.

    Args:
        image (np.ndarray): Image as a NumPy array with 4 channels (RGBA).
        debug (bool): If True, shows debugging windows with the image and cropped result.

    Returns:
        int: Width (in pixels) of the visible region (where alpha > 0).
             Returns 0 if the image is fully transparent.
    """
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Invalid image input.")
    
    if image.shape[2] != 4:
        raise ValueError("Image does not have an alpha channel (not RGBA).")

    # Crop bottom 33% pixels (remove screw base or background)
    image_height, image_width = image.shape[:2]
    cropped_image = image[:int(image_height * 0.67), :]

    if debug:
        cv2.imshow("Original Screw", image)
        cv2.imshow("Cropped Screw without base", cropped_image)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            exit()
        cv2.destroyAllWindows()

    # Extract alpha channel
    alpha_channel = cropped_image[:, :, 3]


    # To find the width of the visible part of the image, we want to find the leftmost and rightmost columns that are visible (not fully transparent).
    # A pixel with alpha = 0 is fully transparent. 
    # Hence, to find the columns that are visible, we look for columns with some pixel having alpha > 0.
    columns_have_visible_pixels = np.any(alpha_channel > 0, axis=0)
    indices_of_visible_columns = np.where(columns_have_visible_pixels)[0]

    # If no visible pixels, return width = 0
    if len(indices_of_visible_columns) == 0:
        return 0

    # Compute visible width
    x_min = indices_of_visible_columns[0]
    x_max = indices_of_visible_columns[-1]
    return x_max - x_min + 1

def screw_is_empty(image: np.ndarray, threshold: Optional[int] = 100, debug: bool = False) -> bool:
    """
    Determines if the screw is empty based on the visible width.

    Args:
        image (np.ndarray): Image as a NumPy array with 4 channels (RGBA).
        threshold (Optional[int]): Minimum visible width to consider the screw as non-empty.
        debug (bool): If True, shows debugging windows with the image and cropped result.

    Returns:
        bool: True if the screw is empty (visible width < threshold), False otherwise.
    """
    visible_width = get_visible_width(image, debug=debug)
    if debug:
        print(f"Visible width: {visible_width}, Threshold: {threshold}")
    return visible_width < threshold

if __name__ == '__main__':
    from glob import glob

    # Find all screw images (PNG with alpha channel)
    screw_image_paths = glob("./Screw_*.png")

    for image_path in screw_image_paths:
        # Load image with alpha channel
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        image_height, image_width = image.shape[:2]

        if image is None:
            print(f"Error: could not load image: {image_path}")
            continue

        try:
            # visible_width = get_visible_width(image, debug=False)
            # print(f"{image_path=}, {image_height=}, {image_width=}, {visible_width=}")
            is_empty = screw_is_empty(image, threshold=100, debug=False)
            print(f"{image_path=}, {image_height=}, {image_width=}, {is_empty=}")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
