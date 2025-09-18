import cv2
import numpy as np
from typing import List, Tuple


def get_nuts_subimages(image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """
    Extracts subimages corresponding to the screws (nuts) from the original image.

    Args:
        image (np.ndarray): The original image.

    Returns:
        List[Tuple[str, np.ndarray]]: A list of tuples containing labels and the corresponding subimages.
    """
    height, width, _ = image.shape

    # Hardcoded values to locate the screws relative to the image shape
    first_nut_height_percentage = 36.2
    middle_nuts_height_percentage = 17.4
    middle_nut_start_percentage = 27.8
    middle_nut_end_percentage = 73.9

    first_nut_height = int(height * first_nut_height_percentage) // 100
    middle_nut_height = int(height * middle_nuts_height_percentage) // 100
    middle_nut_start = int(width * middle_nut_start_percentage) // 100
    middle_nut_end = int(width * middle_nut_end_percentage) // 100

    subimages: List[Tuple[str, np.ndarray]] = []

    # First (top) nut
    first_nut = image[
        first_nut_height - middle_nut_height:first_nut_height,
        middle_nut_start:middle_nut_end
    ]
    subimages.append(("first nut", first_nut))

    # Middle nuts (3 of them)
    for k in range(3):
        y1 = first_nut_height + middle_nut_height * k
        y2 = first_nut_height + middle_nut_height * (k + 1)
        middle_nut = image[y1:y2, middle_nut_start:middle_nut_end]
        subimages.append((f"{k} middle nut", middle_nut))

    return subimages

def get_average_color(image: np.ndarray) -> np.ndarray:
    """
    Calculates the average color of a image.

    Args:
        image (np.ndarray): The image (nut region).

    Returns:
        np.ndarray: The average RGB color as a uint8 array of shape (3,).
    """
    average_color = image.mean(axis=(0, 1)).astype(np.uint8)
    return average_color

def get_nuts_colors_from_screw_image(image: np.ndarray, debug: bool = False) -> List[np.ndarray]:
    """
    Processes the input image to detect screws (nuts) and computes their average colors.

    Args:
        image (np.ndarray): The input image in BGR format.
        debug (bool): If True, displays intermediate results for debugging.

    Returns:
        List[np.ndarray]: List of average RGB colors for each detected nut subimage.
    """
    if image is None or not isinstance(image, np.ndarray):
        print("Error: Invalid image input.")
        return []

    subimages = get_nuts_subimages(image)
    detected_colors: List[np.ndarray] = []

    for label, subimg in subimages:
        avg_color = get_average_color(subimg)
        detected_colors.append(avg_color)

        if debug:
            print(f"{label} color={avg_color}")
            cv2.imshow(label, subimg)

            # Create and display a solid color image with the average color
            color_image = np.zeros_like(subimg, dtype=np.uint8) + avg_color
            cv2.imshow("average color", color_image)

            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                exit()

    if debug:
        cv2.destroyAllWindows()
    
    return detected_colors



if __name__ == "__main__":
    image_path = "./Screw_2.png"
    image = cv2.imread(image_path)

    colors = get_nuts_colors_from_screw_image(image, debug=True)
    print("\nDetected average colors:")
    for idx, color in enumerate(colors):
        print(f"Nut {idx}: Color (BGR) = {color}")

