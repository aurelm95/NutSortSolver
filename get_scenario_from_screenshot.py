import cv2
import numpy as np

import get_screws_from_image
import nut_is_empty
import screw_color_detector


if __name__ == "__main__":
    # Load image
    img = cv2.imread("./level31.jpg")
    if img is None:
        raise FileNotFoundError("Image file not found.")

    screws_contours = get_screws_from_image.get_screws_contours(img, interactive=False)

    for i, contour in enumerate(screws_contours):
        x, y, w, h = cv2.boundingRect(contour)
        roi_bgr = img[y:y + h, x:x + w]
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        shifted_contour = contour - [x, y]
        cv2.drawContours(roi_mask, [shifted_contour], -1, 255, -1)
        roi_rgba = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2BGRA)
        roi_rgba[:, :, 3] = roi_mask

        # Check if the nut is empty
        if nut_is_empty.screw_is_empty(roi_rgba,debug=False):
            print(f"Screw {i} is empty, skipping color detection.")
            cv2.imshow(f"Screw_{i} is empty", roi_rgba)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                exit()
            continue

        # get the average color of the nut
        colors = screw_color_detector.get_nuts_colors_from_screw_image(roi_bgr, debug=True)
        print(f"Screw {i} detected colors: {colors}")
        cv2.imshow(f"Screw_{i}", roi_rgba)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            exit()

