import cv2
import numpy as np

from cv2_image_utils import resize

def get_screws_contours(image: np.ndarray, debug: bool = False, interactive: bool = False) -> None:
    """
    Extracts and returns contours of screws from the input image by segmenting the background
    using HSV color space thresholds.

    Args:
        image (np.ndarray): The input image in BGR format (as read by OpenCV).
        debug (bool, optional): If True, shows intermediate masks for debugging. Defaults to False.
        interactive (bool, optional): If True, opens a window with interactive HSV sliders.
                                      If False, uses predefined HSV thresholds. Defaults to False.

    Raises:
        ValueError: If the input image is None or not a valid image array.
    """
    SCREW_ASPECT_RATIO_MIN = 1.2
    SCREW_ASPECT_RATIO_MAX = 3.0
    SCREW_MIN_CONTOUR_HEIGHT = 100
    SCALE = 0.5
    KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    DEFAULT_MIN_HUE = 90
    DEFAULT_MIN_SAT = 60
    DEFAULT_MIN_VAL = 50
    DEFAULT_MAX_HUE = 130
    DEFAULT_MAX_SAT = 255
    DEFAULT_MAX_VAL = 155

    DEFAULT_LOWER_HSV = np.array([DEFAULT_MIN_HUE, DEFAULT_MIN_SAT, DEFAULT_MIN_VAL])
    DEFAULT_UPPER_HSV = np.array([DEFAULT_MAX_HUE, DEFAULT_MAX_SAT, DEFAULT_MAX_VAL])

    def nothing(x: int) -> None:
        pass

    def _get_screws_contours(image_hsv: np.ndarray, lower_hsv: np.ndarray, upper_hsv: np.ndarray, debug: bool = False, interactive: bool = True):
        """
        Gets contours of screws based on HSV thresholds.
        Args:
            image_hsv (np.ndarray): The input image in HSV color space.
            lower_hsv (np.ndarray): The lower HSV threshold.
            upper_hsv (np.ndarray): The upper HSV threshold.
            debug (bool, optional): If True, shows intermediate masks for debugging. Defaults to False.
            interactive (bool, optional): If True, keeps OpenCV windows open for interaction. Defaults to True.
        
        Returns:
            List of contours that match the screw criteria.
        
        """

        mask_bg = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
        mask_objects = cv2.bitwise_not(mask_bg)
        mask_objects = cv2.morphologyEx(mask_objects, cv2.MORPH_OPEN, KERNEL)
        mask_objects = cv2.morphologyEx(mask_objects, cv2.MORPH_CLOSE, KERNEL)
        contours = cv2.findContours(mask_objects, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
       
        # Filter contours based on aspect ratio and height
        filtered_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w == 0 or h == 0 or h < SCREW_MIN_CONTOUR_HEIGHT:
                continue
            aspect_ratio = h / w
            if SCREW_ASPECT_RATIO_MIN <= aspect_ratio <= SCREW_ASPECT_RATIO_MAX:
                filtered_contours.append(contour)
        
        if debug:

            # display original image and original image with mask applied
            cv2.imshow("Original Image", resize(image, SCALE))

            ## apply mask to original image and set the background transparent
            masked_image = cv2.bitwise_and(image, image, mask=mask_objects)
            cv2.imshow("Masked Image", resize(masked_image, SCALE))

            # show filtered contours on a transparent background
            filtered_contours_image = np.zeros_like(image)
            for contour in filtered_contours:
                x, y, w, h = cv2.boundingRect(contour)
                screw_subimage = image[y:y + h, x:x + w]
                filtered_contours_image[y:y + h, x:x + w] = screw_subimage
                # cv2.drawContours(filtered_contours_image, [contour], -1, (0, 255, 0), 2)
            
            filtered_contours_image = cv2.bitwise_and(filtered_contours_image, filtered_contours_image, mask=mask_objects)
            cv2.imshow("Filtered Contours", resize(filtered_contours_image, SCALE))

            key = cv2.waitKey(1 if interactive else 0) & 0xFF
            if key == 27:  # ESC key to exit
                exit()
            
            if not interactive:
                cv2.destroyAllWindows()

        
        return filtered_contours


    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Input image is invalid or None.")

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if interactive:
        cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Controls", 400, 300)

        # Create HSV trackbars
        cv2.createTrackbar("H_min", "Controls", DEFAULT_MIN_HUE, 179, nothing)
        cv2.createTrackbar("S_min", "Controls", DEFAULT_MIN_SAT, 255, nothing)
        cv2.createTrackbar("V_min", "Controls", DEFAULT_MIN_VAL, 255, nothing)
        cv2.createTrackbar("H_max", "Controls", DEFAULT_MAX_HUE, 179, nothing)
        cv2.createTrackbar("S_max", "Controls", DEFAULT_MAX_SAT, 255, nothing)
        cv2.createTrackbar("V_max", "Controls", DEFAULT_MAX_VAL, 255, nothing)

        while True:
            h_min = cv2.getTrackbarPos("H_min", "Controls")
            s_min = cv2.getTrackbarPos("S_min", "Controls")
            v_min = cv2.getTrackbarPos("V_min", "Controls")
            h_max = cv2.getTrackbarPos("H_max", "Controls")
            s_max = cv2.getTrackbarPos("S_max", "Controls")
            v_max = cv2.getTrackbarPos("V_max", "Controls")

            lower_hsv = np.array([h_min, s_min, v_min])
            upper_hsv = np.array([h_max, s_max, v_max])

            _get_screws_contours(image_hsv, lower_hsv, upper_hsv, debug=debug, interactive=True)

    else:
        
        return _get_screws_contours(image_hsv, DEFAULT_LOWER_HSV, DEFAULT_UPPER_HSV, debug=debug, interactive=False)



if __name__ == "__main__":
    # Load image
    img = cv2.imread("./level31.jpg")
    if img is None:
        raise FileNotFoundError("Image file not found.")

    # Call the function in interactive mode
    # get_screws_contours(img, interactive=True)

    # Call the function in non-interactive mode
    screws_contours = get_screws_contours(img, interactive=False)

    for i, contour in enumerate(screws_contours):
        x, y, w, h = cv2.boundingRect(contour)
        roi_bgr = img[y:y + h, x:x + w]
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        shifted_contour = contour - [x, y]
        cv2.drawContours(roi_mask, [shifted_contour], -1, 255, -1)
        roi_rgba = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2BGRA)
        roi_rgba[:, :, 3] = roi_mask

        filename = f"Screw_{i}.png"
        cv2.imwrite(filename, roi_rgba)
        print(f"Saved: {filename}")
