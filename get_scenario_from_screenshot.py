import cv2
import numpy as np

import get_screws_from_image
import nut_is_empty
import screw_color_detector
import cv2_image_utils


if __name__ == "__main__":
    # Load image
    img = cv2.imread("./level31.jpg")
    if img is None:
        raise FileNotFoundError("Image file not found.")

    screws_contours = get_screws_from_image.get_screws_contours(img, debug=False, interactive=False)

    SCREWS_COLORS=[]
    # create a black image for debugging purposes with the same size as img
    black_image = np.zeros_like(img)

    for i, screw_contour in enumerate(screws_contours):
        x, y, w, h = cv2.boundingRect(screw_contour)
        roi_bgr = img[y:y + h, x:x + w]
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        shifted_contour = screw_contour - [x, y]
        cv2.drawContours(roi_mask, [shifted_contour], -1, 255, -1)
        roi_rgba = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2BGRA)
        roi_rgba[:, :, 3] = roi_mask

        # Check if the nut is empty
        if nut_is_empty.screw_is_empty(roi_rgba,debug=False):
            SCREWS_COLORS.append([])
            # draw a gray rectangle around the screw on the black image
            cv2.rectangle(black_image, (x, y), (x + w, y + h), (128, 128, 128), 2)
            print(f"Screw {i} is empty, skipping color detection.")
            # cv2.imshow(f"Screw_{i} is empty", roi_rgba)
            # key = cv2.waitKey(0) & 0xFF
            # if key == ord('q'):
            #     exit()
            continue

        # get the average color of the nut
        colors = screw_color_detector.get_nuts_colors_from_screw_image(roi_bgr, debug=False)
        print(f"Screw {i} detected colors: {colors}")
        # For every detected color, draw a rectangle around the screw on the black image
        color_recangle_height = h//len(colors) if len(colors) > 0 else h
        for j, color in enumerate(colors):
            print(f"Screw {i} color {j}: {color}")
            #  at this point, color is a numpy array of 3 elements (BGR)
            # draw a rectangle of this color on the black image
            color = tuple(int(c) for c in color)  # convert numpy array to tuple of ints
            # draw the rectangle below the screw bounding box
            cv2.rectangle(black_image, (x, y + j * color_recangle_height), (x + w, y + (j + 1) * color_recangle_height), color, -1)
            # print the color in hex format
            hex_color = "#{:02x}{:02x}{:02x}".format(color[2], color[1], color[0])

        SCREWS_COLORS.append(colors)
        print(f"Screw {i} detected colors: {colors}")
        # cv2.imshow(f"Screw_{i}", roi_rgba)
        # key = cv2.waitKey(0) & 0xFF
        # if key == ord('q'):
        #     exit()

    # display the black image with rectangles
    print("All screws detected colors:", SCREWS_COLORS)
    cv2.imshow("Original Image", cv2_image_utils.resize(img, 0.5))
    cv2.imshow("Detected Screws Colors", cv2_image_utils.resize(black_image, 0.5))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # conver SCREWS_COLORS to an array of colors
    all_colors = [color for sublist in SCREWS_COLORS for color in sublist]

    import color_clusterer
    CC = color_clusterer.ColorClusterer(eps=10, min_samples=1, use_lab=True)
    CC.fit(all_colors)
    CC.display_for_debugging()


    FINAL_DATA=[]
    for i, screw_colors in enumerate(SCREWS_COLORS):
        screw_data = []
        for color in screw_colors:
            cluster_label = CC.predict(color)
            cluster_name = CC.cluster_names_[cluster_label]
            screw_data.append((color, cluster_name))
        FINAL_DATA.append(screw_data)
    print("Final clustered colors data:", FINAL_DATA)