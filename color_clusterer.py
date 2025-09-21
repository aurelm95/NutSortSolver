import numpy as np
from sklearn.cluster import DBSCAN
import cv2
from scipy.spatial.distance import cdist
import webcolors
from typing import List, Tuple, Dict, Optional


class ColorClusterer:
    def __init__(self, eps: float = 10, min_samples: int = 2, use_lab: bool = True):
        """
        Initializes the ColorClusterer.

        Args:
            eps (float): The maximum distance between two samples for them to be considered in the same neighborhood.
            min_samples (int): The number of samples in a neighborhood for a point to be considered a core point.
            use_lab (bool): Whether to convert colors to the LAB color space for clustering.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.use_lab = use_lab
        self.db: Optional[DBSCAN] = None
        self.labels_: Optional[np.ndarray] = None
        self.cluster_centers_: Dict[int, np.ndarray] = {}
        self.cluster_nominal_colors_: Dict[int, np.ndarray] = {}  # BGR
        self.cluster_names_: Dict[int, str] = {}                  # color name
        self.X_original: Optional[np.ndarray] = None
        self.X_transformed: Optional[np.ndarray] = None

    def _convert_colors(self, colors: List[np.ndarray]) -> np.ndarray:
        """
        Converts a list of BGR colors to the chosen color space (LAB or BGR).

        Args:
            colors (List[np.ndarray]): List of BGR color arrays.

        Returns:
            np.ndarray: Array of transformed color values.
        """
        arr = np.array(colors, dtype=np.uint8).reshape(-1, 1, 3)
        if self.use_lab:
            lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB).reshape(-1, 3)
            return lab
        else:
            return arr.reshape(-1, 3)

    def fit(self, colors: List[np.ndarray]) -> None:
        """
        Fits the clustering model to the provided list of colors.

        Args:
            colors (List[np.ndarray]): List of BGR color arrays.
        """
        self.X_original = np.array(colors)
        self.X_transformed = self._convert_colors(colors)

        self.db = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.labels_ = self.db.fit_predict(self.X_transformed)

        unique_labels = np.unique(self.labels_)

        for label in unique_labels:
            if label == -1:
                continue  # Ignore noise points

            indices = np.where(self.labels_ == label)[0]
            cluster_points = self.X_transformed[indices]
            cluster_bgr = self.X_original[indices]

            # Centroid in LAB or original space
            centroid = np.mean(cluster_points, axis=0)
            self.cluster_centers_[label] = centroid

            # Average BGR color (nominal)
            mean_bgr = np.mean(cluster_bgr, axis=0).astype(np.uint8)
            self.cluster_nominal_colors_[label] = mean_bgr

            # Convert BGR -> RGB -> closest color name
            rgb = tuple(int(c) for c in mean_bgr[::-1])  # BGR -> RGB
            name = self._closest_color_name(rgb)
            self.cluster_names_[label] = name

    def predict(self, color: np.ndarray) -> int:
        """
        Predicts the cluster label for a given BGR color.

        Args:
            color (np.ndarray): A single BGR color array.

        Returns:
            int: The cluster label the color belongs to.
        """
        if not self.cluster_centers_:
            raise ValueError("The model has not been trained. Call .fit() first.")

        transformed = self._convert_colors([color])[0]
        centers = np.array(list(self.cluster_centers_.values()))
        labels = list(self.cluster_centers_.keys())

        distances = cdist([transformed], centers)[0]
        closest_idx = np.argmin(distances)
        return labels[closest_idx]

    def get_cluster_names(self) -> Dict[int, Tuple[str, np.ndarray]]:
        """
        Returns a dictionary mapping cluster labels to their closest color name and BGR color.

        Returns:
            Dict[int, Tuple[str, np.ndarray]]: A dictionary in the form {label: (name, bgr_color)}.
        """
        return {
            label: (self.cluster_names_[label], self.cluster_nominal_colors_[label])
            for label in self.cluster_names_
        }

    def _closest_color_name(self, requested_rgb: Tuple[int, int, int]) -> str:
        """
        Finds the closest CSS3 color name to a given RGB value.

        Args:
            requested_rgb (Tuple[int, int, int]): RGB color tuple.

        Returns:
            str: Closest matching CSS3 color name.
        """
        color_distance_to_names = {}
        for name in webcolors.names("css3"):
            r_c, g_c, b_c = webcolors.name_to_rgb(name)
            rd = (r_c - requested_rgb[0]) ** 2
            gd = (g_c - requested_rgb[1]) ** 2
            bd = (b_c - requested_rgb[2]) ** 2
            color_distance_to_names[(rd + gd + bd)] = name
        return color_distance_to_names[min(color_distance_to_names.keys())]

    def display_for_debugging(self, box_width: int = 50, box_height: int = 50, spacing: int = 10) -> None:
        """
        Displays a debug visualization of clustered colors using OpenCV.

        For each cluster:
            - Shows solid color rectangles for each color assigned to that cluster.
            - Draws a bounding box (hollow) in the cluster's mean color.
            - Displays the cluster name as a label.

        Args:
            box_width (int): Width of each color block.
            box_height (int): Height of each color block.
            spacing (int): Spacing between color blocks.
        """
        if self.labels_ is None or self.X_original is None:
            raise ValueError("Model must be trained using .fit() before calling display_for_debugging().")

        clusters = {}
        for idx, label in enumerate(self.labels_):
            if label == -1:
                continue  # Skip noise
            clusters.setdefault(label, []).append(self.X_original[idx])

        # Calculate image size
        margin = 20
        max_colors_in_cluster = max(len(colors) for colors in clusters.values())
        img_width = (box_width + spacing) * max_colors_in_cluster + 2 * margin
        img_height = (box_height + spacing + 30) * len(clusters) + 2 * margin

        image = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255  # white background

        y_offset = margin

        for label, colors in clusters.items():
            x_offset = margin
            top_left = (x_offset - spacing // 2, y_offset - spacing // 2)
            bottom_right = (
                x_offset + len(colors) * (box_width + spacing) - spacing + spacing // 2,
                y_offset + box_height + spacing // 2
            )

            # Draw each color rectangle
            for color in colors:
                color_bgr = tuple(int(c) for c in color)
                cv2.rectangle(image,
                              (x_offset, y_offset),
                              (x_offset + box_width, y_offset + box_height),
                              color_bgr,
                              thickness=-1)
                x_offset += box_width + spacing

            # Draw the cluster boundary box (hollow)
            cluster_color = tuple(int(c) for c in self.cluster_nominal_colors_[label])
            cv2.rectangle(image, top_left, bottom_right, cluster_color, thickness=2)

            # Put cluster name
            name = self.cluster_names_[label]
            text_position = (top_left[0], bottom_right[1] + 20)
            cv2.putText(image, name, text_position, cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 1, lineType=cv2.LINE_AA)

            # Move y_offset for next cluster
            y_offset = bottom_right[1] + spacing + 30

        # Show the image
        cv2.imshow("Cluster Debugging", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # List of BGR colors (each as np.array of length 3)
    colors = [
        np.array([123, 231, 132]),
        np.array([124, 229, 133]),
        np.array([10, 20, 30]),
        np.array([12, 18, 33]),
        np.array([200, 100, 50]),
        np.array([201, 99, 52]),
        np.array([0, 255, 0]),
        np.array([0, 254, 1]),
        np.array([255, 0, 0])
    ]

    # Instantiate the object
    clusterer = ColorClusterer(eps=10, min_samples=2, use_lab=True)

    # Fit the model to the data
    clusterer.fit(colors)

    # Display the clustered colors for debugging
    clusterer.display_for_debugging()

    # Predict the cluster of a new color
    new_color = np.array([122, 230, 130])  # similar to the first group
    cluster_id = clusterer.predict(new_color)

    print(f"The color {new_color} belongs to cluster #{cluster_id} - "
          f"{clusterer.cluster_names_[cluster_id]}")
