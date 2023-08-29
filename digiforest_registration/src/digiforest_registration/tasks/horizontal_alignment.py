import numpy as np
from numpy.typing import NDArray
from numpy import float64
import cv2


class HorizontalRegistration:
    def __init__(
        self, reference_cloud, reference_ground_plane, cloud, cloud_ground_plane
    ):
        self.reference_cloud = reference_cloud
        self.reference_ground_plane = reference_ground_plane
        self.cloud = cloud
        self.cloud_ground_plane = cloud_ground_plane

        self.min_distance_to_ground = 3.0
        self.image_resolution = 0.1

    def point_plane_distance(self, a, b, c, d, array: NDArray[float64]):
        """
        Calculate the distance between a point and a plane
        ax + by + cz + d = 0
        """
        normal = np.tile(np.array([a, b, c]), (array.shape[0], 1))
        d_array = np.tile(d, (array.shape[0], 1))

        # Calculate the distance using the formula
        dot_product = np.sum(array * normal, axis=1)
        dot_product = dot_product[:, np.newaxis]  # make it a column vector
        numerator = np.abs(dot_product + d_array)
        denominator = np.linalg.norm(np.array([a, b, c]))

        distance = numerator / denominator
        return distance

    def compute_canopy_image(self, cloud, a, b, c, d):
        points = np.asarray(cloud.to_legacy().points)
        dist = self.point_plane_distance(a, b, c, d, points)
        idx = dist > self.min_distance_to_ground
        idx = idx.flatten()  # make it a row vector
        canopy_points = points[idx]

        bounding_box = cloud.get_axis_aligned_bounding_box()

        # Get the minimum and maximum points of the bounding box
        min_bound = bounding_box.min_bound.numpy()
        max_bound = bounding_box.max_bound.numpy()

        # Create the canopy image
        height = int(np.ceil((max_bound[1] - min_bound[1]) / self.image_resolution))
        width = int(np.ceil((max_bound[0] - min_bound[0]) / self.image_resolution))
        print("height, width", height, width)

        image = np.zeros((height, width, 1), dtype=np.float32)

        # Fill the image
        dist = self.point_plane_distance(a, b, c, d, canopy_points)
        max_height = max(dist)
        index = 0
        for point in canopy_points:
            x = int(np.floor((point[0] - min_bound[0]) / self.image_resolution))
            y = int(np.floor((point[1] - min_bound[1]) / self.image_resolution))
            v = max(0, dist[index][0] / max_height[0])
            image[y, x] = [max(v, image[y, x])]
            index += 1

        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def filter_image(self, image):
        pass

    def process(self):
        self.compute_canopy_image(self.reference_cloud, *self.reference_ground_plane)
