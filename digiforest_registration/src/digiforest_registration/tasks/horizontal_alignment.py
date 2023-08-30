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

        # cv2.imshow("Image", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return image

    def filter_image(self, image, name="image.png"):
        grayscale_image = (image * 255).astype(np.uint8)

        # Convert to binary image
        # _, binary_image = cv2.threshold(grayscale_image, 1, 255, cv2.THRESH_BINARY)

        return grayscale_image

    def find_feature_matches(self, img1, img2):
        # Initiate SIFT detector
        orb = cv2.ORB_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1, des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)
        # Draw first 10 matches.
        cv2.drawMatches(
            img1,
            kp1,
            img2,
            kp2,
            matches[:10],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        # cv2.imshow("Image", img3)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def find_local_maxima(self, img):
        # the input image is a float image
        # find local maxima in the image
        kernel_size = (25, 25)
        kernel = np.ones(kernel_size, np.uint8)
        dilated_img = cv2.dilate(img, kernel)
        mask = cv2.compare(
            img, dilated_img, cv2.CMP_GE
        )  # set mask to 255 where img >= mask

        white_pixel_indices = np.where(mask == 255)
        white_points_coordinates = list(
            zip(white_pixel_indices[1], white_pixel_indices[0])
        )
        print(len(white_points_coordinates))

        #
        grayscale_image = (img * 255).astype(np.uint8)
        maxima_img = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2RGB)
        for point in white_points_coordinates:
            if grayscale_image[point[1], point[0]] > 0:
                cv2.circle(maxima_img, point, 1, (0, 0, 255), -1)

        cv2.imshow("Image", maxima_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process(self):
        reference_image = self.compute_canopy_image(
            self.reference_cloud, *self.reference_ground_plane
        )
        # img1 = self.filter_image(reference_image, name="drone.png")
        image = self.compute_canopy_image(self.cloud, *self.cloud_ground_plane)
        # img2 = self.filter_image(image, name="frontier.png")

        self.find_local_maxima(reference_image)
        self.find_local_maxima(image)
