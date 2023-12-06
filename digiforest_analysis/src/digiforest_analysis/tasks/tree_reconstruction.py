from typing import Iterable, List, Union, Tuple
import numpy as np
from skimage.transform import hough_circle
from copy import deepcopy
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from plotly import graph_objects as go


class Circle:
    def __init__(
        self,
        center: Union[tuple, list, np.ndarray],
        radius: float,
        normal: np.ndarray = np.array([0, 0, 1]),
    ) -> None:
        self.radius = radius
        self.x = center[0]
        self.y = center[1]
        self.z = center[2]
        if type(center) in (list, tuple):
            center = np.array(center)
        self.center = center

        y_vec = np.array([0, -normal[2], normal[1]]).astype(float)
        y_vec /= np.linalg.norm(y_vec)
        x_vec = np.cross(y_vec, normal)
        self.rot_mat = np.vstack((x_vec, y_vec, normal)).T

    @classmethod
    def from_cloud_hough(
        cls,
        points: np.ndarray,
        grid_res: float,
        min_radius: float,
        max_radius: float,
        point_ratio: float = 0.0,
        previous_center: np.ndarray = None,
        search_radius: float = None,
        entropy_weighting: float = 10.0,
        circle_height: float = 0.0,
        return_pixels_and_votes: bool = False,
        **kwargs,
    ) -> Tuple["Circle", np.ndarray, np.ndarray, np.ndarray]:
        """This function fits circles to the points in a slice using the hough
        transform. If both previous_center and search_radius are given, the search
        for the circle center is constrained to a circle around the previous center.
        If previous_radius is given, the search for the circle radius is constrained
        to be smaller than this radius.

        Adapted from: https://www.sciencedirect.com/science/article/pii/S0168169917301114?ref=pdf_download&fr=RR-2&rr=81fe181ccd14779b

        Args:
            points (np.ndarray): N x 2 array of points in the slice
            grid_res (float): length of pixel for hough map in m
            min_radius (float): minimum radius of circle in m
            max_radius (float): maximum radius of circle in m
            point_ratio (float, optional): ratio of points in a pixel wrt. number in
                most populated pixel to be counted valid. Defaults to 1.0.
            previous_center (np.ndarray, optional): x,y coordinates of the previous
                circle center. Defaults to None.
            search_radius (float, optional): radius around previous center to search
            entropy_weighing (float, optional): weight to weigh the hough votes by
                the entropy of the top 10 votes for that radius. This helps to cope
                with noise at small radii. Defaults to 10.0.
            circle_height (float, optional): height of the slice in m.
                Defaults to 0.0.
            return_pixels_and_votes (bool, optional): If True, an array containing the
                pixels aggregating the points and the votes corresponding to the optimal
                radius casted are returned additionally. Useful for debugging.
                Defaults to False.

        Returns:
            Circle: circle object and if wanted the pixels aggregating the
                points and the entropy_weighted hough votes and the penalty factor.
                The unweighted votes can be obtained by multiplying weighted votes with
                the factor.
        """
        # construct 2D grid with pixel length of grid_res
        # bounding the point cloud of the current slice
        min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
        min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
        # make sure pixels are square
        grid_center = np.array([(max_x + min_x) / 2, (max_y + min_y) / 2])
        grid_width = 1.5 * max(max_x - min_x, max_y - min_y)
        min_x, max_x = grid_center[0] - grid_width / 2, grid_center[0] + grid_width / 2
        min_y, max_y = grid_center[1] - grid_width / 2, grid_center[1] + grid_width / 2
        n_cells = int(grid_width / grid_res)
        # construct the grid
        grid_x, grid_y = np.meshgrid(
            np.linspace(min_x, max_x, n_cells + 1),
            np.linspace(min_y, max_y, n_cells + 1),
        )
        # reshape and remove last row and column
        grid_x = grid_x[None, :-1, :-1]
        grid_y = grid_y[None, :-1, :-1]

        # count how many points are in every cell (numpy version consumes too much mem)
        pixels = np.zeros((n_cells, n_cells))
        for point in points:
            i = int((point[1] - min_y) / grid_res)
            j = int((point[0] - min_x) / grid_res)
            pixels[i, j] += 1
        pixels[pixels < point_ratio * np.max(pixels)] = 0

        # crop image to only include the pixels containing points with a 50% margin
        filled_pixels = np.argwhere(pixels)
        min_x, max_x = np.min(filled_pixels[:, 0]), np.max(filled_pixels[:, 0])
        min_y, max_y = np.min(filled_pixels[:, 1]), np.max(filled_pixels[:, 1])
        min_x = max(0, min_x - int(0.5 + 0.5 * (max_x - min_x)))
        max_x = min(n_cells, max_x + int(0.5 + 0.5 * (max_x - min_x)))
        min_y = max(0, min_y - int(0.5 + 0.5 * (max_y - min_y)))
        max_y = min(n_cells, max_y + int(0.5 + 0.5 * (max_y - min_y)))
        pixels = pixels[min_x:max_x, min_y:max_y]
        grid_x = grid_x[:, min_x:max_x, min_y:max_y]
        grid_y = grid_y[:, min_x:max_x, min_y:max_y]

        # fit circles to the points in every cell using the hough transform
        min_radius_px = int(0.5 + min_radius / grid_res)
        max_radius_px = int(0.5 + max_radius / grid_res)
        # assume that at least a quarter of the points are seen. Then the max radius
        # is the number of pixels in one direction
        max_radius_px = min(max_radius_px, n_cells)
        # TODO unequally space tryradii for efficiency
        try_radii = np.arange(min_radius_px, max_radius_px)
        if try_radii.shape[0] == 0:
            if return_pixels_and_votes:
                return None, None, None, None
            else:
                return None
        hough_res = hough_circle(pixels, try_radii)

        # weigh each radius by the entropy of the top 10 hough votes for that radius
        # for small radii there are many circles in a small area leading to many
        # high values. Which might be higher than for higher radii, which we are
        # looking for more. To avoid this, we weigh each radius by the entropy of
        # the top 10 hough votes for that radius, to reward radii that have a clear
        # vote.
        if entropy_weighting != 0.0:
            # To avoid artefacts where large radii have cropped voting-rings that
            # lead to artificially high entropy, exclude radii where both the
            # following applies:
            #    1. the radius is bigger than 0.5 * nc_cells. Thus, the voting rings
            #       are cropped.
            #    2. compared to the radius with max number of voted pixels, this
            #       radius has less than 50% of the votes.

            vote_fraction = np.count_nonzero(hough_res, axis=(1, 2)) / n_cells**2
            radius_mask = try_radii > 0.5 * n_cells
            votes_mask = vote_fraction < 0.50 * np.max(vote_fraction)
            mask = ~np.logical_and(radius_mask, votes_mask)
            if not np.any(mask):
                # return if there's no radii left
                if return_pixels_and_votes:
                    return None, None, None, None
                else:
                    return None
            hough_res = hough_res[mask]
            try_radii = try_radii[mask]

            hough_flattened = hough_res.reshape(hough_res.shape[0], -1)
            top_10 = np.partition(hough_flattened, -10, axis=1)[:, -10:]
            # discard radii where there's fewer than 10 candidates
            # i.e. where top 10 contains 0
            discard_mask = top_10.min(axis=1) < 1e-3
            top_10_normalized = top_10 / np.sum(top_10, axis=1, keepdims=True)
            top_10_entropy = -np.sum(
                top_10_normalized * np.log(top_10_normalized + 1e-12), axis=1
            )
            # replace NaNs with max value
            top_10_entropy[np.isnan(top_10_entropy)] = -1
            top_10_entropy[top_10_entropy < 0] = top_10_entropy.max()
            top_10_entropy[discard_mask] = top_10_entropy.max()
            # normalize entropy to be between 0.1 and 1 given max reward of 10x
            penalty = 1 / entropy_weighting + (1 - entropy_weighting) * (
                top_10_entropy - np.min(top_10_entropy)
            ) / (np.max(top_10_entropy) - np.min(top_10_entropy))
            hough_res /= penalty[:, None, None]

        # constrain circles to be roughly above previous one
        if previous_center is not None and search_radius is not None:
            # calculate distance of every circle candidate center to the previous
            # center
            dist = np.sqrt(
                (grid_x - previous_center[0]) ** 2 + (grid_y - previous_center[1]) ** 2
            )
            # mask out all circles that are not within the search radius
            hough_res[np.broadcast_to(dist, hough_res.shape) > search_radius] = 0

        # find the circle with the most votes
        i_rad, x_px, y_px = np.unravel_index(np.argmax(hough_res), hough_res.shape)

        # transform pixel coordinates back to world coordinates
        x_c = grid_x[0, x_px, y_px] + grid_res / 2
        y_c = grid_y[0, x_px, y_px] + grid_res / 2
        r = try_radii[i_rad] * grid_res
        circ = cls((x_c, y_c, circle_height), r)

        if return_pixels_and_votes:
            return circ, pixels, hough_res[i_rad], penalty[i_rad, None, None]
        else:
            return circ

    @classmethod
    def from_cloud_bullock(
        cls,
        points: np.ndarray,
        circle_height: float = 0.0,
        **kwargs,
    ) -> "Circle":
        """Fits a circle to a point cloud in the least-squares sense.

        Adapted from https://dtcenter.org/sites/default/files/community-code/met/docs/write-ups/circle_fit.pdf

        Args:
            points (np.ndarray): N x 2 array of points in the slice

        Returns:
            Circle: fitted circle
        """
        # normaliye points
        points_mean = np.mean(points, axis=0)
        u = points[:, 0] - points_mean[0]
        v = points[:, 1] - points_mean[1]
        # pre-calculate summands
        S_uu = np.sum(np.power(u, 2))
        S_vv = np.sum(np.power(v, 2))
        S_uv = np.sum(u * v)
        S_uuu_uvv = np.sum(np.power(u, 3) + u * np.power(v, 2))
        S_vvv_vuu = np.sum(np.power(v, 3) + v * np.power(u, 2))

        # calculate circle center in normalized coordinates and radius
        v_c = (S_uuu_uvv / (2 * S_uu) - S_vvv_vuu / (2 * S_uv)) / (
            S_uv / S_uu - S_vv / S_uv + 1e-12
        )
        u_c = (S_uuu_uvv / (2 * S_uv) - S_vvv_vuu / (2 * S_vv)) / (
            S_uu / S_uv - S_uv / S_vv + 1e-12
        )
        r = np.sqrt(
            np.power(u_c, 2) + np.power(v_c, 2) + (S_uu + S_vv) / points.shape[0]
        )
        # denormalize
        x_c, y_c = points_mean[0] + u_c, points_mean[1] + v_c
        return cls((x_c, y_c, circle_height), r)

    @classmethod
    def from_cloud_ransac(
        cls,
        points: np.ndarray,
        n_iterations: int = 100,
        n_points: int = 3,
        n_inliers: int = 10,
        inlier_threshold: float = 0.01,
        circle_height: float = 0.0,
        **kwargs,
    ) -> Tuple["Circle", np.ndarray]:
        """This function filters outliers using the RANSAC algorithm. It fits a
        circle to a random sample of n_points points and counts how many points are
        inliers. If the number of inliers is larger than n_inliers, the circle is
        accepted and the number of inliers is updated. The algorithm is repeated
        n_iterations times.

        Args:
            points (np.ndarray): N x 2 array of points in the slice
            n_iterations (int): Number of iterations to run the algorithm
            n_points (int): Number of points used to model the circle
            n_inliers (int): Minimum number of inliers to accept the circle
            inlier_threshold (float): thickness of band around circle to qualify
                                    inliers

        Returns:
            tuple: best model Circle and inlier points if model was found, else None
        """
        best_model = None
        best_inliers = None
        for _ in range(n_iterations):
            # randomly sample n_points from the points
            sample = points[np.random.choice(points.shape[0], n_points, replace=False)]
            # fit a circle to the sample
            circle = cls.from_cloud_bullock(sample, circle_height)
            # count how many points are inliers
            dist = np.sqrt(
                (points[:, 0] - circle.x) ** 2 + (points[:, 1] - circle.y) ** 2
            )
            inliers = points[
                np.logical_and(
                    (circle.radius - inlier_threshold) < dist,
                    dist < (circle.radius + inlier_threshold),
                )
            ]
            if inliers.shape[0] > n_inliers:
                best_model = deepcopy(circle)
                best_inliers = inliers
                n_inliers = inliers.shape[0]
        return best_model, best_inliers

    @classmethod
    def from_cloud_lm(  # NOT TESTED!
        cls,
        initial_circle: "Circle",
        points: np.ndarray,
        weights: np.ndarray = None,
        circle_height: float = 0.0,
    ) -> "Circle":
        def residuals(x: np.ndarray, *args, **kwargs) -> np.ndarray:
            if "points" not in kwargs or "weights" not in kwargs:
                raise KeyError("points and weights have to be passed as kwargs")
            points = kwargs["points"][:, :2]
            weights = kwargs["weights"]

            A, D, theta = x
            x = points[:, 0]
            y = points[:, 1]
            radicant = 1 + 4 * A * D
            # if radicant < 0:
            #     print("radicant is negative: ", radicant)
            E = np.sqrt(radicant)
            z = np.sum(np.power(points, 2), axis=1)
            u = x * np.cos(theta) + y * np.sin(theta)
            P = A * z + E * u + D
            d = 2 * P / (1 + np.sqrt(1 + 4 * A * P))
            F = (weights if weights is not None else 1) * np.power(d, 2)
            return F

        def jacobian(vars: np.ndarray, *args, **kwargs) -> np.ndarray:
            if "points" not in kwargs or "weights" not in kwargs:
                raise KeyError("points and weights have to be passed as kwargs")
            points = kwargs["points"][:, :2]
            weights = kwargs["weights"]

            A, D, theta = vars
            x = points[:, 0]
            y = points[:, 1]
            E = np.sqrt(1 + 4 * A * D)
            z = np.sum(np.power(points, 2), axis=1)
            u = x * np.cos(theta) + y * np.sin(theta)
            P = A * z + E * u + D
            d = 2 * P / (1 + np.sqrt(1 + 4 * A * P))
            Q = np.sqrt(1 + 4 * A * P)
            R = 2 * (1 - A * d / Q) / (Q + 1)

            dddA = (z + 2 * D * u / E) * R - np.power(d, 2) / Q
            dddA = (weights if weights is not None else 1) * d * dddA
            dddD = (2 * A * u / E + 1) * R
            dddD = (weights if weights is not None else 1) * d * dddD
            dddtheta = (-x * np.sin(theta) + y * np.cos(theta)) * E * R
            dddtheta = (weights if weights is not None else 1) * d * dddtheta
            return np.vstack((dddA, dddD, dddtheta)).T

        (x_0, y_0, _), R_0 = initial_circle.center, initial_circle.radius
        A_0 = 1 / (2 * R_0)
        # A_min = 1 / (2 * R_max)
        # A_max = 1 / (2 * R_min)
        B_0 = -2 * A_0 * x_0
        # B_min = -2 * A_max * initial_circle.x
        # B_max = -2 * A_min * initial_circle.x
        C_0 = -2 * A_0 * y_0
        # C_min = -2 * A_max * initial_circle.y
        # C_max = -2 * A_min * initial_circle.y
        D_0 = (B_0**2 + C_0**2 - 1) / (4 * A_0)
        theta_0 = np.arccos(B_0 / np.sqrt(1 + 4 * A_0 * D_0))
        result = least_squares(
            fun=residuals,
            x0=[A_0, D_0, theta_0],
            jac=jacobian,
            xtol=1e-15,
            # bounds=([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]),
            kwargs=dict(points=points, weights=weights),
            method="lm",
        )
        A, D, theta = result.x
        radix = np.sqrt(1 + 4 * A * D)
        B, C = radix * np.cos(theta), radix * np.sin(theta)
        x_c, y_c, R = -B / (2 * A), -C / (2 * A), 1 / (2 * np.abs(A))

        return cls((x_c, y_c, circle_height), R)

    def query_point(self, theta: float):
        pointer = np.array([np.cos(theta), -np.sin(theta), 0])
        return self.center + self.radius * self.rot_mat @ pointer

    def update_height(self, new_height: float):
        self.center[2] = new_height
        self.z = new_height

    def get_distance(self, point: np.ndarray, use_z: bool = False):
        if use_z:
            if len(point.shape) == 2:
                return np.linalg.norm(point - self.center, axis=1) - self.radius
            else:
                return np.linalg.norm(point - self.center) - self.radius
        else:
            if len(point.shape) == 2:
                return (
                    np.linalg.norm(point[:, :2] - self.center[:2], axis=1) - self.radius
                )
            else:
                return np.linalg.norm(point[:2] - self.center[:2]) - self.radius

    def plot(self, figure: go.Figure, **kwargs):
        points = [self.query_point(theta) for theta in np.linspace(0, 2 * np.pi, 100)]
        points = np.array(points)
        line_color = kwargs.get("color", "red")
        line_width = kwargs.get("line_width", 2)
        figure.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="lines",
                line=dict(color=line_color, width=line_width),
            )
        )

    def plot_cone_frustum(self, other_circle: "Circle", figure: go.Figure, **kwargs):
        num_verts = 100
        vertices = [
            self.query_point(theta) for theta in np.linspace(0, 2 * np.pi, num_verts)
        ] + [
            other_circle.query_point(theta)
            for theta in np.linspace(0, 2 * np.pi, num_verts)
        ]
        vertices = np.stack(vertices)
        basic_tri_1 = np.array([0, 1, num_verts])  # these triangles repeats 100 times
        basic_tri_2 = np.array([1, num_verts + 1, num_verts])
        tri_indices = [basic_tri_1 + i for i in range(num_verts)] + [
            basic_tri_2 + i for i in range(num_verts)
        ]
        tri_indices = np.stack(tri_indices)[:-1, :]  # last is equal to first
        color = kwargs.get("color", "red")
        figure.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=tri_indices[:, 0],
                j=tri_indices[:, 1],
                k=tri_indices[:, 2],
                color=color,
                opacity=0.25,
            )
        )

    def genereate_cone_frustum_mesh(self, other_circle: "Circle", num_verts: int = 20):
        """This function genereates meshes (vertices and triangle indices) for a frustum
        (cut cone-oid) spanned by the current circle and a given other circle.

        Args:
            other_circle (Circle): Other circle spannig the frustum
            num_verts (int, optional): Number of vertices for vizualizing the top or bottom face. Defaults to 100.

        Returns:
            _type_: _description_
        """
        vertices = [
            self.query_point(theta) for theta in np.linspace(0, 2 * np.pi, num_verts)
        ] + [
            other_circle.query_point(theta)
            for theta in np.linspace(0, 2 * np.pi, num_verts)
        ]
        vertices = np.stack(vertices)
        basic_tri_1 = np.array([0, 1, num_verts])  # these triangles repeats 100 times
        basic_tri_2 = np.array([1, num_verts + 1, num_verts])
        tri_indices = [basic_tri_1 + i for i in range(num_verts)] + [
            basic_tri_2 + i for i in range(num_verts)
        ]
        tri_indices = np.stack(tri_indices)[:-1, :]  # last is equal to first

        return vertices, tri_indices

    def __str__(self) -> str:
        return f"Circle: center: {self.center}, radius: {self.radius}, normal: {self.rot_mat[:, 2]}"

    def apply_transform(self, translation: np.ndarray, rotation: np.ndarray):
        if rotation.shape[0] == 4:
            rot_mat = Rotation.from_quat(rotation).as_matrix()
        elif rotation.shape == (3, 3):
            rot_mat = rotation
        else:
            raise ValueError("rotation must be given as 3x3 matrix or quaternion")
        self.center = rot_mat @ self.center + translation
        self.x, self.y, self.z = self.center
        self.rot_mat = rot_mat @ self.rot_mat


class Tree:
    def __init__(
        self,
        circles: List[Circle],
        point_counts: List[float],
        points: List[np.ndarray] = None,
        hough_points: List[np.ndarray] = None,
        hough_circles: List[Circle] = None,
        hough_pixels: List[np.ndarray] = None,
        hough_votes: List[np.ndarray] = None,
    ) -> None:
        assert len(circles) == len(
            point_counts
        ), "circles and point_counts must have the same length"
        if points is not None:
            assert len(circles) == len(
                points
            ), "circles and points must have the same length"
        if hough_points is not None:
            assert len(circles) == len(
                hough_points
            ), "circles and hough_points must have the same length"
        if hough_votes is not None:
            assert len(circles) == len(
                hough_votes
            ), "circles and votes must have the same length"
        if hough_pixels is not None:
            assert len(circles) == len(
                hough_pixels
            ), "circles and hough_pixels must have the same length"
        if hough_circles is not None:
            assert len(circles) == len(
                hough_circles
            ), "circles and hough_circles must have the same length"

        self.circles = circles
        self.point_counts = point_counts
        self.points = points
        self.hough_points = hough_points
        self.hough_pixels = hough_pixels
        self.hough_votes = hough_votes
        self.hough_circles = hough_circles

    def generate_mesh(self):
        vertices, triangles = np.empty((0, 3)), np.empty((0, 3), dtype=int)
        for i in range(len(self.circles) - 1):
            verts, tris = self.circles[i].genereate_cone_frustum_mesh(
                self.circles[i + 1]
            )
            triangles = np.vstack((triangles, tris + vertices.shape[0]))
            vertices = np.vstack((vertices, verts))
        return vertices, triangles

    def apply_transform(self, translation: np.ndarray, rotation: np.ndarray):
        if rotation.shape[0] == 4:
            rot_mat = Rotation.from_quat(rotation).as_matrix()
        elif rotation.shape == (3, 3):
            rot_mat = rotation
        else:
            raise ValueError("rotation must be given as 3x3 matrix or quaternion")
        for i in range(len(self.circles)):
            self.circles[i].apply_transform(translation, rotation)
            if self.hough_circles is not None:
                self.hough_circles[i].apply_transform(translation, rotation)
            if self.points is not None:
                self.points[i] = self.points[i] @ rot_mat.T + translation.squeeze()
            if self.hough_points is not None:
                self.hough_points[i] = (
                    self.hough_points[i] @ rot_mat.T + translation.squeeze()
                )

    @classmethod
    def from_cluster(
        cls,
        cluster: dict,
        slice_heights: Union[float, Iterable] = 0.5,
        slice_thickness: float = 0.3,
        outlier_radius: float = 0.02,
        max_center_deviation: float = 0.05,
        max_radius_deviation: float = 0.05,
        filter_min_points: int = 10,
        min_hough_vote: float = 0.1,
        grid_res: float = 0.01,
        point_ratio: float = 0.2,
        entropy_weighting: float = 10.0,
        max_consecutive_fails: int = 3,
        max_height: float = 10.0,
        save_points: bool = True,
        save_debug_results: bool = True,
        **kwargs,
    ) -> "Tree":
        """slices the given point cloud at regular intervals or given intervals and
        vilters the slices using the hough transform. After filtering, circles are
        fit to the slices using a least squares fitting. The circles are then

        Args:
            cluster (dict): Cluster result including ["cloud"] and ["info"]["axis"]
            slice_heights (Union[float, Iterable]): if a float [m] is given, the
                point cloud is sliced at regular intervals. If an iterable is given,
                the point cloud is sliced at the given heights. Defaults to 0.5.
            slice_thickness (float, optional): Thickness of the slices in m.
            outlier_radius (float, optional): Least distance from the filter circle
                for a point to be considered an outlier. Defaults to 0.02.
            max_center_deviation (float, optional): Maximal deviation of a circle's
                radius wrt. to the previous circle beneath. Defaults to 0.05.
            max_radius_deviation (float, optional): Maximal deviation of a circle's
                center point wrt. to the previous circle beneath. Defaults to 0.05.
            filter_min_points (int, optional): Minimum number of points in a slice
                to fit a circle. Defaults to 10.
            grid_res (float, optional): Resolution of the hough grid in m. Defaults
                to 0.01.
            point_ratio (float, optional): Ratio of points in a pixel wrt. number in
                most populated pixel to be counted valid. Defaults to 0.2.
            entropy_weighting (float, optional): weight to weigh the hough votes by
                the entropy of the top 10 votes for that radius. This helps to cope
                with noise at small radii. Defaults to 10.0.
            max_consecutive_fails (int, optional): Maximum number of consecutive
                slices that fail to fit a circle before the algorithm stops. Defaults
                to 3.
            max_height (float, optional): Maximum height until where the tree is
                fitted. Defaults to 10.0.
            save_points (bool, optional): If True, the points for each slice are saved
                in the tree object. Defaults to False.
            save_debug_results(bool, optional): If True, intermediate results from the
                hough filtering process are stored in the tree object.

        Returns:
            Tree: Tree object representing a stack of circles
        """
        cloud = deepcopy(cluster["cloud"].point.positions.numpy())
        axis = cluster["info"]["axis"]
        center, rot_mat, r_from_seg = axis["center"], axis["rot_mat"], axis["radius"]

        # move cloud to origin and rotate it upright
        cloud -= center
        cloud = cloud @ rot_mat  # equal to: cloud = (rot_mat.T @ cloud.T).T

        circle_stack = []
        # slice point cloud
        if type(slice_heights) == float:
            slice_heights = np.arange(
                np.min(cloud[:, 2]),
                max_height if max_height is not None else np.max(cloud[:, 2]),
                slice_heights,
            )
        fail_counter = 0  # counts how many slices failed to fit a circle
        for slice_height in slice_heights:
            if fail_counter > max_consecutive_fails:
                break

            search_cylinder_radius = 2 * r_from_seg + max_radius_deviation  # TODO tune
            slice_points = cloud[
                np.logical_and(
                    cloud[:, 2] >= slice_height - slice_thickness / 2,
                    cloud[:, 2] < slice_height + slice_thickness / 2,
                )
            ]  # remove points outside of slice
            slice_points = slice_points[
                np.linalg.norm(slice_points[:, :2], axis=1) < search_cylinder_radius
            ]  # remove points outside of search cylinder

            # check if there are enough points to fit a circle
            if len(slice_points) < filter_min_points:
                print(f"Too few points ({len(slice_points)}) to fit a circle (1)")
                fail_counter += 1
                continue

            # fit hough circle to all points in the slice
            previous_circle = (
                None if len(circle_stack) == 0 else circle_stack[-1]["circle"]
            )
            if previous_circle is not None:
                previous_center = previous_circle.center
                # min_radius = previous_circle.radius * (1 - max_radius_deviation)
                # max_radius = previous_circle.radius * (1 + max_radius_deviation)
            else:
                previous_center = None
            min_radius = 0.75 * r_from_seg
            max_radius = 1.5 * r_from_seg

            hough_circle, pixels, hough_votes, penalty = Circle.from_cloud_hough(
                slice_points,
                grid_res,
                min_radius,
                max_radius,
                point_ratio=point_ratio,
                previous_center=previous_center,
                search_radius=max_center_deviation,
                entropy_weighting=entropy_weighting,
                return_pixels_and_votes=True,
            )

            if hough_circle is None:
                print("No hough circle found")
                fail_counter += 1
                continue

            # check for
            if hough_votes.max() * penalty < min_hough_vote:
                print(f"Hough vote {hough_votes.max() * penalty} not very promising")
                fail_counter += 1
                continue

            # filter points using the hough circle
            pc_filtered = slice_points[
                hough_circle.get_distance(slice_points) < outlier_radius
            ]

            # again, check if there are enough points to fit a circle. This is not a
            # redundant check as the check before will often spare expensive hough
            # calculations, whereas this check is necessary to fit a reasonabole
            # circle
            if len(pc_filtered) < filter_min_points:
                print(f"Too few points ({len(pc_filtered)}) to fit a circle (2)")
                fail_counter += 1
                continue

            # fit circle to filtered points
            bulloc_circle = Circle.from_cloud_bullock(pc_filtered, slice_height)

            # TODO check if circle is plausible. For now, just impose max radius of 1 m
            if bulloc_circle.radius > 1.0:
                print("Radius too large")
                fail_counter += 1
                continue

            # aggregate results
            circle_stack.append(
                {
                    "num_points": pc_filtered.shape[0],
                    "circle": bulloc_circle,
                    "points": slice_points,  # if save_points else None,
                    "hough_circle": hough_circle,  # if save_hough_points else None
                    "hough_points": pc_filtered,  # if save_hough_points else None,
                    "hough_pixels": pixels,  # if save_pixels else None,
                    "votes": hough_votes,  # if save_pixels else None,
                }
            )

            fail_counter = 0

        if save_debug_results:
            tree = cls(
                [t["circle"] for t in circle_stack],
                [t["num_points"] for t in circle_stack],
                points=[t["points"] for t in circle_stack] if save_points else None,
                hough_points=[t["hough_points"] for t in circle_stack],
                hough_circles=[t["hough_circle"] for t in circle_stack],
                hough_pixels=[t["hough_pixels"] for t in circle_stack],
                hough_votes=[t["votes"] for t in circle_stack],
            )
        else:
            tree = cls(
                [t["circle"] for t in circle_stack],
                [t["num_points"] for t in circle_stack],
            )

        # reapply rotation and translation
        cloud = cloud @ rot_mat.T
        cloud += center
        tree.apply_transform(center, rot_mat)

        return tree

        # all_points_filtered = np.concatenate(pc_slices_filtered)
        # tree = {
        #     "axis": {"center": None, "direction": None},
        #     "circles": [],
        # }

        # # helper functions
        # def weighting_function(dist:np.ndarray, r:float=outlier_radius)->np.ndarray:
        #     return np.exp(-np.power(dist, 2) / (2 * np.power(r, 2)))

        # # Step 1.     find principal axis using PCA
        # pca = PCA(n_components=3)
        # pca.fit(all_points_filtered)
        # tree_axis = pca.components_[0]
        # tree["axis"]["direction"] = tree_axis

        # # Step 2.     find the slice with the most points. This is deemed to be the
        # #             most relieable filtering result and called the base slice
        # i_base = np.argmax([slice.shape[0] for slice in pc_slices])
        # # Step 2.1.   fit a precise circle to these inliers using RANSAC
        # base_circle, _ = fit_circle_ransac(pc_slices[i_base], **kwargs)
        # base_circle.update_height(slice_heights[i_base])
        # tree["circles"].append(base_circle)
        # # Step 2.2.   the centre of this circle is the centre of the stem axis
        # tree["axis"]["center"] = base_circle.center
        # tree["base_circle_index"] = 0  # has to be updated when pushing to front

        # # Step 3.     Work in both directions up and down from the base circle
        # # Step 3.1.   Direction up
        # i_iter = i_base

        # while i_iter < len(pc_slices) - 1:
        #     i_iter += 1
        #     # Step 3.1.1. initialize a circle directly above the previous one with
        #     #             the previous radius
        #     circ_iter = deepcopy(tree["circles"][-1])
        #     circ_iter.update_height(slice_heights[i_iter])
        #     # Step 3.1.2. do not eliminate all points far from the hough circle, but
        #     #             weigh them nonlinearly by the distance
        #     pc_slice_iter = pc_slices[i_iter]
        #     if pc_slice_iter.shape[0] < 5:
        #         continue
        #     filter_circle = filter_circles[i_iter]
        #     weights = np.abs(weighting_function(filter_circle.get_distance(pc_slice_iter)))
        #     # Step 3.1.3. fit the circle to the new point slice using the advanced
        #     #             circle-segment-algo restricting the movement of the center
        #     #             and the change in radius.
        #     # R_min = circ_iter.radius - max_radius_deviation
        #     # R_max = circ_iter.radius + max_radius_deviation
        #     # circ_iter = fit_circle_lm(circ_iter, pc_slice_iter, weights, slice_heights[i_iter])
        #     circ_iter = fit_circle_bullock(pc_slice_iter, slice_heights[i_iter])

        #     # Step 3.1.4. limit changes in center position and radius
        #     # Step 3.1.5. check fit plausability / quality
        #     # Step 4.     if fit is good, add circle to tree
        #     tree["circles"].append(circ_iter)

        # # Step 3.2.   Direction down
        # i_iter = i_base
        # while i_iter > 1:
        #     i_iter -= 1
        #     # Step 3.2.1. initialize a circle directly below the previous one with
        #     #             the previous radius
        #     circ_iter = deepcopy(tree["circles"][0])
        #     circ_iter.update_height(slice_heights[i_iter])
        #     # Step 3.2.2. do not eliminate all points far from the hough circle, but
        #     #             weigh them nonlinearly by the distance
        #     pc_slice_iter = pc_slices[i_iter]
        #     if pc_slice_iter.shape[0] < 5:
        #         continue
        #     filter_circle = filter_circles[i_iter]
        #     weights = np.abs(weighting_function(filter_circle.get_distance(pc_slice_iter)))
        #     # Step 3.2.3. fit it to the new point slice using the advanced
        #     #             circle-segment- algo
        #     # circ_iter = fit_circle_lm(circ_iter, pc_slice_iter, weights, slice_heights[i_iter])
        #     circ_iter = fit_circle_bullock(pc_slice_iter, slice_heights[i_iter])
        #     # Step 3.2.4. limit changes in center position and radius
        #     # Step 3.2.5. check fit plausability / quality
        #     # Step 4.     if fit is good, add circle to tree. Also update
        #     #             base_circle_index
        #     # append circ_iter at the beginning of the list
        #     tree["circles"].insert(0, circ_iter)
        #     tree["base_circle_index"] += 1
        # return tree

    def __str__(self) -> str:
        return f"Tree: {len(self.circles)} circles, {len(self.points) if self.points is not None else 0} points"
