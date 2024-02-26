from itertools import product
from typing import Iterable, List, Union, Tuple
import numpy as np
import open3d as o3d
from skimage.transform import hough_circle
from copy import deepcopy
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from plotly import graph_objects as go
from scipy.spatial import cKDTree
import trimesh
from digiforest_analysis.utils.matrix_calc import efficient_inv
from digiforest_analysis.utils.distances import pnts_to_axes_sq_dist


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
        grid_res: float = 0.02,
        min_radius: float = 0.05,
        max_radius: float = 0.5,
        point_ratio: float = 0.0,
        center_region: "Circle" = None,
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
        if len(points) == 0:
            if return_pixels_and_votes:
                return None, None, None, None
            else:
                return None

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

        if n_cells < 5:
            if return_pixels_and_votes:
                return None, None, None, None
            else:
                return None

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
            radius_mask = try_radii < 0.5 * n_cells
            votes_mask = vote_fraction > 0.50 * np.max(vote_fraction)
            mask = np.logical_and(radius_mask, votes_mask)
            if not np.any(mask):
                # return if there's no radii left
                if return_pixels_and_votes:
                    return None, None, None, None
                else:
                    return None
            hough_res = hough_res[mask]
            try_radii = try_radii[mask]

            hough_flattened = hough_res.reshape(hough_res.shape[0], -1)
            # print("######")
            # print(n_cells)
            # print(vote_fraction)
            # print(votes_mask)
            # print(hough_flattened.shape)
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
            # normalize entropy to be between 1/entropy_weighting and 1 given max reward
            # of entropy_weighting
            entropy_range = np.max(top_10_entropy) - np.min(top_10_entropy)
            if entropy_range < 1e-12:
                penalty = np.ones_like(top_10_entropy)
            else:
                penalty = (
                    1 / entropy_weighting
                    + (1 - 1 / entropy_weighting)
                    * (top_10_entropy - np.min(top_10_entropy))
                    / entropy_range
                )
            hough_res /= penalty[:, None, None]

        # constrain circles to be roughly above previous one
        if center_region is not None:
            previous_center = center_region.center[:2]
            search_radius = center_region.radius
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
        fixed_radius: float = None,
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
        if fixed_radius is None:
            # normalize points
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
        else:
            r = fixed_radius
            A = np.hstack([2 * points, np.ones((points.shape[0], 1))])
            b = np.sum(np.power(points, 2), axis=1) - np.power(r, 2)
            c = np.linalg.lstsq(A, b, rcond=None)[0]
            x_c, y_c = c[:2]
            return cls((x_c, y_c, circle_height), r)

    @classmethod
    def from_cloud_ransac(
        cls,
        points: np.ndarray,
        min_radius: float = 0.0,
        max_radius: float = np.inf,
        n_iterations: int = 100,
        min_n_inliers: int = 10,
        center_region: "Circle" = None,
        inlier_threshold: float = 0.01,
        circle_height: float = 0.0,
        sampling: str = "weighted",
        **kwargs,
    ) -> Tuple["Circle", np.ndarray]:
        """This function filters outliers using the RANSAC algorithm. It fits a
        circle to a random sample of n_points points and counts how many points are
        inliers. If the number of inliers is larger than min_n_inliers, the circle is
        accepted and the number of inliers is updated. The algorithm is repeated
        n_iterations times.

        Args:
            points (np.ndarray): N x 2 array of points in the slice
            n_iterations (int): Number of iterations to run the algorithm
            n_points (int): Number of points used to model the circle
            min_n_inliers (int): Minimum number of inliers to accept the circle
            inlier_threshold (float): thickness of band around circle to qualify
                                    inliers

        Returns:
            tuple: best model Circle and inlier points if model was found, else None
        """
        best_model = None
        points = points[:, :2]

        if sampling == "weighted":
            # construct weights by distance to closest neighbor i.e. local density
            dist_to_closest_neighbor = cKDTree(points).query(points, k=2)[0][:, 1]
            probas = np.exp(-dist_to_closest_neighbor)
            probas -= probas.min() - 1e-12
            probas /= probas.max()

        N_circs = n_iterations

        if sampling == "weighted":
            indices = (
                (1 - np.random.rand(N_circs, len(points)) * probas)
            ).argpartition(3, axis=1)[:, :3]
        elif sampling == "random":
            indices = ((1 - np.random.rand(N_circs, len(points)))).argpartition(
                3, axis=1
            )[:, :3]
        elif sampling == "full":
            indices = np.array(list(product(range(len(points)), repeat=3)))
            mask = np.logical_and(
                indices[:, 0] != indices[:, 1],
                indices[:, 0] != indices[:, 2],
                indices[:, 1] != indices[:, 2],
            )
            mask = np.logical_and(mask, indices[:, 1] != indices[:, 2])
            indices = indices[mask]
        else:
            raise ValueError("sampling must be one of 'weighted', 'random', 'full'")

        circle_points = points[indices]  # num_samples x 3 x 2

        circle_points = points[indices]  # n_iters x 3 x 2
        circles = cls.from_3_2d_points(circle_points, method="bisectors")  # n_iters x 3

        circles = circles[circles[:, 2] < max_radius]
        circles = circles[circles[:, 2] > min_radius]
        if center_region is not None:
            dists = np.linalg.norm(circles[:, :2] - center_region.center[:2], axis=1)
            circles = circles[dists < center_region.radius]

        dists = np.linalg.norm(points[None, ...] - circles[:, None, :2], axis=2).T
        dists -= circles[:, 2][None, :]
        inlier_mask = np.logical_and(
            dists < inlier_threshold, dists > -inlier_threshold
        )
        n_inliers = np.sum(inlier_mask, axis=0)
        circles = circles[n_inliers > min_n_inliers]
        n_inliers = n_inliers[n_inliers > min_n_inliers]
        if len(circles) == 0:
            return None

        best_model = circles[np.argmax(n_inliers)]
        best_model = cls((best_model[0], best_model[1], circle_height), best_model[2])
        return best_model

    @classmethod
    def from_cloud_ransahc(
        cls,
        points,
        min_radius: float = 0.0,
        max_radius: float = np.inf,
        center_region: "Circle" = None,
        max_circles: int = 500,
        search_radius: float = 0.05,
        sample_percentage: float = 0.01,
        sampling: str = "weighted",
        circle_height: float = 0.0,
        **kwargs,
    ) -> "Circle":
        if len(points) < 10:
            return None
        points = points[:, :2]
        # sampling triplets of points
        if sampling == "weighted":
            # construct weights by distance to closest neighbor i.e. local density
            dist_to_closest_neighbor = cKDTree(points).query(points, k=2)[0][:, 1]
            probas = np.exp(-dist_to_closest_neighbor)
            probas -= probas.min() - 1e-12
            probas /= probas.max()

        N_points = points.shape[0]
        N_circs = int(N_points * (N_points - 1) * (N_points - 2) * sample_percentage)
        N_circs = max(N_points, min(N_circs, max_circles))
        N_circs = min(max(N_points, N_circs), max_circles)

        if sampling == "weighted":
            indices = (
                (1 - np.random.rand(N_circs, len(points)) * probas)
            ).argpartition(3, axis=1)[:, :3]
        elif sampling == "random":
            indices = ((1 - np.random.rand(N_circs, len(points)))).argpartition(
                3, axis=1
            )[:, :3]
        elif sampling == "full":
            indices = np.array(list(product(range(len(points)), repeat=3)))
            mask = np.logical_and(
                indices[:, 0] != indices[:, 1],
                indices[:, 0] != indices[:, 2],
                indices[:, 1] != indices[:, 2],
            )
            mask = np.logical_and(mask, indices[:, 1] != indices[:, 2])
            indices = indices[mask]
        else:
            raise ValueError("sampling must be one of 'weighted', 'random', 'full'")

        circle_points = points[indices]  # num_samples x 3 x 2

        # fitting circles to triplets
        circles = cls.from_3_2d_points(circle_points, method="bisectors")

        # filter circles using constraints
        circles = circles[circles[:, 2] < max_radius]
        circles = circles[circles[:, 2] > min_radius]
        if center_region is not None:
            dists = np.linalg.norm(circles[:, :2] - center_region.center[:2], axis=1)
            circles = circles[dists < center_region.radius]

        if len(circles) == 0:
            return None

        # hough consensus using kdtree

        # spheres = circles[0][None, :]
        # sphere_members = [[0]]
        # kdtree = cKDTree(spheres)
        # for i_circle in range(1, len(circles)):
        #     dist, index = kdtree.query(circles[i_circle], k=1)
        #     if dist < search_radius:
        #         sphere_members[index].append(i_circle)
        #     else:
        #         spheres = np.vstack((spheres, circles[i_circle]))
        #         sphere_members.append([i_circle])
        #         kdtree = cKDTree(spheres)
        # best_sphere = sphere_members[np.argmax([len(sm) for sm in sphere_members])]
        # best_circles = circles[best_sphere]

        query_circles = circles.copy()
        query_circles[:, 2] *= 2
        kdtree = cKDTree(query_circles)
        neighbors = kdtree.query_ball_tree(kdtree, r=search_radius)
        best_index = neighbors.index(max(neighbors, key=len))
        best_circle_inds = neighbors[best_index] + [best_index]
        best_circles = circles[best_circle_inds]

        # # fitting quality
        # with timer("resampling circles"):
        #     dists = np.linalg.norm(slice - circles[:, None, :2], axis=2) # num_samples x len(slice)
        #     dists = dists - circles[:, 2][:, None]
        #     dists[dists < 0] *= -10
        #     dists = dists.mean(axis=1)
        #     probas = np.exp(-dists)
        #     circles = circles[np.random.rand(circles.shape[0]) < probas]

        # # 3d plot points and color by number of neighbors
        # fig = go.Figure(data=[
        #     go.Scatter3d(
        #         x=circles[:, 0],
        #         y=circles[:, 1],
        #         z=circles[:, 2],
        #         mode='markers',
        #         marker=dict(
        #             size=2,
        #             color="black",                # set color to an array/list of desired values
        #             colorscale='Viridis',   # choose a colorscale
        #             opacity=0.8
        #         )
        #     ),
        #     go.Scatter3d(
        #         x=best_circles[:, 0],
        #         y=best_circles[:, 1],
        #         z=best_circles[:, 2],
        #         mode='markers',
        #         marker=dict(
        #             size=3,
        #             color='red',                # set color to an array/list of desired values
        #             opacity=1
        #         )
        #     ),
        # ])
        # fig['layout']['showlegend'] = False
        # fig.show()

        # best_circle = np.average(best_circles, weights=probas[best_circle_inds], axis=0)
        best_circle = best_circles.mean(axis=0)

        return cls(
            center=(best_circle[0], best_circle[1], circle_height),
            radius=best_circle[2],
        )

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

    @classmethod
    def from_3_2d_points(cls, points, method="bisectors"):
        # from https://qc.edu.hk/math/Advanced%20Level/circle%20given%203%20points.htm
        if points.shape[0] != 3:
            assert points.shape[1] == 3, "each batch must contain 3 points"
        else:
            assert points.shape[0] == 3, "points must contain 3 points"
            points = points[None, ...]
        if method == "bisectors":
            # intersecting bisectors method
            p0 = (points[:, 0, :] + points[:, 1, :]) / 2
            p1 = (points[:, 1, :] + points[:, 2, :]) / 2
            v0 = points[:, 1, :] - points[:, 0, :]
            v0 = np.flip(v0, axis=1) * np.array([1, -1])
            v1 = points[:, 2, :] - points[:, 1, :]
            v1 = np.flip(v1, axis=1) * np.array([1, -1])
            with np.errstate(divide="ignore", invalid="ignore"):
                alpha = (
                    p1[:, 1] - p0[:, 1] + (p0[:, 0] - p1[:, 0]) * v1[:, 1] / v1[:, 0]
                )
                alpha /= v0[:, 1] - v0[:, 0] * v1[:, 1] / v1[:, 0]
            x = p0[:, 0] + alpha * v0[:, 0]
            y = p0[:, 1] + alpha * v0[:, 1]
            r = np.sqrt((x - points[:, 0, 0]) ** 2 + (y - points[:, 0, 1]) ** 2)
        elif method == "determinant":
            # determinant method
            M = np.zeros((points.shape[0], 3, 4))
            M[:, 0, 0] = points[:, 0, 0] ** 2 + points[:, 0, 1] ** 2
            M[:, 0, 1] = points[:, 0, 0]
            M[:, 0, 2] = points[:, 0, 1]
            M[:, 0, 3] = 1
            M[:, 1, 0] = points[:, 1, 0] ** 2 + points[:, 1, 1] ** 2
            M[:, 1, 1] = points[:, 1, 0]
            M[:, 1, 2] = points[:, 1, 1]
            M[:, 1, 3] = 1
            M[:, 2, 0] = points[:, 2, 0] ** 2 + points[:, 2, 1] ** 2
            M[:, 2, 1] = points[:, 2, 0]
            M[:, 2, 2] = points[:, 2, 1]
            M[:, 2, 3] = 1
            denom = np.linalg.det(M[:, :, 1:])
            x = 0.5 * np.linalg.det(M[:, :, [0, 2, 3]]) / denom
            y = -0.5 * np.linalg.det(M[:, :, [0, 1, 3]]) / denom
            r = np.sqrt(x**2 + y**2 + np.linalg.det(M[:, :, :3]) / denom)
        else:
            raise ValueError(f"method {method} not supported")
        circles = np.stack([x, y, r], axis=1)
        if points.shape[0] == 1:
            return cls((circles[0, 0], circles[0, 1], 0), circles[0, 2])
        else:
            return circles

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
        basic_tri_1 = np.array([0, num_verts, 1])  # these triangles repeats 100 times
        basic_tri_2 = np.array([1, num_verts, num_verts + 1])
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
    def __init__(self, id: int, place_holder_height: float = 5) -> None:
        self.id = id
        self.place_holder_height = place_holder_height

        self.reconstructed = False
        self.circles: List[Circle] = None
        self.canopy_mesh = None

        self.clusters = []
        self.DBH = None
        self.number_bends = None
        self.clear_wood = None

        self.num_clusters_after_last_reco = 0
        self.cosys_changed_after_last_reco = False

        self.hue = np.random.rand()
        self.dbh = None

    def add_cluster(self, cluster: dict):
        self.clusters.append(cluster)

    @property
    def axis(self) -> dict:
        """Returns Axis of the tree in the MAP FRAME. The axis is calculated as the
        mean of all axis parameters of the clusters.

        Raises:
            ValueError: If there are no clusters

        Returns:
            dict: mean axis
        """
        if len(self.clusters) == 0:
            raise ValueError("No measurements available yet.")

        cluster_trafos_odom = [
            c["info"]["T_sensor2map"] @ c["info"]["axis"]["transform"]
            for c in self.clusters
        ]

        mean_center = np.mean([c[:3, 3] for c in cluster_trafos_odom], axis=0)
        mean_radius = np.mean(
            [c["info"]["axis"]["radius"] for c in self.clusters], axis=0
        )

        # mean rotation is a bit trickier, scipy takes care of that
        rotation_stack = Rotation.from_matrix([c[:3, :3] for c in cluster_trafos_odom])
        mean_rotation = rotation_stack.mean().as_matrix()

        mean_T = np.eye(4)
        mean_T[:3, :3] = mean_rotation
        mean_T[:3, 3] = mean_center

        return {
            "transform": mean_T,
            "radius": mean_radius,
        }

    @property
    def points(self):
        if len(self.clusters) == 0:
            raise ValueError("No measurements available yet.")

        # align points using axes
        ax_locs_map = np.stack(
            (c["info"]["T_sensor2map"] @ c["info"]["axis"]["transform"][:, 3])
            for c in self.clusters
        )
        ax_locs_axis = (efficient_inv(self.axis["transform"]) @ ax_locs_map.T).T
        delta_translations_axis = np.concatenate(
            (ax_locs_axis[:, :2], np.zeros((len(ax_locs_axis), 2))), axis=1
        )
        delta_translations_map = (self.axis["transform"] @ delta_translations_axis.T).T

        # transform all points to odom frame and then stack them
        return np.vstack(
            [
                cluster["cloud"].point.positions.numpy()
                @ cluster["info"]["T_sensor2map"][:3, :3].T
                + cluster["info"]["T_sensor2map"][:3, 3]
                - delta[:3]
                for cluster, delta in zip(self.clusters, delta_translations_map)
            ]
        )

    def transform_circles(self, translation: np.ndarray, rotation: np.ndarray):
        """applies the transform to all member objects of this tree.

        Args:
            translation (np.ndarray): 3x1 translation vector
            rotation (np.ndarray): Either 3x3 rotation matrix or 4x1 quaternion

        Raises:
            ValueError: if rotation is not given as 3x3 matrix or 4x1 quaternion
        """
        if self.reconstructed:
            for i in range(len(self.circles)):
                self.circles[i].apply_transform(translation, rotation)

    def evaluate_at_height(self, height: float):
        circ_interp = None
        if not self.reconstructed:
            return circ_interp

        # extrapolate up to one slice interval under first slice
        first_frustum_height = self.circles[1].center[2] - self.circles[0].center[2]
        if -first_frustum_height < height - self.circles[0].center[2] < 0:
            alpha = (self.circles[0].center[2] - height) / first_frustum_height
            center = self.circles[0].center + alpha * (
                self.circles[0].center - self.circles[1].center
            )
            radius = self.circles[0].radius + alpha * (
                self.circles[0].radius - self.circles[1].radius
            )
            if np.any(np.isnan(center)) or np.isnan(radius):
                return None
            return Circle(center, radius)

        # interpolate elsewhere
        for i_circ in range(len(self.circles) - 1):
            if (
                height < self.circles[i_circ + 1].center[2]
                and height > self.circles[i_circ].center[2]
            ):
                alpha = (height - self.circles[i_circ].center[2]) / (
                    self.circles[i_circ + 1].center[2] - self.circles[i_circ].center[2]
                )
                center = (1 - alpha) * self.circles[
                    i_circ
                ].center + alpha * self.circles[i_circ + 1].center
                radius = (1 - alpha) * self.circles[
                    i_circ
                ].radius + alpha * self.circles[i_circ + 1].radius
                if np.any(np.isnan(center)) or np.isnan(radius):
                    return None
                return Circle(center, radius)

    def apply_transform(self, translation: np.ndarray, rotation: np.ndarray):
        """applies the transform to all member objects of this tree.

        Args:
            translation (np.ndarray): 3x1 translation vector
            rotation (np.ndarray): Either 3x3 rotation matrix or 4x1 quaternion

        Raises:
            ValueError: if rotation is not given as 3x3 matrix or 4x1 quaternion
        """
        self.transform_circles(translation, rotation)
        T = np.eye(4)
        if rotation.shape[0] == 4:
            T[:3, :3] = Rotation.from_quat(rotation).as_matrix()
        elif rotation.shape == (3, 3):
            T[:3, :3] = rotation
        else:
            raise ValueError("rotation must be given as 3x3 matrix or quaternion")
        T[:3, 3] = translation
        for i in range(len(self.clusters)):
            self.clusters[i]["info"]["axis"]["transform"] = (
                T @ self.clusters[i]["info"]["axis"]["transform"]
            )

    def reconstruct(  # noqa: C901
        self,
        slice_heights: Union[float, Iterable] = 0.5,
        slice_thickness: float = 0.3,
        outlier_radius: float = 0.02,
        max_center_deviation: float = 0.05,
        max_radius_deviation: float = 0.05,
        filter_min_points: int = 10,
        max_consecutive_fails: int = 3,
        max_height: float = 15.0,
        debug_level: int = 0,
        **kwargs,
    ) -> bool:
        """slices the given point cloud at regular intervals or given intervals and
        vilters the slices using the hough transform. After filtering, circles are
        fit to the slices using a least squares fitting. The circles are then

        Args:
            slice_heights (Union[float, Iterable]): if a float [m] is given, the
                point cloud is sliced at regular intervals. If an iterable is given,
                the point cloud is sliced at the given heights. Defaults to 0.5.
            slice_thickness (float, optional): Thickness of the slices in m.
            outlier_radius (float, optional): Least distance from the filter circle
                for a point to be considered an outlier. Defaults to 0.02.
            max_center_deviation (float, optional): Maximal deviation of a circle's
                radius wrt. to the previous circle beneath. Defaults to 0.05.
            max_radius_deviation (float, optional): Maximal deviation of a circle's
                center point wrt. to the previous circle beneath as a fraction.
                Defaults to 0.05.
            filter_min_points (int, optional): Minimum number of points in a slice
                to fit a circle. Defaults to 10.
            max_consecutive_fails (int, optional): Maximum number of consecutive
                slices that fail to fit a circle before the algorithm stops. Defaults
                to 3.
            max_height (float, optional): Maximum height until where the tree is
                fitted. Defaults to 10.0.
            save_points (bool, optional): If True, the points for each slice are saved
                in the tree object. Defaults to False.
            debug_level (int, optional): Verbosity level of debug msgs. Defaults to 0.

        Returns:
            bool: True if reconstruction yielded at least two circles, else False
        """
        self._slice_points = []
        self._hough_points = []
        self._hough_circles = []

        cloud = self.points
        center = self.axis["transform"][:3, 3]
        rot_mat = self.axis["transform"][:3, :3]
        r_from_seg = self.axis["radius"]

        # move cloud to origin and rotate it upright
        cloud -= center
        cloud = cloud @ rot_mat  # equal to: cloud = (rot_mat.T @ cloud.T).T

        circle_stack = []
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

            search_cylinder_radius = 2 * r_from_seg
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
                if debug_level > 0:
                    print(f"Too few points ({len(slice_points)}) to fit a circle (1)")
                fail_counter += 1
                continue

            # fit hough circle to all points in the slice
            if len(circle_stack) == 0:
                center_region = None
            else:
                center_region = Circle(
                    circle_stack[-1]["circle"].center, max_radius_deviation
                )
                min_radius = circle_stack[-1]["circle"].radius * (
                    1 - max_radius_deviation
                )
                max_radius = circle_stack[-1]["circle"].radius * (
                    1 + max_radius_deviation
                )
            min_radius = 0.75 * r_from_seg
            max_radius = 1.5 * r_from_seg

            hough_circle = Circle.from_cloud_ransahc(
                slice_points,
                min_radius,
                max_radius,
                center_region,
                circle_height=slice_height,
            )

            if hough_circle is None:
                if debug_level > 0:
                    print("No hough circle found")
                fail_counter += 1
                continue

            # filter points using the hough circle
            filter_mask = hough_circle.get_distance(slice_points) < outlier_radius
            pc_filtered = slice_points[filter_mask]

            # again, check if there are enough points to fit a circle. This is not a
            # redundant check as the check before will often spare expensive hough
            # calculations, whereas this check is necessary to fit a reasonabole
            # circle
            if len(pc_filtered) < filter_min_points:
                if debug_level > 0:
                    print(f"Too few points ({len(pc_filtered)}) to fit a circle (2)")
                fail_counter += 1
                continue

            # fit circle to filtered points
            bullock_circle = Circle.from_cloud_bullock(pc_filtered, slice_height)

            # TODO check if circle is plausible. For now, just impose max radius of 1 m
            if bullock_circle.radius > 1.0:
                if debug_level > 0:
                    print("Radius too large")
                fail_counter += 1
                continue

            if debug_level > 0:
                print(f"Found circle at height {slice_height:.2f} m")

            # aggregate results
            circle_stack.append(
                {
                    "num_points": pc_filtered.shape[0],
                    "circle": bullock_circle,
                    "slice_points": slice_points,
                    "hough_circle": hough_circle,
                    "hough_points": pc_filtered,
                }
            )

            fail_counter = 0

        if len(circle_stack) < 2:
            if debug_level > 0:
                print("Not enough circles found")
            self.circles = []
            self.reconstructed = False
            return False
        else:
            self.circles = [t["circle"] for t in circle_stack]
            self.reconstructed = True
            # reapply rotation and translation
            self.transform_circles(center, rot_mat)
            return True

    def reconstruct2(  # noqa: C901
        self,
        max_height: float = None,
        slice_heights: Union[float, Iterable] = 1.0,
        slice_thickness: float = 0.2,
        max_center_deviation: float = 0.1,
        force_straight: bool = False,
        max_radius: float = np.inf,
    ):
        circle_stack = []
        cluster_points_map = [
            cluster["cloud"].point.positions.numpy()
            @ cluster["info"]["T_sensor2map"][:3, :3].T
            + cluster["info"]["T_sensor2map"][:3, 3]
            for cluster in self.clusters
        ]
        cluster_points_upright = [
            (points - self.axis["transform"][:3, 3]) @ self.axis["transform"][:3, :3]
            for points in cluster_points_map
        ]
        if type(slice_heights) == float:
            min_point = np.min([cpu.min() for cpu in cluster_points_upright])
            max_point = np.max([cpu.max() for cpu in cluster_points_upright])
            slice_heights = np.arange(
                min_point,
                max_height if max_height is not None else max_point,
                slice_heights,
            )

        fail_counter = 0  # counts how many slices failed to fit a circle
        init_candidates = []
        num_init_candidates = 3
        i_slice_height = 0

        while i_slice_height < len(slice_heights):  # len of slice_heights is changed
            slice_height = slice_heights[i_slice_height]
            if fail_counter == 3:
                break
            ransahc_circles = []
            scores = []
            point_counts = []
            debug_slice_points = []
            if len(circle_stack) == 0:
                center_region = None
                radius_lb = 0.5 * self.axis["radius"]
                radius_ub = min(
                    3 * self.axis["radius"], max_radius if max_radius else np.inf
                )
            else:
                center_region = Circle(circle_stack[-1].center, max_center_deviation)
                radius_lb = 0.9 * circle_stack[-1].radius
                radius_ub = min(
                    1.05 * circle_stack[-1].radius, max_radius if max_radius else np.inf
                )

            for i_cluster, cluster_points in enumerate(cluster_points_upright):
                # print(f"Cluster Number {i_cluster}")
                slice_points = cluster_points[
                    np.logical_and(
                        cluster_points[:, 2] >= slice_height - slice_thickness / 2,
                        cluster_points[:, 2] < slice_height + slice_thickness / 2,
                    )
                ]
                debug_slice_points.append(slice_points)
                if len(slice_points) < 10:
                    continue

                ransahc_circle = Circle.from_cloud_ransahc(
                    slice_points,
                    radius_lb,
                    radius_ub,
                    center_region,
                    circle_height=slice_height,
                )
                if ransahc_circle is None:
                    continue
                ransahc_circles.append(ransahc_circle)

                dists_center = np.linalg.norm(
                    slice_points[:, :2] - ransahc_circle.center[:2], axis=1
                )
                dists_circle = np.abs(dists_center - ransahc_circle.radius)

                score = np.sum(dists_circle < 0.05 * ransahc_circle.radius) / np.sum(
                    dists_center < 1.05 * ransahc_circle.radius
                )

                point_count = np.sum(dists_circle < 0.05 * ransahc_circle.radius)
                scores.append(score)
                point_counts.append(point_count)

            if len(ransahc_circles) == 0:
                i_slice_height += 1
                fail_counter += 1
                continue

            if np.sum(scores) < 1e-12:
                i_slice_height += 1
                fail_counter += 1
                continue

            if np.mean(scores) < 0.33333:
                fail_counter += 1
                i_slice_height += 1
                continue

            # from matplotlib import pyplot as plt
            # import colorsys
            # fig, ax = plt.subplots()
            # ax.set_aspect("equal", adjustable="box")
            # ax.set_xlabel("x")
            # ax.set_ylabel("y")
            # ax.set_title(f"Slice at height {slice_height:.2f} m")
            # hues = np.linspace(0, 1, len(debug_slice_points))
            # for points, circle, hue in zip(debug_slice_points, ransahc_circles, hues):
            #     color = colorsys.hls_to_rgb(hue, 0.45, 0.8)
            #     ax.scatter(
            #         points[:, 0],
            #         points[:, 1],
            #         c=[color] * len(points),
            #         s=1,
            #     )
            #     ax.add_artist(
            #         plt.Circle(
            #             (circle.x, circle.y),
            #             circle.radius,
            #             color=color,
            #             fill=False,
            #         )
            #     )
            # ax.scatter([0], [0], c="black", s=10)
            # fig.show()
            # plt.show()

            if len(circle_stack) == 0:
                if len(init_candidates) == 0:
                    # add additional slice heights for NMS candidates here
                    candidate_heights = np.linspace(
                        slice_heights[i_slice_height],
                        slice_heights[i_slice_height + 1],
                        num_init_candidates + 1,
                    )[1:-1]
                    for candidate_height in candidate_heights[::-1]:
                        slice_heights = np.insert(
                            slice_heights, i_slice_height + 1, candidate_height
                        )
                if len(init_candidates) < num_init_candidates:
                    # aggregate results
                    init_candidates.append(
                        {
                            "circles": np.array(ransahc_circles, dtype=object),
                            "scores": np.array(scores),
                            "point_counts": np.array(point_counts),
                        }
                    )
                    i_slice_height += 1
                if len(init_candidates) == num_init_candidates:
                    init_circles = []
                    init_scores = []
                    # evaluate results aka supress non-maxima
                    for init_candidate in init_candidates:
                        mask = init_candidate["point_counts"] > 0.5 * np.max(
                            init_candidate["point_counts"]
                        )  #
                        average_center = np.average(
                            [c.center for c in init_candidate["circles"][mask]],
                            weights=init_candidate["scores"][mask],
                            axis=0,
                        )
                        average_radius = np.average(
                            [c.radius for c in init_candidate["circles"][mask]],
                            weights=init_candidate["scores"][mask],
                            axis=0,
                        )
                        total_point_count = np.sum(init_candidate["point_counts"][mask])
                        init_circles.append(Circle(average_center, average_radius))
                        init_scores.append(
                            np.mean(init_candidate["scores"][mask]) * total_point_count
                        )
                    # add the best candidate to the stack
                    circle_stack.append(init_circles[np.argmax(init_scores)])
                    # print(f"found init, using {np.argmax(init_scores)}")
                    i_slice_height += 1
            else:
                # aggregate models
                average_center = np.average(
                    [c.center for c in ransahc_circles], weights=scores, axis=0
                )
                average_radius = np.average(
                    [c.radius for c in ransahc_circles], weights=scores, axis=0
                )
                average_circle = Circle(average_center, average_radius)
                # mean_r_prev_3 = np.mean([c.radius for c in circle_stack[-3:]], axis=0)
                # if average_circle.radius > mean_r_prev_3:
                #     fail_counter += 1
                #     average_circle.radius = mean_r_prev_3

                circle_stack.append(average_circle)
                i_slice_height += 1
            fail_counter = 0

        if len(circle_stack) < 2:
            self.reconstructed = False
            return False

        if force_straight:
            connecting_vecs = np.diff([c.center for c in circle_stack], axis=0)
            connecting_vecs = (
                connecting_vecs / np.linalg.norm(connecting_vecs, axis=1)[:, None]
            )
            mean_dir = np.mean(connecting_vecs, axis=0)
            for circle in circle_stack:
                alpha = circle.center[2] / mean_dir[2]
                circle.center = circle_stack[0].center + mean_dir * alpha
        self.circles = circle_stack
        self.reconstructed = True
        self.transform_circles(
            self.axis["transform"][:3, 3], self.axis["transform"][:3, :3]
        )
        return True

    def _filter_slice(
        self,
        sliced_clusters,
        radius_lb,
        radius_ub,
        center_region,
        slice_height,
        filter_radius,
    ):
        ransahc_circles = []
        scores = []
        point_counts = []
        remove_inds = []
        for i_cluster, points in enumerate(sliced_clusters):
            # too few points
            if len(points) < 10:
                remove_inds.append(i_cluster)
                continue

            # fit circle
            # filter_circle = Circle.from_cloud_ransahc(
            #     points,
            #     min_radius=radius_lb,
            #     max_radius=radius_ub,
            #     center_region=center_region,
            #     circle_height=slice_height,
            # )
            # filter_circle = Circle.from_cloud_hough(
            #     points,
            #     min_radius=radius_lb,
            #     max_radius=radius_ub,
            #     center_region=center_region,
            #     entropy_weighting=0.0,
            #     circle_height=slice_height,
            # )
            filter_circle = Circle.from_cloud_ransac(
                points,
                circle_height=slice_height,
                min_radius=radius_lb,
                max_radius=radius_ub,
                center_region=center_region,
                sampling="random",
                n_iterations=1000,
            )

            if filter_circle is None:
                remove_inds.append(i_cluster)
                continue  # no fit found
            ransahc_circles.append(filter_circle)

            # compute fitness score for circle fit
            dists_center = np.linalg.norm(
                points[:, :2] - filter_circle.center[:2], axis=1
            )
            dists_circle = np.abs(dists_center - filter_circle.radius)
            score = np.sum(dists_circle < 0.05 * filter_circle.radius) / np.sum(
                dists_center < 1.05 * filter_circle.radius
            )
            scores.append(score)
            point_count = np.sum(dists_circle < 0.05 * filter_circle.radius)
            point_counts.append(point_count)
        for i in remove_inds[::-1]:
            sliced_clusters.pop(i)

        # no fit found
        if len(ransahc_circles) == 0:
            return None

        # bad fit
        if np.mean(scores) < 0.33333 or np.sum(scores) < 1e-12:
            return None

        # filter points using the ransahc circle
        remove_inds = []
        for i_cluster in range(len(sliced_clusters)):
            filter_circle: Circle = ransahc_circles[i_cluster]
            filter_mask = (
                filter_circle.get_distance(sliced_clusters[i_cluster]) < filter_radius
            )
            pc_filtered = sliced_clusters[i_cluster][filter_mask]
            if len(pc_filtered) < 10:
                remove_inds.append(i_cluster)
            sliced_clusters[i_cluster] = pc_filtered
        for i in remove_inds[::-1]:
            sliced_clusters.pop(i)
            ransahc_circles.pop(i)
            scores.pop(i)
            point_counts.pop(i)
        if len(sliced_clusters) == 0:
            return None  # no slices left
        if np.sum(point_counts) < 10:
            return None  # to few points left for fitting

        return {
            "ransahc_circles": ransahc_circles,
            "scores": np.array(scores),
            "point_counts": np.array(point_counts),
            "filtered_points": deepcopy(sliced_clusters),
        }

    def reconstruct3(  # noqa: C901
        self,
        max_height: float = None,
        slice_heights: Union[float, Iterable] = 1.0,
        slice_thickness: float = 0.2,
        max_center_deviation: float = 0.1,
        force_straight: bool = False,
        max_radius: float = np.inf,
        filter_radius: float = 0.05,
    ):
        circle_stack = []
        cluster_points_map = [
            cluster["cloud"].point.positions.numpy()
            @ cluster["info"]["T_sensor2map"][:3, :3].T
            + cluster["info"]["T_sensor2map"][:3, 3]
            for cluster in self.clusters
        ]
        cluster_points_upright = [
            (points - self.axis["transform"][:3, 3]) @ self.axis["transform"][:3, :3]
            for points in cluster_points_map
        ]
        if type(slice_heights) == float:
            min_point = np.min([cpu.min() for cpu in cluster_points_upright])
            max_point = np.max([cpu.max() for cpu in cluster_points_upright])
            slice_heights = np.arange(
                min_point,
                max_height if max_height is not None else max_point,
                slice_heights,
            )
            if len(slice_heights) < 3:
                return False  # too few slices

        fail_counter = 0  # counts how many slices failed to fit a circle
        init_candidates = []
        num_init_candidates = 3
        i_slice_height = 0

        while i_slice_height < len(slice_heights):  # len of slice_heights is changed
            slice_height = slice_heights[i_slice_height]
            if fail_counter == 5:
                break

            # determine boundary conditions for ransahc fits
            if len(circle_stack) == 0:
                center_region = None
                radius_lb = 0.5 * self.axis["radius"]
                radius_ub = min(
                    3 * self.axis["radius"], max_radius if max_radius else np.inf
                )
            else:
                center_region = Circle(circle_stack[-1].center, max_center_deviation)
                radius_lb = 0.9 * circle_stack[-1].radius
                radius_ub = min(
                    1.05 * circle_stack[-1].radius, max_radius if max_radius else np.inf
                )

            # slice clusters
            sliced_clusters = [
                cluster_points[
                    np.logical_and(
                        cluster_points[:, 2] >= slice_height - slice_thickness / 2,
                        cluster_points[:, 2] < slice_height + slice_thickness / 2,
                    )
                ]
                for cluster_points in cluster_points_upright
            ]

            # try to fit hough circles to the slices
            slice_circles = self._filter_slice(
                sliced_clusters,
                radius_lb,
                radius_ub,
                center_region,
                slice_height,
                filter_radius,
            )
            if slice_circles is None:
                fail_counter += 1
                i_slice_height += 1
                continue

            # fit bulloc circles with fixed radius of mean ransahc circles
            weights = slice_circles["scores"] * slice_circles["point_counts"]
            if np.sum(weights) < 1e-12:
                fail_counter += 1
                i_slice_height += 1
                continue
            average_radius = np.average(
                [c.radius for c in slice_circles["ransahc_circles"]], weights=weights
            )
            slice_circles["bullock_circles"] = [
                Circle.from_cloud_bullock(
                    pc, fixed_radius=average_radius, circle_height=slice_height
                )
                for pc in slice_circles["filtered_points"]
            ]

            # from matplotlib import pyplot as plt
            # import colorsys

            # fig, ax = plt.subplots()
            # ax.set_aspect("equal", adjustable="box")
            # ax.set_xlabel("x")
            # ax.set_ylabel("y")
            # ax.set_title(f"Slice at height {slice_height:.2f} m")
            # hues = np.linspace(0, 1, len(slice_circles["filtered_points"]))
            # for points, circle, hue in zip(
            #     slice_circles["filtered_points"], slice_circles["bullock_circles"], hues
            # ):
            #     color = colorsys.hls_to_rgb(hue, 0.45, 0.8)
            #     ax.scatter(
            #         points[:, 0],
            #         points[:, 1],
            #         c=[color] * len(points),
            #         s=1,
            #     )
            #     ax.add_artist(
            #         plt.Circle(
            #             (circle.x, circle.y),
            #             circle.radius,
            #             color=color,
            #             fill=False,
            #         )
            #     )
            # ax.scatter([0], [0], c="black", s=10)
            # fig.show()
            # plt.show()

            # shift all circles to using the bulloc circles
            mean_center = np.mean(
                np.vstack([c.center for c in slice_circles["bullock_circles"]]), axis=0
            )
            shifts = mean_center - np.vstack(
                [c.center for c in slice_circles["bullock_circles"]]
            )
            for i in range(len(slice_circles["filtered_points"])):
                slice_circles["filtered_points"][i] += shifts[i]

            # fit final bullock circle
            points = np.vstack(slice_circles["filtered_points"])
            slice_circles["final_circle"] = Circle.from_cloud_bullock(
                points, circle_height=slice_height
            )

            # reject bad fits
            if np.any(np.isnan(slice_circles["final_circle"].center)):
                fail_counter += 1
                i_slice_height += 1
                continue
            if slice_circles["final_circle"].radius > max_radius:
                fail_counter += 1
                i_slice_height += 1
                continue
            if (
                len(circle_stack)
                and slice_circles["final_circle"].radius > 2 * circle_stack[-1].radius
            ):
                fail_counter += 1
                i_slice_height += 1
                continue

            # from matplotlib import pyplot as plt
            # import colorsys

            # fig, ax = plt.subplots()
            # ax.set_aspect("equal", adjustable="box")
            # ax.set_xlabel("x")
            # ax.set_ylabel("y")
            # ax.set_title(f"Slice at height {slice_height:.2f} m")
            # points = np.vstack(slice_circles["filtered_points"])
            # ax.scatter(
            #     points[:, 0],
            #     points[:, 1],
            #     s=1,
            # )
            # ax.add_artist(
            #     plt.Circle(
            #         (slice_circles["final_circle"].x, slice_circles["final_circle"].y),
            #         slice_circles["final_circle"].radius,
            #         color="r",
            #         fill=False,
            #     )
            # )
            # ax.scatter([0], [0], c="black", s=10)
            # fig.show()
            # plt.show()

            if len(circle_stack) == 0:
                if len(init_candidates) == 0:
                    if i_slice_height == len(slice_heights) - 1:
                        return False  # no init can be found, too few good slices
                    # add additional slice heights for NMS candidates here
                    candidate_heights = np.linspace(
                        slice_heights[i_slice_height],
                        slice_heights[i_slice_height + 1],
                        num_init_candidates + 1,
                    )[1:-1]
                    for candidate_height in candidate_heights[::-1]:
                        slice_heights = np.insert(
                            slice_heights, i_slice_height + 1, candidate_height
                        )
                if len(init_candidates) < num_init_candidates:
                    # aggregate results
                    init_candidates.append(
                        {
                            "circle": slice_circles["final_circle"],
                            "scores": slice_circles["scores"],
                            "point_counts": slice_circles["point_counts"],
                        }
                    )
                    i_slice_height += 1
                if len(init_candidates) == num_init_candidates:
                    init_circles = []
                    init_scores = []
                    # evaluate results aka supress non-maxima
                    for init_candidate in init_candidates:
                        init_circles.append(slice_circles["final_circle"])
                        init_scores.append(
                            np.mean(
                                init_candidate["scores"]
                                * init_candidate["point_counts"]
                            )
                        )
                    # add the best candidate to the stack
                    circle_stack.append(init_circles[np.argmax(init_scores)])
                    # print(f"found init, using {np.argmax(init_scores)}")
                    i_slice_height += 1
            else:
                # mean_r_prev_3 = np.mean([c.radius for c in circle_stack[-3:]], axis=0)
                # if average_circle.radius > mean_r_prev_3:
                #     fail_counter += 1
                #     average_circle.radius = mean_r_prev_3

                circle_stack.append(slice_circles["final_circle"])
                i_slice_height += 1
            fail_counter = 0

        if len(circle_stack) < 2:
            self.reconstructed = False
            return False

        if force_straight:
            connecting_vecs = np.diff([c.center for c in circle_stack], axis=0)
            connecting_vecs = (
                connecting_vecs / np.linalg.norm(connecting_vecs, axis=1)[:, None]
            )
            mean_dir = np.mean(connecting_vecs, axis=0)
            for circle in circle_stack:
                alpha = circle.center[2] / mean_dir[2]
                circle.center = circle_stack[0].center + mean_dir * alpha
        self.circles = circle_stack
        self.reconstructed = True
        self.transform_circles(
            self.axis["transform"][:3, 3], self.axis["transform"][:3, :3]
        )
        return True

    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """generates a mesh for the tree by connecting the circles with cone frustums

        Returns:
            np.ndarray: vertices of the mesh
            np.ndarray: triangle indices of the mesh
        """
        if self.reconstructed:
            vertices, triangles = np.empty((0, 3)), np.empty((0, 3), dtype=int)
            for i in range(len(self.circles) - 1):
                verts, tris = self.circles[i].genereate_cone_frustum_mesh(
                    self.circles[i + 1]
                )
                triangles = np.vstack((triangles, tris + vertices.shape[0]))
                vertices = np.vstack((vertices, verts))

            return vertices, triangles
        else:
            lower_center = self.axis["transform"][:3, 3]
            cylinder_radius = self.axis["radius"]
            cylinder_axis = self.axis["transform"][:3, 2]
            cylinder_height = self.place_holder_height
            upper_center = lower_center + cylinder_axis * cylinder_height

            bottom_circle = Circle(lower_center, cylinder_radius, cylinder_axis)
            top_circle = Circle(upper_center, cylinder_radius, cylinder_axis)

            retval = bottom_circle.genereate_cone_frustum_mesh(top_circle)
            mesh = trimesh.Trimesh(vertices=retval[0], faces=retval[1])
            return mesh.vertices, mesh.faces

    def generate_canopy(self):
        if not self.reconstructed:
            return False
        # filter out points close to floor (by map z axis)
        canopy_points = self.points[self.points[:, 2] > 2.0]
        if len(canopy_points) < 10:
            return False
        # filter out points close to reconstruction
        axis = np.concatenate(
            (self.axis["transform"][:3, 2], self.axis["transform"][:3, 3])
        )
        sq_dists = pnts_to_axes_sq_dist(canopy_points, axis[None, :]).flatten()
        mask = np.logical_or(
            sq_dists > (2 * self.axis["radius"]) ** 2,
            canopy_points[:, 2] > np.max(canopy_points[:, 2]) - 0.5,
        )
        canopy_points = canopy_points[mask]

        if len(canopy_points) < 10:
            return False

        # make o3d point cloud from canopy points
        pcd = o3d.t.geometry.PointCloud(canopy_points)
        hull = pcd.compute_convex_hull()
        self.canopy_mesh = {
            "verts": hull.vertex.positions.numpy(),
            "tris": hull.triangle.indices.numpy(),
        }

    def compute_dbh(self, terrain_height):
        reco_dbh = None
        query_height = terrain_height + 1.3
        dbh_circle = self.evaluate_at_height(query_height)
        if dbh_circle is not None:
            reco_dbh = dbh_circle.radius * 2
            self.dbh = reco_dbh

    def __str__(self) -> str:
        if self.circles is not None:
            return f"Tree {self.id} with {len(self.circles)} circles and {len(self.clusters)} clusters"
        else:
            return f"Tree {self.id} with no circles and {len(self.clusters)} clusters"

    def __setstate__(self, state):
        if "dbh" not in state:
            state["dbh"] = None
        self.__dict__.update(state)
