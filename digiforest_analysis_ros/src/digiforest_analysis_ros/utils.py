from digiforest_analysis.utils.timing import Timer
import numpy as np
from scipy.spatial.transform import Rotation
import open3d as o3d
import rospy

from sensor_msgs.msg import PointCloud2
from digiforest_analysis.tasks.terrain_fitting import TerrainFitting
from digiforest_analysis.tasks.tree_segmentation_voronoi import TreeSegmentationVoronoi


def pose2T(orientation, position):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(
        np.array(
            [
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
            ]
        )
    ).as_matrix()
    T[:3, 3] = np.array([position.x, position.y, position.z])
    return T


def efficient_inv(T):
    assert T.shape == (4, 4), "T must be a 4x4 matrix"
    T_inv = np.eye(4)
    T_inv[:3, :3] = T[:3, :3].T
    T_inv[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return T_inv


def pc2_to_o3d(cloud: PointCloud2):
    cloud_numpy = np.frombuffer(cloud.data, dtype=np.float32).reshape(-1, 12)
    # clear points with at least one nan
    cloud_numpy = cloud_numpy[~np.isnan(cloud_numpy).any(axis=1)]
    cloud = o3d.t.geometry.PointCloud(cloud_numpy[:, :3])
    cloud.point.normals = cloud_numpy[:, 4:7]
    return cloud


def radius_crop_pc(
    cloud: o3d.t.geometry.PointCloud,
    center_pose: np.ndarray,
    radius: float,
):
    """Crops a point cloud using a max distance from the sensor. The sensor pose

    Args:
        cloud (o3d.t.geometry.PointCloud): Point cloud to be cropped
        sensor_pose (np.ndarray): 4x4 transformation matrix from sensor to odom
        radius (float): maximum distance from sensor in m
    """
    # Transform the point cloud to sensor frame
    cloud = cloud.transform(efficient_inv(center_pose))
    # Calculate the distance from the sensor
    distances = np.linalg.norm(cloud.point.positions.numpy()[:, :2], axis=1)
    cloud = cloud.select_by_mask(distances <= radius)
    cloud = cloud.transform(center_pose)

    return cloud


def clustering_worker_fun(
    tf_buffer,
    cloud_msg,
    terrain_enabled,
    terrain_smoothing,
    terrain_cloth_cell_size,
    clustering_crop_radius,
    path_odom,
    pose_graph_stamps,
    hough_filter_radius,
    crop_lower_bound,
    crop_upper_bound,
    max_cluster_radius,
    point_fraction,
    debug_level,
    map_frame_id,
    odom_frame_id,
):
    terrain_fitter = TerrainFitting(
        sloop_smooth=terrain_smoothing,
        cloth_cell_size=terrain_cloth_cell_size,
        debug_level=debug_level,
    )

    tree_segmenter = TreeSegmentationVoronoi(
        debug_level=debug_level,
    )

    # find index closest to center of path that also is in posegraph
    center_index = len(path_odom.poses) // 2
    center_index_found = False
    for i, step_size in enumerate(range(len(path_odom.poses))):
        center_index += step_size * (-1) ** i
        if path_odom.poses[center_index].header.stamp in pose_graph_stamps:
            center_index_found = True
            break
    if not center_index_found:
        rospy.logerr("Could not find any cloud's stamp in posegraph")
        return

    center_pose = path_odom.poses[center_index].pose
    center_stamp = path_odom.poses[center_index].header.stamp
    T_sensor2odom = pose2T(center_pose.orientation, center_pose.position)
    pose_odom2map = tf_buffer.lookup_transform(
        map_frame_id,
        odom_frame_id,
        rospy.Duration(0),
        rospy.Duration(1),
    )
    T_odom2map = pose2T(
        pose_odom2map.transform.rotation, pose_odom2map.transform.translation
    )
    T_sensor2map = T_odom2map @ T_sensor2odom

    timer = Timer()
    with timer("cw"):
        with timer("cw/conversion"):
            cloud = pc2_to_o3d(cloud_msg)
            cloud = radius_crop_pc(
                cloud,
                center_pose=T_sensor2odom,
                radius=clustering_crop_radius,
            )
        if terrain_enabled:
            with timer("cw/terrain"):
                terrain = terrain_fitter.process(cloud=cloud)
        else:
            terrain = None

        with timer("cw/clustering"):
            clusters = tree_segmenter.process(
                cloud=cloud,
                cloth=terrain,
                hough_filter_radius=hough_filter_radius,
                crop_lower_bound=crop_lower_bound,
                crop_upper_bound=crop_upper_bound,
                max_cluster_radius=max_cluster_radius,
                point_fraction=point_fraction,
            )
        # convert clusters into stamped sensor frame
        with timer("cw/transform"):
            for i in range(len(clusters)):
                clusters[i]["cloud"].transform(efficient_inv(T_sensor2odom))
                clusters[i]["info"]["axis"]["transform"] = (
                    efficient_inv(T_sensor2odom)
                    @ clusters[i]["info"]["axis"]["transform"]
                )
                clusters[i]["info"]["T_sensor2map"] = T_sensor2map
                clusters[i]["info"]["time_stamp"] = center_stamp

    rospy.loginfo(f"Timing results of clustering:\n{timer}")

    return clusters, terrain


def apply_transform(
    points: np.ndarray,
    translation: np.ndarray,
    rotation: np.ndarray,
    inverse: bool = False,
) -> np.ndarray:
    """Transforms the given points by the given translation and rotation.
    Optionally performs the inverse transformation.

    Args:
        points (np.ndarray): Nx3 array of points to be transformed
        translation (np.ndarray): 3x1 array of translation
        rotation (np.ndarray): 3x3 rotation matrix or 4x1 quaternion
        inverse (bool, optional): Flag to calculate inverse. Defaults to False.

    Returns:
        np.ndarray: Nx3 array of transformed points
    """
    points = points.copy()
    if rotation.shape[0] == 4:
        rot_mat = Rotation.from_quat(rotation).as_matrix()
    elif rotation.shape == (3, 3):
        rot_mat = rotation
    else:
        raise ValueError("rotation must be given as 3x3 matrix or quaternion")
    if inverse:
        points = (points - translation) @ rot_mat
    else:
        points = points @ rot_mat.T + translation

    return points


def set_axes_equal(ax):
    # from https://stackoverflow.com/questions/13685386/how-to-set-the-equal-aspect-ratio-for-all-axes-x-y-z
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
