from digiforest_analysis.utils.timing import Timer
import numpy as np
from scipy.spatial.transform import Rotation
import open3d as o3d
import rospy

from sensor_msgs.msg import PointCloud2
from digiforest_analysis.tasks.terrain_fitting import TerrainFitting
from digiforest_analysis.tasks.tree_segmentation_voronoi import TreeSegmentation


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


def transform_clusters(clusters, T_new2old, time_stamp=None):
    for i in range(len(clusters)):
        clusters[i]["cloud"].transform(efficient_inv(T_new2old))
        clusters[i]["info"]["axis"]["transform"] = (
            efficient_inv(T_new2old) @ clusters[i]["info"]["axis"]["transform"]
        )
        clusters[i]["info"]["sensor_transform"] = T_new2old
        if time_stamp:
            clusters[i]["info"]["time_stamp"] = time_stamp


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
    mask = distances <= radius
    cloud = cloud.select_by_mask(mask)
    cloud = cloud.transform(center_pose)

    return cloud


def clustering_worker_fun(
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
):
    terrain_fitter = TerrainFitting(
        sloop_smooth=terrain_smoothing,
        cloth_cell_size=terrain_cloth_cell_size,
        debug_level=debug_level,
    )

    tree_segmenter = TreeSegmentation(
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
            transform_clusters(
                clusters, T_new2old=T_sensor2odom, time_stamp=center_stamp
            )

    rospy.loginfo(f"Timing results of clustering:\n{timer}")

    return clusters, terrain
