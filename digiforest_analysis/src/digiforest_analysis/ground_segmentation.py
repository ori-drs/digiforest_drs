import digiforest_analysis.terrain_mapping as df
import pcl
from typing import Tuple

class GroundSegmentation:
    def __init__(self, cloud_filename: str):
        self.cloud_filename = cloud_filename

    def remove_normals(self, cloud: pcl.PointCloud_PointNormal) -> pcl.PointCloud:
        array_xyz = cloud.to_array()[:, 0:3]
        cloud = pcl.PointCloud()
        cloud.from_array(array_xyz)
        return cloud 

    def generate_height_map(self)-> Tuple[pcl.PointCloud, pcl.PointCloud]:
        cloud_pc = pcl.PointCloud_PointNormal()
        cloud_pc._from_pcd_file(self.cloud_filename.encode('utf-8'))

        # remove non-up points
        ground_cloud = df.filterUpNormal(cloud_pc, 0.95)

        forest_cloud = df.filterUpNormal(cloud_pc, 0.95, keepUp=False)

        # drop from xyznormal to xyz
        # TODO necessary?
        ground_cloud = self.remove_normals(ground_cloud)
        forest_cloud = self.remove_normals(forest_cloud)
        
        # get the terrain height
        heights_array_raw = df.getTerrainCloud(ground_cloud)
        ground_cloud = pcl.PointCloud()
        ground_cloud.from_list(heights_array_raw)

        return ground_cloud, forest_cloud
        
