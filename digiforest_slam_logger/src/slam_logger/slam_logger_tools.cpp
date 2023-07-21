
#include <pcl/filters/passthrough.h>
#include <pcl/filters/crop_box.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/io/ply_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>

#include <slam_logger/slam_logger_tools.hpp>


std::string getEnvVar( std::string const & key )
{
    char * val = getenv( key.c_str() );
    return val == NULL ? std::string("") : std::string(val);
}

std::string getTimeAsString(){
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];
    time (&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(buffer,sizeof(buffer),"%Y-%m-%d-%H-%M-%S",timeinfo);
    std::string str(buffer);
    return str;
}

// Returns filtered cloud: crop cloud using box (centered at origin)
// This filter reduces size of the input cloud
// input: negative takes the opposite set (those outside the box), by default this is NOT done.
void getPointsInOrientedBox(IcpCloudType::Ptr& cloud, float min, float max,
                            Eigen::Matrix4f& origin, bool negative) {
  pcl::CropBox<IcpPointType> box_filter;

  Eigen::Vector3f position, orientation;
  position << origin(0, 3), origin(1, 3), origin(2, 3);
  orientation =
      origin.block<3, 3>(0, 0).eulerAngles(0, 1, 2);  // (rx,ry,rz) in radians

  box_filter.setMin(Eigen::Vector4f(min, min, min, 1.0));  // minX, minY, minZ
  box_filter.setMax(Eigen::Vector4f(max, max, max, 1.0));
  box_filter.setRotation(orientation);
  box_filter.setTranslation(position);
  box_filter.setInputCloud(cloud);
  box_filter.setNegative(negative);
  box_filter.filter(*cloud);
}

// pose = world_to_cloud_corrected_
void calculateCloudNormals(IcpCloudType::Ptr& cloud,
                           const Eigen::Vector3d& sensorOrigin) {
  if (areNormalsSet(cloud)) {
    return;
  }

  // Create the normal estimation class, and pass the input dataset to it

  // Note: Using OMP version which is approx 6x quicker than the normal version
  // through parallel computation (6ms vs 36ms)
  pcl::NormalEstimationOMP<IcpPointType, pcl::Normal> ne;
  ne.setInputCloud(cloud);

  // Create an empty kdtree representation, and pass it to the normal estimation
  // object. Its content will be filled inside the object, based on the given
  // input dataset (as no other search surface is given).
  auto tree = boost::make_shared<pcl::search::KdTree<IcpPointType>>();
  ne.setSearchMethod(tree);

  // ne.setRadiusSearch (0.25); // Use all neighbors in a sphere of radius
  ne.setKSearch(20);

  // Important to direct all normals towards origin of the sensor
  ne.setViewPoint(sensorOrigin(0), sensorOrigin(1), sensorOrigin(2));

  pcl::PointCloud<pcl::Normal> normals;
  ne.compute(normals);

  // Assign to output cloud
  assert(normals.size() == cloud->size());
  for (size_t i = 0; i < cloud->size(); ++i) {
    cloud->points[i].normal_x = normals.points[i].normal_x;
    cloud->points[i].normal_y = normals.points[i].normal_y;
    cloud->points[i].normal_z = normals.points[i].normal_z;
  }
}

/// Compute normals in the voxelized cloud
/// This should have better normals by having a more evenly distributed cloud
/// (especially for individual lidar scans)
void calculateCloudNormalsVoxelized(IcpCloudType::Ptr& cloud,
                                    const Eigen::Vector3d& sensorOrigin,
                                    const double leaf_size) {
  if (areNormalsSet(cloud)) {
    return;
  }

  auto cloud_voxelized = boost::make_shared<IcpCloudType>();

  // Perform voxel filtering, saving the result to speed up assocation of the
  // voxelised points to non-voxelised points
  pcl::VoxelGrid<IcpPointType> sor;
  sor.setInputCloud(cloud);
  sor.setLeafSize (leaf_size, leaf_size, leaf_size); // was hard coded at 0.08
  sor.setDownsampleAllData(false); // only downsample XYZ fields
  sor.setSaveLeafLayout(true); // save internal results for getCentroidIndex()
  sor.filter(*cloud_voxelized);

  calculateCloudNormals(cloud_voxelized, sensorOrigin);

  // Copy the normals to the original cloud
  // To keep it simple, just copy the normal of the closest point
  for (auto &p : cloud->points) {
    const int ind = sor.getCentroidIndex(p);

    p.normal_x = cloud_voxelized->points[ind].normal_x;
    p.normal_y = cloud_voxelized->points[ind].normal_y;
    p.normal_z = cloud_voxelized->points[ind].normal_z;
  }

}

bool areNormalsSet(const IcpCloudType::ConstPtr &cloud) {
  // Check if they are already set
  if (cloud->empty()) {
    return false;
  }

  const double nx = cloud->points[0].normal_x;
  const double ny = cloud->points[0].normal_y;
  const double nz = cloud->points[0].normal_z;
  const double n_norm = sqrt(nx * nx + ny * ny + nz * nz);

  // If normal is set already then don't do anything
  if (std::abs(n_norm - 1) < 0.001) {
    return true;
  }

  return false;
}