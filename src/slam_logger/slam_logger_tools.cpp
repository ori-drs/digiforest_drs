
#include <pcl/filters/passthrough.h>
#include <pcl/filters/crop_box.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/io/ply_io.h>

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