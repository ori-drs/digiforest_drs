#pragma once
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <pcl/filters/passthrough.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/io/ply_io.h>

using IcpPointType = pcl::PointXYZINormal;
using IcpCloudType = pcl::PointCloud<IcpPointType>;

void getPointsInOrientedBox(IcpCloudType::Ptr& cloud, float min, float max,
                            Eigen::Matrix4f& origin, bool negative = false);

std::string getEnvVar( std::string const & key );

std::string getTimeAsString();


// Time data structure (not to be confused with ros::Time)
struct Time
{
   uint32_t sec, nsec;

   // constructor
   Time(const uint32_t _sec = 0, const uint32_t _nsec = 0)
       : sec(_sec), nsec(_nsec) {}

   // custom equality operator
   inline bool operator==(const Time& rhs) const {
     return (sec == rhs.sec) && (nsec == rhs.nsec);
   }

   inline bool operator!=(const Time& rhs) const {
     return !(*this == rhs);
   }
};


class IsometryWithTime
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
public:

  IsometryWithTime(){
    pose = Eigen::Isometry3d::Identity();
    sec = 0;
    nsec = 0;
    key = 0;
  }

  IsometryWithTime(Eigen::Isometry3d pose_in,
         int sec_in,
         int nsec_in,
         uint64_t key_in){
    pose = pose_in;
    sec = sec_in;
    nsec = nsec_in;
    key = key_in;
  }

  ~IsometryWithTime(){
  }

  Eigen::Isometry3d pose;
  uint32_t sec;
  uint32_t nsec;
  uint64_t key;
};