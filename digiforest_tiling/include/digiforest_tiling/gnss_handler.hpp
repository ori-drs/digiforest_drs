#pragma once

#include <Eigen/Geometry>
#include <GeographicLib/LocalCartesian.hpp>
#include <GeographicLib/TransverseMercator.hpp>

#include "digiforest_tiling/slam_mission.hpp"

namespace digiforest_tiling {

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

///
/// There are three main coordinate frames used in this class.
/// LLA:
/// - Latitude, longitude, altitude (global Earth-centered coordinate system)
/// - This is what is measured by the GNSS sensor.
///
/// ENU:
/// - East, north, up (global Earth-centered coordinate system)
/// - An Euclidean coordinate system
///
/// Map:
/// - VILENS SLAM local coordinate system
///
class GnssHandler {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Positions =
      std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>;
  using Isometry3dVector =
      std::vector<Eigen::Isometry3d,
                  Eigen::aligned_allocator<Eigen::Isometry3d>>;

  GnssHandler();
  ~GnssHandler() = default;

  // Initialization
  void initializeLLA(const double lat, const double lon, const double alt);

  // Manually initialize ENU to Map
  void initializeENUtoMap(const Eigen::Isometry3d &T_enu_map);

  //
  bool initializeFromFile(const std::string &g2oFile);

  // Compute initialization
  void computeEnuLocalTransform(const std::vector<digiforest_tiling::Time>& slam_times,
                                const Isometry3dVector& slam_poses,
                                const bool use_2d_alignment);

  // LLA <> Map
  void transformMapToLLA(const Eigen::Vector3d& position, double& lat,
                         double& lon, double& alt) const;

  void transformLLAtoMap(const double lat, const double lon, const double alt,
                         Eigen::Vector3d& position) const;

  // LLA <> ENU
  void transformLLAtoENU(const double lat, const double lon, const double alt,
                         Eigen::Vector3d& position) const;
  void transformENUtoLLA(const Eigen::Vector3d& position, double& lat,
                         double& lon, double& alt) const;

  // LLA <> EPSG-3067 (Finland coordinate system)
  void convertWGS84toEPSG3067(const double lat, const double lon,
                              double& easting, double& northing) const;

  // LLA <> EPSG-25833 (Europe between 12E and 18E, including Germany)
  void convertWGS84toEPSG25833(const double lat, const double lon,
                               double& easting, double& northing) const;

  // LLA <> EPSG-25832 (Europe between 6E and 12E, including Switzerland)
  void convertWGS84toEPSG25832(const double lat, const double lon,
                               double& easting, double& northing) const;

  // LLA <> EPSG-25830 (Europe between 0E and 6E, including UK)
  void convertWGS84toEPSG25830(const double lat, const double lon,
                               double& easting, double& northing) const;

  // Writing to file
  void writeKmlFile(const std::string& data_directory_path,
                    const Isometry3dVector& slam_poses) const;

  void writeSlamPosesGNSS(const std::string& data_directory_path,
                          const Isometry3dVector& slam_poses) const;

  void writeSlamPosesUTM(const std::string& data_directory_path,
                         const PoseVector& slam_poses) const;

  void writeRawGnssMeasurementsENU(const std::string& data_directory_path) const;

  // Adding new measurements
  void addGnssMeasurement(const double lat, const double lon, const double alt,
                          const digiforest_tiling::Time& time, const std::vector<digiforest_tiling::Time>& slam_times,
                          const Isometry3dVector& slam_poses);

  // Getters
  Eigen::Isometry3d getEnuWorldTransform() const {
    return T_EnuW_;
  }

  bool isInitialized() const {
    return (enu_ref_initialized_ && transform_enu_world_exists_);
  }

  void getGnssMeasurementsInMapFrame(Isometry3dVector &poses) const;

  int getNumGnssMeasurements() const { return enu_fix_buffer_.size(); }

  double getLatRef() const;
  double getLonRef() const;
  double getAltRef() const;

 protected:

  bool enu_ref_initialized_;
  bool transform_enu_world_exists_;
  bool addGnss_;

  // Used for converting lat/lon to cartesian frame
  GeographicLib::LocalCartesian enuRef_;

  // transformation from ENU ref to world frame
  Eigen::Isometry3d T_EnuW_ = Eigen::Isometry3d::Identity();

  // slam pose when ENU Ref is initialized
  Eigen::Isometry3d T_enuRef_pose_;

  Positions enu_fix_buffer_;
  std::vector<digiforest_tiling::Time> enu_fix_times_buffer_;

  // Keep objects to speed up conversions to other coordinate systems
  std::unique_ptr<GeographicLib::TransverseMercator> tm_epsg3067_;
  std::unique_ptr<GeographicLib::TransverseMercator> tm_epsg25833_;
  std::unique_ptr<GeographicLib::TransverseMercator> tm_epsg25832_;
  std::unique_ptr<GeographicLib::TransverseMercator> tm_epsg25830_;

};

}  // namespace digiforest_tiling
