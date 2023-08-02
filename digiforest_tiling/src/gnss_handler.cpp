#include <fstream> // for ofstream
#include <iomanip> // for setprecision
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/transformation_estimation_2D.h>

#include "digiforest_tiling/gnss_handler.hpp"

namespace digiforest_tiling {

GnssHandler::GnssHandler()
    : enu_ref_initialized_(false),
      transform_enu_world_exists_(false) {
  // std::cout << "[GNSS] GNSS handler initialized\n";

  // Preallocate geographiclib objects for efficiency
  // https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system

  // UTM35N Finland EPSG3067 Constants from: https://epsg.io/3067
  {
  const double radius = 6378137;
  const double scale_factor = 0.9996;
  const double inverse_flattening = 298.257222101;
  const double flattening = 1 / inverse_flattening;

  tm_epsg3067_ = std::make_unique<GeographicLib::TransverseMercator>(
      radius, flattening, scale_factor);
  }

  // UTM33N Eastern Germany EPSG25833 Constants from: https://epsg.io/25833
  {
  const double radius = 6378137;
  const double scale_factor = 0.9996;
  const double inverse_flattening = 298.257222101;
  const double flattening = 1 / inverse_flattening;

  tm_epsg25833_ = std::make_unique<GeographicLib::TransverseMercator>(
      radius, flattening, scale_factor);
  }

  // UTM32N Switzerland EPSG25832 Constants from: https://epsg.io/25832
  {
  const double radius = 6378137;
  const double scale_factor = 0.9996;
  const double inverse_flattening = 298.257222101;
  const double flattening = 1 / inverse_flattening;

  tm_epsg25832_ = std::make_unique<GeographicLib::TransverseMercator>(
      radius, flattening, scale_factor);
  }

  // UTM30N UK EPSG25830 Constants from: https://epsg.io/25830
  {
  const double radius = 6378137;
  const double scale_factor = 0.9996;
  const double inverse_flattening = 298.257222101;
  const double flattening = 1 / inverse_flattening;

  tm_epsg25830_ = std::make_unique<GeographicLib::TransverseMercator>(
      radius, flattening, scale_factor);
  }

}

/// Note: It defines easting and northing different from the usual convention,
/// i.e., [easting, northing, altitude]
void GnssHandler::convertWGS84toEPSG3067(const double lat, const double lon,
                                         double& easting,
                                         double& northing) const {
  const double ref_lat = 0; // deg
  const double ref_lon = 27; // deg
  const double false_easting = 500000; // m
  const double false_northing = 0; // m
  tm_epsg3067_->Forward(ref_lon, lat, lon, easting, northing);

  // Apply false_easting and false_northing
  easting += false_easting;
  northing += false_northing;
}

void GnssHandler::convertWGS84toEPSG25833(const double lat, const double lon,
                                          double& easting,
                                          double& northing) const {
  const double ref_lat = 0; // deg
  const double ref_lon = 15; // deg
  const double false_easting = 500000; // m
  const double false_northing = 0; // m
  tm_epsg25833_->Forward(ref_lon, lat, lon, easting, northing);

  // Apply false_easting and false_northing
  easting += false_easting;
  northing += false_northing;
}

void GnssHandler::convertWGS84toEPSG25832(const double lat, const double lon,
                                          double& easting,
                                          double& northing) const {
  const double ref_lat = 0; // deg
  const double ref_lon = 9; // deg
  const double false_easting = 500000; // m
  const double false_northing = 0; // m
  tm_epsg25832_->Forward(ref_lon, lat, lon, easting, northing);

  // Apply false_easting and false_northing
  easting += false_easting;
  northing += false_northing;
}

void GnssHandler::convertWGS84toEPSG25830(const double lat, const double lon,
                                          double& easting,
                                          double& northing) const {
  const double ref_lat = 0; // deg
  const double ref_lon = -3; // deg
  const double false_easting = 500000; // m
  const double false_northing = 0; // m
  tm_epsg25830_->Forward(ref_lon, lat, lon, easting, northing);

  // Apply false_easting and false_northing
  easting += false_easting;
  northing += false_northing;
}

void GnssHandler::transformMapToLLA(const Eigen::Vector3d& position,
                                           double& lat, double& lon,
                                           double& alt) const {
  if (!transform_enu_world_exists_ || !enu_ref_initialized_) {
    throw std::runtime_error(
        "Cannot convert to lat, lon, alt before transform between ENU and map "
        "frame has been calculated.");
  }
  const Eigen::Vector3d pos_enu = T_EnuW_.inverse() * position;
  enuRef_.Reverse(pos_enu[0], pos_enu[1], pos_enu[2], lat, lon, alt);
}

void GnssHandler::transformLLAtoMap(const double lat, const double lon,
                                    const double alt,
                                    Eigen::Vector3d& position) const {
  if (!transform_enu_world_exists_ || !enu_ref_initialized_) {
    throw std::runtime_error(
        "Cannot convert to lat, lon, alt before transform between ENU and map "
        "frame has been calculated.");
  }

  Eigen::Vector3d pos_enu = Eigen::Vector3d::Zero();
  transformLLAtoENU(lat, lon, alt, pos_enu);
  position = T_EnuW_ * pos_enu;
}

void GnssHandler::transformLLAtoENU(const double lat, const double lon,
                                    const double alt,
                                    Eigen::Vector3d& position) const {
  if (!enu_ref_initialized_) {
    throw std::runtime_error(
        "Cannot convert to ENU before enu_ref_ is initialized");
  }
  enuRef_.Forward(lat, lon, alt, position[0], position[1], position[2]);
}

void GnssHandler::transformENUtoLLA(const Eigen::Vector3d& position,
                                    double& lat, double& lon,
                                    double& alt) const {
  if (!enu_ref_initialized_) {
    throw std::runtime_error(
        "Cannot convert to LLA before enu_ref_ is initialized");
  }
  enuRef_.Reverse(position[0], position[1], position[2], lat, lon, alt);
}

void GnssHandler::initializeLLA(const double lat, const double lon, const double alt) {
  // Only initialize once
  if (enu_ref_initialized_) {
    return;
  }

  std::cout << "[GNSS] Initializing the ENU frame. "
            << " LAT: " << lat << " LON: " << lon << " ALT: " << alt << "\n";
  enuRef_.Reset(lat, lon, alt);
  enu_ref_initialized_ = true;
}

void GnssHandler::initializeENUtoMap(const Eigen::Isometry3d &T_enu_map) {
  std::cout << "[GNSS] Initializing the ENU to Map transform:\n"
            << T_enu_map.matrix() << "\n";

  T_EnuW_ = T_enu_map;
  transform_enu_world_exists_ = true;
}

bool GnssHandler::initializeFromFile(const std::string &g2oFile) {
  // open the file
  std::ifstream inFile(g2oFile);
  if (!inFile) {
    std::cout << "[GNSS] Error, could not open g2o file: " << g2oFile;
    return false;
  }

  std::cout << "[GNSS] Initializing gnss handler from file: " << g2oFile << "\n";

  double lat, lon, alt;
  double x, y, z, qx, qy, qz, qw;
  bool found_gnss_lla_ref = false, found_gnss_lla_to_map = false;

  // go through each line of the file
  while (!inFile.eof()) {
    std::string type;
    inFile >> type;

    if (type == "GNSS_LLA_REF") {
      found_gnss_lla_ref = true;
      inFile >> lat >> lon >> alt;
      initializeLLA(lat, lon, alt);
    } else if (type == "GNSS_LLA_TO_MAP") {
      found_gnss_lla_to_map = true;
      inFile >> x >> y >> z >> qx >> qy >> qz >> qw;
      Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
      T.translate(Eigen::Vector3d(x, y, z));
      T.rotate(Eigen::Quaterniond(qw, qx, qy, qz).matrix());
      initializeENUtoMap(T);
    } else {
      // ignore other lines
      getline(inFile, type);
    }
  }

  if (!found_gnss_lla_ref || !found_gnss_lla_to_map) {
    return false;
  }

  return isInitialized();
}

// TODO: Do we need this file?
void GnssHandler::writeSlamPosesGNSS(const std::string& data_directory_path,
                                     const Isometry3dVector& slam_poses) const {
  // Write the georeferenced SLAM poses to file:
  std::string slam_poses_gnss_filename =
      data_directory_path + "/slam_poses_gnss.csv";
  std::ofstream slam_poses_gnss_file;
  slam_poses_gnss_file.open(slam_poses_gnss_filename);
  slam_poses_gnss_file << std::setprecision(15);
  slam_poses_gnss_file << "type,latitude,longitude,altitude,desc,color\n";
  for (size_t i = 0; i < slam_poses.size(); ++i) {
    // convert to GNSS coordinates
    double lat, lon, alt;
    transformMapToLLA(slam_poses[i].translation(), lat, lon, alt);
    slam_poses_gnss_file << "T," << lat << "," << lon << "," << alt
                         << ",track,red\n";
  }
  slam_poses_gnss_file.close();
}

void GnssHandler::writeSlamPosesUTM(const std::string& data_directory_path,
                                    const PoseVector& slam_poses) const {
  // Write the georeferenced SLAM poses to file:
  std::string slam_poses_utm_filename =
      data_directory_path + "/slam_poses_utm.csv";
  std::ofstream slam_poses_utm_file;
  slam_poses_utm_file.open(slam_poses_utm_filename);
  slam_poses_utm_file << std::setprecision(15);
  slam_poses_utm_file << "#counter,sec,nsec,x,y,z,qx,qy,qz,qw\n";
  for (size_t i = 0; i < slam_poses.size(); ++i) {
    Eigen::Quaterniond q(slam_poses[i].pose.rotation());

    slam_poses_utm_file << std::to_string(i) << "," << slam_poses[i].sec << "," << slam_poses[i].nsec << ","
                        << slam_poses[i].pose.translation().x() << ","
                        << slam_poses[i].pose.translation().y() << ","
                        << slam_poses[i].pose.translation().z() << ","
                        << q.x() << "," << q.y() << "," << q.z() << "," << q.w() << "\n";
  }
  slam_poses_utm_file.close();
}

// TODO: Do we need this file?
/// Write raw GNSS measurements in ENU reference frame
void GnssHandler::writeRawGnssMeasurementsENU(
    const std::string& data_directory_path) const {
  std::string slam_poses_enu_filename =
      data_directory_path + "/raw_gnss_measurements_enu.csv";
  std::ofstream slam_poses_enu_file;
  slam_poses_enu_file.open(slam_poses_enu_filename);
  slam_poses_enu_file << std::setprecision(15);
  slam_poses_enu_file << "# counter, sec, nsec, x, y, z, qx, qy, qz, qw\n";
  for (size_t i = 0; i < enu_fix_buffer_.size(); ++i) {
    const Eigen::Vector3d pos = enu_fix_buffer_[i];
    const digiforest_tiling::Time time = enu_fix_times_buffer_[i];
    slam_poses_enu_file << i << ", " << time.sec << ", " << time.nsec << ", ";
    slam_poses_enu_file << pos[0] << ", " << pos[1] << ", " << pos[2] << ", ";
    slam_poses_enu_file << 0.0 << ", " << 0.0 << ", " << 0.0 << ", " << 1.0;
    slam_poses_enu_file << "\n";
  }
  slam_poses_enu_file.close();
}

/// Write the georeferenced SLAM poses to file:
/// This file can be loaded directly into Google Earth for simple visualisation
/// https://earth.google.com/
void GnssHandler::writeKmlFile(const std::string& data_directory_path,
                               const Isometry3dVector& slam_poses) const {
  const std::string slam_poses_kml_filename =
      data_directory_path + "/slam_poses_gnss.kml";
  std::ofstream slam_poses_kml_file;
  slam_poses_kml_file.open(slam_poses_kml_filename);
  slam_poses_kml_file << std::setprecision(15);
  slam_poses_kml_file
      << ""
         "<?xml version=\"1.0\" encoding=\"utf-8\" standalone=\"yes\"?>\n"
         "<kml xmlns=\"http://www.opengis.net/kml/2.2\">\n"
         "  <Document>\n"
         "    <name><![CDATA[slam_poses_gnss]]></name>\n"
         "    <visibility>1</visibility>\n"
         "    <open>1</open>\n"
         "    <Folder id=\"Tracks\">\n"
         "      <name>Tracks</name>\n"
         "      <visibility>1</visibility>\n"
         "      <open>0</open>\n"
         "      <Placemark>\n"
         "        <name><![CDATA[slam_poses_gnss]]></name>\n"
         "        <description><![CDATA[track]]></description>\n"
         "        <Style>\n"
         "          <LineStyle>\n"
         "            <color>ff0000ff</color>\n"
         "            <width>4</width>\n"
         "          </LineStyle>\n"
         "        </Style>\n"
         "        <LineString>\n"
         "          <tessellate>1</tessellate>\n"
         "          <altitudeMode>clampToGround</altitudeMode>\n"
         "          <coordinates> ";

  // convert slam poses to GNSS coordinates
  for (size_t i = 0; i < slam_poses.size(); ++i) {
    double lat, lon, alt;
    transformMapToLLA(slam_poses[i].translation(), lat, lon, alt);
    slam_poses_kml_file << " " << lon << "," << lat << "," << alt;
  }

  slam_poses_kml_file << " </coordinates>\n"
                         "        </LineString>\n"
                         "      </Placemark>\n"
                         "    </Folder>\n"
                         "  </Document>\n"
                         "</kml>";

  slam_poses_kml_file.close();
}


///
/// This function assumes that we have a set of synchronized GNSS and SLAM poses
/// with the same timestamp.
///
/// These PCL alignment methods assume temporally aligned pairs of measurements
/// which is not true in general.
///
/// const RegisteredCloudList& registered_cloud_list,
void GnssHandler::computeEnuLocalTransform(const std::vector<digiforest_tiling::Time>& slam_times,
                                           const Isometry3dVector& slam_poses,
                                           const bool use_2d_alignment) {
  // Compute if min number of gnss measurements received
  // and only calculate every 10 measurements (to save on computation)
  if ((enu_fix_buffer_.size() < 10) ||
      (enu_fix_buffer_.size() % 10 != 0)) {
    return;
  }

  // Check inputs
  assert(slam_times.size() == slam_poses.size());
  assert(slam_poses.size() == enu_fix_buffer_.size());

  auto pts_enu = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  pts_enu->width = enu_fix_buffer_.size();
  pts_enu->height = 1;
  pts_enu->is_dense = false;
  pts_enu->resize(pts_enu->width * pts_enu->height);

  auto pts_world = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  pts_world->width = enu_fix_buffer_.size();
  pts_world->height = 1;
  pts_world->is_dense = false;
  pts_world->resize(pts_world->width * pts_world->height);

  for (size_t i = 0; i < enu_fix_buffer_.size(); i++) {
    const Eigen::Vector3d enuFix = enu_fix_buffer_[i];
    pts_enu->points[i].x = enuFix[0];
    pts_enu->points[i].y = enuFix[1];
    pts_enu->points[i].z = enuFix[2];

    assert(slam_times[i] == enu_fix_times_buffer_[i]);
    const Eigen::Vector3d pos = slam_poses[i].translation();
    pts_world->points[i].x = pos[0];
    pts_world->points[i].y = pos[1];
    pts_world->points[i].z = pos[2];
  }

  Eigen::Matrix4f T_est;
  if (use_2d_alignment) {
    // Note: This method requires z = 0
    for (auto& p : pts_enu->points) {
      p.z = 0;
    }
    for (auto& p : pts_world->points) {
      p.z = 0;
    }

    pcl::registration::TransformationEstimation2D<pcl::PointXYZ,pcl::PointXYZ> TE;
    TE.estimateRigidTransformation (*pts_enu,*pts_world, T_est);
  } else {
    pcl::registration::TransformationEstimationSVD<pcl::PointXYZ,pcl::PointXYZ> TE;
    TE.estimateRigidTransformation(*pts_enu, *pts_world, T_est);
  }

  // set trafo from ENU to World frame
  T_EnuW_.matrix() = T_est.cast<double>();

  // check matrix is valid
  assert(!std::isnan(T_EnuW_.matrix().norm()));
  assert(!std::isinf(T_EnuW_.matrix().norm()));

  transform_enu_world_exists_ = true;
}

void GnssHandler::addGnssMeasurement(const double lat, const double lon,
  // std::cout << "[GNSS] GNSS fix. LAT: " << lat << " LON: " << lon
  //           << " ALT: "<< altitude << "\n";
                                     const double alt, const digiforest_tiling::Time& time,
                                     const std::vector<digiforest_tiling::Time>& slam_times,
                                     const Isometry3dVector& slam_poses) {

  initializeLLA(lat, lon, alt);

  Eigen::Vector3d enu = Eigen::Vector3d::Zero();
  transformLLAtoENU(lat, lon, alt, enu);
  enu_fix_buffer_.push_back(enu);
  enu_fix_times_buffer_.push_back(time);

  if (!transform_enu_world_exists_) {
    computeEnuLocalTransform(slam_times, slam_poses, true);
  }
}

void GnssHandler::getGnssMeasurementsInMapFrame(Isometry3dVector& poses) const {
  poses.clear();
  poses.reserve(enu_fix_buffer_.size());
  for (size_t i = 0; i < enu_fix_buffer_.size(); ++i) {
    poses.push_back(Eigen::Isometry3d::Identity());
    if (isInitialized()) {
      poses[i].translation() = T_EnuW_ * enu_fix_buffer_[i];
    } else {
      poses[i].translation() = enu_fix_buffer_[i];
    }
  }
}

double GnssHandler::getLatRef() const {
  if (!enu_ref_initialized_) {
    throw std::runtime_error("LLA ref not initialized!");
  }
  return enuRef_.LatitudeOrigin();
}

double GnssHandler::getLonRef() const {
  if (!enu_ref_initialized_) {
    throw std::runtime_error("LLA ref not initialized!");
  }
  return enuRef_.LongitudeOrigin();
}

double GnssHandler::getAltRef() const {
  if (!enu_ref_initialized_) {
    throw std::runtime_error("LLA ref not initialized!");
  }
  return enuRef_.HeightOrigin();
}

}  // namespace digiforest_tiling
