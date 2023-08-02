#define PCL_NO_PRECOMPILE
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/impl/voxel_grid.hpp>
#include <pcl/filters/random_sample.h>

#include <boost/filesystem.hpp>
#include <iostream>
#include <string>

#include <ros/ros.h>

#include "digiforest_tiling/gnss_handler.hpp"
#include "digiforest_tiling/slam_mission.hpp"

namespace fs = boost::filesystem;

///////////////////////////////////////////////////////////////////////////////

struct EIGEN_ALIGN16 PointXYZRGBNormalDouble {
  double x;
  double y;
  double z;
  std::uint8_t r;
  std::uint8_t g;
  std::uint8_t b;
  float normal_x;
  float normal_y;
  float normal_z;
  PCL_MAKE_ALIGNED_OPERATOR_NEW

  inline PointXYZRGBNormalDouble(double _x = 0, double _y = 0, double _z = 0,
                               std::uint8_t _r = 0, std::uint8_t _g = 0, std::uint8_t _b = 0,
                               float _normal_x = 0, float _normal_y = 0, float _normal_z = 0)
      : x(_x),
        y(_y),
        z(_z),
        r(_r),
        g(_g),
        b(_b),
        normal_x(_normal_x),
        normal_y(_normal_y),
        normal_z(_normal_z) {}

};

POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointXYZRGBNormalDouble,
    (double, x, x)(double, y, y)(double, z, z)
    (std::uint8_t, r, r)(std::uint8_t, g, g)(std::uint8_t, b, b)
    (float, normal_x, normal_x)(float, normal_y, normal_y)(float, normal_z, normal_z))                                                          

////////////////////////////////////////////////////////////////////////////////

struct Params {

  enum class Frame {
    GNSS_EPSG3067,
    GNSS_EPSG25833,
    GNSS_EPSG25832,
    GNSS_EPSG25830,
  };

  Frame output_frame;
  double tile_size = 20.0;  // meters
  std::string output_folder;
  std::string input_folder;
  bool generate_merged_cloud = true;
};

////////////////////////////////////////////////////////////////////////////////

struct Tile {
  double x_min, y_min;
  double size_x, size_y;

  bool isInside(const double x, const double y) const {
    return (x >= x_min && x < x_min + size_x && y >= y_min &&
            y < y_min + size_y);
  }
};

class TilesGenerator {
 public:
  TilesGenerator() {
    // surpress pcl output
    pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);
  }
  ~TilesGenerator() = default;

  void generateTiling();

  void setParams(const ros::NodeHandle& nh);

 private:
  digiforest_tiling::GnssHandler gnss_;
  pcl::PLYWriter ply_writer_;

  Params p_;

  std::vector<std::vector<Tile> > tiles_;
  double x_min_, y_min_;
  int num_rows_, num_cols_;

  // Gateway function for file I/O - either pcd (binary, default) or ply (ascii)
  // Note: This is templated to allow for different point cloud types to be
  // saved
  template <class T>
  void writePointCloud(const std::string& filename,
                       const pcl::PointCloud<T>& point_cloud) {
    //std::cout << "writing " << point_cloud.size() << " points to "
    //          << filename << "\n\n";
    ply_writer_.write<T>(filename, point_cloud, true, false);
  }

  template <class T>
  void readPointCloud(const std::string& filename,
                      pcl::PointCloud<T>& cloud) const;

  std::string tileFilepath(int row, int col) const;

  bool createDirectory(const fs::path& path) const;

  void generateTiles(const std::string& dir_input);

  void parsePayloadClouds(const std::string& input_folder);

  void parsePayloadCloud(pcl::PointCloud<PointXYZRGBNormalDouble>& cloud);

  /**
   * @brief Save the tiles coordinates to a csv file
  */
  void saveTiling(const std::string& output_dir) const;

  /**
   * @brief Merge and downsample all the tiles together and save the result
   * point cloud
  */
  void saveDownsampledMergedTiles(const std::string& output_dir);

};

template <class T>
void TilesGenerator::readPointCloud(const std::string& filename,
                                    pcl::PointCloud<T>& cloud) const {
  fs::path entry(filename);
  const std::string file_type = entry.extension().string();
  if (file_type == ".ply") {
    if (pcl::io::loadPLYFile<T>(filename, cloud) == -1) {
      std::cerr << "Error couldn't read file: " << filename + "\n";
      return;
    }
  } else if (file_type == ".pcd") {
    if (pcl::io::loadPCDFile<T>(filename, cloud) == -1) {
      std::cerr << "Error couldn't read file: " << filename + "\n";
      return;
    }

  } else {
    throw std::runtime_error(
        "Tried to read unsupported point cloud type from " + filename);
  }
}

std::string TilesGenerator::tileFilepath(int row, int col) const {
  std::string filename =  "tile_" + std::to_string(row*tiles_.size() + col) + ".ply";
  return p_.output_folder + "/" + filename;
}

void TilesGenerator::setParams(const ros::NodeHandle& nh) {
  std::string output_frame;
  if(!nh.getParam("output_frame", output_frame)) {
    throw std::invalid_argument("Output frame not specified.");
  }

  if (output_frame == "GNSS_EPSG3067") {
    p_.output_frame = Params::Frame::GNSS_EPSG3067;
  }else if (output_frame == "GNSS_EPSG25833") {
    p_.output_frame = Params::Frame::GNSS_EPSG25833;
  }else if (output_frame == "GNSS_EPSG25832") {
    p_.output_frame = Params::Frame::GNSS_EPSG25832;
  }else if (output_frame == "GNSS_EPSG25830") {
    p_.output_frame = Params::Frame::GNSS_EPSG25830;
  } else {
    throw std::invalid_argument("Output frame type not supported.");
  }

  nh.param("tile_size", p_.tile_size, 20.0);
  nh.param("generate_merged_cloud", p_.generate_merged_cloud, true);
  if(!nh.getParam("output_folder", p_.output_folder)) {
    throw std::invalid_argument("Output folder not specified.");
  }
  if(!nh.getParam("input_folder", p_.input_folder)) {
    throw std::invalid_argument("Input folder not specified.");
  }
}

bool TilesGenerator::createDirectory(const fs::path& path) const {
    if (!fs::exists(path)) {
        try {
            fs::create_directory(path);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Failed to create directory: " << e.what() << std::endl;
            return false;
        }
    } else if (fs::is_directory(path)) {
        return true;
    } else {
        std::cerr << "Path exists but is not a directory." << std::endl;
        return false;
    }
}

void TilesGenerator::generateTiling() {
  if(!createDirectory(p_.output_folder)) {
    return;
  }
  generateTiles(p_.input_folder);
  parsePayloadClouds(p_.input_folder);
  saveTiling(p_.output_folder);
  if(p_.generate_merged_cloud) {
    saveDownsampledMergedTiles(p_.output_folder);
  }
}


void TilesGenerator::generateTiles(const std::string& dir_input) {
  const std::string g2o_file = dir_input + "/../slam_pose_graph.g2o";
  const SlamMission slam_mission(g2o_file, 0, 0);
  gnss_.initializeFromFile(g2o_file);

  double x_min = std::numeric_limits<double>::max();
  double y_min = std::numeric_limits<double>::max();
  double x_max = std::numeric_limits<double>::min();
  double y_max = std::numeric_limits<double>::min();

  for (const auto& pose_in_map : slam_mission.poses) {
    double lat, lon, alt;
    Eigen::Vector3d position = pose_in_map.pose.translation();
    gnss_.transformMapToLLA(position, lat, lon, alt);
    double easting, northing;

    if (p_.output_frame == Params::Frame::GNSS_EPSG3067) {
      gnss_.convertWGS84toEPSG3067(lat, lon, easting, northing);
    } else if (p_.output_frame == Params::Frame::GNSS_EPSG25833) {
      gnss_.convertWGS84toEPSG25833(lat, lon, easting, northing);
    } else if (p_.output_frame == Params::Frame::GNSS_EPSG25832) {
      gnss_.convertWGS84toEPSG25832(lat, lon, easting, northing);
    } else if (p_.output_frame == Params::Frame::GNSS_EPSG25830) {
      gnss_.convertWGS84toEPSG25830(lat, lon, easting, northing);
    } else {
      throw std::invalid_argument("GNSS frame type not supported!");
    }

    x_min = std::min(x_min, easting);
    y_min = std::min(y_min, northing);
    x_max = std::max(x_max, easting);
    y_max = std::max(y_max, northing);
  }

  // add extra padding
  x_min -= p_.tile_size;
  x_max += p_.tile_size;
  y_min -= p_.tile_size;
  y_max += p_.tile_size;

  // generate tiles
  x_min_ = x_min;
  y_min_ = y_min;
  double x = x_min;
  double y = y_min;
  num_rows_ = std::ceil((x_max - x_min) / p_.tile_size);
  num_cols_ = std::ceil((y_max - y_min) / p_.tile_size);
  tiles_.resize(num_rows_);

  for(int row = 0; row < num_rows_; ++row) {
    y = y_min;
    for(int col = 0; col < num_cols_; ++col) {
      Tile tile;
      tile.x_min = x;
      tile.y_min = y;
      tile.size_x = p_.tile_size;
      tile.size_y = p_.tile_size;
      tiles_[row].push_back(tile);
      y += p_.tile_size; 
    }
    x += p_.tile_size;
  }
}

void TilesGenerator::parsePayloadClouds(const std::string& input_folder) {

  for (const fs::directory_entry& entry :
      fs::directory_iterator(input_folder)) {
    // Check if a file
    const std::string abs_path = entry.path().string();
    if (fs::is_directory(entry.path()) || !fs::is_regular_file(entry.path())) {
      continue;
    }

    // Check if valid extension
    const std::string ext = entry.path().extension().string();
    if ((ext != ".ply") && (ext != ".pcd")) {
      continue;
    }

    // read point cloud
    auto payload_cloud = std::make_shared<pcl::PointCloud<PointXYZRGBNormalDouble> >();
    readPointCloud(abs_path, *payload_cloud);

    if (payload_cloud->empty()) {
      std::cout << "Error: could not read cloud: " << abs_path << "\n";
      continue;
    }

    std::cout << "Processing cloud: " << abs_path << std::endl;
    parsePayloadCloud(*payload_cloud);
  }
}

void TilesGenerator::parsePayloadCloud(pcl::PointCloud<PointXYZRGBNormalDouble>& cloud) {

  typedef std::vector<std::shared_ptr<pcl::PointCloud<PointXYZRGBNormalDouble> > > CloudVector;

  std::vector<CloudVector > tile_clouds(num_rows_,
      CloudVector(num_cols_, nullptr));

  // iterate over each points of payload cloud and add them to the corresponding tiles
  for (const auto& p : cloud.points) {
    int row = int((p.x - x_min_) / p_.tile_size);
    int col = int((p.y - y_min_) / p_.tile_size);

    if(row < 0 || row >= num_rows_ || col < 0 || col >= num_cols_) {
      continue;
    }

    Tile tile = tiles_[row][col];

    if(p.x < tile.x_min || p.x > tile.x_min + tile.size_x || p.y < tile.y_min || p.y > tile.y_min + tile.size_y) {
      continue;
    }

    if (tile_clouds[row][col] == nullptr) {
      auto tile_cloud =
      std::make_shared<pcl::PointCloud<PointXYZRGBNormalDouble>>();

      if (fs::exists(tileFilepath(row, col))) {
        readPointCloud(tileFilepath(row, col), *tile_cloud);
      }
      tile_clouds[row][col] = tile_cloud;
    }

    tile_clouds[row][col]->emplace_back(p);
  }

  // save updated tiles to disk
  for (int row = 0; row < num_rows_; ++row) {
    for (int col = 0; col < num_cols_; ++col) {
      if (tile_clouds[row][col] == nullptr) {
        continue;
      }
      std::string filepath= tileFilepath(row, col);
      writePointCloud(filepath, *tile_clouds[row][col]);
    }
  }
}

void TilesGenerator::saveTiling(const std::string& output_dir) const {
  std::string tiles_filename = output_dir + "/tiles.csv";
  std::ofstream tiles_file;
  tiles_file.open(tiles_filename);
  tiles_file << std::setprecision(15);
  tiles_file << "#counter,x_min,y_min,size_x,size_y\n";
  int count = 0;
  for (size_t row = 0; row < tiles_.size(); ++row) {
    for (size_t col = 0; col < tiles_[row].size(); ++col) {
      const Tile& tile = tiles_[row][col];
      tiles_file << (row * tiles_.size() + col) << "," << tile.x_min << ","
                 << tile.y_min << "," << tile.size_x << "," << tile.size_y
                 << "\n";
      count++;
    }
  }
  tiles_file.close();
}

void TilesGenerator::saveDownsampledMergedTiles(const std::string& output_dir) {
  auto merged_cloud =
      boost::make_shared<pcl::PointCloud<PointXYZRGBNormalDouble>>();

  for (size_t row = 0; row <tiles_.size(); ++row) {
    for (size_t col = 0; col < tiles_[row].size(); ++col) {
      const Tile& tile = tiles_[row][col];
      std::string filepath= tileFilepath(row, col);
      if (fs::exists(filepath)) {
        auto tile_cloud =
            boost::make_shared<pcl::PointCloud<PointXYZRGBNormalDouble>>();

        auto filtered_cloud =
            boost::make_shared<pcl::PointCloud<PointXYZRGBNormalDouble>>();

        readPointCloud(filepath, *tile_cloud);

        // downsampling the cloud
        // the right way to do it is to use a voxel grid filter
        // but it does not work with our custom point type
        pcl::RandomSample<PointXYZRGBNormalDouble> randomSample;
        randomSample.setInputCloud(tile_cloud);
        randomSample.setSample(
            tile_cloud->size() * 0.01);
        randomSample.filter(*filtered_cloud);

        *merged_cloud += *filtered_cloud;
      }
    }
  }

  // save merged cloud
  writePointCloud(output_dir + "/merged_cloud.ply", *merged_cloud);
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "tiles_generator");
  ros::NodeHandle nh("~");

  TilesGenerator gen;
  gen.setParams(nh);
  gen.generateTiling();
  return 0;
}
