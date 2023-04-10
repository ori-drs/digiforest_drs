// Node to listen to point clouds and write to file
// Clouds assumed to be stored in the 

#include <ros/node_handle.h>

#include <tf_conversions/tf_eigen.h>
#include <eigen_conversions/eigen_msg.h>
#include <pcl/filters/passthrough.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/io/ply_io.h>

//#include <tf2_ros/transform_listener.h>
//#include <tf2_ros/buffer.h>
//#include <tf2_eigen/tf2_eigen.h>

#include <tf_conversions/tf_eigen.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>

//#include <icp_odometry/filter_tools.hpp>
//#include <icp_odometry/common.hpp>
//#include <icp_odometry/registered_cloud_list.hpp>
//#include <icp_odometry/ros_param_tools.hpp>
//#include <vilens_slam/simple_viz/simple_viz.hpp>

#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_srvs/Trigger.h>

using namespace std;
//using namespace icp_odometry;

using IcpPointType = pcl::PointXYZINormal;
using IcpCloudType = pcl::PointCloud<IcpPointType>;

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

struct CommandLineConfig
{
    string path_topic;
    string lidar_topic;
    double max_range;
    string estimation_frame;
    int add_cloud_every_n_poses;
    int recompute_every_n_poses;
    bool publish_colorized_map;
    double individual_cloud_voxel_filter_size;
    string output_data_directory;
    string output_file_type;
    bool keep_all_clouds_in_memory;
    bool reset_local_mapping;

    bool auto_logging;
};

std::string getEnvVar( std::string const & key )
{
    char * val = getenv( key.c_str() );
    return val == NULL ? std::string("") : std::string(val);
}

class App{
  public:
    App(ros::NodeHandle& node, const CommandLineConfig &cl_config);
    
    ~App(){
    }    

    void pointcloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg);
    void pathCallBack(const nav_msgs::Path::ConstPtr& msg);

    bool writeMapRequest(std_srvs::Trigger::Request& request, std_srvs::Trigger::Response& response);
    bool updateMapRequest(std_srvs::Trigger::Request& request, std_srvs::Trigger::Response& response);

    bool processLidar(const sensor_msgs::PointCloud2::ConstPtr& cloud_in,
                           IcpCloudType::Ptr& point_cloud_raw_in_base,
                           IcpCloudType::Ptr& point_cloud_in_odometry,
                           Time& time, Eigen::Isometry3d &base_to_odom);


  private:
    ros::NodeHandle& node_;

    CommandLineConfig cl_config_;
    std::string cloud_frame_id_;

    //tf2_ros::Buffer buffer_;
    //tf2_ros::TransformListener listener_;
    tf::TransformListener listener_;
    //std::unique_ptr<SimpleViz> viz_;

    pcl::PCDWriter pcd_writer_;
    pcl::PLYWriter ply_writer_;

    //std::deque<IcpCloudType::Ptr> recent_clouds_;
    //std::deque<Time> recent_times_;  

    IcpCloudType::Ptr slam_map_;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr slam_map_rgb_;

    ros::Publisher last_cloud_pub_; 
    ros::Publisher combined_cloud_pub_, combined_cloud_rgb_pub_; 
    //std::shared_ptr<RegisteredCloudList> registered_cloud_list_;

    ros::ServiceServer write_map_srv_;
    ros::ServiceServer update_map_srv_;

    void writeIndividualClouds(const IcpCloudType::Ptr cloud,
                         const Eigen::Isometry3d &body_pose, const Time &time);
    void addCloudToGraph(const IcpCloudType::Ptr cloud,
                         const Eigen::Isometry3d& body_pose, const Time& time);
    void updateSlamPoses(const nav_msgs::Path::ConstPtr &msg);

    void updateMap(bool downsample);
    void updateMapMain(int inc);
    void updateMapColorized(int inc);

    void publishCombinedMapCloud(const Time &time);

    Time last_update_time_;
    void publishCombinedMapCloudInternal(const Time &time, bool downsample);

    bool isCombinedMapRequested() const {
      return (combined_cloud_pub_.getNumSubscribers() > 0);
    }

    bool isCombinedMapRgbRequested() const {
      return (combined_cloud_rgb_pub_.getNumSubscribers() > 0);
    }

    // This function calls a ROS service to reset /local_mapping
    // This is dangerous if local_mapping is not running or configured properly
    void makeLocalMappingResetRequest();

    // Gateway function for file I/O - either pcd (binary, default) or ply (ascii)
    // Note: This is templated to allow for different point cloud types to be saved
    template <class T>
    void writePointCloud(std::string filename, pcl::PointCloud<T>& point_cloud) {
      if (cl_config_.output_file_type == "ply") {
        ply_writer_.write<T>(std::string(filename + ".ply"), point_cloud, true);
      } else {
        pcd_writer_.write<T>(std::string(filename + ".pcd"), point_cloud, true);
      }
    }

    int counter = 0;
    int slam_update_counter = -1;
};


// Initialise the TF buffer with 60 seconds history to account for delayed local
// map clouds
App::App(ros::NodeHandle& node, const CommandLineConfig &cl_config):
    node_(node), cl_config_(cl_config){

    //viz_ = std::make_unique<SimpleViz>();

    last_cloud_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/vilens_map/last_cloud", 10, true);
    combined_cloud_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/vilens_map/combined_cloud", 10, true);
    combined_cloud_rgb_pub_ = node_.advertise<sensor_msgs::PointCloud2>("/vilens_map/combined_cloud_rgb", 10, true);
    //registered_cloud_list_ = std::make_shared<RegisteredCloudList>();

    write_map_srv_ = node_.advertiseService("/vilens_map/write_map", &App::writeMapRequest, this);
    update_map_srv_ = node_.advertiseService("/vilens_map/update_map", &App::updateMapRequest, this);

    slam_map_ = boost::make_shared<IcpCloudType>();
    slam_map_rgb_ = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
}


bool App::writeMapRequest(std_srvs::Trigger::Request& request, std_srvs::Trigger::Response& response){
    std::cout << "Got request to save complete map. This can take a few minutes\n";
    updateMap(false);
    std::string filename = getEnvVar("HOME") + "/vilens_slam_data/" + "combined_cloud_vilens_map.pcd";
    std::cout << slam_map_->points.size() << " pts. Write to file as " << filename << "\n";
    pcd_writer_.write<IcpPointType> (filename, *slam_map_, true);
    std::cout << "Finished map writing\n";
    return response.success = true;
}


bool App::updateMapRequest(std_srvs::Trigger::Request& request, std_srvs::Trigger::Response& response){
    std::cout << "Got request to update live map.\n";
    publishCombinedMapCloudInternal(last_update_time_, false);
    std::cout << "Finished republish\n";
    return response.success = true;
}


void App::makeLocalMappingResetRequest(){
  std::cout <<"Making local map reset request\n";
  ros::ServiceClient reset_local_map_request = node_.serviceClient<std_srvs::Trigger>("/local_mapping/reset_local_map");
  std_srvs::Trigger srv;
  if (reset_local_map_request.call(srv)){
    ROS_INFO("Succes: %d", (int)srv.response.success);
  }else{
    ROS_ERROR("Failed to call service /local_mapping/reset_local_map");
  }
}


void App::writeIndividualClouds(const IcpCloudType::Ptr cloud,
                          const Eigen::Isometry3d &body_pose, const Time &time){

  IcpCloudType::Ptr output_cloud = boost::make_shared<IcpCloudType>();

  bool write_in_world_frame = false;
  if (write_in_world_frame){
    pcl::transformPointCloudWithNormals(*cloud, *output_cloud, body_pose.matrix().cast<float>() );
  }else{
    output_cloud = cloud;

    // populate the sensor_origin:
    Eigen::Vector4f pose_base_to_map_trans = Eigen::Vector4f::Zero();
    pose_base_to_map_trans.block<3,1>(0,0) = body_pose.translation().cast<float>();
    output_cloud->sensor_origin_ = pose_base_to_map_trans;
    output_cloud->sensor_orientation_ = Eigen::Quaternionf(body_pose.rotation().cast<float>());
  }

  std::stringstream ss_pcdfile;
  ss_pcdfile << cl_config_.output_data_directory << "/cloud_" ;
  ss_pcdfile << time.sec << "_";
  ss_pcdfile << std::setw(9) << std::setfill('0') << time.nsec;
  writePointCloud(ss_pcdfile.str(), *output_cloud);

  if (cl_config_.reset_local_mapping){
    makeLocalMappingResetRequest();
  }

}

bool isIdentity(const Eigen::Isometry3d &transform){
  const double delta_trans = transform.translation().norm();
  const double delta_angle = Eigen::AngleAxisd(transform.rotation()).angle();
  return ((delta_trans < 0.0001) && (delta_angle < 0.0001));
}


void App::addCloudToGraph(const IcpCloudType::Ptr cloud,
                          const Eigen::Isometry3d &body_pose, const Time &time){

}


void App::updateMap(bool downsample){

}

void App::updateMapMain(int inc){

}

void App::updateMapColorized(int inc){

}


void App::publishCombinedMapCloud(const Time &time){

}

void App::publishCombinedMapCloudInternal(const Time &time, bool downsample){

}


void App::updateSlamPoses(const nav_msgs::Path::ConstPtr &msg){

}


void App::pathCallBack(const nav_msgs::Path::ConstPtr &msg){

}


void App::pointcloudCallback(const sensor_msgs::PointCloud2::ConstPtr &msg){

  Eigen::Isometry3d base_to_odom;
  IcpCloudType::Ptr cloud_raw_in_base = boost::make_shared<IcpCloudType>();
  IcpCloudType::Ptr cloud_in_odom = boost::make_shared<IcpCloudType>();
  Time cloud_time;
  if (!processLidar(msg, cloud_raw_in_base, cloud_in_odom, cloud_time, base_to_odom)){
    std::cout << "Skipping Cloud\n";
    return;
  }

}


bool App::processLidar(const sensor_msgs::PointCloud2::ConstPtr& cloud_in,
                       IcpCloudType::Ptr& point_cloud_raw_in_base,
                       IcpCloudType::Ptr& point_cloud_in_odometry,
                       Time& time, Eigen::Isometry3d &base_to_odom){

  std::string param_odom_frame = "odom";
  std::string param_odom_base_frame = "base";

  ros::Time msg_time(cloud_in->header.stamp.sec, cloud_in->header.stamp.nsec);
  time = {cloud_in->header.stamp.sec, cloud_in->header.stamp.nsec};
  Eigen::Isometry3d sensor_to_odom;
  Eigen::Isometry3d sensor_to_base;

  //////////////////////////////////// 1. usually odom_vilens to base_vilens_optimized
  tf::StampedTransform sensor_to_odom_tf;
  try {
    listener_.waitForTransform(param_odom_frame, cloud_in->header.frame_id, msg_time, ros::Duration(1.0));
    listener_.lookupTransform(param_odom_frame, cloud_in->header.frame_id, msg_time, sensor_to_odom_tf);
  } catch (tf::TransformException ex) {
    ROS_ERROR("%s : ", ex.what());
    ROS_ERROR("Skipping point cloud.");
    return false;
  }
  tf::transformTFToEigen(sensor_to_odom_tf, sensor_to_odom);

  //////////////////////////////////// 2 usually odom_vilens to base_vilens_optimized
  tf::StampedTransform base_to_odom_tf;
  try {
    listener_.waitForTransform(param_odom_frame, param_odom_base_frame, msg_time, ros::Duration(1.0));
    listener_.lookupTransform(param_odom_frame, param_odom_base_frame, msg_time, base_to_odom_tf);
  } catch (tf::TransformException ex) {
    ROS_ERROR("%s : ", ex.what());
    ROS_ERROR("Skipping point cloud.");
    return false;
  }
  tf::transformTFToEigen(base_to_odom_tf, base_to_odom);

  //////////////////////////////////// 3 usually base_vilens_optimized to base_vilens_optimized (TODO: should be avoided if possible!)
  tf::StampedTransform sensor_to_base_tf;
  try {
    listener_.waitForTransform(param_odom_base_frame, cloud_in->header.frame_id, msg_time, ros::Duration(1.0));
    listener_.lookupTransform(param_odom_base_frame, cloud_in->header.frame_id, msg_time, sensor_to_base_tf);
  } catch (tf::TransformException ex) {
    ROS_ERROR("%s : ", ex.what());
    ROS_ERROR("Skipping point cloud.");
    return false;
  }
  tf::transformTFToEigen(sensor_to_base_tf, sensor_to_base);


  // Convert from ROS msg
  bool ros_msg_contains_normals = false;
  for (const sensor_msgs::PointField& f : cloud_in->fields) {
    if (f.name == "normal_x") {
      ros_msg_contains_normals = true;
      break;
    }
  }

  // HACK for now:
  pcl::fromROSMsg(*cloud_in, *point_cloud_in_odometry);
  /*
  if (ros_msg_contains_normals) {
    pcl::fromROSMsg(*cloud_in, *point_cloud_in_odometry);
  } else {
    // needed to avoid warnings from PCL when converting XYZ to XYZINormal
    auto cloud_a = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    pcl::fromROSMsg(*cloud_in, *cloud_a);
    pcl::copyPointCloud(*cloud_a, *point_cloud_in_odometry);
    calculateCloudNormalsVoxelized(point_cloud_in_odometry, Eigen::Vector3d::Zero());
  }
  */

  // Remove long and short returns
  //Eigen::Matrix4f tmp = Eigen::MatrixXf::Identity(4, 4);
  //if (slam_params_.min_range > 0){
  //  getPointsInOrientedBox(point_cloud_in_odometry, -slam_params_.min_range, slam_params_.min_range, tmp, true); // remove points inside a box (near origin)
  //}
  //getPointsInOrientedBox(point_cloud_in_odometry, -slam_params_.max_range, slam_params_.max_range, tmp); // remove points outside a box (far from origin)

  // transform point cloud to odometry frame
  pcl::transformPointCloudWithNormals(*point_cloud_in_odometry, *point_cloud_raw_in_base, sensor_to_base.translation().cast<float>(),
               Eigen::Quaternionf(sensor_to_base.rotation().cast<float>())); // transform raw cloud into <BASE> frame // TODO: this is identity usually, could skip
  pcl::transformPointCloudWithNormals(*point_cloud_in_odometry, *point_cloud_in_odometry, sensor_to_odom.translation().cast<float>(),
               Eigen::Quaternionf(sensor_to_odom.rotation().cast<float>())); // transform filtered cloud into odometry frame

  return true;
}



int main( int argc, char** argv ){
    ros::init(argc, argv, "map_accum");
    ros::NodeHandle nh("~");

    CommandLineConfig cl_config;
    cl_config.max_range = 20.0;
    cl_config.lidar_topic ="/os1_cloud_node/points";
    cl_config.path_topic = "/vilens_slam/slam_poses";
    cl_config.estimation_frame = "base";
    cl_config.add_cloud_every_n_poses = 1;
    cl_config.recompute_every_n_poses = 10;
    cl_config.publish_colorized_map = false;
    cl_config.individual_cloud_voxel_filter_size = 0.0; // dont voxel filter
    cl_config.output_file_type = "pcd";
    cl_config.auto_logging = false;
    cl_config.output_data_directory = "";
    cl_config.keep_all_clouds_in_memory = true;
    cl_config.reset_local_mapping = false;

    /*
    getParamOrExit(nh, "max_range", cl_config.max_range);
    getParamOrExit(nh, "lidar_topic", cl_config.lidar_topic);
    getParamOrExit(nh, "path_topic", cl_config.path_topic);
    getParamOrExit(nh, "estimation_frame", cl_config.estimation_frame);
    getParamOrExit(nh, "add_cloud_every_n_poses", cl_config.add_cloud_every_n_poses);
    getParamOrExit(nh, "recompute_every_n_poses", cl_config.recompute_every_n_poses);
    getParamOrExit(nh, "publish_colorized_map", cl_config.publish_colorized_map);
    getParamOrExit(nh, "individual_cloud_voxel_filter_size", cl_config.individual_cloud_voxel_filter_size);
    getParamOrExit(nh, "output_file_type", cl_config.output_file_type);
    getParamOrExit(nh, "auto_logging", cl_config.auto_logging);
    getParamOrExit(nh, "keep_all_clouds_in_memory", cl_config.keep_all_clouds_in_memory);
    getParamOrExit(nh, "reset_local_mapping", cl_config.reset_local_mapping);
    */

    /*
    // loop here until a folder param exists
    while (!nh.hasParam("/vilens_slam/output_data_directory")) {
      ROS_INFO("Looping while waiting for /vilens_slam/output_data_directory param");
      ros::Duration(0.5).sleep();
    }

    getParamOrExit(nh, "/vilens_slam/output_data_directory", cl_config.output_data_directory);
    cl_config.output_data_directory = cl_config.output_data_directory + "/payload_clouds";
    boost::filesystem::create_directories( cl_config.output_data_directory );
    std::cout<< "Writing output to: " << cl_config.output_data_directory << "\n";
    */

    std::shared_ptr<App> app = std::make_shared<App>(nh, cl_config);
    ros::Subscriber lidar_sub = nh.subscribe(cl_config.lidar_topic, 100, &App::pointcloudCallback, app.get());
    ros::Subscriber path_sub = nh.subscribe(cl_config.path_topic, 100, &App::pathCallBack, app.get());

    ROS_INFO_STREAM("map_accum ready");
    ros::spin();
    return 0;
}