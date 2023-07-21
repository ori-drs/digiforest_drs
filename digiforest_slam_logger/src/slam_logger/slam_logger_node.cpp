// Node to listen to point clouds and write SLAM output
#include <ros/node_handle.h>

#include <tf_conversions/tf_eigen.h>
#include <eigen_conversions/eigen_msg.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>

#include <sensor_msgs/PointCloud2.h>
#include <slam_logger/slam_logger_tools.hpp>

using namespace std;

struct CommandLineConfig
{
  double min_range;
  double max_range;
  string lidar_topic;
  string odom_frame;
  string odom_base_frame;
  string output_file_type;
  double reading_dist_threshold;
  int platform_id;
  std::vector<double> odometry_info;
};

class App{
  public:
    App(ros::NodeHandle& node, const CommandLineConfig &cl_config);
    
    ~App(){
    }    

    void pointcloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg);

    bool processLidar(const sensor_msgs::PointCloud2::ConstPtr& cloud_in,
                           IcpCloudType::Ptr& point_cloud_raw_in_base,
                           IcpCloudType::Ptr& point_cloud_in_odometry,
                           Time& time, Eigen::Isometry3d &base_to_odom);

    void writeResult();

  private:
    ros::NodeHandle& node_;

    CommandLineConfig cl_config_;

    std::stringstream data_directory_path_; // Create data folder for output

    Eigen::Isometry3d last_base_to_odom_;
    std::vector<IsometryWithTime, Eigen::aligned_allocator<IsometryWithTime> > all_base_to_odom_;

    tf::TransformListener listener_;
    //std::unique_ptr<SimpleViz> viz_;

    pcl::PCDWriter pcd_writer_;
    pcl::PLYWriter ply_writer_;

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
};


// Initialise the TF buffer with 60 seconds history to account for delayed local
// map clouds
App::App(ros::NodeHandle& node, const CommandLineConfig &cl_config):
    node_(node), cl_config_(cl_config){

  last_base_to_odom_ = Eigen::Isometry3d::Identity();

  data_directory_path_ << getEnvVar("HOME") << "/slam_logger/" << getTimeAsString();
  std::string path = data_directory_path_.str().c_str();
  boost::filesystem::path dir(path);
  if(boost::filesystem::exists(path))
    boost::filesystem::remove_all(path);

  if(boost::filesystem::create_directories(dir))
  {
    cout << "Created SLAM logger data directory: " << string(path) << endl
         << "============================" << endl;
  }

  // Create additional directory for odometry output
  std::string data_directory_path_individual = data_directory_path_.str() + "/individual_clouds";
  boost::filesystem::path dir3(data_directory_path_individual);
  boost::filesystem::create_directories(data_directory_path_individual);
}


void App::writeResult() {
  
  std::ofstream slam_problem_filestream;
  slam_problem_filestream.open( std::string ( data_directory_path_.str() + "/slam_pose_graph.g2o" ) );
  slam_problem_filestream << "# VERTEX_SE3:QUAT_TIME id x y z qx qy qz qw sec nsec\n";
  slam_problem_filestream << "# EDGE_SE3:QUAT tail_id head_id factor_x factor_y factor_z factor_qx factor_qy factor_qz factor_qw info_matrix\n";
  slam_problem_filestream << "# Information matrix is a 21 element upper triangular matrix ordered (r, R) which is not the GTSAM ordering\n";
  //slam_problem_filestream << "# GNSS_LLA_REF latitude longitude altitude\n";
  //slam_problem_filestream << "# These are the LLA coordinates of the initial pose\n";
  //slam_problem_filestream << "# GNSS_LLA_TO_MAP x y z qx qy qz qw \n";
  //slam_problem_filestream << "# This is the transform between the LLA coordinates (ENU frame) and map frame.\n";
  //slam_problem_filestream << "# TAG_DETECTION slam_id tag_id x y z qx qy qz qw sec nsec\n";
  //slam_problem_filestream << "# This is the raw slam-pose-to-tag measurement\n";
  //slam_problem_filestream << "# TAG_POSE slam_id tag_id x y z qx qy qz qw sec nsec\n";
  //slam_problem_filestream << "# This is the pose of the tag in the map frame\n";
  slam_problem_filestream << "PLATFORM_ID "<< cl_config_.platform_id <<"\n";

  for (size_t i=0; i < all_base_to_odom_.size(); i++){
    IsometryWithTime pose = all_base_to_odom_[i];
    const Eigen::Vector3d r = pose.pose.translation();
    const Eigen::Quaterniond q(pose.pose.rotation());

    std::stringstream ss6;
    ss6 << "VERTEX_SE3:QUAT_TIME " << i << " "
        << r[0] << " " << r[1] << " " << r[2] << " "
        << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << " "
        << pose.sec << " " << pose.nsec;
    slam_problem_filestream << ss6.str() << "\n";
  }

  for (size_t i=1; i < all_base_to_odom_.size(); i++){
    IsometryWithTime prev_pose = all_base_to_odom_[i-1];
    IsometryWithTime curr_pose = all_base_to_odom_[i];
    Eigen::Isometry3d pose_delta =  prev_pose.pose.inverse() * curr_pose.pose;

    const Eigen::Vector3d r = pose_delta.translation();
    const Eigen::Quaterniond q(pose_delta.rotation());

    std::stringstream ss6;
    ss6 << "EDGE_SE3:QUAT " << i-1 << " " << i << " "
        << r[0] << " " << r[1] << " " << r[2] << " "
        << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << " "
        << cl_config_.odometry_info[3] << " " << 0 << " " << 0 << " " << 0 << " " << 0 << " " << 0 << " " 
        << cl_config_.odometry_info[4] << " " << 0 << " " << 0 << " " << 0 << " " << 0 << " "
        << cl_config_.odometry_info[5] << " " << 0 << " " << 0 << " " << 0 << " "
        << cl_config_.odometry_info[0] << " " << 0 << " " << 0 << " " 
        << cl_config_.odometry_info[1] << " " << 0 << " " 
        << cl_config_.odometry_info[2];
    slam_problem_filestream << ss6.str() << "\n";
  }

  slam_problem_filestream.close();
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

  ros::Time msg_time(cloud_in->header.stamp.sec, cloud_in->header.stamp.nsec);
  time = {cloud_in->header.stamp.sec, cloud_in->header.stamp.nsec};
  Eigen::Isometry3d sensor_to_odom;
  Eigen::Isometry3d sensor_to_base;

  //////////////////////////////////// 1. usually odom_vilens to base_vilens_optimized
  tf::StampedTransform sensor_to_odom_tf;
  try {
    listener_.waitForTransform(cl_config_.odom_frame, cloud_in->header.frame_id, msg_time, ros::Duration(1.0));
    listener_.lookupTransform(cl_config_.odom_frame, cloud_in->header.frame_id, msg_time, sensor_to_odom_tf);
  } catch (tf::TransformException ex) {
    ROS_ERROR("%s : ", ex.what());
    ROS_ERROR("Skipping point cloud.");
    return false;
  }
  tf::transformTFToEigen(sensor_to_odom_tf, sensor_to_odom);

  //////////////////////////////////// 2 usually odom_vilens to base_vilens_optimized
  tf::StampedTransform base_to_odom_tf;
  try {
    listener_.waitForTransform(cl_config_.odom_frame, cl_config_.odom_base_frame, msg_time, ros::Duration(1.0));
    listener_.lookupTransform(cl_config_.odom_frame, cl_config_.odom_base_frame, msg_time, base_to_odom_tf);
  } catch (tf::TransformException ex) {
    ROS_ERROR("%s : ", ex.what());
    ROS_ERROR("Skipping point cloud.");
    return false;
  }
  tf::transformTFToEigen(base_to_odom_tf, base_to_odom);

  //////////////////////////////////// 3 usually base_vilens_optimized to base_vilens_optimized (TODO: should be avoided if possible!)
  tf::StampedTransform sensor_to_base_tf;
  try {
    listener_.waitForTransform(cl_config_.odom_base_frame, cloud_in->header.frame_id, msg_time, ros::Duration(1.0));
    listener_.lookupTransform(cl_config_.odom_base_frame, cloud_in->header.frame_id, msg_time, sensor_to_base_tf);
  } catch (tf::TransformException ex) {
    ROS_ERROR("%s : ", ex.what());
    ROS_ERROR("Skipping point cloud.");
    return false;
  }
  tf::transformTFToEigen(sensor_to_base_tf, sensor_to_base);

  // How much have you moved since last cloud output:
  Eigen::Isometry3d pose_delta =  last_base_to_odom_.inverse() * base_to_odom;
  double dist_travelled = pose_delta.translation().norm();
  if (all_base_to_odom_.size()==0){
    std::cout << "Write first cloud\n";
  }else if (dist_travelled < cl_config_.reading_dist_threshold){
    return false;
  }
  last_base_to_odom_ = base_to_odom;

  // Convert from ROS msg to PCL and add normals
  bool ros_msg_contains_normals = false;
  for (const sensor_msgs::PointField& f : cloud_in->fields) {
    if (f.name == "normal_x") {
      ros_msg_contains_normals = true;
      break;
    }
  }
  if (ros_msg_contains_normals) {
    pcl::fromROSMsg(*cloud_in, *point_cloud_in_odometry);
  } else {
    // needed to avoid warnings from PCL when converting XYZ to XYZINormal
    auto cloud_a = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    pcl::fromROSMsg(*cloud_in, *cloud_a);
    pcl::copyPointCloud(*cloud_a, *point_cloud_in_odometry);
    calculateCloudNormalsVoxelized(point_cloud_in_odometry, Eigen::Vector3d::Zero());
  }

  // Remove long and short returns
  Eigen::Matrix4f tmp = Eigen::MatrixXf::Identity(4, 4);
  if (cl_config_.min_range > 0){
    getPointsInOrientedBox(point_cloud_in_odometry, -cl_config_.min_range, cl_config_.min_range, tmp, true); // remove points inside a box (near origin)
  }
  getPointsInOrientedBox(point_cloud_in_odometry, -cl_config_.max_range, cl_config_.max_range, tmp); // remove points outside a box (far from origin)

  // transform point cloud to base and odometry frames
  pcl::transformPointCloudWithNormals(*point_cloud_in_odometry, *point_cloud_raw_in_base, sensor_to_base.translation().cast<float>(),
               Eigen::Quaternionf(sensor_to_base.rotation().cast<float>())); // transform raw cloud into <BASE> frame // TODO: this is identity usually, could skip
  pcl::transformPointCloudWithNormals(*point_cloud_in_odometry, *point_cloud_in_odometry, sensor_to_odom.translation().cast<float>(),
               Eigen::Quaternionf(sensor_to_odom.rotation().cast<float>())); // transform filtered cloud into odometry frame

  // Cache pose list
  IsometryWithTime this_base_to_odom = IsometryWithTime(base_to_odom, time.sec, time.nsec, 0); // id not used here
  all_base_to_odom_.push_back(this_base_to_odom);
  std::cout << "Write cloud " << all_base_to_odom_.size() << "\n";

  // Write point clouds and poses
  Eigen::Vector4f pose_trans = Eigen::Vector4f::Zero();
  pose_trans.block<3,1>(0,0) = base_to_odom.translation().cast<float>();
  point_cloud_raw_in_base->sensor_origin_ = pose_trans; // put odometry pose into point cloud VIEWPOINT field
  point_cloud_raw_in_base->sensor_orientation_ = Eigen::Quaternionf(base_to_odom.rotation().cast<float>());
  stringstream ss_pcdfile;
  ss_pcdfile << data_directory_path_.str() << "/individual_clouds/cloud_" ;
  ss_pcdfile << time.sec << "_";
  ss_pcdfile << std::setw(9) << std::setfill('0') << time.nsec;
  writePointCloud(ss_pcdfile.str(), *point_cloud_raw_in_base);
  writeResult();

  return true;
}


void getParamOrExit(const ros::NodeHandle &nh, const std::string &param_field, std::vector<double>& variable){
  if(!nh.getParam(param_field, variable)){
    std::cout << "Exiting. Couldn't find param: " << param_field << "\n";
    exit(-1);
  }

  for (int i=0;i<variable.size();i++){
    if(i==0){
      std::cout << param_field << ": " << variable[i] << ", ";
    }else if (i<variable.size()-1){
      std::cout << variable[i] << ", ";
    }else{
      std::cout << variable[i] << " (vector" << variable.size() <<"d)\n";
    }
  }
}

int main( int argc, char** argv ){
  ros::init(argc, argv, "slam_logger");
  ros::NodeHandle nh("~");

  CommandLineConfig cl_config;
  cl_config.min_range = 2.0;
  cl_config.max_range = 100.0;
  cl_config.lidar_topic ="/vilens/point_cloud_transformed_processed";
  cl_config.odom_frame = "base_vilens";
  cl_config.odom_base_frame = "base_vilens";
  cl_config.output_file_type = "pcd";
  cl_config.reading_dist_threshold = 1.0;
  cl_config.platform_id = 1;
  cl_config.odometry_info = {1e4, 1e4, 1e4, 1e6, 1e6, 1e6}; // order: roll, pitch, yaw, x, y, z

  nh.getParam("min_range", cl_config.min_range);
  nh.getParam("max_range", cl_config.max_range);
  nh.getParam("lidar_topic", cl_config.lidar_topic);
  nh.getParam("odom_frame", cl_config.odom_frame);
  nh.getParam("odom_base_frame", cl_config.odom_base_frame);
  nh.getParam("output_file_type", cl_config.output_file_type);
  nh.getParam("reading_dist_threshold", cl_config.reading_dist_threshold);
  nh.getParam("platform_id", cl_config.platform_id);
  getParamOrExit(nh, "odometry_info", cl_config.odometry_info);

  std::shared_ptr<App> app = std::make_shared<App>(nh, cl_config);
  ros::Subscriber lidar_sub = nh.subscribe(cl_config.lidar_topic, 100, &App::pointcloudCallback, app.get());

  ROS_INFO_STREAM("slam_logger ready");
  ros::spin();
  return 0;
}
