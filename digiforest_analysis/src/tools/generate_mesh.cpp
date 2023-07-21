#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/ply_io.h>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>

int
main (int argc, char** argv)
{
  int dim = 40;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

  if (pcl::io::loadPCDFile<pcl::PointXYZ> ("/tmp/height_map.pcd", *cloud) == -1) //* load the file
  {
    PCL_ERROR ("Couldn't read file height_map.pcd \n");
    return (-1);
  }
  std::cout << "Loaded "
            << cloud->width * cloud->height
            << " data points from height_map.pcd with the following fields: "
            << std::endl;

  cloud->width    = dim;
  cloud->height   = dim;


  /*
  pcl::PointCloud<pcl::PointXYZ> cloud;
  // Fill in the cloud data
  cloud.width    = dim;
  cloud.height   = dim;
  cloud.is_dense = false;
  cloud.points.resize (cloud.width * cloud.height);

  for(int i=0; i < dim; i++){
    for(int j=0; j < dim; j++){
      pcl::PointXYZ pt = cloud.points[i*dim+j];
      pt.x=(float) i*4.;
      pt.y=(float) j*4.;
      pt.z=(float) 5.*(sin(pt.x/10.) * cos(pt.y/10.));
      cloud.points[i*dim+j] = pt;
    }
  }
  */

  //std::cout << dim << " is dim\n";
  std::vector<pcl::Vertices> polygons;
  for(int i=0; i < dim; i++){
    for(int j=0; j < dim; j++){
      pcl::PointXYZ pt = cloud->points[i*dim+j];
      
      if ((i>0) && (j>0)){
        pcl::Vertices face;
        face.vertices.push_back((i*dim)+j);
        face.vertices.push_back((i*dim)+j-1);
        face.vertices.push_back(((i-1)*dim)+j);
        polygons.push_back(face);

        pcl::Vertices face2;
        face2.vertices.push_back(((i-1)*dim)+(j-1));
        face2.vertices.push_back((i*dim)+j-1);
        face2.vertices.push_back(((i-1)*dim)+j);
        polygons.push_back(face2);

      }
    }
  }

  pcl::PCLPointCloud2 point_cloud2;
  pcl::toPCLPointCloud2(*cloud, point_cloud2);

  std::cout << polygons.size() << " polygons\n";
  pcl::PolygonMesh triangles;
  triangles.cloud = point_cloud2;
  triangles.polygons = polygons;

  //pcl::io::savePCDFileASCII ("test_pcd.pcd", *cloud);
  std::cerr << "Saved " << cloud->size () << " data points to height_map.ply." << std::endl;
  pcl::io::savePLYFile("/tmp/height_map.ply", triangles); 


  return (0);
}
