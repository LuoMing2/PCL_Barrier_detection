#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/console/time.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/progressive_morphological_filter.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/distances.h>


#include <unordered_map>
#include <pcl/search/kdtree.h>
#include <Eigen/Core>
#include "radius_kdtree.h"
#include <nlohmann/json.hpp>
#include "earth.h"



typedef pcl::PointXYZI PointTypeIO;
typedef pcl::PointXYZINormal PointTypeFull;

enum STATUS {LEFT, RIGHT};

class MyPoint : public std::array<double, 3>
{
public:
	// dimension of space (or "k" of k-d tree)
	// KDTree class accesses this member
	static const int DIM = 3;
  // bool flag = true;
  int index;
  STATUS status;
	// the constructors
	MyPoint() {}
	MyPoint(PointTypeIO pc, int k, STATUS status_)
	{ 
		(*this)[0] = pc.x;
		(*this)[1] = pc.y;
    (*this)[2] = pc.z;
    index = k;
    status = status_;
	}

	// conversion to Eigen vector3d
	operator Eigen::Vector3d() const { return Eigen::Vector3d((*this)[0], (*this)[1], (*this)[2]); }
};

PointTypeIO GetIntersectPointsForLines(PointTypeIO &line1_p1, PointTypeIO &line1_p2, 
                                          PointTypeIO &line2_p1, PointTypeIO &line2_p2, double z)
{
  Eigen::Vector3d line1, line2;
  line1(0) = line1_p2.y - line1_p1.y;
  line1(1) = line1_p1.x - line1_p2.x;
  line1(2) = line1_p1.y * line1_p2.x - line1_p1.x * line1_p2.y;
  line2(0) = line2_p2.y - line2_p1.y;
  line2(1) = line2_p1.x - line2_p2.x;
  line2(2) = line2_p1.y * line2_p2.x - line2_p1.x * line2_p2.y;
  PointTypeIO intersect_point;
  double norm = line1(0) * line2(1) - line1(1) * line2(0);
  intersect_point.x = (- line2(1) * line1(2) + line2(2) * line1(1)) / norm;
  intersect_point.y = (- line1(0) * line2(2) + line2(0) * line1(2)) / norm;
  intersect_point.z = z;
  return intersect_point;
}


bool enforceIntensitySimilarity (const PointTypeFull& point_a, const PointTypeFull& point_b, float squared_distance)
{
  if (std::abs (point_a.intensity - point_b.intensity) < 5.0f)
    return (true);
  else
    return (false);
}

bool enforceCurvatureOrIntensitySimilarity (const PointTypeFull& point_a, const PointTypeFull& point_b, float squared_distance)
{
  Eigen::Map<const Eigen::Vector3f> point_a_normal = point_a.getNormalVector3fMap (), point_b_normal = point_b.getNormalVector3fMap ();
  if (std::abs (point_a.intensity - point_b.intensity) < 5.0f)
    return (true);
  if (std::abs (point_a_normal.dot (point_b_normal)) < 0.05)
    return (true);
  return (false);
}

bool
customRegionGrowing (const PointTypeFull& point_a, const PointTypeFull& point_b, float squared_distance)
{
  Eigen::Map<const Eigen::Vector3f> point_a_normal = point_a.getNormalVector3fMap (), point_b_normal = point_b.getNormalVector3fMap ();
  if (squared_distance < 10000)
  {
    // if (std::abs (point_a.intensity - point_b.intensity) < 8.0f)
    //   return (true);
    if (std::abs (point_a_normal.dot (point_b_normal)) < 0.06)
      return (true);
  }
  // else
  // {
  //   if (std::abs (point_a.intensity - point_b.intensity) < 3.0f)
  //     return (true);
  // }
  return (false);
}

bool
LineRegionGrowing (const PointTypeFull& point_a, const PointTypeFull& point_b, float squared_distance)
{
    Eigen::Map<const Eigen::Vector3f> point_a_normal = point_a.getNormalVector3fMap (), point_b_normal = point_b.getNormalVector3fMap ();
    if (squared_distance < 0.3)
    {
        // if (std::abs (point_a.intensity - point_b.intensity) < 8.0f)
        // return (true);
        if (std::abs (point_a_normal.dot (point_b_normal)) > 0.99863)
            return (true);
    }
    // else
    // {
    //     // if (std::abs (point_a.intensity - point_b.intensity) < 3.0f)
    //         return (true);
    // }
    return (false);
}

int
main (int argc, char** argv)
{
    std::string input_path = std::string(argv[1]);
    // Data containers used
    pcl::PointCloud<PointTypeIO>::Ptr cloud_in(new pcl::PointCloud<PointTypeIO>), cloud_out (new pcl::PointCloud<PointTypeIO>);
    pcl::PointCloud<PointTypeFull>::Ptr cloud_with_normals (new pcl::PointCloud<PointTypeFull>);
    pcl::IndicesClustersPtr clusters (new pcl::IndicesClusters), small_clusters (new pcl::IndicesClusters), large_clusters (new pcl::IndicesClusters);
    pcl::search::KdTree<PointTypeIO>::Ptr search_tree(new pcl::search::KdTree<PointTypeIO>);
    pcl::console::TicToc tt;

    // Load the input point cloud
    std::cerr << "Loading...\n", tt.tic ();
    pcl::io::loadPCDFile(input_path, *cloud_in);
    std::cerr << ">> Done: " << tt.toc () << " ms, " << cloud_in->size () << " points\n";

    // Downsample the cloud using a Voxel Grid class
    std::cerr << "Downsampling...\n", tt.tic ();
    // pcl::PassThrough<PointTypeIO> pass;
    // pass.setInputCloud (cloud_in);
    // pass.setFilterFieldName ("z");
    // pass.setFilterLimits (0, 2);
    // //pass.setFilterLimitsNegative (true);
    // pass.filter (*cloud_out);
    pcl::StatisticalOutlierRemoval<PointTypeIO> sor;
    sor.setInputCloud (cloud_in);
    sor.setMeanK(50);
    sor.setStddevMulThresh (0.3);
    sor.filter(*cloud_out);
    std::cerr << ">> Done: " << tt.toc () << " ms, " << cloud_out->size () << " points\n";

    // Set up a Normal Estimation class and merge data in cloud_with_normals
    std::cerr << "Computing normals...\n", tt.tic ();
    pcl::copyPointCloud(*cloud_out, *cloud_with_normals);
    pcl::NormalEstimation<PointTypeIO, PointTypeFull> ne;
    ne.setInputCloud (cloud_out);
    ne.setSearchMethod (search_tree);
    // ne.setViewPoint(-1194.94, -809.47, -6.56);
    // ne.setKSearch(1000);
    ne.setRadiusSearch (0.3);
    ne.compute(*cloud_with_normals);
    pcl::io::savePCDFileASCII("output_point_with_normal.pcd", *cloud_with_normals);
    std::cerr << ">> Done: " << tt.toc () << " ms\n";

    // Set up a Conditional Euclidean Clustering class
    std::cerr << "Segmenting to clusters...\n", tt.tic ();
    pcl::ConditionalEuclideanClustering<PointTypeFull> cec (true);
    cec.setInputCloud (cloud_with_normals);
    cec.setConditionFunction (&customRegionGrowing);
    cec.setClusterTolerance (0.3);
    cec.setMinClusterSize (50);
    cec.setMaxClusterSize (50000);
    cec.segment(*clusters);
    cec.getRemovedClusters (small_clusters, large_clusters);
    std::cerr << ">> Done: " << tt.toc () << " ms\n";

    std::cerr << "recording unnormal points index...\n", tt.tic ();
    std::unordered_map<int, int> unnormal_points_index;
    for (int i = 0; i < cloud_with_normals->size (); ++i)
    {
        if(abs((*cloud_with_normals)[i].normal_z) > 0.1){
          unnormal_points_index[i]++;
          continue;
        }

        (*cloud_with_normals)[i].z = 0;
    }   
    std::cerr << ">> Done: " << tt.toc () << " ms\n";

    // Set up a Line Conditional Euclidean Clustering class
    pcl::IndicesClustersPtr line_clusters (new pcl::IndicesClusters), \
        line_small_clusters (new pcl::IndicesClusters), \
        line_large_clusters (new pcl::IndicesClusters);
    std::cerr << "Segmenting to clusters...\n", tt.tic ();
    pcl::ConditionalEuclideanClustering<PointTypeFull> cec_line (true);
    cec_line.setInputCloud(cloud_with_normals);
    cec_line.setConditionFunction (&LineRegionGrowing);
    cec_line.setClusterTolerance (0.3);
    cec_line.setMinClusterSize (50);
    cec_line.setMaxClusterSize (50000);
    cec_line.segment (*line_clusters);
    cec_line.getRemovedClusters (line_small_clusters, line_large_clusters);
    std::cerr << ">> Done: " << tt.toc () << " ms\n";

    pcl::PointCloud<PointTypeIO>::Ptr corner_point(new pcl::PointCloud<PointTypeIO>); //ouput pcd 
    std::vector<std::vector<std::pair<PointTypeIO, bool>>> end_points;
    std::cerr << "Saving...\n", tt.tic ();

    for (int i = 0; i < line_clusters->size (); ++i)
    {
      int label = rand () % 8;
      pcl::PointCloud<PointTypeIO>::Ptr line_cloud_cluster(new pcl::PointCloud<PointTypeIO>);
      for(int j = 0; j < (*line_clusters)[i].indices.size (); ++j)
      {
        if(unnormal_points_index.find((*line_clusters)[i].indices[j]) != unnormal_points_index.end())
        {
          continue; //do not save and use unnormal points 
        }
        (*cloud_out)[(*line_clusters)[i].indices[j]].intensity = label;
        (*cloud_out)[(*line_clusters)[i].indices[j]].z = 0;
        line_cloud_cluster->push_back((*cloud_out)[(*line_clusters)[i].indices[j]]);
      }

      if(line_cloud_cluster->size() == 0) continue;
      std::cout << "PointCloud representing the " << i << "th Cluster: " << line_cloud_cluster->size () << " data points." << std::endl;
      std::string output_ps= "cloud_cluster_" + std::to_string(i) + ".pcd";
      pcl::io::savePCDFileASCII<PointTypeIO>(output_ps, *line_cloud_cluster); //*

      //选端点
      PointTypeIO center;
      for(int k = 0; k < line_cloud_cluster->size(); k++)
      {
        center.x += line_cloud_cluster->points[k].x;
        center.y += line_cloud_cluster->points[k].y;
        center.z += line_cloud_cluster->points[k].z;
      }
      center.x /= line_cloud_cluster->size();
      center.y /= line_cloud_cluster->size();
      center.z /= line_cloud_cluster->size();

      PointTypeIO left_p;
      double max_dis = VTK_DOUBLE_MIN;
      for(int k = 0; k < line_cloud_cluster->size(); k++)
      {
        double cur_dis = pcl::euclideanDistance(center, line_cloud_cluster->points[k]);
        if(cur_dis > max_dis)
        {
          left_p = line_cloud_cluster->points[k];
          max_dis = cur_dis;
        }
      }

      PointTypeIO right_p;
      max_dis = VTK_DOUBLE_MIN;
      for(int k = 0; k < line_cloud_cluster->size(); k++)
      {
        double cur_dis = pcl::euclideanDistance(left_p, line_cloud_cluster->points[k]);
        if(cur_dis > max_dis)
        {
          right_p = line_cloud_cluster->points[k];
          max_dis = cur_dis;
        }
      }

      //line fitting, then project two endpoint to this line 
      pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
      pcl::PointIndices::Ptr inliers(new pcl::PointIndices);  
      pcl::SACSegmentation<PointTypeIO> seg;     
      seg.setOptimizeCoefficients(true);      
      seg.setModelType(pcl::SACMODEL_LINE);  
      seg.setMethodType(pcl::SAC_RANSAC);     
      seg.setDistanceThreshold(0.4);         
      seg.setInputCloud(line_cloud_cluster);              
      seg.segment(*inliers, *coefficients);

      //projection
      Eigen::Vector3d P_(left_p.x, left_p.y, left_p.z);
      Eigen::Vector3d line_direction(coefficients->values[3], coefficients->values[4], coefficients->values[5]);
      Eigen::Vector3d line_origin_p(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
      //left endpoint
      P_ -= line_origin_p;
      double cost = (P_.transpose() * line_direction);
      Eigen::Vector3d projection_p = cost * line_direction;
      projection_p += line_origin_p;
      left_p.x = projection_p(0);
      left_p.y = projection_p(1);
      left_p.z = projection_p(2);
      //right endpoint
      P_ << right_p.x, right_p.y, right_p.z;
      P_ -= line_origin_p;
      cost = (P_.transpose() * line_direction);
      projection_p = cost * line_direction;
      projection_p += line_origin_p;
      right_p.x = projection_p(0);
      right_p.y = projection_p(1);
      right_p.z = projection_p(2);


      std::vector<std::pair<PointTypeIO, bool>> endpoint_pair; //make pair with two point
      endpoint_pair.push_back(std::make_pair(left_p, 1));
      corner_point->push_back(left_p);
      // printf("point location = (%f, %f, %f)\n", right_p.x, right_p.y, right_p.z);
      endpoint_pair.push_back(std::make_pair(right_p, 1));
      corner_point->push_back(right_p);
      end_points.push_back(endpoint_pair);
    }

    std::string output_end_points= "end_points.pcd";
    pcl::io::savePCDFileASCII<PointTypeIO>(output_end_points, *corner_point); //*

    std::vector<MyPoint> cluster_corner_points;
    for(int k = 0; k < end_points.size(); k++)
    {
      // printf("point location = (%f, %f, %f)\n", end_points[k][0].first.x, end_points[k][0].first.y,end_points[k][0].first.z);
      cluster_corner_points.push_back(MyPoint(end_points[k][0].first, k, STATUS::LEFT));
      cluster_corner_points.push_back(MyPoint(end_points[k][1].first, k, STATUS::RIGHT));
    }
    kdt::KDTree<MyPoint> decluster_tree(cluster_corner_points);
    //clustering points, which should belong to one part, like pilars or walls
    printf("end_points size = %d\n", end_points.size());
    for(int k = 0; k < end_points.size(); k++)
    {
      if(end_points[k][0].second == true)
      {
        std::vector<int> indexs = decluster_tree.radiusSearch(MyPoint(end_points[k][0].first, k, STATUS::LEFT), 0.5);
        //hard to distinguish the same endpoint, can't search by nearest search
        int index;
        double dis_min = VTK_DOUBLE_MAX;
        printf("size = %d\n", indexs.size());
        if(indexs.size() == 1)
          continue;
        else
        {
          //find nearest point, kick out itself
          for(auto &id : indexs)
          {
            double distance;
            if(cluster_corner_points[id].status == STATUS::LEFT)
              distance = pcl::euclideanDistance(end_points[k][0].first, end_points[cluster_corner_points[id].index][0].first);
            else 
              distance = pcl::euclideanDistance(end_points[k][0].first, end_points[cluster_corner_points[id].index][1].first);
            if(distance == 0.0)
              continue;
            else if(distance < dis_min)
            {
              dis_min = distance;
              index = id;
            }
          }
          printf("distance = %f\n", dis_min);
        }

        if(cluster_corner_points[index].status == STATUS::LEFT)
        {
          if(end_points[cluster_corner_points[index].index][0].second == false)
            continue;
        }
        else
        {
          if(end_points[cluster_corner_points[index].index][1].second == false)
            continue;
        }
        //small line segement will search itself endpoint  
        if(cluster_corner_points[index].index == k)
          continue;
        
        //intersection point
        if(dis_min < 0.5)
        {
          //make sure two line at the same plane
          end_points[k][1].first.z = end_points[k][0].first.z;
          end_points[cluster_corner_points[index].index][0].first.z = end_points[k][0].first.z;
          end_points[cluster_corner_points[index].index][1].first.z = end_points[k][0].first.z;
          Eigen::Vector2d line1_normal(end_points[k][1].first.x - end_points[k][0].first.x, 
                                       end_points[k][1].first.y - end_points[k][0].first.y);
          Eigen::Vector2d line2_normal(end_points[cluster_corner_points[index].index][1].first.x - end_points[cluster_corner_points[index].index][0].first.x, 
                                       end_points[cluster_corner_points[index].index][1].first.y - end_points[cluster_corner_points[index].index][0].first.y);
          //perpendicular
          std::cout << "line_normal = " << line1_normal.normalized().transpose() << "line2_normal = " << line2_normal.normalized().transpose() << std::endl;
          printf("cos value of two vector = %f\n", line1_normal.normalized().transpose() * line2_normal.normalized());
          if(abs(line1_normal.normalized().transpose() * line2_normal.normalized()) <= 0.8) // small than 30 degree
          {
            PointTypeIO intersection_point = GetIntersectPointsForLines(end_points[k][0].first, end_points[k][1].first,
                                                                        end_points[cluster_corner_points[index].index][0].first,
                                                                        end_points[cluster_corner_points[index].index][1].first,
                                                                        end_points[k][0].first.z);
            end_points[k][0].first = intersection_point;
            printf("point location = (%f, %f, %f)\n", intersection_point.x, intersection_point.y, intersection_point.z);
            end_points[k][0].second = false;
            printf("intersection point + 1\n");
            if(cluster_corner_points[index].status == STATUS::LEFT)
            {
              end_points[cluster_corner_points[index].index][0].first = intersection_point;
              end_points[cluster_corner_points[index].index][0].second = false;
            }                                                              
            else
            {
              end_points[cluster_corner_points[index].index][1].first = intersection_point;
              end_points[cluster_corner_points[index].index][1].second = false; 
            }
          }
          else if(abs(line1_normal.normalized().transpose() * line2_normal.normalized()) > 0.8) //parallel
          {
            PointTypeIO mid_point;
            if(cluster_corner_points[index].status == STATUS::LEFT)
            {
              mid_point.x = (end_points[k][0].first.x + end_points[cluster_corner_points[index].index][0].first.x) / 2;
              mid_point.y = (end_points[k][0].first.y + end_points[cluster_corner_points[index].index][0].first.y) / 2;
              mid_point.z = (end_points[k][0].first.z + end_points[cluster_corner_points[index].index][0].first.z) / 2;
              end_points[cluster_corner_points[index].index][0].first = mid_point;
              end_points[k][0].first = mid_point;
              end_points[k][0].second = false;
              end_points[cluster_corner_points[index].index][0].second = false;
            }                                                              
            else
            {
              mid_point.x = (end_points[k][0].first.x + end_points[cluster_corner_points[index].index][1].first.x) / 2;
              mid_point.y = (end_points[k][0].first.y + end_points[cluster_corner_points[index].index][1].first.y) / 2;
              mid_point.z = (end_points[k][0].first.z + end_points[cluster_corner_points[index].index][1].first.z) / 2;
              end_points[cluster_corner_points[index].index][1].first = mid_point;
              end_points[k][0].first = mid_point;
              end_points[k][0].second = false;
              end_points[cluster_corner_points[index].index][1].second = false;
            }
          }
        }
      }
      if(end_points[k][1].second == true)
      {
        std::vector<int> indexs = decluster_tree.radiusSearch(MyPoint(end_points[k][1].first, k, STATUS::RIGHT), 0.5);
        
        int index;
        double dis_min = VTK_DOUBLE_MAX;
        printf("size = %d\n", indexs.size());
        if(indexs.size() == 1)
          continue;
        else
        {
          //find nearest point 
          for(auto &id : indexs)
          {
            double distance;
            if(cluster_corner_points[id].status == STATUS::LEFT)
              distance = pcl::euclideanDistance(end_points[k][1].first, end_points[cluster_corner_points[id].index][0].first);
            else 
              distance = pcl::euclideanDistance(end_points[k][1].first, end_points[cluster_corner_points[id].index][1].first);
            if(distance == 0.0)
              continue;
            else if(distance < dis_min)
            {
              dis_min = distance;
              index = id;
            }
          }
          printf("distance = %f\n", dis_min);
        }
        
        if(cluster_corner_points[index].status == STATUS::LEFT)
        {
          if(end_points[cluster_corner_points[index].index][0].second == false)
            continue;
        }
        else
        {
          if(end_points[cluster_corner_points[index].index][1].second == false)
            continue;
        }
        //small line segement will search itself endpoint  
        if(cluster_corner_points[index].index == k)
          continue;
        
        //intersection point
        if(dis_min < 0.5)
        {
          //make sure two line at the same plane
          end_points[k][1].first.z = end_points[k][0].first.z;
          end_points[cluster_corner_points[index].index][0].first.z = end_points[k][0].first.z;
          end_points[cluster_corner_points[index].index][1].first.z = end_points[k][0].first.z;
          Eigen::Vector2d line1_normal(end_points[k][1].first.x - end_points[k][0].first.x, 
                                       end_points[k][1].first.y - end_points[k][0].first.y);
          Eigen::Vector2d line2_normal(end_points[cluster_corner_points[index].index][1].first.x - end_points[cluster_corner_points[index].index][0].first.x, 
                                       end_points[cluster_corner_points[index].index][1].first.y - end_points[cluster_corner_points[index].index][0].first.y);
          //perpendicular
          if(abs(line1_normal.normalized().transpose() * line2_normal.normalized()) <= 0.8) // small than 3 degree
          {
            PointTypeIO intersection_point = GetIntersectPointsForLines(end_points[k][0].first, end_points[k][1].first,
                                                                        end_points[cluster_corner_points[index].index][0].first,
                                                                        end_points[cluster_corner_points[index].index][1].first,
                                                                        end_points[k][0].first.z);
            //change status after the endpoint has been dealt
            end_points[k][1].first = intersection_point;
            // printf("point location = (%f, %f, %f)\n", intersection_point.x,intersection_point.y, intersection_point.z);
            end_points[k][1].second = false;
            // printf("intersection point + 1\n");
            if(cluster_corner_points[index].status == STATUS::LEFT)
            {
              end_points[cluster_corner_points[index].index][0].first = intersection_point;
              end_points[cluster_corner_points[index].index][0].second = false;
            }                                                              
            else
            {
              end_points[cluster_corner_points[index].index][1].first = intersection_point;
              end_points[cluster_corner_points[index].index][1].second = false; 
            }
            printf("intersection point for endpoint = (%f, %f, %f)\n", end_points[k][1].first.x,end_points[k][1].first.y, end_points[k][1].first.z);
            printf("point location = (%f, %f, %f)\n", end_points[cluster_corner_points[index].index][1].first.x,
                                                      end_points[cluster_corner_points[index].index][1].first.y, 
                                                      end_points[cluster_corner_points[index].index][1].first.z);
          }
          else if(abs(line1_normal.normalized().transpose() * line2_normal.normalized()) > 0.8) //parallel
          {
            PointTypeIO mid_point;
            if(cluster_corner_points[index].status == STATUS::LEFT)
            {
              mid_point.x = (end_points[k][1].first.x + end_points[cluster_corner_points[index].index][0].first.x) / 2;
              mid_point.y = (end_points[k][1].first.y + end_points[cluster_corner_points[index].index][0].first.y) / 2;
              mid_point.z = (end_points[k][1].first.z + end_points[cluster_corner_points[index].index][0].first.z) / 2;
              end_points[cluster_corner_points[index].index][0].first = mid_point;
              end_points[k][1].first = mid_point;
              end_points[k][1].second = false;
              end_points[cluster_corner_points[index].index][0].second = false;
            }                                                              
            else
            {
              mid_point.x = (end_points[k][1].first.x + end_points[cluster_corner_points[index].index][1].first.x) / 2;
              mid_point.y = (end_points[k][1].first.y + end_points[cluster_corner_points[index].index][1].first.y) / 2;
              mid_point.z = (end_points[k][1].first.z + end_points[cluster_corner_points[index].index][1].first.z) / 2;
              end_points[cluster_corner_points[index].index][1].first = mid_point;
              end_points[k][1].first = mid_point;
              end_points[k][1].second = false;
              end_points[cluster_corner_points[index].index][1].second = false;
            }
          }
        }
      }
    }

    for(int k = 0; k < end_points.size(); k++)
    {
      end_points[k][0].second = true;
      end_points[k][1].second = true;
    }
    
    //construct kd tree again
    std::vector<MyPoint> cluster_corner_points_new;
    for(int k = 0; k < end_points.size(); k++)
    {
      // printf("point location = (%f, %f, %f)\n", end_points[k][0].first.x, end_points[k][0].first.y,end_points[k][0].first.z);
      cluster_corner_points_new.push_back(MyPoint(end_points[k][0].first, k, STATUS::LEFT));
      cluster_corner_points_new.push_back(MyPoint(end_points[k][1].first, k, STATUS::RIGHT));
    }
    kdt::KDTree<MyPoint> cluster_tree(cluster_corner_points_new);

    std::vector<std::vector<PointTypeIO>> merge_result;

    for(int k = 0; k < end_points.size(); k++)
    {
      std::vector<int> points = cluster_tree.radiusSearch(MyPoint(end_points[k][0].first, k, STATUS::LEFT), 0.001);
      printf("find merge size = %d\n", points.size());
      if(points.size() > 0)
      {
        // int index = points[0];
        // printf("point location = (%f, %f, %f)\n", end_points[k][1].first.x,end_points[k][1].first.y, end_points[k][1].first.z);
        // printf("point location = (%f, %f, %f)\n", end_points[k][0].first.x,end_points[k][0].first.y, end_points[k][0].first.z);
        // printf("point location = (%f, %f, %f)\n", end_points[cluster_corner_points[index].index][1].first.x,
        //                                           end_points[cluster_corner_points[index].index][1].first.y, 
        //                                           end_points[cluster_corner_points[index].index][1].first.z);
        // printf("point location = (%f, %f, %f)\n", end_points[cluster_corner_points[index].index][0].first.x,
        //                                           end_points[cluster_corner_points[index].index][0].first.y, 
        //                                           end_points[cluster_corner_points[index].index][0].first.z);
        if(points.size() == 1)
        {
          printf("find the same point\n");
          std::vector<PointTypeIO> obj;
          obj.push_back(end_points[k][0].first);
          obj.push_back(end_points[k][1].first);
          end_points[k][1].second = false;
          end_points[k][0].second = false;
          merge_result.push_back(obj);
        }
        else
        {
          std::vector<PointTypeIO> obj;
          if(points[0] != k)
          {
            obj.push_back(end_points[k][1].first);
            obj.push_back(end_points[k][0].first);
            end_points[k][0].second = false;
            end_points[k][1].second = false;  //TODO
            int index = points[0];    //judge is the same point
            while(1)
            {
              if(cluster_corner_points_new[index].status == STATUS::LEFT && end_points[cluster_corner_points_new[index].index][0].second == true)
              {
                end_points[cluster_corner_points_new[index].index][0].second = false;
                // obj.push_back(end_points[index][0].first);
                end_points[cluster_corner_points_new[index].index][1].second = false;
                obj.push_back(end_points[cluster_corner_points_new[index].index][1].first);
                //find the other endpoint
                points = cluster_tree.radiusSearch(MyPoint(end_points[cluster_corner_points_new[index].index][1].first, k, STATUS::LEFT), 0.001);
              }
              else if(cluster_corner_points_new[index].status == STATUS::RIGHT && end_points[cluster_corner_points_new[index].index][1].second == true)
              {
                end_points[cluster_corner_points_new[index].index][1].second = false;
                // obj.push_back(end_points[index][1].first);
                end_points[cluster_corner_points_new[index].index][0].second = false;
                obj.push_back(end_points[cluster_corner_points_new[index].index][0].first);
                points = cluster_tree.radiusSearch(MyPoint(end_points[cluster_corner_points_new[index].index][0].first, k, STATUS::LEFT), 0.001);
              }
              else
              {
                break;
              }

              if(points.size() != 2)
                break;
              index = (points[0] != index) ? points[0] : points[1]; //judge find intersect which line 
            }
            printf("find the same point\n");
            merge_result.push_back(obj);
          }
          else if(points[1] != k)
          {
            obj.push_back(end_points[k][1].first);
            obj.push_back(end_points[k][0].first);
            end_points[k][0].second = false;
            end_points[k][1].second = false;  //TODO
            int index = points[1];    //judge is the same point
            while(1)
            {
              if(cluster_corner_points_new[index].status == STATUS::LEFT && end_points[cluster_corner_points_new[index].index][0].second == true)
              {
                end_points[cluster_corner_points_new[index].index][0].second = false;
                // obj.push_back(end_points[index][0].first);
                end_points[cluster_corner_points_new[index].index][1].second = false;
                obj.push_back(end_points[cluster_corner_points_new[index].index][1].second);
                //find the other endpoint
                points = cluster_tree.radiusSearch(MyPoint(end_points[cluster_corner_points_new[index].index][1].first, k, STATUS::LEFT), 0.001);
              }
              else if(cluster_corner_points_new[index].status == STATUS::RIGHT && end_points[cluster_corner_points_new[index].index][1].second == true)
              {
                end_points[cluster_corner_points_new[index].index][1].second = false;
                // obj.push_back(end_points[index][1].first);
                end_points[cluster_corner_points_new[index].index][0].second = false;
                obj.push_back(end_points[cluster_corner_points_new[index].index][0].first);
                points = cluster_tree.radiusSearch(MyPoint(end_points[cluster_corner_points_new[index].index][0].first, k, STATUS::LEFT), 0.001);
              }
              else
              {
                break;
              }

              if(points.size() != 2)
                break;
              index = (points[0] != index) ? points[0] : points[1]; //judge find intersect which line 
            }
            printf("find the same point\n");
            merge_result.push_back(obj);
          }
        }
      }

      //judge another end_point, whether has been dealt with 
      if(end_points[k][1].second == true)
      {
        points = cluster_tree.radiusSearch(MyPoint(end_points[k][1].first, k, STATUS::LEFT), 0.001);
        if(points.size() > 0)
        {
          if(points.size() == 1)
          {
            printf("find the same point\n");
            std::vector<PointTypeIO> obj;
            obj.push_back(end_points[k][1].first);
            obj.push_back(end_points[k][0].first);
            end_points[k][1].second = false;
            end_points[k][0].second = false;
            merge_result.push_back(obj);
          }
          else
          {
            std::vector<PointTypeIO> obj;
            if(points[0] != k)
            {
              obj.push_back(end_points[k][0].first);
              obj.push_back(end_points[k][1].first);
              end_points[k][0].second = false;
              end_points[k][1].second = false;  //TODO
              int index = points[0];    //judge is the same point
              while(1)
              {
                if(cluster_corner_points_new[index].status == STATUS::LEFT && end_points[cluster_corner_points_new[index].index][0].second == true)
                {
                  end_points[cluster_corner_points_new[index].index][0].second = false;
                  // obj.push_back(end_points[index][0].first);
                  end_points[cluster_corner_points_new[index].index][1].second = false;
                  obj.push_back(end_points[cluster_corner_points_new[index].index][1].first);
                  //find the other endpoint
                  points = cluster_tree.radiusSearch(MyPoint(end_points[cluster_corner_points_new[index].index][1].first, k, STATUS::LEFT), 0.001);
                }
                else if(cluster_corner_points_new[index].status == STATUS::RIGHT && end_points[cluster_corner_points_new[index].index][1].second == true)
                {
                  end_points[cluster_corner_points_new[index].index][1].second = false;
                  // obj.push_back(end_points[index][1].first);
                  end_points[cluster_corner_points_new[index].index][0].second = false;
                  obj.push_back(end_points[cluster_corner_points_new[index].index][0].first);
                  points = cluster_tree.radiusSearch(MyPoint(end_points[cluster_corner_points_new[index].index][0].first, k, STATUS::LEFT), 0.001);
                }
                else
                {
                  break;
                }
                if(points.size() != 2)
                  break;
                index = (points[0] != index) ? points[0] : points[1]; //judge find intersect which line 
              }
              printf("find the same point\n");
              merge_result.push_back(obj);
            }
            else if(points[1] != k)
            {
              obj.push_back(end_points[k][0].first);
              obj.push_back(end_points[k][1].first);
              end_points[k][0].second = false;
              end_points[k][1].second = false;  //TODO
              int index = points[1];    //judge is the same point
              while(1)
              {
                if(cluster_corner_points_new[index].status == STATUS::LEFT && end_points[cluster_corner_points_new[index].index][0].second == true)
                {
                  end_points[cluster_corner_points_new[index].index][0].second = false;
                  // obj.push_back(end_points[index][0].first);
                  end_points[cluster_corner_points_new[index].index][1].second = false;
                  obj.push_back(end_points[cluster_corner_points_new[index].index][1].second);
                  //find the other endpoint
                  points = cluster_tree.radiusSearch(MyPoint(end_points[cluster_corner_points_new[index].index][1].first, k, STATUS::LEFT), 0.001);
                }
                else if(cluster_corner_points_new[index].status == STATUS::RIGHT && end_points[cluster_corner_points_new[index].index][1].second == true)
                {
                  end_points[cluster_corner_points_new[index].index][1].second = false;
                  // obj.push_back(end_points[index][1].first);
                  end_points[cluster_corner_points_new[index].index][0].second = false;
                  obj.push_back(end_points[cluster_corner_points_new[index].index][0].first);
                  points = cluster_tree.radiusSearch(MyPoint(end_points[cluster_corner_points_new[index].index][0].first, k, STATUS::LEFT), 0.001);
                }
                else
                {
                  break;
                }
                if(points.size() != 2)
                  break;
                index = (points[0] != index) ? points[0] : points[1]; //judge find intersect which line 
              }
              printf("find the same point\n");
              merge_result.push_back(obj);
            }
          }
        }
      }
    }

    pcl::PointCloud<PointTypeIO>::Ptr cluster_endpoint_cloud_output(new pcl::PointCloud<PointTypeIO>); //ouput pcd 
    for(auto &result: merge_result)
    {
      for(auto & p : result)
      {
        cluster_endpoint_cloud_output->push_back(p);
      }
    }
    
    std::string cluster_endpoint_cloud_output_path = "cluster_endpoint_cloud_output.pcd";
    pcl::io::savePCDFileASCII<PointTypeIO>(cluster_endpoint_cloud_output_path, *cluster_endpoint_cloud_output); //*

    printf("merge result size = %d\n", merge_result.size());

    nlohmann::json output_json_coner;
    output_json_coner["type"] = "FeatureCollection";
    std::string json_path = "barrier_detection_result.json";
    std::ofstream fileout(json_path);
    if(!fileout.is_open())
    {
        printf("no such file");
        exit(-1);
    }

    std::vector<nlohmann::json> features;
    nlohmann::json feature;

    Eigen::Vector3d lla_origin(31.431136, 120.646651, 8.0);
    mapecu::tools::Earth earth_lla;
    
    earth_lla.SetOrigin(lla_origin);
    
    //输出json id 
    int cnt_barrier_id = 0;
    for(auto &result : merge_result)
    {
        feature["type"] = "Feature";
        feature["geometry"]["type"] = "MultiPoint";
        std::vector<std::vector<double>> corner_point(result.size(), std::vector<double>(3,0));
        printf("corner size = %d\n", result.size());
        int cnt = 0;
        for(auto &js : result)
        {
          Eigen::Vector3d p(js.x, js.y, js.z);
          Eigen::Vector3d p_lla = earth_lla.ENU2LLH(p);
          corner_point[cnt][0] = p_lla(1);
          corner_point[cnt][1] = p_lla(0);
          corner_point[cnt][2] = p_lla(2);
          cnt++;
        }
        feature["geometry"]["coordinates"] = corner_point;

        feature["properties"]["confidence"] = (double)1.0;
        feature["properties"]["stroke"] = {255, 255, 0};
        feature["properties"]["id"] = "PS-" + std::to_string(cnt_barrier_id);
        cnt_barrier_id++;
        feature["properties"]["type"] = "parking_space";
        features.push_back(feature);
    }

    output_json_coner["features"] = features;
    fileout << output_json_coner;
    fileout.close();

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Normals"));
    pcl::visualization::PointCloudColorHandlerGenericField<PointTypeIO> fildColor(cloud_out, "x");
    viewer->setBackgroundColor(0, 0, 0);
    // 添加需要显示的点云法向。cloud为原始点云模型，normal为法向信息，1表示需要显示法向的点云间隔，即每1个点显示一次法向，0.003表示法向长度。
    viewer->addPointCloud<PointTypeIO>(cloud_out, fildColor, "bunny cloud");
    viewer->addPointCloudNormals<pcl::PointXYZINormal>(cloud_with_normals, 1, 0.3, "normals");
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }

    // Save the output point cloud
    // std::cerr << "Saving...\n", tt.tic ();
    // // pcl::io::savePCDFile ("output_condition_cluster.pcd", *cloud_out);
    // // pcl::io::savePCDFileASCII ("output_point_with_normal.pcd", *cloud_with_normals);
    // std::cerr << ">> Done: " << tt.toc () << " ms\n";

    return (0);
}