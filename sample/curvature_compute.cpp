
#include <pcl/io/io.h>
#include <pcl/io/obj_io.h>
#include <pcl/PolygonMesh.h>
#include<pcl/ros/conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/io/vtk_lib_io.h>//loadPolygonFileOBJ所属头文件；
 
#include<pcl/features/normal_3d.h>
#include<pcl/features/principal_curvatures.h>
 
vector<PCURVATURE> getModelCurvatures(string modelPath)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PolygonMesh mesh;
	pcl::io::loadPolygonFileOBJ(modelPath, mesh);
	pcl::fromPCLPointCloud2(mesh.cloud, *cloud);
 
	//计算法线--------------------------------------------------------------------------------------
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	ne.setInputCloud(cloud);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	ne.setSearchMethod(tree); //设置搜索方法  
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
	//ne.setRadiusSearch(0.05); //设置半径邻域搜索  
	ne.setKSearch(5);
	ne.compute(*cloud_normals); //计算法向量  
	//计算法线--------------------------------------------------------------------------------------
	//计算曲率-------------------------------------------------------------------------------------
	pcl::PrincipalCurvaturesEstimation<pcl::PointXYZ, pcl::Normal, pcl::PrincipalCurvatures>pc;
	pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr cloud_curvatures(new pcl::PointCloud<pcl::PrincipalCurvatures>);
	pc.setInputCloud(cloud);
	pc.setInputNormals(cloud_normals);
	pc.setSearchMethod(tree);
	//pc.setRadiusSearch(0.05);
	pc.setKSearch(5);
	pc.compute(*cloud_curvatures);
 
	//获取曲率集
	vector<PCURVATURE> tempCV;
	POINT3F tempPoint;
	float curvature = 0.0;
	PCURVATURE pv;
	tempPoint.x = tempPoint.y = tempPoint.z=0.0;
	for (int i = 0; i < cloud_curvatures->size();i++){
		//平均曲率
		//curvature = ((*cloud_curvatures)[i].pc1 + (*cloud_curvatures)[i].pc2) / 2;
		//高斯曲率
		curvature = (*cloud_curvatures)[i].pc1 * (*cloud_curvatures)[i].pc2;
		//pv.cPoint = tempPoint;
		pv.index = i;
		pv.curvature = curvature;
		tempCV.insert(tempCV.end(),pv);
	}
	return tempCV;
