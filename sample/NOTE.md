## 1、PCL滤波和法相计算

* 有用的拷贝操作
```cpp
  pcl::copyPointCloud(*cloud_out, *cloud_with_normals);
```
### 1.1、统计滤波

主要作用是**去除稀疏离群噪点**。在采集点云的过程中，由于测量噪声的影响，会引入部分离群噪点，它们在点云空间中分布稀疏。在估算点云局部特征（例如计算采样点处的法向量和曲率变化率）时，这些噪点可能导致错误的计算结果，从而使点云配准等后期处理失败。统计滤波器的主要思想是假设点云中所有的点与其最近的k个邻居点的平均距离满足高斯分布，那么，根据均值和方差可确定一个距离阈值，当某个点与其最近k个点的平均距离大于这个阈值时，判定该点为离群点并去除。统计滤波器的实现原理如下：
* 首先，遍历点云，计算每个点与其最近的k个邻居点之间的平均距离；
* 其次，计算所有平均距离的均值μ与标准差σ，则距离阈值dmax可表示为dmax=μ＋α×σ，α是一个常数，可称为比例系数，它取决于邻居点的数目；
* 最后，再次遍历选中点云，剔除与k个邻居点的平均距离大于dmax的点。
  
```cpp
  pcl::StatisticalOutlierRemoval<PointTypeIO> sor;
```

### 1.2、直通滤波（可选）

最简单的一种滤波器，它的作用是过滤掉在指定维度方向上取值不在给定值域内的点

```cpp
 pcl::PassThrough<PointTypeIO> pass;

```
### 1、3、法向计算
PCL求平面法向用的是PCA，即主成分分析法。

```cpp

pcl::copyPointCloud(*cloud_out, *cloud_with_normals);
pcl::NormalEstimation<PointTypeIO, PointTypeFull> ne;

```


### 1.4、剔除法相不合格的聚类/点
* 法向若垂直于地面肯定不是柱子、墙壁等障碍物，存入unnormal_points_index，之后剔除
* 之所以不在求法向前，直接剔除再传入条件聚类，是由于点云使用erase()函数执行成本极大，因为其存储是一个std::vector,执行erase()需要先删除该点，并将其后所有点向前移动，极其耗时，更不用说这里牵扯到百万点云，在机器上试过，响应时间极长。。。
  
### 1.5、直线条件聚类
* 相邻点法向差异大于3°的认为不是同一类
* 聚类允许拉入的相邻点最大距离为0.3米
```cpp
pcl::ConditionalEuclideanClustering<PointTypeFull> cec (true);
cec.setInputCloud (cloud_with_normals);
cec.setConditionFunction (&customRegionGrowing);
```

## 2、处理聚类直线
### 2.1、选取每个类合适的端点（都认作为直线特征）
* 输出每个聚类直线
* 剔除上述记录的不合格聚类点，每一类转存位IO点云格式到line_cloud_cluster并输出
* 取直线两端点
  * 求直线中心点
  * 遍历取相距中心点最大的点，为left_p
  * 再次遍历取和left_p相距最远的点，为right_p
* RANSAC拟合直线
  ```cpp
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
      pcl::PointIndices::Ptr inliers(new pcl::PointIndices);  
      pcl::SACSegmentation<PointTypeIO> seg;     
      seg.setOptimizeCoefficients(true);      
      seg.setModelType(pcl::SACMODEL_LINE);  
      seg.setMethodType(pcl::SAC_RANSAC);     
      seg.setDistanceThreshold(0.4);         
      seg.setInputCloud(line_cloud_cluster);              
      seg.segment(*inliers, *coefficients);
  ```
  * 将两端点投影到直线上，如此操作是为了让选取的端点更符合每一类端点，而不是选取到极端值

### 2.2、存储端点
有必要说一说存储端点，前面的条件聚类方法实际上将属于同一柱子、墙壁的轮廓线分割了，后面我们需要聚合在一起，这牵扯到比较复杂的查找（kd-tree实现）和端点融合，需要一个标志位记录是否端点已被处理

**两个端点构成一个pair，键为点云，值为bool，值预先置位true，当被查找并融合后，置位false，排除重复操作行为**
```cpp
std::vector<std::pair<PointTypeIO, bool>> endpoint_pair; 
```

## 3、同类融合
**思路：**
遍历所有的直线类，首先查询左端点附近的最邻近点，该邻近点所属的直线类属于同一个柱子或墙壁。若两条直线相互垂直，求两直线的交点，若相互平行，求两个端点的中点，这两个端点同时赋值为所求的交点（中点），并标记为处理过（false）。需要注意的是，每次为了求交点而查找到的端点并没有同时处理，而后面为了融合，必须要同时处理两个端点，归为同一类，牵扯到递归。
### 3.1、点云构建kd-tree
直接利用pcl提供的kd-tree会有问题：

融合过程每一个类始终有两端点，每一头的端点需要单独处理，处理完后需要标记为false，第一个问题，你并不知道查找到的是左端点还是右端点，第二个问题，处理标记不好记录。

* 因此，构造了一个带接口的kd-tree，该接口允许存入index和左右端点状态status
```cpp
class MyPoint : public std::array<double, 3>
{
public:
	static const int DIM = 3;
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
	operator Eigen::Vector3d() const { return Eigen::Vector3d((*this)[0], (*this)[1], (*this)[2]); }
};
```
### 3.2、查询端点邻近点，求交点（中点）
* 遍历所有直线类，对于左右端点，若没有被查询到，执行kd-tree半径搜索。为什么一定是半径搜索？在构造kd树时，放进去的是所有点云，利用最邻近搜索查询到的实际是自身，因此这里需要排除自身。
* 如果查询到的点被处理过，那么跳过
* 如果查询到最近点符合阈值0.5米，求直线交点或中点
  * 求交点是针对两直线符合柱子或墙壁的拐弯处
  * 求中点是针对墙壁点云发生“断裂”的情况

### 3.3、同一类重新聚合
此处很绕，思考了许久才构造出来
* 首先，所有的处理标志位要重新置位true，用来对融合过的点进行记录
* 再次构造kd-tree
* 遍历所有的直线类，先用半径搜索查询左端点，因为前面的交点被赋值为同一点，这里的查询如果有效会出现两个，一为自身，二为属于同一类的重叠点。
  * while(1)开始处
  * 如下图，若左直线的下端点为左端点（**0**），上端点为右端点(**1**)；
  * 在外层会构造一个vector，**按照顺序**保存同一类的端点；
  * 按照此情况，1会先被推入vector，再是0；
  * 左端点查询到下底直线的一个端点（0'），判断是左端点还是右端点,该重叠端点已经推入0了，0‘不需要记录，后面依次推入1'、0''，0'''，直至另一头的端点查询不到最邻近点了，或者查询到的最近邻点已经全部处理了（即1和0'''对应的情况），就跳出循环
  * 这里一定要注意后一种情况，否则就陷入死循环。
  * 以上所有融合过程中的点，在处理完成后都需要置位false
  
 


## 4、输出到json文件
和psd_tracking一样，复用代码即可
