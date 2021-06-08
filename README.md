# PCL_Barrier_detection
该仓库能够实现3维障碍物自动分割，并投影到2D地面上
首先计算3维障碍物**点云法向**，依据表面法向对障碍物**条件聚类**，对每一类进行单独的直线分割（曲线分割成小段），
随即采用一系列搜索算法，将同一类物体（如柱子、墙面）聚合在一起
[3维点云，检测之前](https://user-images.githubusercontent.com/67543522/121194620-7c8f7b80-c8a1-11eb-9ad7-7f40bd49579a.png)

[检测结果](https://user-images.githubusercontent.com/67543522/121194261-2c181e00-c8a1-11eb-8a7c-f9b063d3b5f4.png)
