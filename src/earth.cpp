#include "earth.h"

namespace mapecu
{
namespace tools
{
    Eigen::Vector3d Earth::_origin = Eigen::Vector3d::Zero();  // ECEF
    Eigen::Matrix3d Earth::_cne = Eigen::Matrix3d::Identity(); //
    bool Earth::_origin_setted = false; // 是否设置过圆心
}
}
