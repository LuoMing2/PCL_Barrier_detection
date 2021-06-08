#pragma once

#include "units.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
namespace mapecu
{
namespace tools
{
class Earth
{
  public:
    static void LLH2ECEF(const double *pos, double *xyz, bool is_deg = false)
    {
        double b = pos[0] * (is_deg ? gl_deg : 1),
               l = pos[1] * (is_deg ? gl_deg : 1), h = pos[2];
        double n = N(b), cb = cos(b);
        xyz[0] = (n + h) * cb * cos(l);
        xyz[1] = (n + h) * cb * sin(l);
        xyz[2] = (n * (1 - _e1 * _e1) + h) * sin(b);
    }

    static Eigen::Vector3d LLH2ECEF(const Eigen::Vector3d &pos,
                                    bool is_deg = false)
    {
        Eigen::Vector3d xyz = {0, 0, 0};
        LLH2ECEF(pos.data(), xyz.data(), is_deg);
        return xyz;
    }

    static void ECEF2LLH(const Eigen::Vector3d &xyz, Eigen::Vector3d *pos,
                         bool to_deg = false)
    {
        const double e1_2 = _e1 * _e1, r2 = xyz.head(2).squaredNorm();
        double v = _a, z = xyz[2], sinp = 0;
        for (double zk = 0; fabs(z - zk) >= 1e-4;)
        {
            zk = z;
            sinp = z / sqrt(r2 + z * z);
            v = _a / sqrt(1 - e1_2 * sinp * sinp);
            z = xyz[2] + v * e1_2 * sinp;
        }
        (*pos)[0] = r2 > 1E-12 ? atan(z / sqrt(r2))
                               : (xyz[2] > 0.0 ? M_PI / 2.0 : -M_PI / 2.0);
        (*pos)[1] = r2 > 1E-12 ? atan2(xyz[1], xyz[0]) : 0.0;
        (*pos)[2] = sqrt(r2 + z * z) - v;
        if (to_deg)
        {
            pos->head(2) /= gl_deg;
        }
        // std::cout << "void ECEF2LLH(const Vector3d &r_, Vector3d *pos, bool
        // to_deg = false)  " << pos->transpose() << std::endl;
    }

    static Eigen::Vector3d ECEF2LLH(const Eigen::Vector3d &xyz,
                                    bool to_deg = false)
    {
        Eigen::Vector3d pos = Eigen::Vector3d::Zero();
        ECEF2LLH(xyz, &pos, to_deg);
        return pos;
    }

    static Eigen::Matrix3d Pos2Cne(const Eigen::Vector3d &pos,
                                   bool is_deg = false)
    {
        using namespace Eigen;
        double b = pos[0] * (is_deg ? gl_deg : 1),
               l = pos[1] * (is_deg ? gl_deg : 1);
        Eigen::Matrix3d re = Eigen::Matrix3d(
            AngleAxisd(-(M_PI / 2 - b), Eigen::Vector3d::UnitX()) *
            AngleAxisd(-(M_PI / 2 + l), Eigen::Vector3d::UnitZ()));
        return re;
    }

    // LLH(默认为经纬度)
    static void SetOrigin(const Eigen::Vector3d pos, bool is_deg = true)
    {
        _origin = LLH2ECEF(pos, is_deg);
        _cne = Pos2Cne(pos, is_deg);
        _origin_setted = true;
    }

    static Eigen::Vector3d GetOrigin(bool to_deg = true)
    {
        if (_origin_setted)
        {
            return ECEF2LLH(_origin, to_deg);
        }
        else
        {
            std::cerr << "please set the origin first.\n";
            return Eigen::Vector3d::Zero();
        }
    }

    static Eigen::Vector3d LLH2ENU(const Eigen::Vector3d &pos,
                                   bool is_deg = false)
    {
        if (_origin_setted)
        {
            return _cne * (LLH2ECEF(pos, is_deg) - _origin);
        }
        else
        {
            std::cerr << "please set the origin first.\n";
            return Eigen::Vector3d::Zero();
        }
    }

    static Eigen::Vector3d ENU2LLH(const Eigen::Vector3d &pos,
                                   bool to_deg = true)
    {
        if (_origin_setted)
        {
            return ECEF2LLH(_origin + _cne.transpose() * pos, to_deg);
        }
        else
        {
            std::cerr << "please set the origin first.\n";
            return Eigen::Vector3d::Zero();
        }
    }

    /**
     * @brief 给定中心，范围，返回相应矩形的左上角与右下角点
     *
     * @param pos 中心点
     * @param dis 范围，[-dis, +dis]
     * @return std::pair<Vector3d, Vector3d> 左上角点，右下角点
     */
    std::pair<Eigen::Vector3d, Eigen::Vector3d> static LLHRangeInDistance(
        Eigen::Vector3d const &pos, const double dis)
    {
        assert(dis > 0);
        double dlat = dis / M(pos[0]), dlon = dis / (N(pos[0]) * cos(pos[0]));
        Eigen::Vector3d p0 = {pos[0] - dlat, pos[1] - dlon, pos[2]};
        Eigen::Vector3d p1 = {pos[0] + dlat, pos[1] + dlon, pos[2]};
        return {p0, p1};
    }

    /**
     * @brief Get the Gn object 获得当前位置的东北天坐标系下重力矢量
     *
     * @param pos
     * @return Eigen::Vector3d
     */
    static Eigen::Vector3d GetGn(Eigen::Vector3d const &pos)
    {
        double lat = pos(0), alt = pos(2), sl = sin(lat);
        double gn_u =
            -(gl_g0 * (1 + 5.27094e-3 * pow(sl, 2) + 2.32718e-5 * pow(sl, 4)) -
              3.086e-6 * alt);
        return {0, 0, gn_u};
    }

    static Eigen::Vector3d GetWnie(Eigen::Vector3d const &pos)
    {
        double lat = pos(0), sl = sin(lat), cl = cos(lat);
        return {0, gl_wie * cl, gl_wie * sl};
    }

    static Eigen::Vector3d GetWnen(Eigen::Vector3d const &pos,
                                   Eigen::Vector3d const &vel)
    {
        double lat = pos(0), alt = pos(2), tl = tan(lat);
        double ve = vel[0], vn = vel[1];
        double rmh = M(lat) + alt, rnh = N(lat) + alt;
        return {-vn / rmh, ve / rnh, ve * tl / rnh};
    }
    /**
     * @brief Get the Rm Rn object 获得子午曲率半径，卯酉曲率半径
     *
     * @param pos
     * @return std::pair<double, double>
     */
    static std::pair<double, double> GetRmRn(Eigen::Vector3d const &pos)
    {
        double b = pos[0];
        return {M(b), N(b)};
    }

    /**
     * @brief 返回pos1-pos2在pos1东北天坐标系下位置
     *
     * @param p1 第一个点大地坐标
     * @param p2 第二个点大地坐标
     * @return Eigen::Vector3d
     */
    static Eigen::Vector3d DeltaPosEnuInFirstPoint(Eigen::Vector3d const &p1,
                                                   Eigen::Vector3d const &p2)
    {
        return Pos2Cne(p1) * (LLH2ECEF(p1) - LLH2ECEF(p2));
    }

    /**
     * @brief 返回pos1-pos2在pos1东北天坐标系下位置
     *
     * @param p1 第一个点大地坐标
     * @param p2 第二个点大地坐标
     * @return Eigen::Vector3d
     */
    static Eigen::Vector3d DeltaPosEnuInSecondPoint(Eigen::Vector3d const &p1,
                                                    Eigen::Vector3d const &p2)
    {
        return Pos2Cne(p2) * (LLH2ECEF(p1) - LLH2ECEF(p2));
    }

    /**
     * @brief 在pos处添加denu的位置变化
     *
     * @param pos :大地坐标
     * @param denu ：位置变化
     * @return Eigen::Vector3d
     */
    static Eigen::Vector3d PlusDeltaEnuAtPos(Eigen::Vector3d const &pos,
                                             Eigen::Vector3d const &denu)
    {
        return ECEF2LLH(LLH2ECEF(pos) + Pos2Cne(pos).transpose() * denu);
    }

  private:
    // double _a, _f, _b, _c, _e1, _e2;     // 长轴 短轴 极点子午曲率半径 扁率
    // 第一偏心率 第二偏心率
    static constexpr double _a = WGS84_RE;
    static constexpr double _f = WGS84_F;
    static constexpr double _b = (1 - _f) * _a;
    static constexpr double _c = _a * _a / _b;
    static constexpr double _e1 = sqrt(_a * _a - _b * _b) / _a;
    static constexpr double _e2 = sqrt(_a * _a - _b * _b) / _b;

    static constexpr double W(const double B_)
    {
        return sqrt(1 - pow(_e1 * sin(B_), 2));
    }
    static constexpr double V(const double B_)
    {
        return sqrt(1 + pow(_e2 * cos(B_), 2));
    }
    static constexpr double M(const double B_)
    {
        return _c / pow(V(B_), 3);
    } // 子午曲率半径
    static constexpr double N(const double B_)
    {
        return _c / V(B_);
    } // 卯酉曲率半径

    static Eigen::Vector3d _origin;     // ECEF
    static Eigen::Matrix3d _cne; //
    static bool _origin_setted;  // 是否设置过圆心
};

/**
 * @brief 航向角转方位角
 *
 * @param yaw:航向角，北偏西为正【-pi,pi】
 * @return double
 */
inline double yawToAzimuth(double const &yaw)
{
    return yaw > 0 ? (2 * M_PI - yaw) : -yaw;
}
/**
 * @brief 方位角转航向角
 *
 * @param azi:方位角，北偏东为正，[0,2pi]
 * @return double
 */
inline double azimuthToYaw(double const &azi)
{
    return azi >= M_PI ? (2 * M_PI - azi) : -azi;
}

/**
 * @brief 欧拉角转坐标余弦矩阵
 *
 * @param euler　顺序为俯仰角[-pi/2,pi/2]、翻滚角[-pi,pi]、方位角[0,2pi]
 * @return Eigen::Matrix3d
 */
inline Eigen::Matrix3d pitchRollAzimuthToMatrix(Eigen::Vector3d const &euler)
{
    using namespace Eigen;
    double yaw = azimuthToYaw(euler(2));
    double si = sin(euler(0)), ci = cos(euler(0)); // pit
    double sj = sin(euler(1)), cj = cos(euler(1)); // rol
    double sk = sin(yaw), ck = cos(yaw);           // yaw
    Eigen::Matrix3d mat;
    mat(0, 0) = cj * ck - si * sj * sk;
    mat(0, 1) = -ci * sk;
    mat(0, 2) = sj * ck + si * cj * sk;
    mat(1, 0) = cj * sk + si * sj * ck;
    mat(1, 1) = ci * ck;
    mat(1, 2) = sj * sk - si * cj * ck;
    mat(2, 0) = -ci * sj;
    mat(2, 1) = si;
    mat(2, 2) = ci * cj;
    return mat;
}

inline Eigen::Vector3d MatrixToPitchRollAzimuth(Eigen::Matrix3d const &mat)
{
    using namespace Eigen;
    Eigen::Vector3d euler = {0, 0, 0};
    euler[0] = asin(mat(2, 1));

    if (std::abs(mat(2, 1)) <= 0.999999)
    {
        euler[1] = -atan2(mat(2, 0), mat(2, 2));
        euler[2] = -atan2(mat(0, 1), mat(1, 1));
    }
    else
    {
        euler[1] = atan2(mat(0, 2), mat(0, 0));
        euler[2] = 0;
    }
    euler[2] = yawToAzimuth(euler[2]);
    return euler;
}

template <typename T>
inline Eigen::Matrix<T, 3, 3> EulerToMatrix(Eigen::Matrix<T, 3, 1> const &euler)
{
    using namespace Eigen;
    T si = sin(euler(0)), ci = cos(euler(0)); // pit
    T sj = sin(euler(1)), cj = cos(euler(1)); // rol
    T sk = sin(euler(2)), ck = cos(euler(2)); // yaw
    Matrix<T, 3, 3> mat;
    mat(0, 0) = cj * ck - si * sj * sk;
    mat(0, 1) = -ci * sk;
    mat(0, 2) = sj * ck + si * cj * sk;
    mat(1, 0) = cj * sk + si * sj * ck;
    mat(1, 1) = ci * ck;
    mat(1, 2) = sj * sk - si * cj * ck;
    mat(2, 0) = -ci * sj;
    mat(2, 1) = si;
    mat(2, 2) = ci * cj;
    return mat;
}

// inline Eigen::Matrix3d EulerToMatrix(Eigen::Vector3d const &euler)
//{
//  using namespace Eigen;
//  double si = sin(euler(0)), ci = cos(euler(0)); // pit
//  double sj = sin(euler(1)), cj = cos(euler(1)); // rol
//  double sk = sin(euler(2)), ck = cos(euler(2)); // yaw
//  Matrix3d mat;
//  mat(0, 0) = cj * ck - si * sj * sk;
//  mat(0, 1) = -ci * sk;
//  mat(0, 2) = sj * ck + si * cj * sk;
//  mat(1, 0) = cj * sk + si * sj * ck;
//  mat(1, 1) = ci * ck;
//  mat(1, 2) = sj * sk - si * cj * ck;
//  mat(2, 0) = -ci * sj;
//  mat(2, 1) = si;
//  mat(2, 2) = ci * cj;
//  return mat;
//}

inline Eigen::Vector3d MatrixToEuler(Eigen::Matrix3d const &mat)
{
    using namespace Eigen;
    Eigen::Vector3d euler = {0, 0, 0};
    euler[0] = asin(mat(2, 1));

    if (std::abs(mat(2, 1)) <= 0.999999)
    {
        euler[1] = -atan2(mat(2, 0), mat(2, 2));
        euler[2] = -atan2(mat(0, 1), mat(1, 1));
    }
    else
    {
        euler[1] = atan2(mat(0, 2), mat(0, 0));
        euler[2] = 0;
    }
    return euler;
}
} // namespace tools
} // namespace mapecu
