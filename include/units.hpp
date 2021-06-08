#pragma once
#include <cmath>

namespace mapecu
{
namespace tools
{

// ellipsoid para
constexpr double WGS84_RE = 6378137.0;
constexpr double WGS84_F = (1.0 / 298.257223563);

constexpr double gl_wie = 7.2921151467e-5;
constexpr double gl_meru = gl_wie / 1000;
constexpr double gl_g0 = 9.7803267714;
constexpr double gl_mg = 1e-3 * gl_g0;
constexpr double gl_ug = 1e-3 * gl_mg;
constexpr double gl_mGal = 1e-3 * 0.01; // milli Gal = 1cm/s^2 ~= 1.0E-6*g0
constexpr double gl_ugpg2 = gl_ug / gl_g0 / gl_g0;
constexpr double gl_ppm = 1e-6;
constexpr double gl_deg = M_PI / 180;  // arcdeg
constexpr double gl_min = gl_deg / 60; // arcmin
constexpr double gl_sec = gl_min / 60; // arcsec
constexpr double gl_hur = 3600;
constexpr double gl_dps = gl_deg / 1;             // arcdeg / second
constexpr double gl_dph = gl_deg / gl_hur;        // arcdeg / hour
constexpr double gl_dpss = gl_deg / sqrt(1.0);    // arcdeg / sqrt(second)
constexpr double gl_dpsh = gl_deg / sqrt(gl_hur); // arcdeg / sqrt(hour)
constexpr double gl_dphpsh = gl_dph / sqrt(gl_hur);
constexpr double gl_Hz = 1 / 1; // Hertz
constexpr double gl_dphpsHz = gl_dph / gl_Hz;
constexpr double gl_mgpsHz = gl_mg / sqrt(gl_Hz);
constexpr double gl_ugpsHz = gl_ug / sqrt(gl_Hz);
constexpr double gl_ugpsh = gl_ug / sqrt(gl_hur); // ug / sqrt(hour)
constexpr double gl_mpsh = 1 / 1 / sqrt(gl_hur);
constexpr double gl_ppmpsh = gl_ppm / sqrt(gl_hur);
constexpr double gl_mil = 2 * M_PI / 6000;
constexpr double gl_nm = 1853;           // nautical mile
constexpr double gl_kn = gl_nm / gl_hur; // 海里每小时
// added
constexpr double gl_mps = 1 / 1;
constexpr double gl_km = 1000;
constexpr double gl_kmph = gl_km / gl_hur;
constexpr double gl_mpr = 1 / WGS84_RE;
constexpr double gl_m = 1.0;
constexpr double gl_cm = gl_m / 100;  // cm
constexpr double gl_cmps = gl_cm / 1; // cm /s

constexpr double operator"" _deg(long double x) { return x / 180 * M_PI; }
constexpr double operator"" _m(long double x) { return x * gl_m; }
constexpr double operator"" _mps(long double x) { return x * gl_mps; }
constexpr double operator"" _mpr(long double x) { return x * gl_mpr; }
constexpr double operator"" _mg(long double x) { return x * gl_mg; }
constexpr double operator"" _dps(long double x) { return x * gl_dps; }
constexpr double operator"" _ugpsHz(long double x) { return x * gl_ugpsh; }
constexpr double operator"" _dpss(long double x) { return x * gl_dpss; }
constexpr double operator"" _pixel(long double x) { return x; }
// 时间相关
constexpr double gl_SEC = 1;     // seconds
constexpr double gl_MSEC = 1e-3; // miliseconds
constexpr double gl_USEC = 1e-6; // microseconds
constexpr double gl_NSEC = 1e-9; // Nanosecond

constexpr double operator"" _SEC(long double x) { return x; }            // 秒
constexpr double operator"" _MSEC(long double x) { return x * gl_MSEC; } // 毫秒
constexpr double operator"" _USEC(long double x) { return x * gl_USEC; } // 微妙
constexpr double operator"" _MIN(long double x) { return x * 60; }   // 分钟
constexpr double operator"" _HOU(long double x) { return x * 3600; } // 小时

constexpr double operator""_hz(long double x) { return x; }

} // namespace tools
} // namespace mapecu
