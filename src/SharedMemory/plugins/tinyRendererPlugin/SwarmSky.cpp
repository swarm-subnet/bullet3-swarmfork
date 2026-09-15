#include "SwarmSky.h"

#include <math.h>
#include <string.h>

#include "SwarmDaylight.h"
#include "../../../TinyRenderer/SwarmGamma.h"

namespace
{
const int kFace = 256;  // texels per cube face side: one texel per pixel of a 256 px, 90 degree camera
const double kPi = 3.14159265358979323846;
const double kTurbidity = 2.5;      // Preetham turbidity of a clear day
const double kExposure = 0.12;      // scale on kcd/m^2 before the tone curve: a noon zenith lands near 0.5
const double kHorizonCos = 0.02;    // rays below the horizon read the sky just above it
const double kSunDiscRadius = 1.0 * kPi / 180.0;
const double kSunDiscEdge = 0.25;   // share of the radius over which the disc fades into the sky
const double kAmbientSkyShare = 0.5;
const double kCloudScale = 1.5;     // noise cells across the sky at 45 degrees up
const double kCloudHorizonFade = 0.12;
const double kCloudShade = 0.3;     // how much darker the thickest cloud is than the sun colour
const int kCloudOctaves = 4;

// exp, acos, sin and cos in plain double arithmetic. The C library picks an FMA or a non-FMA
// variant of these per CPU at run time, which could move a texel byte between two validators.
double skyExp(double x)
{
	return swarmExp(x);
}

double skyAcos(double x)
{
	if (x > 1.0)
		x = 1.0;
	if (x < -1.0)
		x = -1.0;
	const double a = x < 0.0 ? -x : x;
	const double r = sqrt(1.0 - a) * (1.5707288 + a * (-0.2121144 + a * (0.0742610 + a * -0.0187293)));
	return x < 0.0 ? kPi - r : r;
}

double skySin(double x)
{
	const double x2 = x * x;
	return x * (1.0 - x2 / 6.0 * (1.0 - x2 / 20.0 * (1.0 - x2 / 42.0 * (1.0 - x2 / 72.0 * (1.0 - x2 / 110.0 * (1.0 - x2 / 156.0))))));
}

double skyCos(double x)
{
	const double x2 = x * x;
	return 1.0 - x2 / 2.0 * (1.0 - x2 / 12.0 * (1.0 - x2 / 30.0 * (1.0 - x2 / 56.0 * (1.0 - x2 / 90.0 * (1.0 - x2 / 132.0)))));
}

double clamp01(double v)
{
	return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
}

// One channel of the Perez sky luminance distribution, coefficients from the turbidity.
struct Perez
{
	double A, B, C, D, E;

	double at(double cosTheta, double gamma, double cosGamma) const
	{
		return (1.0 + A * skyExp(B / cosTheta)) * (1.0 + C * skyExp(D * gamma) + E * cosGamma * cosGamma);
	}
};

// The Preetham daylight model for one sun: zenith values and the three Perez distributions for
// luminance Y and the chromaticities x and y, each divided by its value towards the sun.
struct Preetham
{
	Perez m_Y, m_x, m_y;
	double m_Yz, m_xz, m_yz;
	double m_normY, m_normx, m_normy;

	Preetham(double thetaSun)
	{
		const double T = kTurbidity;
		const Perez Y = {0.1787 * T - 1.4630, -0.3554 * T + 0.4275, -0.0227 * T + 5.3251, 0.1206 * T - 2.5771, -0.0670 * T + 0.3703};
		const Perez x = {-0.0193 * T - 0.2592, -0.0665 * T + 0.0008, -0.0004 * T + 0.2125, -0.0641 * T - 0.8989, -0.0033 * T + 0.0452};
		const Perez y = {-0.0167 * T - 0.2608, -0.0950 * T + 0.0092, -0.0079 * T + 0.2102, -0.0441 * T - 1.6537, -0.0109 * T + 0.0529};
		m_Y = Y;
		m_x = x;
		m_y = y;

		const double chi = (4.0 / 9.0 - T / 120.0) * (kPi - 2.0 * thetaSun);
		const double Yz = (4.0453 * T - 4.9710) * (skySin(chi) / skyCos(chi)) - 0.2155 * T + 2.4192;
		m_Yz = Yz > 0.0 ? Yz : 0.0;
		const double t = thetaSun, t2 = t * t, t3 = t2 * t;
		m_xz = T * T * (0.00166 * t3 - 0.00375 * t2 + 0.00209 * t) + T * (-0.02903 * t3 + 0.06377 * t2 - 0.03202 * t + 0.00394) + (0.11693 * t3 - 0.21196 * t2 + 0.06052 * t + 0.25886);
		m_yz = T * T * (0.00275 * t3 - 0.00610 * t2 + 0.00317 * t) + T * (-0.04214 * t3 + 0.08970 * t2 - 0.04153 * t + 0.00516) + (0.15346 * t3 - 0.26756 * t2 + 0.06670 * t + 0.26688);

		const double cosSun = skyCos(thetaSun);
		m_normY = m_Y.at(1.0, thetaSun, cosSun);
		m_normx = m_x.at(1.0, thetaSun, cosSun);
		m_normy = m_y.at(1.0, thetaSun, cosSun);
	}

	// Display colour, 0..1 per channel, of the sky at a zenith angle and an angle from the sun.
	void colour(double cosTheta, double gamma, double cosGamma, double rgb[3]) const
	{
		const double Y = m_Yz * m_Y.at(cosTheta, gamma, cosGamma) / m_normY;
		const double x = m_xz * m_x.at(cosTheta, gamma, cosGamma) / m_normx;
		const double y = m_yz * m_y.at(cosTheta, gamma, cosGamma) / m_normy;
		const double X = x / y * Y;
		const double Z = (1.0 - x - y) / y * Y;
		const double lin[3] = {3.2406 * X - 1.5372 * Y - 0.4986 * Z,
							   -0.9689 * X + 1.8758 * Y + 0.0415 * Z,
							   0.0557 * X - 0.2040 * Y + 1.0570 * Z};
		for (int i = 0; i < 3; i++)
		{
			const double c = (lin[i] > 0.0 ? lin[i] : 0.0) * kExposure;
			rgb[i] = sqrt(c / (1.0 + c));
		}
	}

	// The same colour as linear radiance, before the tone curve above, for the daylight sky.
	void linear(double cosTheta, double gamma, double cosGamma, double rgb[3]) const
	{
		const double Y = m_Yz * m_Y.at(cosTheta, gamma, cosGamma) / m_normY;
		const double x = m_xz * m_x.at(cosTheta, gamma, cosGamma) / m_normx;
		const double y = m_yz * m_y.at(cosTheta, gamma, cosGamma) / m_normy;
		const double X = x / y * Y;
		const double Z = (1.0 - x - y) / y * Y;
		const double lin[3] = {3.2406 * X - 1.5372 * Y - 0.4986 * Z,
							   -0.9689 * X + 1.8758 * Y + 0.0415 * Z,
							   0.0557 * X - 0.2040 * Y + 1.0570 * Z};
		for (int i = 0; i < 3; i++)
			rgb[i] = (lin[i] > 0.0 ? lin[i] : 0.0) * kExposure;
	}
};

// Integer hash noise, so the same seed gives the same clouds everywhere.
double hashNoise(int ix, int iy, unsigned seed)
{
	unsigned h = (unsigned)ix * 374761393u + (unsigned)iy * 668265263u + seed * 1274126177u;
	h = (h ^ (h >> 13)) * 1274126177u;
	h ^= h >> 16;
	return (h & 0xFFFFu) / 65535.0;
}

double valueNoise(double u, double v, unsigned seed)
{
	int iu = (int)u, iv = (int)v;
	if (u < iu)
		iu--;
	if (v < iv)
		iv--;
	const double fu = u - iu, fv = v - iv;
	const double su = fu * fu * (3.0 - 2.0 * fu), sv = fv * fv * (3.0 - 2.0 * fv);
	const double a = hashNoise(iu, iv, seed), b = hashNoise(iu + 1, iv, seed);
	const double c = hashNoise(iu, iv + 1, seed), d = hashNoise(iu + 1, iv + 1, seed);
	const double low = a + (b - a) * su, high = c + (d - c) * su;
	return low + (high - low) * sv;
}

// Cloud density, 0..1, along a unit direction: layered noise on a plane over the viewer, faded
// out at the horizon where the plane runs off to infinity.
double cloudDensity(const double dir[3], int up, unsigned seed, double coverage)
{
	const double h = dir[up];
	if (h <= 0.0)
		return 0.0;
	const double fade = clamp01(h / kCloudHorizonFade);
	if (fade <= 0.0)
		return 0.0;
	const double u = dir[up == 0 ? 1 : 0] / h * kCloudScale;
	const double v = dir[up == 2 ? 1 : 2] / h * kCloudScale;
	double sum = 0.0, amp = 0.5, freq = 1.0, total = 0.0;
	for (int k = 0; k < kCloudOctaves; k++)
	{
		sum += amp * valueNoise(u * freq, v * freq, seed + (unsigned)k);
		total += amp;
		amp *= 0.5;
		freq *= 2.0;
	}
	const double t = clamp01((sum / total - coverage) / 0.2);
	return t * t * (3.0 - 2.0 * t) * fade;
}
}  // namespace

SwarmSky::SwarmSky()
	: m_clouds(false), m_cloudSeed(0), m_upAxis(2), m_built(false),
	  m_daylightClouds(false), m_daylightCloudSeed(0), m_daylightUpAxis(2),
	  m_photoRgb(0), m_photoWidth(0), m_photoHeight(0), m_photoYaw(0.0f), m_exposure(1.0f), m_daylightBuilt(false)
{
	for (int i = 0; i < 3; i++)
	{
		m_ambient[i] = m_horizon[i] = m_zenith[i] = 1.0f;
		m_sunDir[i] = m_daylightSunDir[i] = 0.0f;
		m_sunColor[i] = m_daylightSunColor[i] = 1.0f;
	}
	memset(m_sh, 0, sizeof(m_sh));
}

void SwarmSky::prepare(const float sunDir[3], const float sunColor[3], bool clouds, unsigned cloudSeed, int upAxis)
{
	if (m_built && m_clouds == clouds && m_cloudSeed == cloudSeed && m_upAxis == upAxis &&
		memcmp(m_sunDir, sunDir, sizeof(m_sunDir)) == 0 && memcmp(m_sunColor, sunColor, sizeof(m_sunColor)) == 0)
		return;
	memcpy(m_sunDir, sunDir, sizeof(m_sunDir));
	memcpy(m_sunColor, sunColor, sizeof(m_sunColor));
	m_clouds = clouds;
	m_cloudSeed = cloudSeed;
	m_upAxis = upAxis;
	m_built = true;

	double sun[3] = {sunDir[0], sunDir[1], sunDir[2]};
	const double sunLen = sqrt(sun[0] * sun[0] + sun[1] * sun[1] + sun[2] * sun[2]);
	for (int i = 0; i < 3; i++)
		sun[i] = sunLen > 0.0 ? sun[i] / sunLen : (i == upAxis ? 1.0 : 0.0);
	// The model is defined for a sun above the horizon; a lower one is lit as if just above it.
	const double sunUp = sun[upAxis] > kHorizonCos ? sun[upAxis] : kHorizonCos;
	const Preetham model(skyAcos(sunUp));
	double disc[3];
	for (int i = 0; i < 3; i++)
		disc[i] = clamp01(sunColor[i]);
	const double coverage = 0.40 + 0.25 * hashNoise(7, 11, cloudSeed);

	m_texels.resize((size_t)6 * kFace * kFace * 3);
	// Sums of the finished bytes over the upper hemisphere, the horizon band and the zenith cap.
	double sums[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
	long counts[3] = {0, 0, 0};
	for (int face = 0; face < 6; face++)
	{
		const int axis = face / 2;
		const double sign = (face & 1) ? -1.0 : 1.0;
		for (int j = 0; j < kFace; j++)
		{
			const double t = (j + 0.5) / kFace * 2.0 - 1.0;
			for (int i = 0; i < kFace; i++)
			{
				const double s = (i + 0.5) / kFace * 2.0 - 1.0;
				// The face axis takes the sign; s and t fill the other two axes in index order.
				double dir[3];
				dir[axis] = sign;
				dir[axis == 0 ? 1 : 0] = s;
				dir[axis == 2 ? 1 : 2] = t;
				const double len = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
				for (int k = 0; k < 3; k++)
					dir[k] /= len;

				const double cosTheta = dir[upAxis] > kHorizonCos ? dir[upAxis] : kHorizonCos;
				const double cosGamma = dir[0] * sun[0] + dir[1] * sun[1] + dir[2] * sun[2];
				const double gamma = skyAcos(cosGamma);
				double rgb[3];
				model.colour(cosTheta, gamma, cosGamma, rgb);
				if (gamma < kSunDiscRadius)
				{
					const double f = clamp01((kSunDiscRadius - gamma) / (kSunDiscRadius * kSunDiscEdge));
					for (int k = 0; k < 3; k++)
						rgb[k] += (disc[k] - rgb[k]) * f;
				}
				if (clouds)
				{
					const double density = cloudDensity(dir, upAxis, cloudSeed, coverage);
					for (int k = 0; k < 3; k++)
						rgb[k] += (disc[k] * (1.0 - kCloudShade * density) - rgb[k]) * density;
				}

				unsigned char* out = &m_texels[(((size_t)face * kFace + j) * kFace + i) * 3];
				const double h = dir[upAxis];
				const bool bands[3] = {h > 0.0, h > -0.03 && h < 0.03, h > 0.95};
				for (int k = 0; k < 3; k++)
				{
					out[k] = (unsigned char)(clamp01(rgb[k]) * 255.0 + 0.5);
					for (int b = 0; b < 3; b++)
						if (bands[b])
							sums[b][k] += out[k] / 255.0;
				}
				for (int b = 0; b < 3; b++)
					counts[b] += bands[b];
			}
		}
	}

	double brightest = 0.0;
	for (int k = 0; k < 3; k++)
	{
		for (int b = 0; b < 3; b++)
			sums[b][k] = counts[b] ? sums[b][k] / counts[b] : 1.0;
		brightest = sums[0][k] > brightest ? sums[0][k] : brightest;
		m_horizon[k] = (float)sums[1][k];
		m_zenith[k] = (float)sums[2][k];
	}
	for (int k = 0; k < 3; k++)
	{
		const double tint = brightest > 0.0 ? sums[0][k] / brightest : 1.0;
		m_ambient[k] = (float)(1.0 + (tint - 1.0) * kAmbientSkyShare);
	}
}

void SwarmSky::lookup(float x, float y, float z, unsigned char out[3]) const
{
	const float ax = x < 0.f ? -x : x, ay = y < 0.f ? -y : y, az = z < 0.f ? -z : z;
	int face;
	float m, s, t;
	if (ax >= ay && ax >= az)
	{
		face = x < 0.f ? 1 : 0;
		m = ax;
		s = y;
		t = z;
	}
	else if (ay >= az)
	{
		face = y < 0.f ? 3 : 2;
		m = ay;
		s = x;
		t = z;
	}
	else
	{
		face = z < 0.f ? 5 : 4;
		m = az;
		s = x;
		t = y;
	}
	int i = 0, j = 0;
	if (m > 0.f)
	{
		const float inv = 1.f / m;
		i = (int)((s * inv + 1.f) * (0.5f * kFace));
		j = (int)((t * inv + 1.f) * (0.5f * kFace));
		i = i < 0 ? 0 : (i >= kFace ? kFace - 1 : i);
		j = j < 0 ? 0 : (j >= kFace ? kFace - 1 : j);
	}
	const unsigned char* texel = &m_texels[(((size_t)face * kFace + j) * kFace + i) * 3];
	out[0] = texel[0];
	out[1] = texel[1];
	out[2] = texel[2];
}

// The linear sky of ER_SWARM_DAYLIGHT.
namespace
{
const int kDayFace = 512;               // texels per face side: four texels per pixel of a 256 px, 90 degree camera
const double kSunDiscRadiance = 12.0;   // linear radiance of the analytic sun disc, times the sun colour
const double kCloudRadiance = 1.2;      // a sunlit cloud top, times the sun colour, in the analytic sky
const double kPhotoMeanLuminance = 1.0; // a photograph under a high sun is scaled so its upper hemisphere averages this
const double kPhotoDuskShare = 0.2;     // a sky at the horizon keeps this share of it; full brightness from a sun 30 degrees up
const double kPhotoFullSunUp = 0.5;
const double kGroundBounce = 0.25;      // the ground below the horizon returns this share of the mean sky
const double kPhotoHighlight = 8.0;     // a texel clipped to white in the photograph near the sun is the sun or its glare, this much brighter
const double kPhotoHighlightCos = 0.9961947;  // within 5 degrees of the sun
const int kPhotoClipByte = 254;

// Face, in-face coordinates and the four texels and weights of a direction on a cube of `face` texels a side.
struct CubeSample
{
	int m_face;
	int m_i0, m_i1, m_j0, m_j1;
	float m_wi, m_wj;
};

CubeSample cubeSample(float x, float y, float z, int faceSize)
{
	const float ax = x < 0.f ? -x : x, ay = y < 0.f ? -y : y, az = z < 0.f ? -z : z;
	CubeSample cs;
	float m, s, t;
	if (ax >= ay && ax >= az)
	{
		cs.m_face = x < 0.f ? 1 : 0;
		m = ax;
		s = y;
		t = z;
	}
	else if (ay >= az)
	{
		cs.m_face = y < 0.f ? 3 : 2;
		m = ay;
		s = x;
		t = z;
	}
	else
	{
		cs.m_face = z < 0.f ? 5 : 4;
		m = az;
		s = x;
		t = y;
	}
	float fi = 0.f, fj = 0.f;
	if (m > 0.f)
	{
		const float inv = 1.f / m;
		fi = (s * inv + 1.f) * (0.5f * faceSize) - 0.5f;
		fj = (t * inv + 1.f) * (0.5f * faceSize) - 0.5f;
	}
	const float last = (float)(faceSize - 1);
	fi = fi < 0.f ? 0.f : (fi > last ? last : fi);
	fj = fj < 0.f ? 0.f : (fj > last ? last : fj);
	cs.m_i0 = (int)fi;
	cs.m_j0 = (int)fj;
	cs.m_i1 = cs.m_i0 + 1 < faceSize ? cs.m_i0 + 1 : cs.m_i0;
	cs.m_j1 = cs.m_j0 + 1 < faceSize ? cs.m_j0 + 1 : cs.m_j0;
	cs.m_wi = fi - (float)cs.m_i0;
	cs.m_wj = fj - (float)cs.m_j0;
	return cs;
}

inline size_t texelIndex(int face, int i, int j, int faceSize)
{
	return (((size_t)face * faceSize + j) * faceSize + i) * 3;
}

// The nine real spherical harmonics of a unit direction, in the order the irradiance formula reads them.
void shBasis(const double d[3], double y[9])
{
	y[0] = 0.282095;
	y[1] = 0.488603 * d[1];
	y[2] = 0.488603 * d[2];
	y[3] = 0.488603 * d[0];
	y[4] = 1.092548 * d[0] * d[1];
	y[5] = 1.092548 * d[1] * d[2];
	y[6] = 0.315392 * (3.0 * d[2] * d[2] - 1.0);
	y[7] = 1.092548 * d[0] * d[2];
	y[8] = 0.546274 * (d[0] * d[0] - d[1] * d[1]);
}

// One texel of a photograph, blended from its four nearest samples, wrapped across the seam and clamped at the poles, decoded to linear light.
void photoSample(const SwarmSky::Photo& photo, double u, double v, bool nearSun, double out[3])
{
	const int w = photo.m_width, h = photo.m_height;
	u -= (double)(long long)u;
	if (u < 0.0)
		u += 1.0;
	const double fx = u * w - 0.5;
	double fy = v * h - 0.5;
	fy = fy < 0.0 ? 0.0 : (fy > (double)(h - 1) ? (double)(h - 1) : fy);
	int x0 = (int)fx;
	if (fx < x0)
		x0--;
	const int y0 = (int)fy;
	const double wx = fx - x0, wy = fy - y0;
	x0 = ((x0 % w) + w) % w;
	const int x1 = x0 + 1 < w ? x0 + 1 : 0;
	const int y1 = y0 + 1 < h ? y0 + 1 : y0;
	const unsigned char* p00 = photo.m_rgb + ((size_t)y0 * w + x0) * 3;
	const unsigned char* p10 = photo.m_rgb + ((size_t)y0 * w + x1) * 3;
	const unsigned char* p01 = photo.m_rgb + ((size_t)y1 * w + x0) * 3;
	const unsigned char* p11 = photo.m_rgb + ((size_t)y1 * w + x1) * 3;
	for (int k = 0; k < 3; k++)
	{
		const double a = kSwarmSrgbToLinear[p00[k]] + (kSwarmSrgbToLinear[p10[k]] - kSwarmSrgbToLinear[p00[k]]) * wx;
		const double b = kSwarmSrgbToLinear[p01[k]] + (kSwarmSrgbToLinear[p11[k]] - kSwarmSrgbToLinear[p01[k]]) * wx;
		out[k] = a + (b - a) * wy;
	}
	// Eight bits stop at white; where all four samples sit there next to the sun, the photograph was brighter still.
	const unsigned char* corners[4] = {p00, p10, p01, p11};
	bool clipped = nearSun;
	for (int c = 0; c < 4 && clipped; c++)
		clipped = corners[c][0] >= kPhotoClipByte && corners[c][1] >= kPhotoClipByte && corners[c][2] >= kPhotoClipByte;
	if (clipped)
		for (int k = 0; k < 3; k++)
			out[k] *= kPhotoHighlight;
}
}  // namespace

void SwarmSky::prepareDaylight(const float sunDir[3], const float sunColor[3], bool clouds, unsigned cloudSeed, int upAxis,
							   const Photo* photo, float exposure)
{
	const unsigned char* photoRgb = photo ? photo->m_rgb : 0;
	const int photoWidth = photo ? photo->m_width : 0, photoHeight = photo ? photo->m_height : 0;
	const float photoYaw = photo ? photo->m_yaw : 0.0f;
	if (m_daylightBuilt && m_daylightClouds == clouds && m_daylightCloudSeed == cloudSeed && m_daylightUpAxis == upAxis &&
		memcmp(m_daylightSunDir, sunDir, sizeof(m_daylightSunDir)) == 0 && memcmp(m_daylightSunColor, sunColor, sizeof(m_daylightSunColor)) == 0 &&
		m_photoRgb == photoRgb && m_photoWidth == photoWidth && m_photoHeight == photoHeight && m_photoYaw == photoYaw)
	{
		if (m_exposure != exposure)
		{
			m_exposure = exposure;
			buildDisplay();
		}
		return;
	}
	memcpy(m_daylightSunDir, sunDir, sizeof(m_daylightSunDir));
	memcpy(m_daylightSunColor, sunColor, sizeof(m_daylightSunColor));
	m_daylightClouds = clouds;
	m_daylightCloudSeed = cloudSeed;
	m_daylightUpAxis = upAxis;
	m_photoRgb = photoRgb;
	m_photoWidth = photoWidth;
	m_photoHeight = photoHeight;
	m_photoYaw = photoYaw;
	m_exposure = exposure;
	m_daylightBuilt = true;

	double sun[3] = {sunDir[0], sunDir[1], sunDir[2]};
	const double sunLen = sqrt(sun[0] * sun[0] + sun[1] * sun[1] + sun[2] * sun[2]);
	for (int i = 0; i < 3; i++)
		sun[i] = sunLen > 0.0 ? sun[i] / sunLen : (i == upAxis ? 1.0 : 0.0);
	const double sunUp = sun[upAxis] > kHorizonCos ? sun[upAxis] : kHorizonCos;
	const Preetham model(skyAcos(sunUp));
	double disc[3];
	for (int i = 0; i < 3; i++)
		disc[i] = clamp01(sunColor[i]);
	const double coverage = 0.40 + 0.25 * hashNoise(7, 11, cloudSeed);
	const bool usePhoto = photo && photo->m_rgb && photo->m_width > 0 && photo->m_height > 0;
	const int side = upAxis == 0 ? 1 : 0, other = upAxis == 2 ? 1 : 2;

	const int n = kDayFace;
	m_radiance.assign((size_t)6 * n * n * 3, 0.0f);
	// Pass one: the upper hemisphere, and the solid-angle-weighted mean of its luminance.
	double meanLum = 0.0, upperWeight = 0.0;
	for (int face = 0; face < 6; face++)
	{
		const int axis = face / 2;
		const double sign = (face & 1) ? -1.0 : 1.0;
		for (int j = 0; j < n; j++)
		{
			const double t = (j + 0.5) / n * 2.0 - 1.0;
			for (int i = 0; i < n; i++)
			{
				const double s = (i + 0.5) / n * 2.0 - 1.0;
				double dir[3];
				dir[axis] = sign;
				dir[axis == 0 ? 1 : 0] = s;
				dir[axis == 2 ? 1 : 2] = t;
				const double len2 = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
				const double len = sqrt(len2);
				for (int k = 0; k < 3; k++)
					dir[k] /= len;
				if (dir[upAxis] <= 0.0)
					continue;
				double rgb[3];
				if (usePhoto)
				{
					const double azimuth = swarmAtan2(dir[other], dir[side]);
					const double u = azimuth / (2.0 * kPi) + (double)photo->m_yaw / 360.0;
					const double v = 0.5 - swarmAsin(dir[upAxis]) / kPi;
					const double cosGamma = dir[0] * sun[0] + dir[1] * sun[1] + dir[2] * sun[2];
					photoSample(*photo, u, v, cosGamma > kPhotoHighlightCos, rgb);
				}
				else
				{
					const double cosTheta = dir[upAxis] > kHorizonCos ? dir[upAxis] : kHorizonCos;
					const double cosGamma = dir[0] * sun[0] + dir[1] * sun[1] + dir[2] * sun[2];
					const double gamma = skyAcos(cosGamma);
					model.linear(cosTheta, gamma, cosGamma, rgb);
					if (clouds)
					{
						const double density = cloudDensity(dir, upAxis, cloudSeed, coverage);
						for (int k = 0; k < 3; k++)
							rgb[k] += (disc[k] * kCloudRadiance * (1.0 - kCloudShade * density) - rgb[k]) * density;
					}
					if (gamma < kSunDiscRadius)
					{
						const double f = clamp01((kSunDiscRadius - gamma) / (kSunDiscRadius * kSunDiscEdge));
						for (int k = 0; k < 3; k++)
							rgb[k] += (disc[k] * kSunDiscRadiance - rgb[k]) * f;
					}
				}
				float* out = &m_radiance[texelIndex(face, i, j, n)];
				for (int k = 0; k < 3; k++)
					out[k] = (float)rgb[k];
				const double weight = 1.0 / (len2 * len);
				meanLum += (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) * weight;
				upperWeight += weight;
			}
		}
	}
	meanLum = upperWeight > 0.0 ? meanLum / upperWeight : 0.0;
	// Photographs are scaled to one mean luminance so every sky in a pack lights the world alike.
	// A low sun lights a dimmer sky: the target falls with the sun's height, as the real sky does.
	const double sunShare = sun[upAxis] > kPhotoFullSunUp ? 1.0 : (sun[upAxis] > 0.0 ? sun[upAxis] / kPhotoFullSunUp : 0.0);
	const double target = kPhotoMeanLuminance * (kPhotoDuskShare + (1.0 - kPhotoDuskShare) * sunShare);
	const double scale = (usePhoto && meanLum > 0.0) ? target / meanLum : 1.0;
	if (scale != 1.0)
		for (size_t k = 0; k < m_radiance.size(); k++)
			m_radiance[k] = (float)(m_radiance[k] * scale);
	// Pass two: the ground below the horizon, one colour, the mean of the sky it bounces back.
	double ground[3] = {0.0, 0.0, 0.0};
	double groundWeight = 0.0;
	for (int face = 0; face < 6; face++)
		for (int j = 0; j < n; j++)
			for (int i = 0; i < n; i++)
			{
				const double s = (i + 0.5) / n * 2.0 - 1.0, t = (j + 0.5) / n * 2.0 - 1.0;
				const double len2 = 1.0 + s * s + t * t;
				const int axis = face / 2;
				double up;
				if (axis == upAxis)
					up = (face & 1) ? -1.0 : 1.0;
				else
					up = (upAxis == (axis == 0 ? 1 : 0)) ? s : t;
				if (up <= 0.0)
					continue;
				const float* rad = &m_radiance[texelIndex(face, i, j, n)];
				const double weight = 1.0 / (len2 * sqrt(len2));
				for (int k = 0; k < 3; k++)
					ground[k] += rad[k] * weight;
				groundWeight += weight;
			}
	for (int k = 0; k < 3; k++)
		ground[k] = groundWeight > 0.0 ? ground[k] / groundWeight * kGroundBounce : 0.0;
	// Pass three: fill the ground and project the whole sphere onto the nine harmonics.
	double sh[9][3];
	memset(sh, 0, sizeof(sh));
	for (int face = 0; face < 6; face++)
	{
		const int axis = face / 2;
		const double sign = (face & 1) ? -1.0 : 1.0;
		for (int j = 0; j < n; j++)
		{
			const double t = (j + 0.5) / n * 2.0 - 1.0;
			for (int i = 0; i < n; i++)
			{
				const double s = (i + 0.5) / n * 2.0 - 1.0;
				double dir[3];
				dir[axis] = sign;
				dir[axis == 0 ? 1 : 0] = s;
				dir[axis == 2 ? 1 : 2] = t;
				const double len2 = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
				const double len = sqrt(len2);
				for (int k = 0; k < 3; k++)
					dir[k] /= len;
				float* rad = &m_radiance[texelIndex(face, i, j, n)];
				if (dir[upAxis] <= 0.0)
					for (int k = 0; k < 3; k++)
						rad[k] = (float)ground[k];
				const double weight = (4.0 / ((double)n * n)) / (len2 * len);
				double basis[9];
				shBasis(dir, basis);
				for (int c = 0; c < 9; c++)
					for (int k = 0; k < 3; k++)
						sh[c][k] += basis[c] * rad[k] * weight;
			}
		}
	}
	for (int c = 0; c < 9; c++)
		for (int k = 0; k < 3; k++)
			m_sh[c][k] = (float)sh[c][k];
	buildDisplay();
}

void SwarmSky::forgetPhoto()
{
	if (m_photoRgb)
		m_daylightBuilt = false;
	m_photoRgb = 0;
}

void SwarmSky::buildDisplay()
{
	m_display.resize(m_radiance.size());
	for (size_t k = 0; k < m_radiance.size(); k += 3)
	{
		const float lin[3] = {m_radiance[k] * m_exposure, m_radiance[k + 1] * m_exposure, m_radiance[k + 2] * m_exposure};
		float display[3];
		SwarmAgx::apply(lin, display);
		for (int c = 0; c < 3; c++)
			m_display[k + c] = SwarmAgx::toByte(display[c]);
	}
}

void SwarmSky::radiance(float x, float y, float z, float out[3]) const
{
	if (!m_daylightBuilt)
	{
		out[0] = out[1] = out[2] = 1.0f;
		return;
	}
	const CubeSample cs = cubeSample(x, y, z, kDayFace);
	const float* p00 = &m_radiance[texelIndex(cs.m_face, cs.m_i0, cs.m_j0, kDayFace)];
	const float* p10 = &m_radiance[texelIndex(cs.m_face, cs.m_i1, cs.m_j0, kDayFace)];
	const float* p01 = &m_radiance[texelIndex(cs.m_face, cs.m_i0, cs.m_j1, kDayFace)];
	const float* p11 = &m_radiance[texelIndex(cs.m_face, cs.m_i1, cs.m_j1, kDayFace)];
	for (int k = 0; k < 3; k++)
	{
		const float a = p00[k] + (p10[k] - p00[k]) * cs.m_wi;
		const float b = p01[k] + (p11[k] - p01[k]) * cs.m_wi;
		out[k] = a + (b - a) * cs.m_wj;
	}
}

void SwarmSky::irradiance(const float normal[3], float out[3]) const
{
	if (!m_daylightBuilt)
	{
		out[0] = out[1] = out[2] = 1.0f;
		return;
	}
	// Ramamoorthi and Hanrahan's closed form, divided by pi so a uniform sky of radiance L gives L.
	const float c1 = 0.429043f, c2 = 0.511664f, c3 = 0.743125f, c4 = 0.886227f, c5 = 0.247708f;
	const float x = normal[0], y = normal[1], z = normal[2];
	for (int k = 0; k < 3; k++)
	{
		float e = c1 * m_sh[8][k] * (x * x - y * y) + c3 * m_sh[6][k] * z * z + c4 * m_sh[0][k] - c5 * m_sh[6][k];
		e += 2.0f * c1 * (m_sh[4][k] * x * y + m_sh[7][k] * x * z + m_sh[5][k] * y * z);
		e += 2.0f * c2 * (m_sh[3][k] * x + m_sh[1][k] * y + m_sh[2][k] * z);
		e *= 0.3183098861837907f;
		out[k] = e > 0.0f ? e : 0.0f;
	}
}

void SwarmSky::lookupDisplay(float x, float y, float z, unsigned char out[3]) const
{
	if (!m_daylightBuilt)
	{
		out[0] = out[1] = out[2] = 255;
		return;
	}
	const CubeSample cs = cubeSample(x, y, z, kDayFace);
	const unsigned char* p00 = &m_display[texelIndex(cs.m_face, cs.m_i0, cs.m_j0, kDayFace)];
	const unsigned char* p10 = &m_display[texelIndex(cs.m_face, cs.m_i1, cs.m_j0, kDayFace)];
	const unsigned char* p01 = &m_display[texelIndex(cs.m_face, cs.m_i0, cs.m_j1, kDayFace)];
	const unsigned char* p11 = &m_display[texelIndex(cs.m_face, cs.m_i1, cs.m_j1, kDayFace)];
	const int wi = (int)(cs.m_wi * 256.f), wj = (int)(cs.m_wj * 256.f);
	const int w00 = (256 - wi) * (256 - wj), w10 = wi * (256 - wj), w01 = (256 - wi) * wj, w11 = wi * wj;
	for (int k = 0; k < 3; k++)
		out[k] = (unsigned char)((p00[k] * w00 + p10[k] * w10 + p01[k] * w01 + p11[k] * w11 + 32768) >> 16);
}
