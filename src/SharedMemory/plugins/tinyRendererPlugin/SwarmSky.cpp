#include "SwarmSky.h"

#include <math.h>
#include <string.h>

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
	const double ln2 = 0.6931471805599453;
	int n = (int)(x / ln2 + (x < 0.0 ? -0.5 : 0.5));
	if (n < -1000)
		return 0.0;
	if (n > 1000)
		n = 1000;
	const double r = x - n * ln2;
	const double p = 1.0 + r * (1.0 + r * (1.0 / 2.0 + r * (1.0 / 6.0 + r * (1.0 / 24.0 + r * (1.0 / 120.0 + r * (1.0 / 720.0 + r * (1.0 / 5040.0 + r * (1.0 / 40320.0 + r * (1.0 / 362880.0)))))))));
	unsigned long long bits = (unsigned long long)(n + 1023) << 52;
	double scale;
	memcpy(&scale, &bits, sizeof(scale));
	return p * scale;
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
	: m_clouds(false), m_cloudSeed(0), m_upAxis(2), m_built(false)
{
	for (int i = 0; i < 3; i++)
	{
		m_ambient[i] = m_horizon[i] = m_zenith[i] = 1.0f;
		m_sunDir[i] = 0.0f;
		m_sunColor[i] = 1.0f;
	}
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
