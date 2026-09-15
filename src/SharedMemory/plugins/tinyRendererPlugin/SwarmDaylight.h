#ifndef SWARM_DAYLIGHT_H
#define SWARM_DAYLIGHT_H

#include <math.h>
#include <string.h>

// The daylight arithmetic as polynomials and series in plain operations, since the C library's exp, log and atan pick an FMA path per CPU.

// exp(x) as a range reduction by ln 2 and a ninth-order series on the remainder.
inline double swarmExp(double x)
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

// log2(x) for x > 0: the exponent from the float's bits, the mantissa through 2 atanh((m - 1) / (m + 1)) / ln 2.
inline float swarmLog2(float x)
{
	unsigned int bits;
	memcpy(&bits, &x, sizeof(bits));
	const int e = (int)((bits >> 23) & 255) - 127;
	bits = (bits & 0x007fffffu) | 0x3f800000u;
	float m;
	memcpy(&m, &bits, sizeof(m));
	const float y = (m - 1.0f) / (m + 1.0f);
	const float y2 = y * y;
	const float atanh = y * (1.0f + y2 * (1.0f / 3.0f + y2 * (1.0f / 5.0f + y2 * (1.0f / 7.0f + y2 * (1.0f / 9.0f + y2 * (1.0f / 11.0f))))));
	return (float)e + atanh * 2.8853900817779268f;
}

// atan(t) for |t| <= 1, an odd polynomial good to about 1e-5 radians.
inline double swarmAtanUnit(double t)
{
	const double t2 = t * t;
	return t * (0.99997726 + t2 * (-0.33262347 + t2 * (0.19354346 + t2 * (-0.11643287 + t2 * (0.05265332 + t2 * -0.01172120)))));
}

// atan2(y, x) in (-pi, pi], from the unit-range polynomial and the octant identities.
inline double swarmAtan2(double y, double x)
{
	const double pi = 3.14159265358979323846;
	const double ax = x < 0.0 ? -x : x, ay = y < 0.0 ? -y : y;
	if (ax == 0.0 && ay == 0.0)
		return 0.0;
	double r = ax >= ay ? swarmAtanUnit(ay / ax) : pi * 0.5 - swarmAtanUnit(ax / ay);
	if (x < 0.0)
		r = pi - r;
	return y < 0.0 ? -r : r;
}

// asin(z) for z in [-1, 1], through atan2 so it shares the one polynomial.
inline double swarmAsin(double z)
{
	z = z > 1.0 ? 1.0 : (z < -1.0 ? -1.0 : z);
	return swarmAtan2(z, sqrt(1.0 - z * z));
}

// The AgX film curve as its public polynomial approximation: inset matrix, log2 over 16.5 stops, sigmoid, outset matrix, display 0..1 out.
struct SwarmAgx
{
	static void apply(const float linear[3], float display[3])
	{
		// Rows of the inset (linear sRGB into the AgX working space) and outset matrices; each row sums to one.
		static const float kInset[3][3] = {{0.544814746488245f, 0.373787398372697f, 0.0813978551390581f},
										   {0.140416948464053f, 0.754137554567394f, 0.105445496968552f},
										   {0.0888104196149096f, 0.178871756420858f, 0.732317823964232f}};
		static const float kOutset[3][3] = {{1.96488741169489f, -0.855988495690215f, -0.108898916004672f},
											{-0.299313364904742f, 1.32639796461980f, -0.0270845997150571f},
											{-0.164352742528393f, -0.238183969428088f, 1.40253671195648f}};
		const float kMinEv = -12.47393f;
		const float kMaxEv = 4.026069f;
		float encoded[3];
		for (int i = 0; i < 3; i++)
		{
			float v = (kInset[i][0] * linear[0] + kInset[i][1] * linear[1]) + kInset[i][2] * linear[2];
			v = v > 1e-10f ? v : 1e-10f;
			v = (swarmLog2(v) - kMinEv) / (kMaxEv - kMinEv);
			v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
			const float x2 = v * v, x4 = x2 * x2;
			encoded[i] = ((((15.5f * x4 * x2 - 40.14f * x4 * v) + 31.96f * x4) - 6.868f * x2 * v) + 0.4298f * x2) + (0.1191f * v - 0.00232f);
		}
		for (int i = 0; i < 3; i++)
		{
			const float v = (kOutset[i][0] * encoded[0] + kOutset[i][1] * encoded[1]) + kOutset[i][2] * encoded[2];
			display[i] = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
		}
	}

	// The byte for a display value: rounded to nearest, as the sky's bytes are.
	static unsigned char toByte(float display)
	{
		return (unsigned char)(display * 255.0f + 0.5f);
	}
};

#endif  // SWARM_DAYLIGHT_H
