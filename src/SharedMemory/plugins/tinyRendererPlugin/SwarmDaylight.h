#ifndef SWARM_DAYLIGHT_H
#define SWARM_DAYLIGHT_H

#include <math.h>
#include <string.h>

// The daylight arithmetic as polynomials and series in plain operations, since the C library's exp, log and atan pick an FMA path per CPU.

// exp(x) as a range reduction by ln 2 and a ninth-order series on the remainder.
inline double swarmExp(double x)
{
	const double ln2 = 0.6931471805599453;
	// Within 0.345 of zero the reduction always rounds to n = 0 (x / ln2 stays inside +-0.4978), so it needs no division.
	int n = 0;
	if (!(x > -0.345 && x < 0.345))
		n = (int)(x / ln2 + (x < 0.0 ? -0.5 : 0.5));
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

#if defined(__GNUC__)
// Four floats in one vector register, with per-lane IEEE arithmetic.
typedef float SwarmLanes __attribute__((vector_size(16)));
typedef int SwarmLaneMask __attribute__((vector_size(16)));

// Per lane: a where the mask is set, b elsewhere, as the scalar ternary picks.
inline SwarmLanes swarmSelect(SwarmLaneMask mask, SwarmLanes a, SwarmLanes b)
{
	return (SwarmLanes)((mask & (SwarmLaneMask)a) | (~mask & (SwarmLaneMask)b));
}

// swarmLog2 on each lane, for lanes above zero: the same steps on the same bits.
inline SwarmLanes swarmLog2(SwarmLanes x)
{
	const SwarmLaneMask bits = (SwarmLaneMask)x;
	const SwarmLaneMask e = ((bits >> 23) & 255) - 127;
	const SwarmLanes m = (SwarmLanes)((bits & 0x007fffff) | 0x3f800000);
	const SwarmLanes y = (m - 1.0f) / (m + 1.0f);
	const SwarmLanes y2 = y * y;
	const SwarmLanes atanh = y * (1.0f + y2 * (1.0f / 3.0f + y2 * (1.0f / 5.0f + y2 * (1.0f / 7.0f + y2 * (1.0f / 9.0f + y2 * (1.0f / 11.0f))))));
	return __builtin_convertvector(e, SwarmLanes) + atanh * 2.8853900817779268f;
}

// Eight floats, eight ints, four doubles and four 64-bit ints in one vector register each, with per-lane IEEE arithmetic.
typedef float SwarmLanes8 __attribute__((vector_size(32)));
typedef int SwarmLaneMask8 __attribute__((vector_size(32)));
typedef double SwarmDoubles4 __attribute__((vector_size(32)));
typedef long long SwarmLongs4 __attribute__((vector_size(32)));

// Per lane: a where the mask is set, b elsewhere.
inline SwarmLanes8 swarmSelect(SwarmLaneMask8 mask, SwarmLanes8 a, SwarmLanes8 b)
{
	return (SwarmLanes8)((mask & (SwarmLaneMask8)a) | (~mask & (SwarmLaneMask8)b));
}

// swarmLog2 on each of eight lanes above zero: the same steps on the same bits.
inline SwarmLanes8 swarmLog2(SwarmLanes8 x)
{
	const SwarmLaneMask8 bits = (SwarmLaneMask8)x;
	const SwarmLaneMask8 e = ((bits >> 23) & 255) - 127;
	const SwarmLanes8 m = (SwarmLanes8)((bits & 0x007fffff) | 0x3f800000);
	const SwarmLanes8 y = (m - 1.0f) / (m + 1.0f);
	const SwarmLanes8 y2 = y * y;
	const SwarmLanes8 atanh = y * (1.0f + y2 * (1.0f / 3.0f + y2 * (1.0f / 5.0f + y2 * (1.0f / 7.0f + y2 * (1.0f / 9.0f + y2 * (1.0f / 11.0f))))));
	return __builtin_convertvector(e, SwarmLanes8) + atanh * 2.8853900817779268f;
}

// swarmExp on each of four lanes: the same reduction, the same series and the same exponent bits as one call.
inline SwarmDoubles4 swarmExp(SwarmDoubles4 x)
{
	const double ln2 = 0.6931471805599453;
	const SwarmDoubles4 zero = {0.0, 0.0, 0.0, 0.0};
	const SwarmLongs4 outside = ~((x > -0.345) & (x < 0.345));
	// With every lane inside the window n is 0: x less 0 * ln2 is x and the series times 2^0 is itself, so only the series
	// is left, the same bits without the reduction.
	if (!(outside[0] | outside[1] | outside[2] | outside[3]))
		return 1.0 + x * (1.0 + x * (1.0 / 2.0 + x * (1.0 / 6.0 + x * (1.0 / 24.0 + x * (1.0 / 120.0 + x * (1.0 / 720.0 + x * (1.0 / 5040.0 + x * (1.0 / 40320.0 + x * (1.0 / 362880.0)))))))));
	const SwarmLongs4 below = x < 0.0;
	const SwarmDoubles4 half = (SwarmDoubles4)((below & (SwarmLongs4)(zero - 0.5)) | (~below & (SwarmLongs4)(zero + 0.5)));
	SwarmLongs4 n = __builtin_convertvector(x / ln2 + half, SwarmLongs4) & outside;
	const SwarmLongs4 under = n < -1000;
	n = n > 1000 ? (SwarmLongs4){1000, 1000, 1000, 1000} : n;
	const SwarmDoubles4 r = x - __builtin_convertvector(n, SwarmDoubles4) * ln2;
	const SwarmDoubles4 p = 1.0 + r * (1.0 + r * (1.0 / 2.0 + r * (1.0 / 6.0 + r * (1.0 / 24.0 + r * (1.0 / 120.0 + r * (1.0 / 720.0 + r * (1.0 / 5040.0 + r * (1.0 / 40320.0 + r * (1.0 / 362880.0)))))))));
	const SwarmDoubles4 scaled = p * (SwarmDoubles4)((n + 1023) << 52);
	return (SwarmDoubles4)(~under & (SwarmLongs4)scaled);
}
#endif

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

// cos(x) for x in [-pi, pi], the Taylor series to the x^20 term, below 1e-12 over the range.
inline double swarmCos(double x)
{
	const double x2 = x * x;
	double term = 1.0, sum = 1.0;
	for (int k = 1; k <= 10; k++)
	{
		term *= -x2 / (double)((2 * k - 1) * (2 * k));
		sum += term;
	}
	return sum;
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
#if defined(__GNUC__)
		// The channels as vector lanes, each running the scalar steps below in order, so IEEE gives the same bits.
		const SwarmLanes zero = {0.0f, 0.0f, 0.0f, 0.0f}, one = {1.0f, 1.0f, 1.0f, 1.0f}, tiny = {1e-10f, 1e-10f, 1e-10f, 1e-10f};
		const SwarmLanes in0 = {kInset[0][0], kInset[1][0], kInset[2][0], 0.0f};
		const SwarmLanes in1 = {kInset[0][1], kInset[1][1], kInset[2][1], 0.0f};
		const SwarmLanes in2 = {kInset[0][2], kInset[1][2], kInset[2][2], 0.0f};
		SwarmLanes v = (in0 * linear[0] + in1 * linear[1]) + in2 * linear[2];
		v = swarmSelect(v > tiny, v, tiny);
		v = (swarmLog2(v) - kMinEv) / (kMaxEv - kMinEv);
		v = swarmSelect(v < zero, zero, swarmSelect(v > one, one, v));
		const SwarmLanes x2 = v * v, x4 = x2 * x2;
		const SwarmLanes encoded = ((((15.5f * x4 * x2 - 40.14f * x4 * v) + 31.96f * x4) - 6.868f * x2 * v) + 0.4298f * x2) + (0.1191f * v - 0.00232f);
		const SwarmLanes out0 = {kOutset[0][0], kOutset[1][0], kOutset[2][0], 0.0f};
		const SwarmLanes out1 = {kOutset[0][1], kOutset[1][1], kOutset[2][1], 0.0f};
		const SwarmLanes out2 = {kOutset[0][2], kOutset[1][2], kOutset[2][2], 0.0f};
		SwarmLanes d = (out0 * encoded[0] + out1 * encoded[1]) + out2 * encoded[2];
		d = swarmSelect(d < zero, zero, swarmSelect(d > one, one, d));
		for (int i = 0; i < 3; i++)
			display[i] = d[i];
#else
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
#endif
	}

#if defined(__GNUC__)
	// apply on eight colours at once, one colour per lane and one vector per channel: each lane runs the scalar steps
	// above in their order, so IEEE gives each colour the bits apply gives it.
	static void apply8(const SwarmLanes8 linear[3], SwarmLanes8 display[3])
	{
		static const float kInset[3][3] = {{0.544814746488245f, 0.373787398372697f, 0.0813978551390581f},
										   {0.140416948464053f, 0.754137554567394f, 0.105445496968552f},
										   {0.0888104196149096f, 0.178871756420858f, 0.732317823964232f}};
		static const float kOutset[3][3] = {{1.96488741169489f, -0.855988495690215f, -0.108898916004672f},
											{-0.299313364904742f, 1.32639796461980f, -0.0270845997150571f},
											{-0.164352742528393f, -0.238183969428088f, 1.40253671195648f}};
		const float kMinEv = -12.47393f;
		const float kMaxEv = 4.026069f;
		const SwarmLanes8 zero = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, one = zero + 1.0f, tiny = zero + 1e-10f;
		SwarmLanes8 encoded[3];
		for (int i = 0; i < 3; i++)
		{
			SwarmLanes8 v = (kInset[i][0] * linear[0] + kInset[i][1] * linear[1]) + kInset[i][2] * linear[2];
			v = swarmSelect(v > tiny, v, tiny);
			v = (swarmLog2(v) - kMinEv) / (kMaxEv - kMinEv);
			v = swarmSelect(v < zero, zero, swarmSelect(v > one, one, v));
			const SwarmLanes8 x2 = v * v, x4 = x2 * x2;
			encoded[i] = ((((15.5f * x4 * x2 - 40.14f * x4 * v) + 31.96f * x4) - 6.868f * x2 * v) + 0.4298f * x2) + (0.1191f * v - 0.00232f);
		}
		for (int i = 0; i < 3; i++)
		{
			const SwarmLanes8 v = (kOutset[i][0] * encoded[0] + kOutset[i][1] * encoded[1]) + kOutset[i][2] * encoded[2];
			display[i] = swarmSelect(v < zero, zero, swarmSelect(v > one, one, v));
		}
	}
#endif

	// The byte for a display value: rounded to nearest, as the sky's bytes are.
	static unsigned char toByte(float display)
	{
		return (unsigned char)(display * 255.0f + 0.5f);
	}
};

#endif  // SWARM_DAYLIGHT_H
