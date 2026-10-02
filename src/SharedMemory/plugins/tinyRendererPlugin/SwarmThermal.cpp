#include "SwarmThermal.h"

#include <math.h>
#include <vector>

#include "SwarmDaylight.h"
#include "SwarmGrain.h"

namespace
{
using SwarmGrain::blur;
using SwarmGrain::gaussian;
using SwarmGrain::gaussianKernel;
using SwarmGrain::mix32;

// The radiance table: every quarter degree from -120 C to 240 C, read with a linear blend.
const float kTableMinC = -120.0f;
const float kTableStepC = 0.25f;
const int kTableSize = 1441;
// The M4TD core's band and the second radiation constant, in micrometres and micrometre kelvin.
const double kBandLowUm = 8.0;
const double kBandHighUm = 14.0;
const double kPlanckC2 = 14387.77;
// Lens: a Gaussian spot, plus a wide glow at quarter resolution that lets a hot body bleed into its surroundings.
const float kLensSigmaPx = 1.0f;
const float kGlowShare = 0.05f;
const float kGlowSigmaQuarterPx = 1.75f;
// Sensor: temporal grain, per-row flicker and a fixed column pattern, in kelvin at the scene's level. The M4TD quotes a
// 50 mK NETD; the values here match the grain of frames from a compressed live stream, which is what the model sees.
const float kNetdK = 0.025f;
const float kRowNoiseK = 0.003f;
const float kColumnNoiseK = 0.006f;
const unsigned int kColumnSalt = 0x5a17c0deu;
// Contrast: 1024 bins between the frame's coldest and hottest points, each capped at kClip times the mean bin with the
// excess spread over every bin, so a flat scene's noise is never stretched past a bounded gain; the equalised curve
// blended with the straight one, then detail lifted against a wider blur, as DJI's AGC and DDE do.
const int kBins = 1024;
const float kClip = 2.0f;
const float kEqualised = 0.7f;
const float kDetailGain = 0.3f;
const float kDetailSigmaPx = 2.0f;

// Band radiance of a black body at `kelvin`, Simpson's rule over the band in 240 steps, arbitrary units.
double bandRadiance(double kelvin)
{
	const int steps = 240;
	const double h = (kBandHighUm - kBandLowUm) / steps;
	double sum = 0.0;
	for (int i = 0; i <= steps; i++)
	{
		const double lambda = kBandLowUm + h * i;
		const double l5 = lambda * lambda * lambda * lambda * lambda;
		const double value = 1.0 / (l5 * (swarmExp(kPlanckC2 / (lambda * kelvin)) - 1.0));
		sum += value * ((i == 0 || i == steps) ? 1.0 : ((i & 1) ? 4.0 : 2.0));
	}
	return sum * h / 3.0;
}

// The radiance table, built once when the library loads, from the polynomial exp so every machine holds the same values.
struct RadianceTable
{
	float m_values[kTableSize];

	RadianceTable()
	{
		const double zero = bandRadiance(273.15);
		for (int i = 0; i < kTableSize; i++)
			m_values[i] = (float)(bandRadiance(273.15 + kTableMinC + kTableStepC * i) / zero);
	}
};

const RadianceTable kRadiance;

// The lens: the Gaussian spot, and a share of glow blurred at quarter resolution and read back bilinearly.
void lens(const float* in, float* out, int width, int height, int threads)
{
	blur(in, out, width, height, gaussianKernel(kLensSigmaPx), threads);
	const int qw = (width + 3) / 4, qh = (height + 3) / 4;
	std::vector<float> quarter((size_t)qw * qh), glow((size_t)qw * qh);
	for (int y = 0; y < qh; y++)
		for (int x = 0; x < qw; x++)
		{
			float sum = 0.0f;
			int count = 0;
			for (int dy = 0; dy < 4 && y * 4 + dy < height; dy++)
				for (int dx = 0; dx < 4 && x * 4 + dx < width; dx++)
				{
					sum += in[(size_t)(y * 4 + dy) * width + x * 4 + dx];
					count++;
				}
			quarter[(size_t)y * qw + x] = sum / (float)count;
		}
	blur(&quarter[0], &glow[0], qw, qh, gaussianKernel(kGlowSigmaQuarterPx), threads);
#pragma omp parallel for num_threads(threads) schedule(static)
	for (int y = 0; y < height; y++)
	{
		float fy = ((float)y + 0.5f) * 0.25f - 0.5f;
		fy = fy < 0.0f ? 0.0f : fy;
		const int y0 = (int)fy < qh - 1 ? (int)fy : qh - 1;
		const int y1 = y0 + 1 < qh ? y0 + 1 : y0;
		const float ay = fy - (float)y0 < 1.0f ? fy - (float)y0 : 1.0f;
		for (int x = 0; x < width; x++)
		{
			float fx = ((float)x + 0.5f) * 0.25f - 0.5f;
			fx = fx < 0.0f ? 0.0f : fx;
			const int x0 = (int)fx < qw - 1 ? (int)fx : qw - 1;
			const int x1 = x0 + 1 < qw ? x0 + 1 : x0;
			const float ax = fx - (float)x0 < 1.0f ? fx - (float)x0 : 1.0f;
			const float top = glow[(size_t)y0 * qw + x0] + (glow[(size_t)y0 * qw + x1] - glow[(size_t)y0 * qw + x0]) * ax;
			const float bottom = glow[(size_t)y1 * qw + x0] + (glow[(size_t)y1 * qw + x1] - glow[(size_t)y1 * qw + x0]) * ax;
			const size_t i = (size_t)y * width + x;
			out[i] = out[i] * (1.0f - kGlowShare) + (top + (bottom - top) * ay) * kGlowShare;
		}
	}
}

// Radiance change per kelvin at the level `level`, from the table around the temperature that radiates it.
float radiancePerKelvin(float level)
{
	int low = 0, high = kTableSize - 1;
	while (high - low > 1)
	{
		const int mid = (low + high) / 2;
		if (kRadiance.m_values[mid] <= level)
			low = mid;
		else
			high = mid;
	}
	return (kRadiance.m_values[high] - kRadiance.m_values[low]) / kTableStepC;
}
}  // namespace

float SwarmThermal::radiance(float celsius)
{
	float x = (celsius - kTableMinC) / kTableStepC;
	x = x < 0.0f ? 0.0f : (x > (float)(kTableSize - 1) ? (float)(kTableSize - 1) : x);
	int i = (int)x;
	i = i < kTableSize - 1 ? i : kTableSize - 2;
	const float f = x - (float)i;
	return kRadiance.m_values[i] + (kRadiance.m_values[i + 1] - kRadiance.m_values[i]) * f;
}

SwarmThermal::Sky SwarmThermal::sky(float airC, float zenithC)
{
	Sky out;
	out.m_air = radiance(airC);
	float clear = 1.0f - radiance(zenithC) / out.m_air;
	clear = clear > 1e-6f ? (clear < 1.0f ? clear : 1.0f) : 1e-6f;
	out.m_logClear = swarmLog2(clear) * 0.6931472f;
	// Twice the integral of emissivity times up over up in [0, 1], by the midpoint rule in 64 steps.
	double sum = 0.0;
	for (int i = 0; i < 64; i++)
	{
		const double up = (i + 0.5) / 64.0;
		sum += (1.0 - swarmExp((double)out.m_logClear / up)) * up;
	}
	out.m_hemisphere = (float)(2.0 * sum / 64.0) * out.m_air;
	return out;
}

float SwarmThermal::skyRadiance(const Sky& sky, float up)
{
	if (!(up > 0.0f))
		return sky.m_air;
	up = up < 1.0f ? up : 1.0f;
	return (float)(1.0 - swarmExp((double)sky.m_logClear / (double)up)) * sky.m_air;
}

void SwarmThermal::develop(const float* radiance, int width, int height, unsigned int seed, int threads, unsigned char* rgb)
{
	const int numPixels = width * height;
	if (numPixels <= 0)
		return;
	// The frame buffers are kept for the next frame and written in full before they are read.
	static thread_local std::vector<float> frameBuffer, greyBuffer, baseBuffer;
	float* frame = SwarmGrain::reuse(frameBuffer, (size_t)numPixels);
	lens(radiance, frame, width, height, threads);

	// The grain is scaled at the frame's own level, where the sensor's NETD is quoted.
	double total = 0.0;
	for (int i = 0; i < numPixels; i++)
		total += frame[(size_t)i];
	const float perKelvin = radiancePerKelvin((float)(total / numPixels));
	std::vector<float> columns((size_t)width), rows((size_t)height);
	for (int x = 0; x < width; x++)
		columns[(size_t)x] = kColumnNoiseK * gaussian(kColumnSalt, (unsigned int)x);
	for (int y = 0; y < height; y++)
		rows[(size_t)y] = kRowNoiseK * gaussian(mix32(seed) ^ 0x2545f491u, (unsigned int)y);
#pragma omp parallel for num_threads(threads) schedule(static)
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
		{
			const size_t i = (size_t)y * width + x;
			frame[i] += perKelvin * ((kNetdK * gaussian(seed, (unsigned int)i) + columns[(size_t)x]) + rows[(size_t)y]);
		}

	// Coldest point black, hottest white; between them the straight line blended with the plateau-equalised curve.
	float lo = frame[0], hi = frame[0];
	for (int i = 1; i < numPixels; i++)
	{
		lo = frame[(size_t)i] < lo ? frame[(size_t)i] : lo;
		hi = frame[(size_t)i] > hi ? frame[(size_t)i] : hi;
	}
	float* grey = SwarmGrain::reuse(greyBuffer, (size_t)numPixels);
	if (!(hi > lo))
		for (int i = 0; i < numPixels; i++)
			grey[(size_t)i] = 0.0f;
	else
	{
		const float toBins = (float)kBins / (hi - lo);
		std::vector<long long> counts((size_t)kBins, 0);
		for (int i = 0; i < numPixels; i++)
		{
			int b = (int)((frame[(size_t)i] - lo) * toBins);
			counts[(size_t)(b < 0 ? 0 : (b >= kBins ? kBins - 1 : b))]++;
		}
		long long cap = (long long)(kClip * (float)numPixels / (float)kBins);
		cap = cap > 0 ? cap : 1;
		long long excess = 0;
		for (int b = 0; b < kBins; b++)
			excess += counts[(size_t)b] > cap ? counts[(size_t)b] - cap : 0;
		const double spread = (double)excess / kBins;
		std::vector<float> below((size_t)kBins + 1, 0.0f);
		double cumulative = 0.0;
		for (int b = 0; b < kBins; b++)
		{
			below[(size_t)b] = (float)cumulative;
			cumulative += (double)(counts[(size_t)b] < cap ? counts[(size_t)b] : cap) + spread;
		}
		below[(size_t)kBins] = (float)cumulative;
		const float invCumulative = 1.0f / (float)cumulative;
		const float invSpan = 1.0f / (hi - lo);
#pragma omp parallel for num_threads(threads) schedule(static)
		for (int i = 0; i < numPixels; i++)
		{
			const float position = (frame[(size_t)i] - lo) * toBins;
			int b = (int)position;
			b = b < 0 ? 0 : (b >= kBins ? kBins - 1 : b);
			const float within = position - (float)b;
			const float equalised = (below[(size_t)b] + (below[(size_t)b + 1] - below[(size_t)b]) * within) * invCumulative;
			const float straight = (frame[(size_t)i] - lo) * invSpan;
			grey[(size_t)i] = straight * (1.0f - kEqualised) + equalised * kEqualised;
		}
	}

	// Detail against a wider blur; the ends stay at 0 and 1, since the blur of values in [0, 1] stays inside it.
	float* base = SwarmGrain::reuse(baseBuffer, (size_t)numPixels);
	blur(grey, base, width, height, gaussianKernel(kDetailSigmaPx), threads);
#pragma omp parallel for num_threads(threads) schedule(static)
	for (int i = 0; i < numPixels; i++)
	{
		float g = grey[(size_t)i] + kDetailGain * (grey[(size_t)i] - base[(size_t)i]);
		g = g < 0.0f ? 0.0f : (g > 1.0f ? 1.0f : g);
		const unsigned char byte = (unsigned char)(g * 255.0f + 0.5f);
		rgb[(size_t)i * 3] = rgb[(size_t)i * 3 + 1] = rgb[(size_t)i * 3 + 2] = byte;
	}
}
