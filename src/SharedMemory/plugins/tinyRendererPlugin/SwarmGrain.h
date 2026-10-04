#ifndef SWARM_GRAIN_H
#define SWARM_GRAIN_H

// Seeded sensor grain and the separable blur the camera chains share (SwarmThermal, SwarmLowLight): integer hashes
// and plain arithmetic, so every validator draws the same grain whatever the thread count.

#include <vector>

#include "SwarmDaylight.h"

namespace SwarmGrain
{
// A 32-bit integer mix with full avalanche, so neighbouring pixels draw unrelated grain.
inline unsigned int mix32(unsigned int x)
{
	x ^= x >> 16;
	x *= 0x7feb352du;
	x ^= x >> 15;
	x *= 0x846ca68bu;
	x ^= x >> 16;
	return x;
}

// A unit Gaussian drawn from (seed, index): four uniforms summed, centred and scaled to unit variance.
inline float gaussian(unsigned int seed, unsigned int index)
{
	unsigned int h = mix32(seed ^ mix32(index + 0x9e3779b9u));
	float sum = 0.0f;
	for (int k = 0; k < 4; k++)
	{
		sum += (float)(h >> 8) * (1.0f / 16777216.0f);
		h = mix32(h + 0x632be5abu);
	}
	return (sum - 2.0f) * 1.7320508f;
}

#if defined(__GNUC__)
typedef unsigned int SwarmWords8 __attribute__((vector_size(32)));

// mix32 on each of eight lanes.
inline SwarmWords8 mix32(SwarmWords8 x)
{
	x ^= x >> 16;
	x *= 0x7feb352du;
	x ^= x >> 15;
	x *= 0x846ca68bu;
	x ^= x >> 16;
	return x;
}

// gaussian for the eight indices from `first` up: the same integer steps and the same float sum on each lane.
inline SwarmLanes8 gaussian8(unsigned int seed, unsigned int first)
{
	const SwarmWords8 step = {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u};
	SwarmWords8 h = mix32(seed ^ mix32(first + step + 0x9e3779b9u));
	SwarmLanes8 sum = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
	for (int k = 0; k < 4; k++)
	{
		// Below 2^24, so the signed conversion gives the same float.
		sum += __builtin_convertvector((SwarmLaneMask8)(h >> 8), SwarmLanes8) * (1.0f / 16777216.0f);
		h = mix32(h + 0x632be5abu);
	}
	return (sum - 2.0f) * 1.7320508f;
}
#endif

// Normalised Gaussian taps for `sigma`, out to three sigma.
inline std::vector<float> gaussianKernel(float sigma)
{
	const int radius = (int)(3.0f * sigma + 0.999f);
	std::vector<float> taps((size_t)(2 * radius + 1));
	double total = 0.0;
	for (int i = -radius; i <= radius; i++)
	{
		taps[(size_t)(i + radius)] = (float)swarmExp(-0.5 * (double)(i * i) / ((double)sigma * sigma));
		total += taps[(size_t)(i + radius)];
	}
	for (size_t i = 0; i < taps.size(); i++)
		taps[i] = (float)(taps[i] / total);
	return taps;
}

// `buffer` grown to `size` floats and kept for the next frame; callers write every element before reading it.
inline float* reuse(std::vector<float>& buffer, size_t size)
{
	if (buffer.size() < size)
		buffer.resize(size);
	return &buffer[0];
}

// The blur's first pass on one row: `line` blurred along itself into `acc`, the edge pixel repeated. `tap` points at
// the middle of 2 * radius + 1 taps. Pixels a full radius from the border skip the clamp, which reads the same samples
// in the same order; each pixel adds its taps from -radius up.
inline void blurRow(const float* line, float* acc, int width, const float* tap, int radius)
{
	const int inner0 = radius < width ? radius : width;
	const int inner1 = width - radius > inner0 ? width - radius : inner0;
	for (int x = 0; x < width; x++)
	{
		if (x == inner0)
			x = inner1;
		if (x >= width)
			break;
		float sum = 0.0f;
		for (int k = -radius; k <= radius; k++)
		{
			int sx = x + k;
			sx = sx < 0 ? 0 : (sx >= width ? width - 1 : sx);
			sum += tap[k] * line[sx];
		}
		acc[x] = sum;
	}
	int x = inner0;
#if defined(__GNUC__)
	// Eight pixels at a time, their sums kept in a register.
	for (; x + 8 <= inner1; x += 8)
	{
		SwarmLanes8 sum = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
		for (int k = -radius; k <= radius; k++)
		{
			SwarmLanes8 v;
			memcpy(&v, line + x + k, sizeof(v));
			sum += tap[k] * v;
		}
		memcpy(acc + x, &sum, sizeof(sum));
	}
#endif
	for (; x < inner1; x++)
	{
		float sum = 0.0f;
		for (int k = -radius; k <= radius; k++)
			sum += tap[k] * line[x + k];
		acc[x] = sum;
	}
}

// The blur's second pass for output row `y`: the first pass's rows blurred down the columns into `acc`, the edge row
// repeated; each pixel adds its taps from -radius up.
inline void blurColumn(const float* rows, float* acc, int width, int height, int y, const float* tap, int radius)
{
	int x = 0;
#if defined(__GNUC__)
	for (; x + 8 <= width; x += 8)
	{
		SwarmLanes8 sum = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
		for (int k = -radius; k <= radius; k++)
		{
			int sy = y + k;
			sy = sy < 0 ? 0 : (sy >= height ? height - 1 : sy);
			SwarmLanes8 v;
			memcpy(&v, rows + (size_t)sy * width + x, sizeof(v));
			sum += tap[k] * v;
		}
		memcpy(acc + x, &sum, sizeof(sum));
	}
#endif
	for (; x < width; x++)
	{
		float sum = 0.0f;
		for (int k = -radius; k <= radius; k++)
		{
			int sy = y + k;
			sy = sy < 0 ? 0 : (sy >= height ? height - 1 : sy);
			sum += tap[k] * rows[(size_t)sy * width + x];
		}
		acc[x] = sum;
	}
}

// Separable blur with the edge pixel repeated, rows then columns; each output pixel sums its taps in one fixed order.
inline void blur(const float* in, float* out, int width, int height, const std::vector<float>& taps, int threads)
{
	const int radius = (int)taps.size() / 2;
	const float* tap = &taps[(size_t)radius];
	static thread_local std::vector<float> rowsBuffer;
	float* rows = reuse(rowsBuffer, (size_t)width * height);
#pragma omp parallel for num_threads(threads) schedule(static)
	for (int y = 0; y < height; y++)
		blurRow(in + (size_t)y * width, rows + (size_t)y * width, width, tap, radius);
#pragma omp parallel for num_threads(threads) schedule(static)
	for (int y = 0; y < height; y++)
		blurColumn(rows, out + (size_t)y * width, width, height, y, tap, radius);
}
}  // namespace SwarmGrain

#endif  // SWARM_GRAIN_H
