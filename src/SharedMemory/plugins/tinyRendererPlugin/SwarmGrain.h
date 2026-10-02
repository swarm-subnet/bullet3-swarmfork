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

// Separable blur with the edge pixel repeated, rows then columns; each output pixel sums its taps in one fixed order.
// Pixels a full radius from the border skip the clamp, which reads the same samples in the same order.
inline void blur(const float* in, float* out, int width, int height, const std::vector<float>& taps, int threads)
{
	const int radius = (int)taps.size() / 2;
	const float* tap = &taps[(size_t)radius];
	const int inner0 = radius < width ? radius : width;
	const int inner1 = width - radius > inner0 ? width - radius : inner0;
	static thread_local std::vector<float> rowsBuffer;
	float* rows = reuse(rowsBuffer, (size_t)width * height);
#pragma omp parallel for num_threads(threads) schedule(static)
	for (int y = 0; y < height; y++)
	{
		const float* line = in + (size_t)y * width;
		float* acc = &rows[(size_t)y * width];
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
		// The inner run one tap at a time: each pixel still adds its taps from -radius up.
		for (int x = inner0; x < inner1; x++)
			acc[x] = 0.0f;
		for (int k = -radius; k <= radius; k++)
		{
			const float t = tap[k];
			for (int x = inner0; x < inner1; x++)
				acc[x] += t * line[x + k];
		}
	}
	// Columns a whole row at a time: each pixel still adds its taps from -radius up, so the sums are the same.
#pragma omp parallel for num_threads(threads) schedule(static)
	for (int y = 0; y < height; y++)
	{
		float* acc = out + (size_t)y * width;
		for (int x = 0; x < width; x++)
			acc[x] = 0.0f;
		for (int k = -radius; k <= radius; k++)
		{
			int sy = y + k;
			sy = sy < 0 ? 0 : (sy >= height ? height - 1 : sy);
			const float* line = &rows[(size_t)sy * width];
			const float t = tap[k];
			for (int x = 0; x < width; x++)
				acc[x] += t * line[x];
		}
	}
}
}  // namespace SwarmGrain

#endif  // SWARM_GRAIN_H
