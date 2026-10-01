#include "SwarmLowLight.h"

#include <math.h>
#include <vector>

#include "../../../TinyRenderer/SwarmGamma.h"
#include "SwarmGrain.h"

namespace
{
// Auto exposure brings the metered luminance to mid grey, the meter weighting the middle third of the frame as much as
// the whole, as a centre-weighted meter does, so a lamp's spot in the middle is not burnt out by the dark around it.
const float kMidGrey = 0.18f;
const double kCentreWeight = 0.5;
// The grain after demosaicing and scaling to the output frame: a fine luma grain, and a weaker colour grain.
const float kLumaGrainSigmaPx = 0.6f;
const float kChromaGrain = 0.4f;
// The camera's noise reduction, tuned against M4T Night Scene live view over unlit farmland: colour is always
// smoothed into soft blotches, and the luma keeps its own detail only where the signal outweighs the noise, half of it
// at this signal-to-noise ratio, the rest smoothed, so a dim scene turns soft rather than speckled.
const float kLumaSmoothSigmaPx = 2.0f;
const float kChromaSmoothSigmaPx = 4.0f;
const float kDetailSnr = 5.0f;
// Colour fades as the noise swamps it: half the saturation is left at this signal-to-noise ratio.
const float kColourFadeSnr = 3.0f;
const unsigned int kChromaSaltB = 0x3c6ef372u;
const unsigned int kChromaSaltR = 0xa54ff53au;

// Rec. 709 luminance of linear light.
inline float luma(const float* c)
{
	return (0.2126f * c[0] + 0.7152f * c[1]) + 0.0722f * c[2];
}

// A field of grain blurred to `sigma` pixels into `field`: white Gaussian draws and the blur. The return is the factor
// that puts back the blur's loss of variance, which for a separable kernel is the sum of its squared taps once per axis.
float grainField(unsigned int seed, int width, int height, float sigma, int threads, float* field)
{
	const size_t numPixels = (size_t)width * height;
	static thread_local std::vector<float> whiteBuffer;
	float* white = SwarmGrain::reuse(whiteBuffer, numPixels);
#pragma omp parallel for num_threads(threads) schedule(static)
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
		{
			const size_t i = (size_t)y * width + x;
			white[i] = SwarmGrain::gaussian(seed, (unsigned int)i);
		}
	const std::vector<float> taps = SwarmGrain::gaussianKernel(sigma);
	SwarmGrain::blur(white, field, width, height, taps, threads);
	float squares = 0.0f;
	for (size_t k = 0; k < taps.size(); k++)
		squares += taps[k] * taps[k];
	return 1.0f / squares;
}
}  // namespace

void SwarmLowLight::develop(unsigned char* rgb, int width, int height, const Settings& settings, int threads)
{
	const int numPixels = width * height;
	if (numPixels <= 0)
		return;

	// Linear light from the bytes, white balanced; in near infrared one grey channel. Every buffer below is kept for the
	// next frame and written in full before it is read.
	static thread_local std::vector<float> lightBuffer, fineBuffer, yBuffer, sigmaBuffer, detailBuffer, keepBuffer, ySmoothBuffer;
	static thread_local std::vector<float> cbBuffer, crBuffer, cbSmoothBuffer, crSmoothBuffer;
	float* light = SwarmGrain::reuse(lightBuffer, (size_t)numPixels * 3);
#pragma omp parallel for num_threads(threads) schedule(static)
	for (int i = 0; i < numPixels; i++)
	{
		float* c = &light[(size_t)i * 3];
		for (int k = 0; k < 3; k++)
			c[k] = kSwarmSrgbToLinear[rgb[(size_t)i * 3 + k]] * (settings.m_lowLight && !settings.m_grey ? settings.m_whiteBalance[k] : 1.0f);
		if (settings.m_grey)
			c[0] = c[1] = c[2] = luma(c);
	}

	if (!settings.m_lowLight)
	{
#pragma omp parallel for num_threads(threads) schedule(static)
		for (int i = 0; i < numPixels * 3; i++)
			rgb[(size_t)i] = swarmLinearToSrgb(light[(size_t)i]);
		return;
	}

	// The meter reads the luminance before any grain, summed in pixel order: the whole frame and its middle third.
	double total = 0.0, centre = 0.0;
	long long centrePixels = 0;
	for (int row = 0; row < height; row++)
		for (int col = 0; col < width; col++)
		{
			const double value = luma(&light[((size_t)row * width + col) * 3]);
			total += value;
			if (3 * row >= height && 3 * row < 2 * height && 3 * col >= width && 3 * col < 2 * width)
			{
				centre += value;
				centrePixels++;
			}
		}
	const double whole = total / numPixels;
	const float mean = (float)(centrePixels > 0 ? (1.0 - kCentreWeight) * whole + kCentreWeight * centre / (double)centrePixels : whole);
	float gain = settings.m_gainCap;
	if (mean > 0.0f && kMidGrey / mean < gain)
		gain = kMidGrey / mean;

	if (!(settings.m_photons > 0.0f))
	{
#pragma omp parallel for num_threads(threads) schedule(static)
		for (int i = 0; i < numPixels * 3; i++)
			rgb[(size_t)i] = swarmLinearToSrgb(light[(size_t)i] * gain);
		return;
	}

	// The luma with its grain: electrons collected and the noise on them, the photons' own spread plus the read noise,
	// in linear light. Beside it, how much luma detail and colour the camera keeps by the signal-to-noise ratio.
	const bool colour = !settings.m_grey;
	float* fine = SwarmGrain::reuse(fineBuffer, (size_t)numPixels);
	const float restore = grainField(settings.m_seed, width, height, kLumaGrainSigmaPx, threads, fine);
	const float photons = settings.m_photons;
	const float readVariance = settings.m_readNoise * settings.m_readNoise;
	const float detailSnr2 = kDetailSnr * kDetailSnr, fadeSnr2 = kColourFadeSnr * kColourFadeSnr;
	float* y = SwarmGrain::reuse(yBuffer, (size_t)numPixels);
	float* sigma = SwarmGrain::reuse(sigmaBuffer, (size_t)numPixels);
	float* detail = SwarmGrain::reuse(detailBuffer, (size_t)numPixels);
	float* keep = colour ? SwarmGrain::reuse(keepBuffer, (size_t)numPixels) : 0;
#pragma omp parallel for num_threads(threads) schedule(static)
	for (int i = 0; i < numPixels; i++)
	{
		const float clean = luma(&light[(size_t)i * 3]);
		const float electrons = clean > 0.0f ? clean * photons : 0.0f;
		const float spread = sqrtf(electrons + readVariance);
		const float snr2 = (electrons * electrons) / (spread * spread);
		sigma[(size_t)i] = spread / photons;
		detail[(size_t)i] = snr2 / (snr2 + detailSnr2);
		y[(size_t)i] = clean + sigma[(size_t)i] * (fine[(size_t)i] * restore);
		if (colour)
			keep[(size_t)i] = snr2 / (snr2 + fadeSnr2);
	}
	float* ySmooth = SwarmGrain::reuse(ySmoothBuffer, (size_t)numPixels);
	SwarmGrain::blur(y, ySmooth, width, height, SwarmGrain::gaussianKernel(kLumaSmoothSigmaPx), threads);

	// Colour at half resolution, as the camera carries it: each 2 x 2 block's Rec. 709 blue and red differences with the
	// grain of their mean, smoothed there by half the full-size radius, then read back bilinearly.
	const int hw = (width + 1) / 2, hh = (height + 1) / 2;
	float *cb = 0, *cr = 0, *cbSmooth = 0, *crSmooth = 0;
	if (colour)
	{
		cb = SwarmGrain::reuse(cbBuffer, (size_t)hw * hh);
		cr = SwarmGrain::reuse(crBuffer, (size_t)hw * hh);
		const unsigned int blueSeed = SwarmGrain::mix32(settings.m_seed) ^ kChromaSaltB;
		const unsigned int redSeed = SwarmGrain::mix32(settings.m_seed) ^ kChromaSaltR;
#pragma omp parallel for num_threads(threads) schedule(static)
		for (int by = 0; by < hh; by++)
			for (int bx = 0; bx < hw; bx++)
			{
				float b = 0.0f, r = 0.0f, s = 0.0f;
				int count = 0;
				for (int dy = 0; dy < 2 && 2 * by + dy < height; dy++)
					for (int dx = 0; dx < 2 && 2 * bx + dx < width; dx++)
					{
						const size_t i = (size_t)(2 * by + dy) * width + 2 * bx + dx;
						const float* c = &light[i * 3];
						const float clean = luma(c);
						b += (c[2] - clean) / 1.8556f;
						r += (c[0] - clean) / 1.5748f;
						s += sigma[i];
						count++;
					}
				const size_t j = (size_t)by * hw + bx;
				const float n = (float)count;
				// The mean of four pixels carries half their grain.
				const float grain = kChromaGrain * (s / n) * 0.5f;
				cb[j] = b / n + grain * SwarmGrain::gaussian(blueSeed, (unsigned int)j);
				cr[j] = r / n + grain * SwarmGrain::gaussian(redSeed, (unsigned int)j);
			}
		const std::vector<float> taps = SwarmGrain::gaussianKernel(0.5f * kChromaSmoothSigmaPx);
		cbSmooth = SwarmGrain::reuse(cbSmoothBuffer, (size_t)hw * hh);
		crSmooth = SwarmGrain::reuse(crSmoothBuffer, (size_t)hw * hh);
		SwarmGrain::blur(cb, cbSmooth, hw, hh, taps, threads);
		SwarmGrain::blur(cr, crSmooth, hw, hh, taps, threads);
	}

	// The noise reduction: smoothed luma takes back the detail the signal can carry; colour is faded by the same ratio.
#pragma omp parallel for num_threads(threads) schedule(static)
	for (int row = 0; row < height; row++)
	{
		float fy = ((float)row + 0.5f) * 0.5f - 0.5f;
		fy = fy < 0.0f ? 0.0f : fy;
		const int y0 = (int)fy < hh - 1 ? (int)fy : hh - 1;
		const int y1 = y0 + 1 < hh ? y0 + 1 : y0;
		const float ay = fy - (float)y0 < 1.0f ? fy - (float)y0 : 1.0f;
		for (int col = 0; col < width; col++)
		{
			const size_t i = (size_t)row * width + col;
			const float luminance = ySmooth[i] + (y[i] - ySmooth[i]) * detail[i];
			float out[3] = {luminance, luminance, luminance};
			if (colour)
			{
				float fx = ((float)col + 0.5f) * 0.5f - 0.5f;
				fx = fx < 0.0f ? 0.0f : fx;
				const int x0 = (int)fx < hw - 1 ? (int)fx : hw - 1;
				const int x1 = x0 + 1 < hw ? x0 + 1 : x0;
				const float ax = fx - (float)x0 < 1.0f ? fx - (float)x0 : 1.0f;
				const size_t a = (size_t)y0 * hw, c = (size_t)y1 * hw;
				const float bTop = cbSmooth[a + x0] + (cbSmooth[a + x1] - cbSmooth[a + x0]) * ax;
				const float bBottom = cbSmooth[c + x0] + (cbSmooth[c + x1] - cbSmooth[c + x0]) * ax;
				const float rTop = crSmooth[a + x0] + (crSmooth[a + x1] - crSmooth[a + x0]) * ax;
				const float rBottom = crSmooth[c + x0] + (crSmooth[c + x1] - crSmooth[c + x0]) * ax;
				const float b = (bTop + (bBottom - bTop) * ay) * keep[i];
				const float r = (rTop + (rBottom - rTop) * ay) * keep[i];
				out[0] = luminance + 1.5748f * r;
				out[1] = luminance - 0.1873f * b - 0.4681f * r;
				out[2] = luminance + 1.8556f * b;
			}
			for (int k = 0; k < 3; k++)
				rgb[i * 3 + k] = swarmLinearToSrgb(out[k] * gain);
		}
	}
}
