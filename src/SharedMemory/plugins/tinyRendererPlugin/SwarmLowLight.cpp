#include "SwarmLowLight.h"

#include <math.h>
#include <vector>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

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

// The auto exposure's gain: the meter reads the luminance before any grain, summed in pixel order, the whole frame and
// its middle third.
float exposureGain(const float* lum, int width, int height, float gainCap)
{
	double total = 0.0, centre = 0.0;
	long long centrePixels = 0;
	for (int row = 0; row < height; row++)
		for (int col = 0; col < width; col++)
		{
			const double value = lum[(size_t)row * width + col];
			total += value;
			if (3 * row >= height && 3 * row < 2 * height && 3 * col >= width && 3 * col < 2 * width)
			{
				centre += value;
				centrePixels++;
			}
		}
	const double whole = total / (width * height);
	const float mean = (float)(centrePixels > 0 ? (1.0 - kCentreWeight) * whole + kCentreWeight * centre / (double)centrePixels : whole);
	float gain = gainCap;
	if (mean > 0.0f && kMidGrey / mean < gain)
		gain = kMidGrey / mean;
	return gain;
}

// One row of linear light, already scaled by the gain, encoded to sRGB bytes three to a pixel.
void encodeRow(const float* red, const float* green, const float* blue, int width, unsigned char* rgb)
{
	for (int x = 0; x < width; x++)
	{
		rgb[(size_t)x * 3] = swarmLinearToSrgb(red[x]);
		rgb[(size_t)x * 3 + 1] = swarmLinearToSrgb(green[x]);
		rgb[(size_t)x * 3 + 2] = swarmLinearToSrgb(blue[x]);
	}
}

// A row of linear light's luma and its Rec. 709 blue and red differences.
void lumaRow(const float* __restrict c0, const float* __restrict c1, const float* __restrict c2, int width, float* __restrict lum,
			 float* __restrict blue, float* __restrict red)
{
	for (int col = 0; col < width; col++)
	{
		const float c[3] = {c0[col], c1[col], c2[col]};
		const float clean = luma(c);
		lum[col] = clean;
		blue[col] = (c[2] - clean) / 1.8556f;
		red[col] = (c[0] - clean) / 1.5748f;
	}
}

// A row of linear light turned to one grey channel, and the luma of that grey.
void greyRow(const float* __restrict c0, const float* __restrict c1, const float* __restrict c2, int width, float* __restrict lum)
{
	for (int col = 0; col < width; col++)
	{
		const float c[3] = {c0[col], c1[col], c2[col]};
		const float g = luma(c);
		const float grey[3] = {g, g, g};
		lum[col] = luma(grey);
	}
}

// A row's linear colour from its luminance and its two colour rows above and below, faded by `keep` and times the gain.
void colourRow(const float* __restrict luminance, const float* __restrict bTop, const float* __restrict bBottom, const float* __restrict rTop,
			   const float* __restrict rBottom, const float* __restrict keep, float ay, float gain, int width, float* __restrict outR,
			   float* __restrict outG, float* __restrict outB)
{
	for (int col = 0; col < width; col++)
	{
		const float b = (bTop[col] + (bBottom[col] - bTop[col]) * ay) * keep[col];
		const float r = (rTop[col] + (rBottom[col] - rTop[col]) * ay) * keep[col];
		outR[col] = (luminance[col] + 1.5748f * r) * gain;
		outG[col] = (luminance[col] - 0.1873f * b - 0.4681f * r) * gain;
		outB[col] = (luminance[col] + 1.8556f * b) * gain;
	}
}

// The low-light chain with grain, in three passes over the frame, so the render threads meet only twice in between.
// Each pass computes every value from the passes before it with the same steps in the same order, so the bytes are
// the same at any thread count and whichever thread takes a row.
void developGrain(unsigned char* rgb, int width, int height, const SwarmLowLight::Settings& settings, int threads)
{
	const size_t numPixels = (size_t)width * height;
	const bool colour = !settings.m_grey;
	const int hw = (width + 1) / 2, hh = (height + 1) / 2;
	static thread_local std::vector<float> lumBuffer, sigmaBuffer, detailBuffer, keepBuffer, blueBuffer, redBuffer;
	static thread_local std::vector<float> fineRowsBuffer, yBuffer, yRowsBuffer, cbRowsBuffer, crRowsBuffer, cbWideBuffer, crWideBuffer;
	// Each render thread's own scratch rows.
	static thread_local std::vector<float> lineBuffer, outBuffer;
	float* lum = SwarmGrain::reuse(lumBuffer, numPixels);
	float* sigma = SwarmGrain::reuse(sigmaBuffer, numPixels);
	float* detail = SwarmGrain::reuse(detailBuffer, numPixels);
	float* keep = colour ? SwarmGrain::reuse(keepBuffer, numPixels) : 0;
	float* blue = colour ? SwarmGrain::reuse(blueBuffer, numPixels) : 0;
	float* red = colour ? SwarmGrain::reuse(redBuffer, numPixels) : 0;
	float* fineRows = SwarmGrain::reuse(fineRowsBuffer, numPixels);
	float* y = SwarmGrain::reuse(yBuffer, numPixels);
	float* yRows = SwarmGrain::reuse(yRowsBuffer, numPixels);
	float* cbRows = colour ? SwarmGrain::reuse(cbRowsBuffer, (size_t)hw * hh) : 0;
	float* crRows = colour ? SwarmGrain::reuse(crRowsBuffer, (size_t)hw * hh) : 0;
	float* cbWide = colour ? SwarmGrain::reuse(cbWideBuffer, (size_t)width * hh) : 0;
	float* crWide = colour ? SwarmGrain::reuse(crWideBuffer, (size_t)width * hh) : 0;

	const std::vector<float> fineTaps = SwarmGrain::gaussianKernel(kLumaGrainSigmaPx);
	const std::vector<float> lumaTaps = SwarmGrain::gaussianKernel(kLumaSmoothSigmaPx);
	const std::vector<float> chromaTaps = SwarmGrain::gaussianKernel(0.5f * kChromaSmoothSigmaPx);
	const int fineRadius = (int)fineTaps.size() / 2, lumaRadius = (int)lumaTaps.size() / 2, chromaRadius = (int)chromaTaps.size() / 2;
	const float* fineTap = &fineTaps[(size_t)fineRadius];
	const float* lumaTap = &lumaTaps[(size_t)lumaRadius];
	const float* chromaTap = &chromaTaps[(size_t)chromaRadius];
	// The factor that puts back the variance the grain's blur loses.
	float squares = 0.0f;
	for (size_t k = 0; k < fineTaps.size(); k++)
		squares += fineTaps[k] * fineTaps[k];
	const float restore = 1.0f / squares;
	const float photons = settings.m_photons;
	const float readVariance = settings.m_readNoise * settings.m_readNoise;
	const float detailSnr2 = kDetailSnr * kDetailSnr, fadeSnr2 = kColourFadeSnr * kColourFadeSnr;
	const unsigned int blueSeed = SwarmGrain::mix32(settings.m_seed) ^ kChromaSaltB;
	const unsigned int redSeed = SwarmGrain::mix32(settings.m_seed) ^ kChromaSaltR;
#if defined(__GNUC__)
	const SwarmLanes8 zero = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#endif

	// Pass 1, a pair of rows at a time: linear light, white balanced, in near infrared one grey channel; its luma, the
	// noise on it and how much luma detail and colour the camera keeps by the signal-to-noise ratio; the fine grain
	// blurred along its rows. Then colour at half resolution, as the camera carries it: each 2 x 2 block's Rec. 709
	// blue and red differences with the grain of their mean, blurred along its row.
	float balanced[3][256];
	for (int k = 0; k < 3; k++)
		for (int b = 0; b < 256; b++)
			balanced[k][b] = kSwarmSrgbToLinear[b] * (colour ? settings.m_whiteBalance[k] : 1.0f);
	const size_t lineFloats = (size_t)3 * width;
#pragma omp parallel for num_threads(threads) schedule(static)
	for (int by = 0; by < hh; by++)
	{
		float* line = SwarmGrain::reuse(lineBuffer, lineFloats);
		for (int row = 2 * by; row < 2 * by + 2 && row < height; row++)
		{
			const size_t start = (size_t)row * width;
			float* __restrict c0 = line;
			float* __restrict c1 = line + width;
			float* __restrict c2 = line + 2 * width;
			for (int col = 0; col < width; col++)
			{
				const unsigned char* bytes = rgb + (start + col) * 3;
				c0[col] = balanced[0][bytes[0]];
				c1[col] = balanced[1][bytes[1]];
				c2[col] = balanced[2][bytes[2]];
			}
			if (colour)
				lumaRow(c0, c1, c2, width, lum + start, blue + start, red + start);
			else
				greyRow(c0, c1, c2, width, lum + start);
			// Electrons collected and the noise on them, the photons' own spread plus the read noise, in linear light.
			int col = 0;
#if defined(__AVX2__)
			for (; col + 8 <= width; col += 8)
			{
				SwarmLanes8 clean;
				memcpy(&clean, lum + start + col, sizeof(clean));
				const SwarmLanes8 product = clean * photons;
				const SwarmLanes8 electrons = clean > zero ? product : zero;
				const SwarmLanes8 spread = (SwarmLanes8)_mm256_sqrt_ps((__m256)(electrons + readVariance));
				const SwarmLanes8 snr2 = (electrons * electrons) / (spread * spread);
				const SwarmLanes8 noise = spread / photons, kept = snr2 / (snr2 + detailSnr2);
				memcpy(sigma + start + col, &noise, sizeof(noise));
				memcpy(detail + start + col, &kept, sizeof(kept));
				memcpy(line + col, &snr2, sizeof(snr2));
			}
#endif
			for (; col < width; col++)
			{
				const size_t i = start + col;
				const float clean = lum[i];
				const float electrons = clean > 0.0f ? clean * photons : 0.0f;
				const float spread = sqrtf(electrons + readVariance);
				const float snr2 = (electrons * electrons) / (spread * spread);
				sigma[i] = spread / photons;
				detail[i] = snr2 / (snr2 + detailSnr2);
				line[col] = snr2;
			}
			if (colour)
				for (int col = 0; col < width; col++)
					keep[start + col] = line[col] / (line[col] + fadeSnr2);
			int grainCol = 0;
#if defined(__GNUC__)
			for (; grainCol + 8 <= width; grainCol += 8)
			{
				const SwarmLanes8 grain = SwarmGrain::gaussian8(settings.m_seed, (unsigned int)(start + grainCol));
				memcpy(line + grainCol, &grain, sizeof(grain));
			}
#endif
			for (; grainCol < width; grainCol++)
				line[grainCol] = SwarmGrain::gaussian(settings.m_seed, (unsigned int)(start + grainCol));
			SwarmGrain::blurRow(line, fineRows + start, width, fineTap, fineRadius);
		}
		if (!colour)
			continue;
		float* cb = line;
		float* cr = line + hw;
		const int whole = 2 * by + 1 < height ? width / 2 : 0;
		int bx = 0;
#if defined(__GNUC__)
		// Eight whole blocks at a time, each lane adding its block's four pixels in the same order.
		const SwarmLaneMask8 evens = {0, 2, 4, 6, 8, 10, 12, 14}, odds = evens + 1;
		for (; bx + 8 <= whole; bx += 8)
		{
			const size_t i = (size_t)(2 * by) * width + 2 * bx, j = (size_t)by * hw + bx;
			SwarmLanes8 sum[3] = {zero, zero, zero};
			const float* planes[3] = {blue, red, sigma};
			for (int p = 0; p < 3; p++)
				for (size_t at : {i, i + width})
				{
					SwarmLanes8 first, second;
					memcpy(&first, planes[p] + at, sizeof(first));
					memcpy(&second, planes[p] + at + 8, sizeof(second));
					sum[p] += __builtin_shuffle(first, second, evens);
					sum[p] += __builtin_shuffle(first, second, odds);
				}
			const SwarmLanes8 grain = kChromaGrain * (sum[2] / 4.0f) * 0.5f;
			const SwarmLanes8 b = sum[0] / 4.0f + grain * SwarmGrain::gaussian8(blueSeed, (unsigned int)j);
			const SwarmLanes8 r = sum[1] / 4.0f + grain * SwarmGrain::gaussian8(redSeed, (unsigned int)j);
			memcpy(cb + bx, &b, sizeof(b));
			memcpy(cr + bx, &r, sizeof(r));
		}
#endif
		for (; bx < whole; bx++)
		{
			const size_t i = (size_t)(2 * by) * width + 2 * bx, j = (size_t)by * hw + bx;
			float b = 0.0f, r = 0.0f, s = 0.0f;
			for (size_t at : {i, i + 1, i + width, i + width + 1})
			{
				b += blue[at];
				r += red[at];
				s += sigma[at];
			}
			const float n = 4.0f;
			// The mean of four pixels carries half their grain.
			const float grain = kChromaGrain * (s / n) * 0.5f;
			cb[bx] = b / n + grain * SwarmGrain::gaussian(blueSeed, (unsigned int)j);
			cr[bx] = r / n + grain * SwarmGrain::gaussian(redSeed, (unsigned int)j);
		}
		// Blocks cut short by the frame's edge.
		for (int bx = whole; bx < hw; bx++)
		{
			float b = 0.0f, r = 0.0f, s = 0.0f;
			int count = 0;
			for (int dy = 0; dy < 2 && 2 * by + dy < height; dy++)
				for (int dx = 0; dx < 2 && 2 * bx + dx < width; dx++)
				{
					const size_t i = (size_t)(2 * by + dy) * width + 2 * bx + dx;
					b += blue[i];
					r += red[i];
					s += sigma[i];
					count++;
				}
			const size_t j = (size_t)by * hw + bx;
			const float n = (float)count;
			const float grain = kChromaGrain * (s / n) * 0.5f;
			cb[bx] = b / n + grain * SwarmGrain::gaussian(blueSeed, (unsigned int)j);
			cr[bx] = r / n + grain * SwarmGrain::gaussian(redSeed, (unsigned int)j);
		}
		SwarmGrain::blurRow(cb, cbRows + (size_t)by * hw, hw, chromaTap, chromaRadius);
		SwarmGrain::blurRow(cr, crRows + (size_t)by * hw, hw, chromaTap, chromaRadius);
	}

	// Each full-width column's two half-width neighbours and its weight between them, the same for every row.
	std::vector<int> upLeft(colour ? (size_t)width : 0), upRight(colour ? (size_t)width : 0);
	std::vector<float> upWeight(colour ? (size_t)width : 0);
	for (int col = 0; colour && col < width; col++)
	{
		float fx = ((float)col + 0.5f) * 0.5f - 0.5f;
		fx = fx < 0.0f ? 0.0f : fx;
		const int x0 = (int)fx < hw - 1 ? (int)fx : hw - 1;
		upLeft[(size_t)col] = x0;
		upRight[(size_t)col] = x0 + 1 < hw ? x0 + 1 : x0;
		upWeight[(size_t)col] = fx - (float)x0 < 1.0f ? fx - (float)x0 : 1.0f;
	}

	// Pass 2: the meter on one thread while the others start; the fine grain blurred down its columns and laid on the
	// luma, which is blurred along its rows; the colour blurred down its columns and read across bilinearly to full width.
	float gain = settings.m_gainCap;
#pragma omp parallel num_threads(threads)
	{
#pragma omp single nowait
		gain = exposureGain(lum, width, height, settings.m_gainCap);
		float* line = SwarmGrain::reuse(lineBuffer, lineFloats);
#pragma omp for schedule(dynamic, 8) nowait
		for (int row = 0; row < height; row++)
		{
			const size_t start = (size_t)row * width;
			SwarmGrain::blurColumn(fineRows, line, width, height, row, fineTap, fineRadius);
			for (int col = 0; col < width; col++)
				y[start + col] = lum[start + col] + sigma[start + col] * (line[col] * restore);
			SwarmGrain::blurRow(y + start, yRows + start, width, lumaTap, lumaRadius);
		}
		if (colour)
		{
			float* cbSmooth = line;
			float* crSmooth = line + hw;
#pragma omp for schedule(dynamic, 8) nowait
			for (int hy = 0; hy < hh; hy++)
			{
				SwarmGrain::blurColumn(cbRows, cbSmooth, hw, hh, hy, chromaTap, chromaRadius);
				SwarmGrain::blurColumn(crRows, crSmooth, hw, hh, hy, chromaTap, chromaRadius);
				float* cbOut = cbWide + (size_t)hy * width;
				float* crOut = crWide + (size_t)hy * width;
				for (int col = 0; col < width; col++)
				{
					const int x0 = upLeft[col], x1 = upRight[col];
					const float ax = upWeight[col];
					cbOut[col] = cbSmooth[x0] + (cbSmooth[x1] - cbSmooth[x0]) * ax;
					crOut[col] = crSmooth[x0] + (crSmooth[x1] - crSmooth[x0]) * ax;
				}
			}
		}
	}

	// Pass 3, the noise reduction: smoothed luma takes back the detail the signal can carry; colour is faded by the same
	// ratio, read down bilinearly; the exposure's gain, and the bytes.
#pragma omp parallel for num_threads(threads) schedule(static)
	for (int row = 0; row < height; row++)
	{
		float* ySmooth = SwarmGrain::reuse(lineBuffer, lineFloats);
		float* out = SwarmGrain::reuse(outBuffer, (size_t)width * 3);
		SwarmGrain::blurColumn(yRows, ySmooth, width, height, row, lumaTap, lumaRadius);
		float fy = ((float)row + 0.5f) * 0.5f - 0.5f;
		fy = fy < 0.0f ? 0.0f : fy;
		const int y0 = (int)fy < hh - 1 ? (int)fy : hh - 1;
		const int y1 = y0 + 1 < hh ? y0 + 1 : y0;
		const float ay = fy - (float)y0 < 1.0f ? fy - (float)y0 : 1.0f;
		const size_t start = (size_t)row * width;
		float* outR = out;
		float* outG = out + width;
		float* outB = out + 2 * width;
		for (int col = 0; col < width; col++)
			ySmooth[col] = ySmooth[col] + (y[start + col] - ySmooth[col]) * detail[start + col];
		const float* luminance = ySmooth;
		if (colour)
			colourRow(luminance, cbWide + (size_t)y0 * width, cbWide + (size_t)y1 * width, crWide + (size_t)y0 * width,
					  crWide + (size_t)y1 * width, keep + start, ay, gain, width, outR, outG, outB);
		else
			for (int col = 0; col < width; col++)
				outR[col] = outG[col] = outB[col] = luminance[col] * gain;
		encodeRow(outR, outG, outB, width, rgb + start * 3);
	}
}
}  // namespace

void SwarmLowLight::develop(unsigned char* rgb, int width, int height, const Settings& settings, int threads)
{
	const int numPixels = width * height;
	if (numPixels <= 0)
		return;
	if (settings.m_lowLight && settings.m_photons > 0.0f)
	{
		developGrain(rgb, width, height, settings, threads);
		return;
	}

	// Linear light from the bytes, white balanced; in near infrared one grey channel.
	static thread_local std::vector<float> lightBuffer, lumBuffer;
	float* light = SwarmGrain::reuse(lightBuffer, (size_t)numPixels * 3);
	float* lum = SwarmGrain::reuse(lumBuffer, (size_t)numPixels);
#pragma omp parallel for num_threads(threads) schedule(static)
	for (int i = 0; i < numPixels; i++)
	{
		float* c = &light[(size_t)i * 3];
		for (int k = 0; k < 3; k++)
			c[k] = kSwarmSrgbToLinear[rgb[(size_t)i * 3 + k]] * (settings.m_lowLight && !settings.m_grey ? settings.m_whiteBalance[k] : 1.0f);
		if (settings.m_grey)
			c[0] = c[1] = c[2] = luma(c);
		lum[i] = luma(c);
	}

	if (!settings.m_lowLight)
	{
#pragma omp parallel for num_threads(threads) schedule(static)
		for (int i = 0; i < numPixels * 3; i++)
			rgb[(size_t)i] = swarmLinearToSrgb(light[(size_t)i]);
		return;
	}

	// Without grain the frame is only exposed.
	const float gain = exposureGain(lum, width, height, settings.m_gainCap);
#pragma omp parallel for num_threads(threads) schedule(static)
	for (int i = 0; i < numPixels * 3; i++)
		rgb[(size_t)i] = swarmLinearToSrgb(light[(size_t)i] * gain);
}
