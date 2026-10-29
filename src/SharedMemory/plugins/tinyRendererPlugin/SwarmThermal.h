#ifndef SWARM_THERMAL_H
#define SWARM_THERMAL_H

// ER_SWARM_THERMAL: a long-wave camera like the M4TD's uncooled 640 x 512 core. Surfaces are shaded into in-band
// radiance (8 to 14 um), and the frame then goes through the camera: lens blur and glow, sensor grain and column
// stripes, a per-frame stretch from its coldest point (black) to its hottest (white), and detail enhancement.
namespace SwarmThermal
{
// In-band radiance of a black body at `celsius`, in units of the radiance at 0 C, from a table built at load time.
float radiance(float celsius);

// The clear sky as the camera sees it: along a direction whose sine of elevation is `up` its emissivity is
// 1 - (1 - e)^(1 / up), e the zenith's, so it radiates as the air near the horizon and as the zenith straight up.
struct Sky
{
	float m_air;       // radiance of the air
	float m_logClear;  // natural log of 1 - e, the zenith's share of the air's radiance it lacks
	float m_hemisphere;  // cosine-weighted mean over the upper hemisphere, what a flat diffuse surface reflects
};

// The sky for an air temperature and the apparent temperature straight up, both in degrees Celsius.
Sky sky(float airC, float zenithC);

// Radiance of the sky along a direction whose sine of elevation is `up`; the air's at the horizon and below it.
float skyRadiance(const Sky& sky, float up);

// The camera chain from a width x height frame of in-band radiance to 8-bit White Hot written as R = G = B into
// `rgb` (3 bytes a pixel, rows as given). `seed` draws the frame's grain; the stripes belong to the camera and
// never change. Rows are dealt to threads, and every output byte is a function of the input alone.
void develop(const float* radiance, int width, int height, unsigned int seed, int threads, unsigned char* rgb);
}  // namespace SwarmThermal

#endif  // SWARM_THERMAL_H
