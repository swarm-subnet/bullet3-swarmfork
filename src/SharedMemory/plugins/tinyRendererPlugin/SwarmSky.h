#ifndef SWARM_SKY_H
#define SWARM_SKY_H

#include <vector>

// A daylight sky computed once from the sun and read per pixel. The Preetham model is evaluated
// into six cube faces of display bytes, with the sun disc and, when seeded, a cloud layer; the map
// also yields the colour the sky lends the ambient term. Every value comes from plain double
// arithmetic, so the bytes are the same on every machine that runs the binary.
// Under ER_SWARM_DAYLIGHT it also builds a linear sky: float radiance faces, a sky-light table, and display bytes through the film curve.
class SwarmSky
{
public:
	SwarmSky();

	// Rebuilds the map only when the sun direction, sun colour, cloud seed or up axis changed.
	// sunDir points towards the sun and need not be unit; sunColor is 0..1 per channel.
	void prepare(const float sunDir[3], const float sunColor[3], bool clouds, unsigned cloudSeed, int upAxis);

	// The sky bytes along a world direction, which need not be unit.
	void lookup(float x, float y, float z, unsigned char out[3]) const;

	// Tint for the ambient term: the average colour of the sky above the horizon, scaled so its
	// brightest channel is 1 and blended half way to white for the light the ground bounces back.
	const float* ambientColor() const { return m_ambient; }

	// The average sky colour at the horizon and straight up, 0..1, for the two-colour readers.
	const float* horizonColor() const { return m_horizon; }
	const float* zenithColor() const { return m_zenith; }

	// A sky photograph: an equirectangular RGB byte image, its top row the zenith, its columns running the full turn; yaw turns it about the up axis, in degrees.
	struct Photo
	{
		const unsigned char* m_rgb;
		int m_width;
		int m_height;
		float m_yaw;
	};

	// Builds the linear sky from the photo, or from the sun with a bright disc; a new exposure alone redoes the display bytes.
	void prepareDaylight(const float sunDir[3], const float sunColor[3], bool clouds, unsigned cloudSeed, int upAxis,
						 const Photo* photo, float exposure);

	// Linear radiance along a world direction, blended from the four nearest texels.
	void radiance(float x, float y, float z, float out[3]) const;

	// Light a surface facing the unit normal receives from the whole sky; a uniform sky of radiance L gives L.
	void irradiance(const float normal[3], float out[3]) const;

	// The display bytes of the linear sky along a world direction, at the exposure it was built with.
	void lookupDisplay(float x, float y, float z, unsigned char out[3]) const;

	bool daylightBuilt() const { return m_daylightBuilt; }

	// Drops a linear sky built from a photograph, for when the textures it points into are freed.
	void forgetPhoto();

private:
	std::vector<unsigned char> m_texels;
	float m_ambient[3];
	float m_horizon[3];
	float m_zenith[3];
	float m_sunDir[3];
	float m_sunColor[3];
	bool m_clouds;
	unsigned m_cloudSeed;
	int m_upAxis;
	bool m_built;

	std::vector<float> m_radiance;
	std::vector<unsigned char> m_display;
	float m_sh[9][3];
	float m_daylightSunDir[3];
	float m_daylightSunColor[3];
	bool m_daylightClouds;
	unsigned m_daylightCloudSeed;
	int m_daylightUpAxis;
	const unsigned char* m_photoRgb;
	int m_photoWidth;
	int m_photoHeight;
	float m_photoYaw;
	float m_exposure;
	bool m_daylightBuilt;

	void buildDisplay();
};

#endif  // SWARM_SKY_H
