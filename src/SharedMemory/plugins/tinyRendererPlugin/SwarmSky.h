#ifndef SWARM_SKY_H
#define SWARM_SKY_H

#include <vector>

// A daylight sky computed once from the sun and read per pixel. The Preetham model is evaluated
// into six cube faces of display bytes, with the sun disc and, when seeded, a cloud layer; the map
// also yields the colour the sky lends the ambient term. Every value comes from plain double
// arithmetic, so the bytes are the same on every machine that runs the binary.
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
};

#endif  // SWARM_SKY_H
