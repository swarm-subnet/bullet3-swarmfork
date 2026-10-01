#ifndef SWARM_LOW_LIGHT_H
#define SWARM_LOW_LIGHT_H

// ER_SWARM_LOW_LIGHT: a colour camera with too little light, as a drone's camera is at night. The finished frame is
// white balanced to its light, exposed to mid grey by a gain the camera caps, and given the grain of the light each
// pixel actually collected: photon noise that grows as the light falls, read noise, a coarser colour noise, and
// colour that fades as the noise swamps it. ER_SWARM_NEAR_INFRARED writes the frame in grey.
namespace SwarmLowLight
{
struct Settings
{
	bool m_lowLight;           // the whole camera chain; off, the frame is only turned to grey
	bool m_grey;               // near infrared: one grey channel, no white balance, no colour noise
	float m_photons;           // photo-electrons a pixel collects in one exposure for linear light 1.0; 0 is noiseless
	float m_readNoise;         // read noise of one pixel, in electrons
	float m_gainCap;           // the largest gain the auto exposure may apply
	unsigned int m_seed;       // draws the frame's grain
	float m_whiteBalance[3];   // per-channel gains that turn the frame's light to white
};

// The camera chain over a width x height frame of sRGB bytes (3 a pixel), in place. Rows are dealt to threads, and
// every output byte is a function of the input and the settings alone.
void develop(unsigned char* rgb, int width, int height, const Settings& settings, int threads);
}  // namespace SwarmLowLight

#endif  // SWARM_LOW_LIGHT_H
