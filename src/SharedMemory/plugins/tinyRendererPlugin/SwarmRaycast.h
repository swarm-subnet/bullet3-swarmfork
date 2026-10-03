#ifndef SWARM_RAYCAST_H
#define SWARM_RAYCAST_H

#include "../../../TinyRenderer/TinyRenderer.h"

struct TinyRenderObjectData;
class btTransform;
class btVector3;
class SwarmSky;

// Light and output for a colour render on the ray-cast path: the same terms TinyRenderer's shader
// takes, so a frame lit through either backend uses one light model.
struct SwarmRaycastShading
{
	float m_lightDir[3];  // unit vector towards the light, world space
	float m_lightColor[3];
	float m_ambientCoeff;
	float m_ambientColor[3];  // tint on the ambient term, white unless a sky lends its colour
	float m_diffuseCoeff;
	float m_specularCoeff;
	// Share of the diffuse and specular light a shadowed hit keeps: 0.8 is the floor TinyRenderer's
	// shader applies where its shadow buffer says blocked, 0 is a full shadow lit by ambient alone.
	float m_shadowLightCoeff;
	// One occlusion ray towards the light per hit. Any drawn surface stops it, whichever way it is
	// wound; a body hidden by a zero alpha lets the light through and so casts no shadow.
	bool m_shadow;
	// With m_shadow: the light's view of the static bodies is cast once into a depth grid and each hit
	// looks itself up there instead of casting a ray. The grid is rebuilt when the light direction
	// changes; a body that moves or is hidden has only its own cells recast. Movers cast no shadow
	// from the map itself.
	bool m_shadowMap;
	// Shadow rays and the map pass through double-sided cut-out surfaces, the leaf and grass cards.
	bool m_leafNoShadow;
	// With m_shadowMap: a hit the map calls lit also casts one occlusion ray against the small tree of
	// the bodies that moved since the world was built, so movers cast shadows that follow them.
	bool m_moverShadow;
	bool m_textureFilter;  // bilinear and mipmapped texture reads instead of the nearest texel
	// After the one ray per pixel, a pixel whose object id or 1/depth breaks with a neighbour is
	// re-composited from the exact share of the pixel each nearby triangle covers, with one probe ray
	// for whatever share is left. Depth and segmentation keep the first ray; every other pixel keeps
	// its bytes.
	bool m_edgeAntialias;
	// ER_SWARM_EDGE_OUTLINE: with m_edgeAntialias, a pixel inside one body is an edge only where the depth jumps.
	bool m_edgeOutline;
	// ER_SWARM_CREASE_FILL: with m_edgeAntialias, a crease fills its uncovered share from its neighbours, not a probe ray.
	bool m_creaseFill;
	// ER_SWARM_RASTER: a lone camera paints the triangles in view, keeping the nearest per pixel, and casts a ray only
	// where a forest tree or a mesh too dense to paint may lie nearer than what was painted there.
	bool m_raster;
	// ER_SWARM_EDGE_BEHIND: with m_edgeAntialias, the uncovered share is shaded on the nearest of the pixel's and its
	// neighbours' triangles under the probe point, and searched only where none of them lies there.
	bool m_edgeBehind;
	// The lighting, the glint and the edge blend run on linear light decoded from the bytes through
	// one fixed table, and the result is encoded back on the write; off, the arithmetic runs on the
	// encoded bytes as TinyRenderer's shader does.
	bool m_linearLight;
	// ER_SWARM_DAYLIGHT: the daylight terms below replace TinyRenderer's formula; m_sky gives radiance, sky light and haze colour, or flat white without one.
	bool m_daylight;
	const SwarmSky* m_sky;
	float m_exposure;
	float m_hazeDistance;      // metres at which a hit is 63 % haze; 0 turns the haze off
	float m_shadowCoreRadius;  // half side of the fine shadow grid about the world origin; 0 for one grid
	TinyRenderGlint m_glint;
	// ER_SWARM_THERMAL: every pixel is in-band radiance from the surface temperatures, the camera chain makes the
	// 8-bit White Hot frame, and the colour terms above are not read. The sky is m_airTemperature at the horizon and
	// m_skyTemperature straight up; m_thermalSeed draws the frame's sensor grain.
	bool m_thermal;
	float m_airTemperature;
	float m_skyTemperature;
	unsigned int m_thermalSeed;
	// A spot light at m_spotPosition along the unit m_spotDirection, under linear light off the daylight path: half its
	// intensity at the edge of its cone, none past m_spotCosOuter, falling with the square of the distance and ending
	// smoothly at m_spotRange. It casts no shadow, since a lamp beside the lens lights what the lens sees.
	bool m_spot;
	float m_spotPosition[3];
	float m_spotDirection[3];
	float m_spotCosInner;
	float m_spotCosOuter;
	float m_spotRange;
	float m_spotIntensity;
	// ER_SWARM_NEAR_INFRARED: surfaces reflect by their near-infrared albedo, the lights are grey, the frame is grey.
	bool m_nearInfrared;
	// ER_SWARM_LOW_LIGHT: the camera chain on the finished frame (SwarmLowLight), with the white balance of the light.
	bool m_lowLight;
	float m_sensorPhotons;
	float m_sensorReadNoise;
	float m_sensorGainCap;
	unsigned int m_sensorSeed;
	float m_whiteBalance[3];
};

// Ray-cast backend beside TinyRenderer. Every render object is an instance of a shared mesh
// tree inside one top-level tree, so a moved body costs one transform update and the map is never
// rebuilt. Depth lands in TinyRenderer's clip-z convention so the same copy-out serves both paths.
class SwarmRaycast
{
public:
	SwarmRaycast();
	~SwarmRaycast();

	// Creates or refreshes the instance for renderObj from its model, world transform and scaling.
	void syncObject(TinyRenderObjectData* renderObj, const btTransform& worldTransform, const btVector3& localScaling);
	// Records renderObj's rewritten vertices for a refit at the next commit; true when other render objects share the tree.
	bool meshChanged(TinyRenderObjectData* renderObj);
	void removeObject(TinyRenderObjectData* renderObj);
	void removeAll();
	// Rebuilds the top-level tree after a batch of sync calls; the mover tree only when moverShadows asks for it.
	void commit(bool moverShadows);

	// What a colour pixel shows where no drawn surface is shaded, asked for that pixel alone; row in output order.
	struct Background
	{
		virtual void pixel(int row, int col, unsigned char out[3]) const = 0;
	};

	// One camera of a render: its view matrix and the buffers it writes. m_seg may be null, and m_rgb
	// may be null only on a render with no shading; it is width * height * 3 bytes, rows in output
	// order, and only hit pixels are written.
	struct Target
	{
		const float* m_view;
		float* m_depth;
		int* m_seg;
		unsigned char* m_rgb;
		// When given, fills every colour pixel no surface shades; without it such a pixel keeps its bytes.
		const Background* m_background;
	};

	// One ray per pixel at the pixel corner, like TinyRenderer, rows already in output order. A hit
	// writes -z_clip into m_depth, objectIndex + ((linkIndex + 1) << 24) into m_seg when given,
	// and the shaded colour into m_rgb when shading is given.
	// Every camera shares projMat, the frame size and the light. The pixels of all cameras are cut
	// into fixed tiles before the frame starts and each thread takes the next untraced tile, so the
	// bytes never depend on the thread count or on which thread traced which tile. With edge anti-aliasing
	// the same tiles are walked a second time once every first ray has landed.
	// With alphaCutout a hit on a texel whose texture alpha is below the cut-out threshold is not a hit:
	// the ray, and a shadow ray, carry on behind it, so colour, depth and shadow share the same holes.
	// With reuseKey, a lone camera asked again for an unchanged frame gets it back untraced; its camera chain still runs.
	void render(const Target* targets, int numTargets, const float projMat[16], int width, int height,
				const SwarmRaycastShading* shading, int threads, bool alphaCutout = false,
				const void* reuseKey = 0, size_t reuseKeyBytes = 0) const;

private:
	struct Data;
	Data* m_data;
};

#endif  // SWARM_RAYCAST_H
