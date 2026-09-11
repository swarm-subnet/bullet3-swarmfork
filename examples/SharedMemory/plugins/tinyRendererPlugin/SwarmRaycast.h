#ifndef SWARM_RAYCAST_H
#define SWARM_RAYCAST_H

struct TinyRenderObjectData;
class btTransform;
class btVector3;

// Light and output for a colour render on the ray-cast path: the same terms TinyRenderer's shader
// takes, so a frame lit through either backend uses one light model.
struct SwarmRaycastShading
{
	float m_lightDir[3];  // unit vector towards the light, world space
	float m_lightColor[3];
	float m_ambientCoeff;
	float m_diffuseCoeff;
	float m_specularCoeff;
	// One occlusion ray towards the light per hit. Any drawn surface stops it, whichever way it is
	// wound; a body hidden by a zero alpha lets the light through and so casts no shadow.
	bool m_shadow;
	bool m_textureFilter;  // bilinear and mipmapped texture reads instead of the nearest texel
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
	// Records that renderObj's vertices were rewritten in place; its tree is refitted at the next commit.
	void meshChanged(TinyRenderObjectData* renderObj);
	void removeObject(TinyRenderObjectData* renderObj);
	void removeAll();
	// Rebuilds the top-level tree after a batch of sync calls.
	void commit();

	// One camera of a render: its view matrix and the buffers it writes. m_seg may be null, and m_rgb
	// may be null only on a render with no shading; it is width * height * 3 bytes, rows in output
	// order, and only hit pixels are written.
	struct Target
	{
		const float* m_view;
		float* m_depth;
		int* m_seg;
		unsigned char* m_rgb;
	};

	// One ray per pixel at the pixel corner, like TinyRenderer, rows already in output order. A hit
	// writes -z_clip into m_depth, objectIndex + ((linkIndex + 1) << 24) into m_seg when given,
	// and the shaded colour into m_rgb when shading is given.
	// Every camera shares projMat, the frame size and the light. The pixels of all cameras are cut
	// into fixed tiles before the frame starts and tile k always goes to thread k mod threads, so the
	// bytes never depend on the thread count or on the order threads finish.
	void render(const Target* targets, int numTargets, const float projMat[16], int width, int height,
				const SwarmRaycastShading* shading, int threads) const;

private:
	struct Data;
	Data* m_data;
};

#endif  // SWARM_RAYCAST_H
