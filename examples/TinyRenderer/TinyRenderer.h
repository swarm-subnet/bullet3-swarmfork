#ifndef TINY_RENDERER_H
#define TINY_RENDERER_H

#include "geometry.h"
#include "model.h"
#include "Bullet3Common/b3AlignedObjectArray.h"
#include "Bullet3Common/b3Vector3.h"
#include "LinearMath/btAlignedObjectArray.h"
#include "LinearMath/btVector3.h"

#include "tgaimage.h"

// ER_SPECULAR_GLINT: a hit blends towards the sky colour of its mirrored view direction, weighted by the
// object's specular colour times a glass Fresnel curve, so a surface seen at a grazing angle reflects
// more sky than one seen straight on. The sky is the horizon-to-zenith gradient of the render, or
// white when the render has none.
struct TinyRenderGlint
{
	bool m_enabled;
	float m_skyHorizon[3];
	float m_skyZenith[3];
	int m_upAxis;

	TinyRenderGlint() : m_enabled(false), m_upAxis(2)
	{
		for (int i = 0; i < 3; i++)
			m_skyHorizon[i] = m_skyZenith[i] = 1.f;
	}

	// Blends lit (0..255 per channel) towards the sky seen along the mirror of toCamera about normal,
	// both unit length, for a surface with the given specular colour. Shared by both colour paths.
	void apply(const float normal[3], const float toCamera[3], const float specular[3], float lit[3]) const
	{
		float nDotV = (normal[0] * toCamera[0] + normal[1] * toCamera[1]) + normal[2] * toCamera[2];
		float side = 1.f;
		if (nDotV < 0.f)
		{
			side = -1.f;
			nDotV = -nDotV;
		}
		const float up = (normal[m_upAxis] * side) * (2.f * nDotV) - toCamera[m_upAxis];
		const float t = up < 0.f ? 0.f : up;
		const float f = 1.f - nDotV;
		const float f2 = f * f;
		const float fresnel = 0.04f + 0.96f * (f2 * f2 * f);
		for (int i = 0; i < 3; i++)
		{
			const float sky = 255.f * (m_skyHorizon[i] + (m_skyZenith[i] - m_skyHorizon[i]) * t);
			const float w = specular[i] * fresnel;
			lit[i] = lit[i] + (sky - lit[i]) * w;
		}
	}
};

struct TinyRenderObjectData
{
	//Camera
	TinyRender::Matrix m_viewMatrix;
	TinyRender::Matrix m_projectionMatrix;
	TinyRender::Matrix m_viewportMatrix;
	btVector3 m_localScaling;
	btVector3 m_lightDirWorld;
	btVector3 m_lightColor;
	float m_lightDistance;
	float m_lightAmbientCoeff;
	float m_lightDiffuseCoeff;
	float m_lightSpecularCoeff;

	//Model (vertices, indices, textures, shader)
	TinyRender::Matrix m_modelMatrix;
	TinyRender::Model* m_model;
	//class IShader* m_shader; todo(erwincoumans) expose the shader, for now we use a default shader

	//Output

	TGAImage& m_rgbColorBuffer;
	b3AlignedObjectArray<float>& m_depthBuffer;              //required, hence a reference
	b3AlignedObjectArray<float>* m_shadowBuffer;             //optional, hence a pointer
	b3AlignedObjectArray<int>* m_segmentationMaskBufferPtr;  //optional, hence a pointer

	TinyRenderObjectData(TGAImage& rgbColorBuffer, b3AlignedObjectArray<float>& depthBuffer);
	TinyRenderObjectData(TGAImage& rgbColorBuffer, b3AlignedObjectArray<float>& depthBuffer, b3AlignedObjectArray<int>* segmentationMaskBuffer, int objectIndex);
	TinyRenderObjectData(TGAImage& rgbColorBuffer, b3AlignedObjectArray<float>& depthBuffer, b3AlignedObjectArray<float>* shadowBuffer);
	TinyRenderObjectData(TGAImage& rgbColorBuffer, b3AlignedObjectArray<float>& depthBuffer, b3AlignedObjectArray<float>* shadowBuffer, b3AlignedObjectArray<int>* segmentationMaskBuffer, int objectIndex, int linkIndex);
	virtual ~TinyRenderObjectData();

	void loadModel(const char* fileName, struct CommonFileIOInterface* fileIO);
	void createCube(float HalfExtentsX, float HalfExtentsY, float HalfExtentsZ, struct CommonFileIOInterface* fileIO=0);
	void registerMeshShape(const float* vertices, int numVertices, const int* indices, int numIndices, const float rgbaColor[4],
						   unsigned char* textureImage = 0, int textureWidth = 0, int textureHeight = 0, const unsigned char* textureAlpha = 0);

	void registerMesh2(btAlignedObjectArray<btVector3>& vertices, btAlignedObjectArray<btVector3>& normals, btAlignedObjectArray<int>& indices, struct CommonFileIOInterface* fileIO);

	void* m_userData;
	int m_userIndex;
	int m_objectIndex;
	int m_linkIndex;
	bool m_doubleSided;
	bool m_textureFilter;  // ER_TEXTURE_FILTER: bilinear + mipmap sampling instead of nearest texel
	TinyRenderGlint m_glint;

	btVector3 m_localAABBMin;
	btVector3 m_localAABBMax;
	bool m_hasLocalAABB;

	void computeLocalAABB();
};

class TinyRenderer
{
public:
	static void renderObjectDepth(TinyRenderObjectData& renderData);
	static void renderObject(TinyRenderObjectData& renderData);
	static void renderObjectCameraDepthOnly(TinyRenderObjectData& renderData);
	// Thread-safe depth-only render: explicit matrices and target buffer,
	// no mutation of renderData, so cameras can render concurrently.
	static void renderObjectCameraDepthOnlyInto(const TinyRenderObjectData& renderData,
												const TinyRender::Matrix& viewMatrix,
												const TinyRender::Matrix& projMatrix,
												const TinyRender::Matrix& modelMatrix,
												const TinyRender::Vec3f& localScaling,
												float* zbufferPtr, int width, int height);
};

// Worker thread count for depth-only render/copy loops (SWARM_RENDER_THREADS, default 2).
int b3GetSwarmRenderThreads();

#endif  // TINY_RENDERER_Hbla
