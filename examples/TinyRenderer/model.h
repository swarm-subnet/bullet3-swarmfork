#ifndef __MODEL_H__
#define __MODEL_H__
#include <vector>
#include <string>
#include "geometry.h"
#include "tgaimage.h"

namespace TinyRender
{
struct SharedMesh;
struct SharedTexture;

class Model
{
private:
	// Mesh arrays and the diffuse texture are reference counted and shared
	// between every Model built from identical data; colour stays per Model.
	SharedMesh* m_mesh;
	SharedTexture* m_diffuse;
	TGAImage normalmap_;
	TGAImage specularmap_;
	Vec4f m_colorRGBA;

	void load_texture(std::string filename, const char* suffix, TGAImage& img);
	void detachMesh();

public:
	Model(const char* filename);
	Model();
	void setColorRGBA(const float rgba[4])
	{
		for (int i = 0; i < 4; i++)
			m_colorRGBA[i] = rgba[i];
	}

	const Vec4f& getColorRGBA() const
	{
		return m_colorRGBA;
	}
	void loadDiffuseTexture(const char* relativeFileName);
	void setDiffuseTextureFromData(unsigned char* textureImage, int textureWidth, int textureHeight);
	// Vertex stride is 9 floats: xyz, w (ignored), normal xyz, uv.
	void setMeshFromArrays(const float* vertices, int numVertices, const int* indices, int numIndices);
	void reserveMemory(int numVertices, int numIndices);
	void addVertex(float x, float y, float z, float normalX, float normalY, float normalZ, float u, float v);
	void addTriangle(int vertexposIndex0, int normalIndex0, int uvIndex0,
					 int vertexposIndex1, int normalIndex1, int uvIndex1,
					 int vertexposIndex2, int normalIndex2, int uvIndex2);
	bool getLocalAABB(Vec3f& aabbMin, Vec3f& aabbMax);

	~Model();
	int nverts();
	int nnormals();
	int nfaces();

	Vec3f normal(int iface, int nthvert);
	// The stored corner normal as loaded, read without writing back to the shared array.
	Vec3f storedNormal(int iface, int nthvert) const;
	Vec3f normal(Vec2f uv);
	Vec3f vert(int i);
	Vec3f vert(int iface, int nthvert);
	Vec3f* readWriteVertices();
	Vec3f* readWriteNormals();
	// Identity of the shared vertex block; two Models with the same key draw the same triangles.
	const void* meshKey() const { return m_mesh; }

	Vec2f uv(int iface, int nthvert);
	TGAColor diffuse(Vec2f uv);
	TGAColor diffuseFiltered(Vec2f uv, Vec2f duvdx, Vec2f duvdy);
	// Builds the mip chain now, so later diffuseFiltered calls only read; safe to call from many threads after this.
	void buildMipmaps();
	float specular(Vec2f uv);
	std::vector<int> face(int idx);
};
}

#endif  //__MODEL_H__
