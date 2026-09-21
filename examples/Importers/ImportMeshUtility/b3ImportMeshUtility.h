#ifndef B3_IMPORT_MESH_UTILITY_H
#define B3_IMPORT_MESH_UTILITY_H

#include <string>
#include "Bullet3Common/b3AlignedObjectArray.h"

enum b3ImportMeshDataFlags
{
	B3_IMPORT_MESH_HAS_RGBA_COLOR=1,
	B3_IMPORT_MESH_HAS_SPECULAR_COLOR=2,
};

//one material of a mesh: the index range it covers in m_gfxShape plus its colour and texture
struct b3ImportMeshMaterialGroup
{
	int m_indexStart;
	int m_indexCount;
	double m_rgbaColor[4];
	double m_specularColor[4];
	unsigned char* m_textureImage;  //in 3 component 8-bit RGB data, 0 when the material has no texture
	unsigned char* m_textureAlpha;  //one byte per texel, 0 when the texture file carries no alpha channel
	bool m_isCached;
	int m_textureWidth;
	int m_textureHeight;
	//the resolved texture file; identity of the texels, and set even when the renderer already holds them
	std::string m_textureName;
};

struct b3ImportMeshData
{
	struct GLInstanceGraphicsShape* m_gfxShape;

	unsigned char* m_textureImage1;  //in 3 component 8-bit RGB data
	unsigned char* m_textureAlpha;   //one byte per texel, 0 when the texture file carries no alpha channel
	bool m_isCached;
	int m_textureWidth;
	int m_textureHeight;
	double m_rgbaColor[4];
	double m_specularColor[4];
	int m_flags;
	//the resolved texture file; identity of the texels, and set even when the renderer already holds them
	std::string m_textureName;
	//filled only when loading with splitOnMaterial; m_textureImage1 stays 0 then
	b3AlignedObjectArray<b3ImportMeshMaterialGroup> m_materialGroups;

	b3ImportMeshData()
		:m_gfxShape(0),
		m_textureImage1(0),
		m_textureAlpha(0),
		m_isCached(false),
		m_textureWidth(0),
		m_textureHeight(0),
		m_flags(0)
	{
	}

};

class b3ImportMeshUtility
{
public:
	// With textureHandover a texture the renderer already holds comes back as its file name alone, with no
	// texels: the caller looks it up by that name instead of the loader reading and decoding the file again.
	static bool loadAndRegisterMeshFromFileInternal(const std::string& fileName, b3ImportMeshData& meshData, struct CommonFileIOInterface* fileIO, bool splitOnMaterial = false, bool textureHandover = false);
	// The alpha channel of an image file held in memory, one malloc'd byte per texel, or 0 when the
	// file has no alpha channel or every texel is opaque. width and height must be the file's own.
	static unsigned char* loadTextureAlpha(const unsigned char* bytes, int size, int width, int height);
	// Drops the texels cached under this file name once the renderer has taken them over.
	static void releaseCachedTexture(const char* textureName);
	// Undoes that, so the next load of this file reads it again.
	static void forgetCachedTexture(const char* textureName);
};

#endif  //B3_IMPORT_MESH_UTILITY_H
