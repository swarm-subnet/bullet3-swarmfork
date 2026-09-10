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
	bool m_isCached;
	int m_textureWidth;
	int m_textureHeight;
};

struct b3ImportMeshData
{
	struct GLInstanceGraphicsShape* m_gfxShape;

	unsigned char* m_textureImage1;  //in 3 component 8-bit RGB data
	bool m_isCached;
	int m_textureWidth;
	int m_textureHeight;
	double m_rgbaColor[4];
	double m_specularColor[4];
	int m_flags;
	//filled only when loading with splitOnMaterial; m_textureImage1 stays 0 then
	b3AlignedObjectArray<b3ImportMeshMaterialGroup> m_materialGroups;

	b3ImportMeshData()
		:m_gfxShape(0),
		m_textureImage1(0),
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
	static bool loadAndRegisterMeshFromFileInternal(const std::string& fileName, b3ImportMeshData& meshData, struct CommonFileIOInterface* fileIO, bool splitOnMaterial = false);
};

#endif  //B3_IMPORT_MESH_UTILITY_H
