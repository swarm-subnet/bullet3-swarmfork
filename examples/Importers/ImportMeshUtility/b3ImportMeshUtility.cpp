#include "b3ImportMeshUtility.h"

#include <map>
#include <vector>
#include "../../ThirdPartyLibs/Wavefront/tiny_obj_loader.h"
#include "LinearMath/btVector3.h"
#include "../ImportObjDemo/Wavefront2GLInstanceGraphicsShape.h"
#include "../../Utils/b3ResourcePath.h"
#include "Bullet3Common/b3FileUtils.h"
#include "stb_image/stb_image.h"
#include "../ImportObjDemo/LoadMeshFromObj.h"
#include "Bullet3Common/b3HashMap.h"
#include "../../CommonInterfaces/CommonFileIOInterface.h"

struct CachedTextureResult
{
	std::string m_textureName;

	int m_width;
	int m_height;
	unsigned char* m_pixels;
	CachedTextureResult()
		: m_width(0),
		  m_height(0),
		  m_pixels(0)
	{
	}
};

static b3HashMap<b3HashString, CachedTextureResult> gCachedTextureResults;
struct CachedTextureManager
{
	CachedTextureManager()
	{
	}
	virtual ~CachedTextureManager()
	{
		for (int i = 0; i < gCachedTextureResults.size(); i++)
		{
			CachedTextureResult* res = gCachedTextureResults.getAtIndex(i);
			if (res)
			{
				free(res->m_pixels);
			}
		}
	}
};
static CachedTextureManager sTexCacheMgr;

//loads the diffuse texture named in a material, trying the mesh folder and the data folders; returns 0 when none loads
static unsigned char* loadDiffuseTexture(const char* filename, const char* pathPrefix, CommonFileIOInterface* fileIO, int& width, int& height, bool& isCached)
{
	unsigned char* image = 0;
	isCached = false;

	const char* prefix[] = {pathPrefix, "./", "./data/", "../data/", "../../data/", "../../../data/", "../../../../data/"};
	int numprefix = sizeof(prefix) / sizeof(const char*);

	for (int i = 0; !image && i < numprefix; i++)
	{
		char relativeFileName[1024];
		sprintf(relativeFileName, "%s%s", prefix[i], filename);
		char relativeFileName2[1024];
		if (fileIO->findResourcePath(relativeFileName, relativeFileName2, 1024))
		{
			if (b3IsFileCachingEnabled())
			{
				CachedTextureResult* texture = gCachedTextureResults[relativeFileName];
				if (texture)
				{
					image = texture->m_pixels;
					width = texture->m_width;
					height = texture->m_height;
					isCached = true;
				}
			}

			if (image == 0)
			{
				int n;
				b3AlignedObjectArray<char> buffer;
				buffer.reserve(1024);
				int fileId = fileIO->fileOpen(relativeFileName,"rb");
				if (fileId>=0)
				{
					int size = fileIO->getFileSize(fileId);
					if (size>0)
					{
						buffer.resize(size);
						int actual = fileIO->fileRead(fileId,&buffer[0],size);
						if (actual != size)
						{
							b3Warning("STL filesize mismatch!\n");
							buffer.resize(0);
						}
					}
					fileIO->fileClose(fileId);
				}

				if (buffer.size())
				{
					image = stbi_load_from_memory((const unsigned char*)&buffer[0], buffer.size(), &width, &height, &n, 3);
				}
				//image = stbi_load(relativeFileName, &width, &height, &n, 3);

				if (image)
				{
					if (b3IsFileCachingEnabled())
					{
						CachedTextureResult result;
						result.m_textureName = relativeFileName;
						result.m_width = width;
						result.m_height = height;
						result.m_pixels = image;
						isCached = true;
						gCachedTextureResults.insert(relativeFileName, result);
					}
				}
				else
				{
					b3Warning("Unsupported texture image format [%s]\n", relativeFileName);

					break;
				}
			}
		}
		else
		{
			b3Warning("not found [%s]\n", relativeFileName);
		}
	}
	return image;
}

//reorders the shapes so every shape of one material is adjacent, materials kept in first-seen order
static void groupShapesByMaterial(std::vector<bt_tinyobj::shape_t>& shapes)
{
	std::map<std::string, size_t> bucketOf;
	std::vector<std::vector<bt_tinyobj::shape_t> > buckets;
	for (size_t i = 0; i < shapes.size(); i++)
	{
		std::map<std::string, size_t>::iterator it = bucketOf.find(shapes[i].material.name);
		if (it == bucketOf.end())
		{
			it = bucketOf.insert(std::make_pair(shapes[i].material.name, buckets.size())).first;
			buckets.push_back(std::vector<bt_tinyobj::shape_t>());
		}
		buckets[it->second].push_back(shapes[i]);
	}
	shapes.clear();
	for (size_t b = 0; b < buckets.size(); b++)
	{
		for (size_t i = 0; i < buckets[b].size(); i++)
		{
			shapes.push_back(buckets[b][i]);
		}
	}
}

//one group per material, covering the index range its shapes occupy in the flattened mesh
static void buildMaterialGroups(const std::vector<bt_tinyobj::shape_t>& shapes, const char* pathPrefix, CommonFileIOInterface* fileIO, b3AlignedObjectArray<b3ImportMeshMaterialGroup>& groups)
{
	int indexStart = 0;
	for (size_t i = 0; i < shapes.size(); i++)
	{
		const bt_tinyobj::material_t& material = shapes[i].material;
		int indexCount = (int)shapes[i].mesh.indices.size();
		if (groups.size() && material.name == shapes[i - 1].material.name)
		{
			groups[groups.size() - 1].m_indexCount += indexCount;
		}
		else
		{
			b3ImportMeshMaterialGroup group;
			group.m_indexStart = indexStart;
			group.m_indexCount = indexCount;
			group.m_rgbaColor[0] = material.diffuse[0];
			group.m_rgbaColor[1] = material.diffuse[1];
			group.m_rgbaColor[2] = material.diffuse[2];
			group.m_rgbaColor[3] = material.transparency;
			group.m_specularColor[0] = material.specular[0];
			group.m_specularColor[1] = material.specular[1];
			group.m_specularColor[2] = material.specular[2];
			group.m_specularColor[3] = 1;
			group.m_textureImage = 0;
			group.m_isCached = false;
			group.m_textureWidth = 0;
			group.m_textureHeight = 0;
			if (material.diffuse_texname.length() > 0)
			{
				group.m_textureImage = loadDiffuseTexture(material.diffuse_texname.c_str(), pathPrefix, fileIO, group.m_textureWidth, group.m_textureHeight, group.m_isCached);
			}
			groups.push_back(group);
		}
		indexStart += indexCount;
	}
}

bool b3ImportMeshUtility::loadAndRegisterMeshFromFileInternal(const std::string& fileName, b3ImportMeshData& meshData, struct CommonFileIOInterface* fileIO, bool splitOnMaterial)
{
	B3_PROFILE("loadAndRegisterMeshFromFileInternal");
	meshData.m_gfxShape = 0;
	meshData.m_textureImage1 = 0;
	meshData.m_textureHeight = 0;
	meshData.m_textureWidth = 0;
	meshData.m_flags = 0;
	meshData.m_isCached = false;
	meshData.m_materialGroups.clear();

	char relativeFileName[1024];
	if (fileIO->findResourcePath(fileName.c_str(), relativeFileName, 1024))
	{
		char pathPrefix[1024];

		b3FileUtils::extractPath(relativeFileName, pathPrefix, 1024);
		btVector3 shift(0, 0, 0);

		std::vector<bt_tinyobj::shape_t> shapes;
		bt_tinyobj::attrib_t attribute;
		{
			B3_PROFILE("tinyobj::LoadObj");
			std::string err = LoadFromCachedOrFromObj(attribute, shapes, relativeFileName, pathPrefix, fileIO, splitOnMaterial);
			//std::string err = tinyobj::LoadObj(shapes, relativeFileName, pathPrefix);
		}
		if (splitOnMaterial)
		{
			groupShapesByMaterial(shapes);
		}

		GLInstanceGraphicsShape* gfxShape = btgCreateGraphicsShapeFromWavefrontObj(attribute, shapes);
		if (splitOnMaterial)
		{
			B3_PROFILE("Load Textures");
			buildMaterialGroups(shapes, pathPrefix, fileIO, meshData.m_materialGroups);
		}
		else
		{
			B3_PROFILE("Load Texture");
			//int textureIndex = -1;
			//try to load some texture
			for (int i = 0; meshData.m_textureImage1 == 0 && i < shapes.size(); i++)
			{
				const bt_tinyobj::shape_t& shape = shapes[i];
				meshData.m_rgbaColor[0] = shape.material.diffuse[0];
				meshData.m_rgbaColor[1] = shape.material.diffuse[1];
				meshData.m_rgbaColor[2] = shape.material.diffuse[2];
				meshData.m_rgbaColor[3] = shape.material.transparency;
				meshData.m_flags |= B3_IMPORT_MESH_HAS_RGBA_COLOR;

				meshData.m_specularColor[0] = shape.material.specular[0];
				meshData.m_specularColor[1] = shape.material.specular[1];
				meshData.m_specularColor[2] = shape.material.specular[2];
				meshData.m_specularColor[3] = 1;
				meshData.m_flags |= B3_IMPORT_MESH_HAS_SPECULAR_COLOR;

				if (shape.material.diffuse_texname.length() > 0)
				{
					meshData.m_textureImage1 = loadDiffuseTexture(shape.material.diffuse_texname.c_str(), pathPrefix, fileIO, meshData.m_textureWidth, meshData.m_textureHeight, meshData.m_isCached);
				}
			}
		}
		meshData.m_gfxShape = gfxShape;
		return true;
	}
	else
	{
		b3Warning("Cannot find %s\n", fileName.c_str());
	}

	return false;
}
