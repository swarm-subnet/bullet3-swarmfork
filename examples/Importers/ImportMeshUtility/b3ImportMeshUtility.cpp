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
#include "Bullet3Common/b3DiskCache.h"
#include "../../CommonInterfaces/CommonFileIOInterface.h"

struct CachedTextureResult
{
	std::string m_textureName;

	int m_width;
	int m_height;
	unsigned char* m_pixels;
	unsigned char* m_alpha;
	//the renderer took these texels over and keeps them under this file name; the size above is all that is left here
	bool m_handedOver;
	CachedTextureResult()
		: m_width(0),
		  m_height(0),
		  m_pixels(0),
		  m_alpha(0),
		  m_handedOver(false)
	{
	}
};

static b3HashMap<b3HashString, CachedTextureResult> gCachedTextureResults;

void b3ImportMeshUtility::releaseCachedTexture(const char* textureName)
{
	CachedTextureResult* texture = textureName ? gCachedTextureResults[textureName] : 0;
	if (!texture)
		return;
	free(texture->m_pixels);
	free(texture->m_alpha);
	texture->m_pixels = 0;
	texture->m_alpha = 0;
	texture->m_handedOver = true;
}

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
				free(res->m_alpha);
			}
		}
	}
};
static CachedTextureManager sTexCacheMgr;

unsigned char* b3ImportMeshUtility::loadTextureAlpha(const unsigned char* bytes, int size, int width, int height)
{
	int w, h, n;
	if (!stbi_info_from_memory(bytes, size, &w, &h, &n) || (n != 2 && n != 4) || w != width || h != height)
		return 0;
	unsigned char* rgba = stbi_load_from_memory(bytes, size, &w, &h, &n, 4);
	if (!rgba)
		return 0;
	const size_t texels = (size_t)width * height;
	unsigned char* alpha = (unsigned char*)malloc(texels);
	bool opaque = true;
	for (size_t i = 0; i < texels; i++)
	{
		alpha[i] = rgba[i * 4 + 3];
		opaque = opaque && alpha[i] == 255;
	}
	stbi_image_free(rgba);
	if (!opaque)
		return alpha;
	free(alpha);
	return 0;
}

// The decoded texels of an image file as another process of this machine saved them: the colour, and the alpha plane
// when the image has one that is not all opaque. Both malloc'ed like the decoder's own; false when there is no such file.
static bool readTextureCache(const std::string& path, int& width, int& height, unsigned char*& image, unsigned char*& alpha)
{
	std::vector<char> bytes;
	if (!b3DiskCacheRead(path, bytes))
		return false;
	b3DiskCacheReader in(bytes);
	int w = 0, h = 0, hasAlpha = 0;
	if (!in.value(w) || !in.value(h) || !in.value(hasAlpha) || w <= 0 || h <= 0)
		return false;
	const size_t texels = (size_t)w * h;
	if ((size_t)(in.m_end - in.m_at) != texels * (hasAlpha ? 4 : 3))
		return false;
	image = (unsigned char*)malloc(texels * 3);
	alpha = hasAlpha ? (unsigned char*)malloc(texels) : 0;
	in.take(image, texels * 3);
	if (alpha)
		in.take(alpha, texels);
	width = w;
	height = h;
	return true;
}

// Saves what readTextureCache reads back.
static void writeTextureCache(const std::string& path, int width, int height, const unsigned char* image, const unsigned char* alpha)
{
	b3DiskCacheWriter out;
	const int hasAlpha = alpha ? 1 : 0;
	out.value(width);
	out.value(height);
	out.value(hasAlpha);
	const size_t texels = (size_t)width * height;
	out.put(image, texels * 3);
	if (alpha)
		out.put(alpha, texels);
	b3DiskCacheWrite(path, &out.m_bytes[0], out.m_bytes.size());
}

//loads the diffuse texture named in a material, trying the mesh folder and the data folders; returns 0 when none loads
static unsigned char* loadDiffuseTexture(const char* filename, const char* pathPrefix, CommonFileIOInterface* fileIO, int& width, int& height, bool& isCached, unsigned char*& alpha, std::string& textureName, bool handover)
{
	unsigned char* image = 0;
	isCached = false;
	alpha = 0;

	const char* prefix[] = {pathPrefix, "./", "./data/", "../data/", "../../data/", "../../../data/", "../../../../data/"};
	int numprefix = sizeof(prefix) / sizeof(const char*);

	for (int i = 0; !image && i < numprefix; i++)
	{
		char relativeFileName[1024];
		sprintf(relativeFileName, "%s%s", prefix[i], filename);
		char relativeFileName2[1024];
		if (fileIO->findResourcePath(relativeFileName, relativeFileName2, 1024))
		{
			const CachedTextureResult* texture = gCachedTextureResults[relativeFileName];
			if (texture && texture->m_pixels && b3IsFileCachingEnabled())
			{
				image = texture->m_pixels;
				alpha = texture->m_alpha;
				width = texture->m_width;
				height = texture->m_height;
				textureName = relativeFileName;
				isCached = true;
			}
			else if (texture && texture->m_handedOver && handover)
			{
				// The renderer holds these texels under this name, so nothing has to be read or decoded.
				width = texture->m_width;
				height = texture->m_height;
				textureName = relativeFileName;
				isCached = true;
				return 0;
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

				// Texels another process of this machine decoded from the same bytes are read back instead of decoded again.
				std::string diskPath;
				const bool onDisk = buffer.size() && b3DiskCachePath(b3DiskCacheHash(&buffer[0], buffer.size(), b3DiskCacheHash("SWTEX1", 6)), "texc", diskPath);
				const bool fromDisk = onDisk && readTextureCache(diskPath, width, height, image, alpha);
				if (buffer.size() && !fromDisk)
				{
					image = stbi_load_from_memory((const unsigned char*)&buffer[0], buffer.size(), &width, &height, &n, 3);
				}
				//image = stbi_load(relativeFileName, &width, &height, &n, 3);

				if (image)
				{
					if (!fromDisk)
					{
						alpha = b3ImportMeshUtility::loadTextureAlpha((const unsigned char*)&buffer[0], buffer.size(), width, height);
						if (onDisk)
							writeTextureCache(diskPath, width, height, image, alpha);
					}
					textureName = relativeFileName;
					if (b3IsFileCachingEnabled())
					{
						CachedTextureResult result;
						result.m_textureName = relativeFileName;
						result.m_width = width;
						result.m_height = height;
						result.m_pixels = image;
						result.m_alpha = alpha;
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
static void buildMaterialGroups(const std::vector<bt_tinyobj::shape_t>& shapes, const char* pathPrefix, CommonFileIOInterface* fileIO, b3AlignedObjectArray<b3ImportMeshMaterialGroup>& groups, bool textureHandover)
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
			group.m_textureAlpha = 0;
			group.m_isCached = false;
			group.m_textureWidth = 0;
			group.m_textureHeight = 0;
			if (material.diffuse_texname.length() > 0)
			{
				group.m_textureImage = loadDiffuseTexture(material.diffuse_texname.c_str(), pathPrefix, fileIO, group.m_textureWidth, group.m_textureHeight, group.m_isCached, group.m_textureAlpha, group.m_textureName, textureHandover);
			}
			groups.push_back(group);
		}
		indexStart += indexCount;
	}
}

bool b3ImportMeshUtility::loadAndRegisterMeshFromFileInternal(const std::string& fileName, b3ImportMeshData& meshData, struct CommonFileIOInterface* fileIO, bool splitOnMaterial, bool textureHandover)
{
	B3_PROFILE("loadAndRegisterMeshFromFileInternal");
	meshData.m_gfxShape = 0;
	meshData.m_textureImage1 = 0;
	meshData.m_textureAlpha = 0;
	meshData.m_textureHeight = 0;
	meshData.m_textureWidth = 0;
	meshData.m_flags = 0;
	meshData.m_isCached = false;
	meshData.m_textureName.clear();
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
			buildMaterialGroups(shapes, pathPrefix, fileIO, meshData.m_materialGroups, textureHandover);
		}
		else
		{
			B3_PROFILE("Load Texture");
			//int textureIndex = -1;
			//try to load some texture
			for (int i = 0; meshData.m_textureImage1 == 0 && meshData.m_textureName.empty() && i < shapes.size(); i++)
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
					meshData.m_textureImage1 = loadDiffuseTexture(shape.material.diffuse_texname.c_str(), pathPrefix, fileIO, meshData.m_textureWidth, meshData.m_textureHeight, meshData.m_isCached, meshData.m_textureAlpha, meshData.m_textureName, textureHandover);
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
