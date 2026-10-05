#include "LoadMeshFromObj.h"

#include "../../OpenGLWindow/GLInstanceGraphicsShape.h"
#include <stdio.h>  //fopen
#include "Bullet3Common/b3AlignedObjectArray.h"
#include <string>
#include <vector>
#include "Wavefront2GLInstanceGraphicsShape.h"
#include "Bullet3Common/b3HashMap.h"
#include "Bullet3Common/b3DiskCache.h"
#include "../../CommonInterfaces/CommonFileIOInterface.h"

struct CachedObjResult
{
	std::string m_msg;
	std::vector<bt_tinyobj::shape_t> m_shapes;
	bt_tinyobj::attrib_t m_attribute;
};

// Held by pointer: growing the table must not deep copy every mesh it already carries.
static b3HashMap<b3HashString, CachedObjResult*> gCachedObjResults;
static int gEnableFileCaching = 1;

static void clearCachedObjResults()
{
	for (int i = 0; i < gCachedObjResults.size(); i++)
	{
		CachedObjResult** entry = gCachedObjResults.getAtIndex(i);
		if (entry)
			delete *entry;
	}
	gCachedObjResults.clear();
}

int b3IsFileCachingEnabled()
{
	return gEnableFileCaching;
}
void b3EnableFileCaching(int enable)
{
	gEnableFileCaching = enable;
	if (enable == 0)
	{
		clearCachedObjResults();
	}
}

// The whole of a file read through the importer's file interface; false when it cannot be read.
static bool readWholeFile(CommonFileIOInterface* fileIO, const char* name, std::vector<char>& out)
{
	const int fileId = fileIO ? fileIO->fileOpen(name, "rb") : -1;
	if (fileId < 0)
		return false;
	const int size = fileIO->getFileSize(fileId);
	out.resize(size > 0 ? (size_t)size : 0);
	const bool ok = size >= 0 && (size == 0 || fileIO->fileRead(fileId, &out[0], size) == size);
	fileIO->fileClose(fileId);
	return ok;
}

// Key of a parse on disk: the format, the grouping, the .obj bytes and the bytes of every material library it names,
// so an edit to any of them misses. False when one of them cannot be read, and the parse then skips the disk.
static bool objCacheKey(const char* filename, const char* mtl_basepath, CommonFileIOInterface* fileIO, bool splitOnMaterial,
						unsigned long long& key)
{
	std::vector<char> obj;
	if (!readWholeFile(fileIO, filename, obj))
		return false;
	key = b3DiskCacheHash("SWOBJ1", 6);
	key = b3DiskCacheHash(&splitOnMaterial, sizeof(splitOnMaterial), key);
	key = b3DiskCacheHash(obj.empty() ? 0 : &obj[0], obj.size(), key);
	const char* text = obj.empty() ? 0 : &obj[0];
	for (size_t at = 0; at < obj.size();)
	{
		const char* newline = (const char*)memchr(text + at, '\n', obj.size() - at);
		const size_t end = newline ? (size_t)(newline - text) : obj.size();
		if (end - at > 7 && memcmp(&obj[at], "mtllib", 6) == 0 && (obj[at + 6] == ' ' || obj[at + 6] == '\t'))
		{
			std::string name(&obj[at + 7], end - at - 7);
			while (!name.empty() && (name[name.size() - 1] == '\r' || name[name.size() - 1] == ' ' || name[name.size() - 1] == '\t'))
				name.erase(name.size() - 1);
			std::vector<char> mtl;
			if (!readWholeFile(fileIO, (std::string(mtl_basepath ? mtl_basepath : "") + name).c_str(), mtl))
				return false;
			key = b3DiskCacheHash(name.data(), name.size(), key);
			key = b3DiskCacheHash(mtl.empty() ? 0 : &mtl[0], mtl.size(), key);
		}
		at = end + 1;
	}
	return true;
}

// A parse as bytes: its message, the three attribute blocks, and every shape's name, material and indices.
static void writeObjCache(const std::string& path, const std::string& msg, const bt_tinyobj::attrib_t& attribute,
						  const std::vector<bt_tinyobj::shape_t>& shapes)
{
	b3DiskCacheWriter out;
	out.text(msg);
	out.block(attribute.vertices);
	out.block(attribute.normals);
	out.block(attribute.texcoords);
	out.value((unsigned long long)shapes.size());
	for (size_t i = 0; i < shapes.size(); i++)
	{
		const bt_tinyobj::material_t& m = shapes[i].material;
		out.text(shapes[i].name);
		out.text(m.name);
		out.put(m.ambient, sizeof(m.ambient));
		out.put(m.diffuse, sizeof(m.diffuse));
		out.put(m.specular, sizeof(m.specular));
		out.put(m.transmittance, sizeof(m.transmittance));
		out.put(m.emission, sizeof(m.emission));
		out.value(m.shininess);
		out.value(m.transparency);
		out.text(m.ambient_texname);
		out.text(m.diffuse_texname);
		out.text(m.specular_texname);
		out.text(m.normal_texname);
		out.value((unsigned long long)m.unknown_parameter.size());
		for (std::map<std::string, std::string>::const_iterator it = m.unknown_parameter.begin(); it != m.unknown_parameter.end(); ++it)
		{
			out.text(it->first);
			out.text(it->second);
		}
		out.block(shapes[i].mesh.indices);
	}
	b3DiskCacheWrite(path, &out.m_bytes[0], out.m_bytes.size());
}

// The parse writeObjCache wrote; false for a file that is short or malformed, which is then parsed afresh.
static bool readObjCache(const std::vector<char>& bytes, std::string& msg, bt_tinyobj::attrib_t& attribute, std::vector<bt_tinyobj::shape_t>& shapes)
{
	b3DiskCacheReader in(bytes);
	unsigned long long count = 0;
	if (!in.text(msg) || !in.block(attribute.vertices) || !in.block(attribute.normals) || !in.block(attribute.texcoords) || !in.value(count) ||
		count > bytes.size())
		return false;
	shapes.resize((size_t)count);
	for (size_t i = 0; i < shapes.size(); i++)
	{
		bt_tinyobj::material_t& m = shapes[i].material;
		unsigned long long params = 0;
		if (!in.text(shapes[i].name) || !in.text(m.name) || !in.take(m.ambient, sizeof(m.ambient)) || !in.take(m.diffuse, sizeof(m.diffuse)) ||
			!in.take(m.specular, sizeof(m.specular)) || !in.take(m.transmittance, sizeof(m.transmittance)) ||
			!in.take(m.emission, sizeof(m.emission)) || !in.value(m.shininess) || !in.value(m.transparency) || !in.text(m.ambient_texname) ||
			!in.text(m.diffuse_texname) || !in.text(m.specular_texname) || !in.text(m.normal_texname) || !in.value(params) || params > bytes.size())
			return false;
		m.unknown_parameter.clear();
		for (unsigned long long p = 0; p < params; p++)
		{
			std::string name, value;
			if (!in.text(name) || !in.text(value))
				return false;
			m.unknown_parameter[name] = value;
		}
		if (!in.block(shapes[i].mesh.indices))
			return false;
	}
	return in.m_at == in.m_end;
}

std::string LoadFromCachedOrFromObj(
	bt_tinyobj::attrib_t& attribute,
	std::vector<bt_tinyobj::shape_t>& shapes,  // [output]
	const char* filename,
	const char* mtl_basepath,
	struct CommonFileIOInterface* fileIO,
	bool splitOnMaterial)
{
	// the two shape groupings of one file must not share a cache entry
	std::string cacheKey = splitOnMaterial ? std::string(filename) + "|mtl" : std::string(filename);
	CachedObjResult** resultPtr = gCachedObjResults[cacheKey.c_str()];
	if (resultPtr && *resultPtr)
	{
		const CachedObjResult& result = **resultPtr;
		shapes = result.m_shapes;
		attribute = result.m_attribute;
		return result.m_msg;
	}

	// A parse another process of this machine already did is read back whole instead of parsing the text again.
	std::string err;
	unsigned long long diskKey = 0;
	std::string diskPath;
	std::vector<char> diskBytes;
	const char* diskDir = getenv("SWARM_BVH_CACHE_DIR");
	const bool onDisk = diskDir && *diskDir && objCacheKey(filename, mtl_basepath, fileIO, splitOnMaterial, diskKey) &&
						b3DiskCachePath(diskKey, "objc", diskPath);
	if (!(onDisk && b3DiskCacheRead(diskPath, diskBytes) && readObjCache(diskBytes, err, attribute, shapes)))
	{
		attribute = bt_tinyobj::attrib_t();
		shapes.clear();
		err = bt_tinyobj::LoadObj(attribute, shapes, filename, mtl_basepath, fileIO, splitOnMaterial);
		if (onDisk)
			writeObjCache(diskPath, err, attribute, shapes);
	}
	if (gEnableFileCaching)
	{
		CachedObjResult* result = new CachedObjResult;
		result->m_msg = err;
		result->m_shapes = shapes;
		result->m_attribute = attribute;
		gCachedObjResults.insert(cacheKey.c_str(), result);
	}
	return err;
}

GLInstanceGraphicsShape* LoadMeshFromObj(const char* relativeFileName, const char* materialPrefixPath, struct CommonFileIOInterface* fileIO)
{
	B3_PROFILE("LoadMeshFromObj");
	std::vector<bt_tinyobj::shape_t> shapes;
	bt_tinyobj::attrib_t attribute;
	{
		B3_PROFILE("bt_tinyobj::LoadObj2");
		std::string err = LoadFromCachedOrFromObj(attribute, shapes, relativeFileName, materialPrefixPath, fileIO);
	}

	{
		B3_PROFILE("btgCreateGraphicsShapeFromWavefrontObj");
		GLInstanceGraphicsShape* gfxShape = btgCreateGraphicsShapeFromWavefrontObj(attribute, shapes);
		return gfxShape;
	}
}
