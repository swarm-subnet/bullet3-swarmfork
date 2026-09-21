#include "LoadMeshFromObj.h"

#include "../../OpenGLWindow/GLInstanceGraphicsShape.h"
#include <stdio.h>  //fopen
#include "Bullet3Common/b3AlignedObjectArray.h"
#include <string>
#include <vector>
#include "Wavefront2GLInstanceGraphicsShape.h"
#include "Bullet3Common/b3HashMap.h"

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

	std::string err = bt_tinyobj::LoadObj(attribute, shapes, filename, mtl_basepath, fileIO, splitOnMaterial);
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
