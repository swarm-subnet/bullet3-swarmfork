#ifndef LOAD_MESH_FROM_OBJ_H
#define LOAD_MESH_FROM_OBJ_H

struct GLInstanceGraphicsShape;

#include "../../ThirdPartyLibs/Wavefront/tiny_obj_loader.h"
#include <memory>

struct CachedObjResult
{
	std::string m_msg;
	std::vector<bt_tinyobj::shape_t> m_shapes;
	bt_tinyobj::attrib_t m_attribute;
};

// A parse the cache kept, alive for as long as anyone holds it, even once the cache drops it.
typedef std::shared_ptr<const CachedObjResult> b3HeldObjParse;

int b3IsFileCachingEnabled();
void b3EnableFileCaching(int enable);

// held, when given, receives the cache's entry for the parse returned, or nothing when file caching is off.
std::string LoadFromCachedOrFromObj(
	bt_tinyobj::attrib_t& attribute,
	std::vector<bt_tinyobj::shape_t>& shapes,  // [output]
	const char* filename,
	const char* mtl_basepath,
	struct CommonFileIOInterface* fileIO,
	bool splitOnMaterial = false,
	b3HeldObjParse* held = 0);

GLInstanceGraphicsShape* LoadMeshFromObj(const char* relativeFileName, const char* materialPrefixPath,struct CommonFileIOInterface* fileIO);

#endif  //LOAD_MESH_FROM_OBJ_H
