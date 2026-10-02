#include "SwarmRaycast.h"

#include <embree4/rtcore.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <atomic>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <xmmintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef _WIN32
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

#include "../../../TinyRenderer/TinyRenderer.h"
#include "../../../TinyRenderer/SwarmGamma.h"
#include "SwarmDaylight.h"
#include "SwarmLowLight.h"
#include "SwarmSky.h"
#include "SwarmThermal.h"
#include "Bullet3Common/b3Logging.h"
#include "LinearMath/btTransform.h"

namespace
{
// Embree reads vertices 16 bytes at a time, so every vertex block carries one spare slot.
const size_t kVertexPadding = 4;
// Side of the square pixel tiles that are dealt out to the render threads.
const int kTileSize = 16;
// The shadow map never exceeds this many cells a side, and a cell is never smaller than this: a small
// scene builds fast and a large one keeps a bounded map.
const int kShadowMapMaxCells = 4096;
const float kShadowMapMinCell = 0.005f;
// Side of the square blocks a shadow map is cast in, each the first time a lookup reads one of its cells.
const int kShadowBlock = 32;
// Daylight: the sun share a leaf passes to its back, and the coated glass curve: flat until the view grazes, then a full mirror.
const float kLeafTransmit = 0.35f;
const float kGlassFlat = 0.055f;
const float kGlassFlatUntilCos = 0.35f;
const float kGlassMirrorFromCos = 0.20f;
// A thin pane: plain glass Fresnel at each of its two faces, how far behind a pane the next ray starts, how many panes a ray passes.
const float kPaneF0 = 0.04f;
const float kPaneBias = 2e-3f;
const int kPaneDepth = 3;
// VISUAL_SHAPE_GLASS_BACKED: the white backsheet behind a module's cells, as an sRGB byte, as the solar park's racking paints it.
const float kPaneBacking = 254.0f;
// A texel with alpha below this is a hole when cut-outs are on.
const unsigned char kAlphaCutoff = 128;
// A cut-out read over many texels is a veil the ray stops on from this much coverage of the pixel, and shades by it.
const unsigned char kVeilCoverage = 8;
// Near infrared: a leaf reflects about half of it, far more than of visible green, which is why foliage glows there.
const float kLeafNearInfrared = 0.5f;

inline float dot3(const float a[3], const float b[3])
{
	return (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2];
}

inline void cross3(const float a[3], const float b[3], float out[3])
{
	out[0] = a[1] * b[2] - a[2] * b[1];
	out[1] = a[2] * b[0] - a[0] * b[2];
	out[2] = a[0] * b[1] - a[1] * b[0];
}

// Scales to unit length the way TinyRenderer's vectors do, by one reciprocal; a zero vector stays zero.
inline void normalize3(float v[3])
{
	const float length = sqrtf(dot3(v, v));
	if (length > 0.0f)
	{
		const float inv = 1.0f / length;
		for (int i = 0; i < 3; i++)
			v[i] *= inv;
	}
}

struct Instance;
struct Batch;

// The light's view of the static tree: a grid laid across the map at right angles to the light, and
// for each cell how far a ray from the light side travels before it meets a drawn surface.
struct ShadowMap
{
	float m_lightDir[3];
	// Unit grid axes, both at right angles to the light, and the world corner of cell (0, 0) on the
	// plane the rays start from.
	float m_axisU[3];
	float m_axisV[3];
	float m_origin[3];
	float m_cell;
	int m_cols;
	int m_rows;
	// Cast block by block while frames read the map, hence mutable behind the const lookups.
	mutable std::unique_ptr<float[]> m_depth;  // INFINITY where the ray met nothing
	// Per block of kShadowBlock x kShadowBlock cells: 0 not cast, 1 being cast, 2 cast.
	mutable std::unique_ptr<std::atomic<unsigned char>[]> m_blocks;
	int m_blockCols;
	// The tree a block is cast against and what the hit filter reads, set before every frame.
	RTCScene m_scene;
	const std::vector<Instance*>* m_instances;
	const std::vector<Batch*>* m_batches;
	unsigned m_forestId;
	bool m_built;
	// Whether the cells were cast with cut-outs on; the map is recast when a frame asks for the other.
	bool m_alphaCutout;
	// Whether the cells were cast with leaf cards letting the light through; the map is recast likewise.
	bool m_leafNoShadow;
};

// A body that never moved since the first frame: its triangles sit in world space inside the one
// static tree. When it moves later it is retired here and carries on as a mover instance.
struct StaticMember
{
	TinyRenderObjectData* m_obj;
	RTCGeometry m_geometry;
	unsigned m_geomId;
	std::vector<float> m_vertices;
	std::vector<unsigned> m_indices;
	// Per-vertex unit normals already in world space, indexed like m_vertices.
	std::vector<float> m_normals;
	// uv pairs indexed like m_vertices: the model's own array when its corners are indexed by vertex,
	// which never moves once a mesh is registered, and m_uvsOwned when they have to be gathered.
	const float* m_uvs;
	std::vector<float> m_uvsOwned;
	const void* m_meshKey;
	float m_transform[16];
	// Row-major rotation of the body: what TinyRenderer's inverse-transpose model matrix does to a normal.
	float m_rotation[9];
	bool m_doubleSided;
	bool m_visible;
	bool m_retired;
	// VISUAL_SHAPE_GLASS: under daylight a hit reads the sky by Fresnel and the surface behind through the tint.
	bool m_glass;
	// The texture carries an alpha plane, so hits may be cut out.
	bool m_hasAlpha;
	// The render object's texture revision the shadow maps last saw.
	unsigned m_textureRevision;
	int m_segmentation;
};

struct CachedTree;
// The process's mesh trees by vertex and triangle count, and the idle ones, least recently used first.
typedef std::multimap<std::pair<size_t, size_t>, CachedTree*> TreeCache;
typedef std::list<CachedTree*> IdleTrees;

// A mesh tree committed over its own copy of the vertex and index arrays and kept for the life of the process. A build
// runs the same steps on the same arrays, so a later world handing over equal arrays is given the tree it would build.
struct CachedTree
{
	RTCScene m_scene;
	RTCGeometry m_geometry;
	size_t m_triangles;
	// Worlds drawing the tree now; at zero the tree waits at m_idle for a later world.
	int m_users;
	TreeCache::iterator m_entry;
	IdleTrees::iterator m_idle;
};

// One tree per distinct mesh, shared by every mover instance drawn from that mesh. A world tree
// instead holds one static body's triangles in world space, drawn through an identity instance,
// so its tree can be kept on disk under the body's mesh and pose.
struct MeshTree
{
	RTCScene m_scene;
	RTCGeometry m_geometry;
	// The process-wide tree m_scene and m_geometry belong to, until the mesh is rewritten; 0 when they are this tree's own.
	CachedTree* m_cached;
	std::vector<float> m_vertices;
	std::vector<unsigned> m_indices;
	// Per-vertex unit normals in the mesh's own frame, and uv pairs, indexed like m_vertices.
	std::vector<float> m_normals;
	std::vector<float> m_uvs;
	int m_refs;
	bool m_dirty;
	bool m_world;
	// Set once the mesh has been rewritten: its tree then lives in a scene that refits instead of rebuilding.
	bool m_refitting;
	// The body transform a world tree was built for; the body counts as moved once it differs.
	float m_pose[16];
};

// A world tree on disk: <SWARM_BVH_CACHE_DIR>/<key>.rtree holds the world-space vertex, index,
// normal and uv blocks followed by Embree's image of the built tree, keyed by the mesh content
// hash, the body pose and scale.
struct TreeCacheHeader
{
	char m_magic[8];
	unsigned long long m_key;
	unsigned int m_numVertices;
	unsigned int m_numTriangles;
	unsigned int m_hasNormals;
	unsigned int m_reserved;
	unsigned long long m_treeBytes;
};

const char* const kTreeCacheMagic = "SWRTREE1";

struct Instance
{
	TinyRenderObjectData* m_obj;
	RTCGeometry m_geometry;
	// The same instance again inside the mover scene, so a shadow ray can skip the static tree.
	RTCGeometry m_shadowGeometry;
	unsigned m_geomId;
	MeshTree* m_tree;
	const void* m_meshKey;
	float m_transform[16];
	float m_rotation[9];
	bool m_enabled;
	bool m_doubleSided;
	bool m_hasAlpha;
	bool m_glass;
	// Sign of the instance determinant: a mirroring scale flips which side of a face is the front.
	float m_facingSign;
	bool m_staticShared;
	unsigned m_textureRevision;
	int m_segmentation;
};

// A forest batch: every placement of one shared mesh tree in a single Embree instance array. Batches live in
// the forest scene, which is built once and reached through one instance, so a moving body never re-sorts them.
struct Batch
{
	TinyRenderObjectData* m_obj;
	RTCGeometry m_geometry;
	unsigned m_geomId;
	MeshTree* m_tree;
	// One 3x4 column-major world transform per placement, shared with Embree.
	std::vector<float> m_transforms;
	// Row-major inverse transpose of each placement's 3x3 part, which carries the mesh normals into world space.
	std::vector<float> m_normalRotations;
	float m_bodyTransform[16];
	bool m_enabled;
	bool m_doubleSided;
	bool m_hasAlpha;
	bool m_glass;
	unsigned m_textureRevision;
	int m_segmentation;
};

struct ObjectState
{
	StaticMember* m_member;
	Instance* m_instance;
	bool m_deformed;
	Batch* m_batch;
};

struct QueryContext
{
	RTCRayQueryContext m_context;
	const std::vector<Instance*>* m_instances;
	// The forest's batches by id inside the forest scene, and the forest's own id in the top scene.
	const std::vector<Batch*>* m_batches;
	unsigned m_forestId;
	// Set for the rays that build the shadow map: any drawn face stops them, like a shadow ray.
	bool m_anyWinding;
	bool m_alphaCutout;
	// Set when leaf cards cast no shadow: the map rays and the occlusion rays pass through them.
	bool m_leafNoShadow;
	// Angle one pixel spans on the filtered camera rays, 0 on every other ray: a cut-out is then read over the
	// pixel's footprint instead of one texel, so a far chain-link or leaf card fades instead of breaking into moire.
	float m_pixelSpread;
	// How far along its ray any camera ray of this thread got this frame, for the region a kept frame depends on.
	float m_farthest;
};

// Casts the cells [col0, col1) x [row0, row1) of the shadow map on this thread, from the start plane along -light
// into the static tree. Each cell is one ray that depends only on the map and the tree, so neither the thread nor
// the moment it is cast can change its bytes.
void castShadowCells(const ShadowMap& map, int col0, int col1, int row0, int row1)
{
	const float dir[3] = {-map.m_lightDir[0], -map.m_lightDir[1], -map.m_lightDir[2]};
	QueryContext ctx;
	rtcInitRayQueryContext(&ctx.m_context);
	ctx.m_instances = map.m_instances;
	ctx.m_batches = map.m_batches;
	ctx.m_forestId = map.m_forestId;
	ctx.m_anyWinding = true;
	ctx.m_alphaCutout = map.m_alphaCutout;
	ctx.m_leafNoShadow = map.m_leafNoShadow;
	ctx.m_pixelSpread = 0.0f;
	RTCIntersectArguments args;
	rtcInitIntersectArguments(&args);
	args.context = &ctx.m_context;
	for (int row = row0; row < row1; row++)
	{
		const float v = ((float)row + 0.5f) * map.m_cell;
		for (int col = col0; col < col1; col++)
		{
			const float u = ((float)col + 0.5f) * map.m_cell;
			RTCRayHit rayhit;
			rayhit.ray.org_x = map.m_origin[0] + map.m_axisU[0] * u + map.m_axisV[0] * v;
			rayhit.ray.org_y = map.m_origin[1] + map.m_axisU[1] * u + map.m_axisV[1] * v;
			rayhit.ray.org_z = map.m_origin[2] + map.m_axisU[2] * u + map.m_axisV[2] * v;
			rayhit.ray.dir_x = dir[0];
			rayhit.ray.dir_y = dir[1];
			rayhit.ray.dir_z = dir[2];
			rayhit.ray.tnear = 0.0f;
			rayhit.ray.tfar = INFINITY;
			rayhit.ray.time = 0.0f;
			rayhit.ray.mask = (unsigned)-1;
			rayhit.ray.id = 0;
			rayhit.ray.flags = 0;
			rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
			rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
			rtcIntersect1(map.m_scene, &rayhit, &args);
			map.m_depth[(size_t)row * map.m_cols + col] = rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID ? INFINITY : rayhit.ray.tfar;
		}
	}
}

// Casts every block not cast yet that holds a cell of [col0, col1] x [row0, row1]. The first thread to claim a
// block casts it and any other waits for it, so a cell is cast once, by one ray, whichever thread asks first.
void castShadowBlocks(const ShadowMap& map, int col0, int col1, int row0, int row1)
{
	col0 = col0 < 0 ? 0 : col0;
	row0 = row0 < 0 ? 0 : row0;
	col1 = col1 >= map.m_cols ? map.m_cols - 1 : col1;
	row1 = row1 >= map.m_rows ? map.m_rows - 1 : row1;
	for (int blockRow = row0 / kShadowBlock; blockRow <= row1 / kShadowBlock && row0 <= row1; blockRow++)
		for (int blockCol = col0 / kShadowBlock; blockCol <= col1 / kShadowBlock && col0 <= col1; blockCol++)
		{
			std::atomic<unsigned char>& state = map.m_blocks[(size_t)blockRow * map.m_blockCols + blockCol];
			if (state.load(std::memory_order_acquire) == 2)
				continue;
			unsigned char expected = 0;
			if (state.compare_exchange_strong(expected, 1, std::memory_order_acquire))
			{
				const int c0 = blockCol * kShadowBlock, r0 = blockRow * kShadowBlock;
				castShadowCells(map, c0, c0 + kShadowBlock < map.m_cols ? c0 + kShadowBlock : map.m_cols, r0,
								r0 + kShadowBlock < map.m_rows ? r0 + kShadowBlock : map.m_rows);
				state.store(2, std::memory_order_release);
			}
			else
				while (state.load(std::memory_order_acquire) != 2)
					std::this_thread::yield();
		}
}

// Makes sure the cells [col0, col1] x [row0, row1] are cast; cells inside one block already cast cost one load.
inline void ensureShadowCells(const ShadowMap& map, int col0, int col1, int row0, int row1)
{
	if (col0 >= 0 && row0 >= 0 && col1 < map.m_cols && row1 < map.m_rows && col0 / kShadowBlock == col1 / kShadowBlock &&
		row0 / kShadowBlock == row1 / kShadowBlock &&
		map.m_blocks[(size_t)(row0 / kShadowBlock) * map.m_blockCols + col0 / kShadowBlock].load(std::memory_order_acquire) == 2)
		return;
	castShadowBlocks(map, col0, col1, row0, row1);
}

// A double-sided cut-out surface is a leaf or grass card, which casts nothing when leaf shadows are off.
inline bool leafCard(bool doubleSided, bool hasAlpha)
{
	return doubleSided && hasAlpha;
}

// True when the hit lands on a texel the texture marks as see-through. The uv is accumulated in the
// same order as the shader, so the texel tested here is the texel the colour would read.
bool cutOut(const TinyRender::Model* model, const float* uvs, const std::vector<unsigned>& indices, const RTCHit* hit)
{
	const unsigned* ids = &indices[(size_t)hit->primID * 3];
	const float weights[3] = {1.0f - hit->u - hit->v, hit->u, hit->v};
	TinyRender::Vec2f uv(0.0f, 0.0f);
	for (int j = 0; j < 3; j++)
	{
		uv.x += uvs[(size_t)ids[j] * 2] * weights[j];
		uv.y += uvs[(size_t)ids[j] * 2 + 1] * weights[j];
	}
	return model->alpha(uv) < kAlphaCutoff;
}

// The same test read over the pixel's footprint. In the hit's own frame the ray's length times the pixel angle is the
// footprint's width, stretched by the grazing angle; the triangle's uv area over its area turns that into uv units.
// Over many texels the surface is a veil: kept from kVeilCoverage, and blended by its coverage where it is shaded.
bool cutOutAt(const TinyRender::Model* model, const float* vertices, const float* uvs, const std::vector<unsigned>& indices,
			  const RTCHit* hit, const RTCRay* ray, float spread)
{
	if (!(spread > 0.0f))
		return cutOut(model, uvs, indices, hit);
	const unsigned* ids = &indices[(size_t)hit->primID * 3];
	const float weights[3] = {1.0f - hit->u - hit->v, hit->u, hit->v};
	TinyRender::Vec2f uv(0.0f, 0.0f);
	for (int j = 0; j < 3; j++)
	{
		uv.x += uvs[(size_t)ids[j] * 2] * weights[j];
		uv.y += uvs[(size_t)ids[j] * 2 + 1] * weights[j];
	}
	const float* p0 = vertices + (size_t)ids[0] * 3;
	const float* p1 = vertices + (size_t)ids[1] * 3;
	const float* p2 = vertices + (size_t)ids[2] * 3;
	const float e1[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
	const float e2[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
	float cross[3];
	cross3(e1, e2, cross);
	const float area2 = sqrtf(dot3(cross, cross));
	const float* t0 = uvs + (size_t)ids[0] * 2;
	const float* t1 = uvs + (size_t)ids[1] * 2;
	const float* t2 = uvs + (size_t)ids[2] * 2;
	const float uvArea2 = fabsf((t1[0] - t0[0]) * (t2[1] - t0[1]) - (t2[0] - t0[0]) * (t1[1] - t0[1]));
	const float dir[3] = {ray->dir_x, ray->dir_y, ray->dir_z};
	const float length = sqrtf(dot3(dir, dir));
	const float normal[3] = {hit->Ng_x, hit->Ng_y, hit->Ng_z};
	const float normalLength = sqrtf(dot3(normal, normal));
	if (!(area2 > 0.0f) || !(length > 0.0f) || !(normalLength > 0.0f))
		return cutOut(model, uvs, indices, hit);
	float grazing = fabsf(dot3(dir, normal)) / (length * normalLength);
	grazing = grazing > 0.05f ? grazing : 0.05f;
	const float width = ray->tfar * length * spread;
	bool averaged = false;
	const unsigned char alpha = model->alphaFiltered(uv, width * width / grazing * (uvArea2 / area2), &averaged);
	return alpha < (averaged ? kVeilCoverage : kAlphaCutoff);
}

// The instance a hit belongs to, or null for the static tree and for an id the scene does not know.
const Instance* hitInstance(const QueryContext* ctx, const RTCHit* hit)
{
	const unsigned instId = hit->instID[0];
	if (instId == RTC_INVALID_GEOMETRY_ID || instId >= ctx->m_instances->size())
		return 0;
	return (*ctx->m_instances)[instId];
}

// The batch a hit inside the forest belongs to and the placement it landed on; null for any other hit.
const Batch* hitBatch(const QueryContext* ctx, const RTCHit* hit, unsigned& placement)
{
	if (!ctx->m_batches || ctx->m_forestId == RTC_INVALID_GEOMETRY_ID || hit->instID[0] != ctx->m_forestId)
		return 0;
	if (hit->instID[1] >= ctx->m_batches->size())
		return 0;
	placement = hit->instPrimID[1];
	return (*ctx->m_batches)[hit->instID[1]];
}

// Sign of a 3x4 column-major placement's determinant: a mirroring scale flips which side of a face is the front.
inline float placementSign(const float* t)
{
	const float det = t[0] * (t[4] * t[8] - t[7] * t[5]) - t[3] * (t[1] * t[8] - t[7] * t[2]) + t[6] * (t[1] * t[5] - t[4] * t[2]);
	return det < 0.0f ? -1.0f : 1.0f;
}

// A hinted search on this thread. Embree drops a box once its rounded entry distance passes the nearest hit so far, so
// of two hits closer than that rounding the full search keeps whichever it met first, and a search cut short may meet
// them in another order. The filter therefore notes the nearest kept hit and the next one, and keeps the search going a
// window past the nearest, wide enough that every surface that rounding could put first is met; the ray goes to the
// full search when the next hit lies inside the window or the nearest is seen too obliquely for the window to hold.
struct HintedSearch
{
	// The ray's world direction, and the coordinate size its window allows for.
	float m_dir[3];
	float m_largest;
	float m_limit;
	float m_window;
	float m_nearest;
	float m_next;
	bool m_oblique;
	RTCHit m_hit;
};
thread_local HintedSearch* t_hintedSearch = 0;
// Below this cosine between the ray and a face, for coordinates of the window's size, the face's own distance rounds
// beyond the window.
const float kHintMinFacing = 1.0f / 16.0f;

// TinyRenderer drops a single-sided face whose winding normal points away from the camera; the
// filter does the same in object space, where Embree hands over both the ray and the hit. Static
// members also drop out here when retired or fully transparent, so the static tree is never rebuilt.
// With cut-outs on, a hit on a see-through texel drops out too and the ray carries on behind it.
void keepHit(const RTCFilterFunctionNArguments* args)
{
	if (args->N != 1)
		return;
	const RTCHit* hit = (const RTCHit*)args->hit;
	const RTCRay* ray = (const RTCRay*)args->ray;
	const QueryContext* ctx = (const QueryContext*)args->context;
	const float facing = hit->Ng_x * ray->dir_x + hit->Ng_y * ray->dir_y + hit->Ng_z * ray->dir_z;
	if (args->geometryUserPtr)
	{
		const StaticMember* member = (const StaticMember*)args->geometryUserPtr;
		if (member->m_retired || !member->m_visible || (!member->m_doubleSided && !ctx->m_anyWinding && facing >= 0.0f))
			args->valid[0] = 0;
		else if (ctx->m_anyWinding && ctx->m_leafNoShadow && leafCard(member->m_doubleSided, member->m_hasAlpha))
			args->valid[0] = 0;
		else if (member->m_hasAlpha && ctx->m_alphaCutout &&
				 cutOutAt(member->m_obj->m_model, member->m_vertices.data(), member->m_uvs, member->m_indices, hit, ray, ctx->m_pixelSpread))
			args->valid[0] = 0;
		return;
	}
	unsigned placement = 0;
	const Batch* batch = hitBatch(ctx, hit, placement);
	if (batch)
	{
		if (!batch->m_doubleSided && !ctx->m_anyWinding && facing * placementSign(&batch->m_transforms[(size_t)placement * 12]) >= 0.0f)
			args->valid[0] = 0;
		else if (ctx->m_anyWinding && ctx->m_leafNoShadow && leafCard(batch->m_doubleSided, batch->m_hasAlpha))
			args->valid[0] = 0;
		else if (batch->m_hasAlpha && ctx->m_alphaCutout &&
				 cutOutAt(batch->m_obj->m_model, batch->m_tree->m_vertices.data(), batch->m_tree->m_uvs.data(), batch->m_tree->m_indices, hit,
						  ray, ctx->m_pixelSpread))
			args->valid[0] = 0;
		return;
	}
	const Instance* inst = hitInstance(ctx, hit);
	if (!inst)
		return;
	if (!inst->m_doubleSided && !ctx->m_anyWinding && facing * inst->m_facingSign >= 0.0f)
		args->valid[0] = 0;
	else if (ctx->m_anyWinding && ctx->m_leafNoShadow && leafCard(inst->m_doubleSided, inst->m_hasAlpha))
		args->valid[0] = 0;
	else if (inst->m_hasAlpha && ctx->m_alphaCutout &&
			 cutOutAt(inst->m_obj->m_model, inst->m_tree->m_vertices.data(), inst->m_tree->m_uvs.data(), inst->m_tree->m_indices, hit, ray,
					  ctx->m_pixelSpread))
		args->valid[0] = 0;
}

// A shadow ray is stopped by any surface it meets, whichever way that surface is wound: a single-sided
// roof hides the sun from the ground even though the camera would see through its underside. Only the
// bodies that are not drawn at all, retired static members and fully transparent ones, let light past,
// and so does a see-through texel when cut-outs are on.
void shadowFilter(const RTCFilterFunctionNArguments* args)
{
	if (args->N != 1)
		return;
	const RTCHit* hit = (const RTCHit*)args->hit;
	const QueryContext* ctx = (const QueryContext*)args->context;
	if (args->geometryUserPtr)
	{
		const StaticMember* member = (const StaticMember*)args->geometryUserPtr;
		if (member->m_retired || !member->m_visible)
			args->valid[0] = 0;
		else if (ctx->m_leafNoShadow && leafCard(member->m_doubleSided, member->m_hasAlpha))
			args->valid[0] = 0;
		else if (member->m_hasAlpha && ctx->m_alphaCutout && cutOut(member->m_obj->m_model, member->m_uvs, member->m_indices, hit))
			args->valid[0] = 0;
		return;
	}
	unsigned placement = 0;
	const Batch* batch = hitBatch(ctx, hit, placement);
	if (batch)
	{
		if (ctx->m_leafNoShadow && leafCard(batch->m_doubleSided, batch->m_hasAlpha))
			args->valid[0] = 0;
		else if (batch->m_hasAlpha && ctx->m_alphaCutout && cutOut(batch->m_obj->m_model, batch->m_tree->m_uvs.data(), batch->m_tree->m_indices, hit))
			args->valid[0] = 0;
		return;
	}
	const Instance* inst = hitInstance(ctx, hit);
	if (!inst)
		return;
	if (ctx->m_leafNoShadow && leafCard(inst->m_doubleSided, inst->m_hasAlpha))
		args->valid[0] = 0;
	else if (inst->m_hasAlpha && ctx->m_alphaCutout && cutOut(inst->m_obj->m_model, inst->m_tree->m_uvs.data(), inst->m_tree->m_indices, hit))
		args->valid[0] = 0;
}

// The cosine between a kept hit's face and the ray in world space, where its distance rounds, scaled down by how far the
// coordinates of the hit's own frame outgrow the window's: the face's normal goes to world space through the inverse
// transpose of its placement or instance, and the ray's origin there, over the ray's stretch, sizes the coordinates.
float hintFacing(const RTCFilterFunctionNArguments* args, const RTCHit* hit, const RTCRay* ray, float t, const HintedSearch& search)
{
	const QueryContext* ctx = (const QueryContext*)args->context;
	const float local[3] = {hit->Ng_x, hit->Ng_y, hit->Ng_z};
	float normal[3] = {local[0], local[1], local[2]};
	if (!args->geometryUserPtr)
	{
		unsigned placement = 0;
		const Batch* batch = hitBatch(ctx, hit, placement);
		const Instance* inst = batch ? 0 : hitInstance(ctx, hit);
		if (batch)
		{
			const float* m = &batch->m_normalRotations[(size_t)placement * 9];
			for (int r = 0; r < 3; r++)
				normal[r] = dot3(m + r * 3, local);
		}
		else if (inst)
		{
			// Cofactors of the instance's 3x3 part: its inverse transpose up to a scale the cosine drops.
			const float* m = inst->m_transform;
			const float a[3] = {m[0], m[1], m[2]}, b[3] = {m[4], m[5], m[6]}, c[3] = {m[8], m[9], m[10]};
			float rows[3][3];
			cross3(b, c, rows[0]);
			cross3(c, a, rows[1]);
			cross3(a, b, rows[2]);
			for (int r = 0; r < 3; r++)
				normal[r] = (rows[0][r] * local[0] + rows[1][r] * local[1]) + rows[2][r] * local[2];
		}
		else
			return 0.0f;
	}
	const float org[3] = {ray->org_x, ray->org_y, ray->org_z};
	const float dir[3] = {ray->dir_x, ray->dir_y, ray->dir_z};
	const float size = sqrtf(dot3(org, org) / dot3(dir, dir)) + t;
	const float scale = size > search.m_largest ? search.m_largest / size : 1.0f;
	return fabsf(dot3(search.m_dir, normal)) / sqrtf(dot3(normal, normal)) * scale;
}

// The camera rays' filter: keepHit, and under a hinted search the note of each kept hit.
void hitFilter(const RTCFilterFunctionNArguments* args)
{
	keepHit(args);
	if (args->N != 1 || !args->valid[0])
		return;
	HintedSearch* search = t_hintedSearch;
	if (!search)
		return;
	RTCRay* ray = (RTCRay*)args->ray;
	const RTCHit* hit = (const RTCHit*)args->hit;
	const float t = ray->tfar;
	if (t < search->m_nearest)
	{
		search->m_next = search->m_nearest;
		search->m_nearest = t;
		search->m_hit = *hit;
		search->m_oblique = !(hintFacing(args, hit, ray, t, *search) >= kHintMinFacing);
	}
	else if (t < search->m_next)
		search->m_next = t;
	const float end = search->m_nearest + 2.0f * search->m_window;
	ray->tfar = end < search->m_limit ? end : search->m_limit;
}

void reportError(void* userPtr, RTCError code, const char* message)
{
	(void)userPtr;
	b3Warning("SwarmRaycast: Embree error %d: %s", (int)code, message ? message : "");
}

// World transform times local scaling, column-major, the order TinyRenderer's vertex shader applies.
void composeTransform(const btTransform& worldTransform, const btVector3& localScaling, float out[16])
{
	ATTRIBUTE_ALIGNED16(btScalar gl[16]);
	worldTransform.getOpenGLMatrix(gl);
	for (int c = 0; c < 4; c++)
	{
		const float scale = (c < 3) ? (float)localScaling[c] : 1.0f;
		for (int r = 0; r < 4; r++)
			out[c * 4 + r] = (float)gl[c * 4 + r] * scale;
	}
}

// Vertex positions of the model, in its own frame, padded for Embree.
void copyLocalVertices(TinyRender::Model* model, std::vector<float>& out)
{
	const int numVerts = model->nverts();
	out.assign((size_t)numVerts * 3 + kVertexPadding, 0.0f);
	for (int i = 0; i < numVerts; i++)
	{
		const TinyRender::Vec3f v = model->vert(i);
		out[(size_t)i * 3] = v.x;
		out[(size_t)i * 3 + 1] = v.y;
		out[(size_t)i * 3 + 2] = v.z;
	}
}

// Vertex positions in world space, accumulated in the same order as TinyRenderer's vertex shader so
// both eyes place every corner on the same float.
void copyWorldVertices(TinyRender::Model* model, const btTransform& worldTransform, const btVector3& localScaling, std::vector<float>& out)
{
	ATTRIBUTE_ALIGNED16(btScalar gl[16]);
	worldTransform.getOpenGLMatrix(gl);
	float m[16];
	for (int i = 0; i < 16; i++)
		m[i] = (float)gl[i];
	const float sx = (float)localScaling[0], sy = (float)localScaling[1], sz = (float)localScaling[2];
	const int numVerts = model->nverts();
	out.assign((size_t)numVerts * 3 + kVertexPadding, 0.0f);
	for (int i = 0; i < numVerts; i++)
	{
		const TinyRender::Vec3f v = model->vert(i);
		const float x = v.x * sx, y = v.y * sy, z = v.z * sz;
		for (int r = 0; r < 3; r++)
		{
			float acc = 0.0f + m[12 + r] * 1.0f;
			acc = acc + m[8 + r] * z;
			acc = acc + m[4 + r] * y;
			acc = acc + m[r] * x;
			out[(size_t)i * 3 + r] = acc;
		}
	}
}

// TinyRenderer draws the first three corners of a face, whatever the file said.
void copyIndices(TinyRender::Model* model, std::vector<unsigned>& out)
{
	const int numFaces = model->nfaces();
	out.resize((size_t)numFaces * 3);
	for (int f = 0; f < numFaces; f++)
	{
		int face[3];
		model->faceVertices(f, face);
		for (int j = 0; j < 3; j++)
			out[(size_t)f * 3 + j] = (unsigned)face[j];
	}
}

// Per-vertex uv and unit normal, gathered through the faces: every model the plugin builds names the
// same index for a corner's position, normal and uv, so each vertex slot is written with its own
// attributes. Normals are rotated into world space when a rotation is given. A model without normals
// leaves the normal block empty and shades with the face normal instead.
void copyAttributes(TinyRender::Model* model, const std::vector<unsigned>& indices, const float rotation[9],
					std::vector<float>& normals, std::vector<float>& uvs, bool gatherUvs, bool unitNormals = true)
{
	const int numVerts = model->nverts();
	const bool hasNormals = model->nnormals() > 0;
	uvs.assign(gatherUvs ? (size_t)numVerts * 2 : 0, 0.0f);
	normals.assign(hasNormals ? (size_t)numVerts * 3 : 0, 0.0f);
	if (!gatherUvs && !hasNormals)
		return;
	for (size_t f = 0; f * 3 + 2 < indices.size(); f++)
		for (int j = 0; j < 3; j++)
		{
			const unsigned v = indices[f * 3 + j];
			if (v >= (unsigned)numVerts)
				continue;
			if (gatherUvs)
			{
				const TinyRender::Vec2f uv = model->uv((int)f, j);
				uvs[(size_t)v * 2] = uv.x;
				uvs[(size_t)v * 2 + 1] = uv.y;
			}
			if (!hasNormals)
				continue;
			const TinyRender::Vec3f n = model->storedNormal((int)f, j);
			const float local[3] = {n.x, n.y, n.z};
			float out[3];
			for (int r = 0; r < 3; r++)
				out[r] = rotation ? (rotation[r * 3] * local[0] + rotation[r * 3 + 1] * local[1]) + rotation[r * 3 + 2] * local[2] : local[r];
			const float length = sqrtf((out[0] * out[0] + out[1] * out[1]) + out[2] * out[2]);
			for (int r = 0; r < 3; r++)
				normals[(size_t)v * 3 + r] = unitNormals ? (length > 0.0f ? out[r] / length : 0.0f) : out[r];
		}
}

RTCGeometry newTriangles(RTCDevice device, std::vector<float>& vertices, std::vector<unsigned>& indices)
{
	RTCGeometry geometry = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
	rtcSetSharedGeometryBuffer(geometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
							   &vertices[0], 0, 3 * sizeof(float), (vertices.size() - kVertexPadding) / 3);
	rtcSetSharedGeometryBuffer(geometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
							   &indices[0], 0, 3 * sizeof(unsigned), indices.size() / 3);
	rtcSetGeometryIntersectFilterFunction(geometry, hitFilter);
	rtcSetGeometryOccludedFilterFunction(geometry, shadowFilter);
	return geometry;
}

// One Embree device for the process: a cached tree outlives the world that built it, and an instance can only draw a
// scene of its own device. threads=1: Embree starts no thread of its own, so a build runs only on the threads that
// commit it. Never released, since the cached trees live as long as the process.
RTCDevice processDevice()
{
	static const RTCDevice device = rtcNewDevice("threads=1,set_affinity=0");
	return device;
}

// Commits a scene on up to two render threads. Embree cuts a build into the same tasks however many threads run them,
// and each task's result depends only on its own primitives, so the tree is the one a single thread builds.
void joinCommit(RTCScene scene)
{
	// The internal scheduler of a one-thread device has room for two threads in a build.
	const int threads = b3GetSwarmRenderThreads() < 2 ? 1 : 2;
	// A joining thread builds with the caller's MXCSR (rounding, flush to zero, denormals are zero), then gets its own back.
	const unsigned int callerCsr = _mm_getcsr();
#pragma omp parallel num_threads(threads)
	{
		const unsigned int ownCsr = _mm_getcsr();
		_mm_setcsr(callerCsr);
		rtcJoinCommitScene(scene);
		_mm_setcsr(ownCsr);
	}
}

// Idle cached trees are kept up to this weight, the least recently used dropped first. A tree weighs its triangles plus
// a fixed share for its scene, geometry and allocator blocks, so many tiny trees cannot pile up either.
const size_t kTreeCacheIdleWeight = 2000000;
const size_t kTreeCacheEntryWeight = 1000;

TreeCache gTreeCache;
IdleTrees gIdleTrees;
size_t gIdleWeight = 0;

// The cached tree over arrays equal byte for byte to these, built here when the process has none yet.
CachedTree* acquireCachedTree(RTCDevice device, const std::vector<float>& vertices, const std::vector<unsigned>& indices)
{
	const size_t numVertices = (vertices.size() - kVertexPadding) / 3, numTriangles = indices.size() / 3;
	const std::pair<size_t, size_t> key(numVertices, numTriangles);
	const std::pair<TreeCache::iterator, TreeCache::iterator> range = gTreeCache.equal_range(key);
	for (TreeCache::iterator it = range.first; it != range.second; ++it)
	{
		CachedTree* tree = it->second;
		if (memcmp(rtcGetGeometryBufferData(tree->m_geometry, RTC_BUFFER_TYPE_VERTEX, 0), &vertices[0], numVertices * 3 * sizeof(float)) != 0 ||
			memcmp(rtcGetGeometryBufferData(tree->m_geometry, RTC_BUFFER_TYPE_INDEX, 0), &indices[0], indices.size() * sizeof(unsigned)) != 0)
			continue;
		if (tree->m_users++ == 0)
		{
			gIdleTrees.erase(tree->m_idle);
			gIdleWeight -= numTriangles + kTreeCacheEntryWeight;
		}
		return tree;
	}
	CachedTree* tree = new CachedTree;
	tree->m_triangles = numTriangles;
	tree->m_users = 1;
	tree->m_scene = rtcNewScene(device);
	rtcSetSceneFlags(tree->m_scene, RTC_SCENE_FLAG_ROBUST);
	rtcSetSceneBuildQuality(tree->m_scene, RTC_BUILD_QUALITY_MEDIUM);
	// Embree owns the copies, so they last as long as any instance still draws the tree, evicted or not.
	tree->m_geometry = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
	memcpy(rtcSetNewGeometryBuffer(tree->m_geometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 3 * sizeof(float), numVertices),
		   &vertices[0], numVertices * 3 * sizeof(float));
	memcpy(rtcSetNewGeometryBuffer(tree->m_geometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof(unsigned), numTriangles),
		   &indices[0], indices.size() * sizeof(unsigned));
	rtcSetGeometryIntersectFilterFunction(tree->m_geometry, hitFilter);
	rtcSetGeometryOccludedFilterFunction(tree->m_geometry, shadowFilter);
	rtcCommitGeometry(tree->m_geometry);
	rtcAttachGeometry(tree->m_scene, tree->m_geometry);
	joinCommit(tree->m_scene);
	tree->m_entry = gTreeCache.insert(std::make_pair(key, tree));
	return tree;
}

// A world stops drawing the tree; once idle it stays for a later world while the idle ones fit the bound.
void releaseCachedTree(CachedTree* tree)
{
	if (--tree->m_users > 0)
		return;
	tree->m_idle = gIdleTrees.insert(gIdleTrees.end(), tree);
	gIdleWeight += tree->m_triangles + kTreeCacheEntryWeight;
	while (gIdleWeight > kTreeCacheIdleWeight)
	{
		CachedTree* drop = gIdleTrees.front();
		gIdleTrees.pop_front();
		gIdleWeight -= drop->m_triangles + kTreeCacheEntryWeight;
		gTreeCache.erase(drop->m_entry);
		rtcReleaseGeometry(drop->m_geometry);
		rtcReleaseScene(drop->m_scene);
		delete drop;
	}
}

// Rotation part of the body transform, row-major, for turning a model normal into world space.
void copyRotation(const btTransform& worldTransform, float out[9])
{
	const btMatrix3x3& basis = worldTransform.getBasis();
	for (int r = 0; r < 3; r++)
		for (int c = 0; c < 3; c++)
			out[r * 3 + c] = (float)basis[r][c];
}

unsigned long long fnv1a(const void* data, size_t len, unsigned long long hash)
{
	const unsigned char* bytes = (const unsigned char*)data;
	for (size_t i = 0; i < len; i++)
	{
		hash ^= bytes[i];
		hash *= 1099511628211ULL;
	}
	return hash;
}

// Key of a world tree: the mesh content hash and the exact floats copyWorldVertices takes.
unsigned long long treeCacheKey(TinyRender::Model* model, const btTransform& worldTransform, const btVector3& localScaling)
{
	ATTRIBUTE_ALIGNED16(btScalar gl[16]);
	worldTransform.getOpenGLMatrix(gl);
	float pose[19];
	for (int i = 0; i < 16; i++)
		pose[i] = (float)gl[i];
	for (int i = 0; i < 3; i++)
		pose[16 + i] = (float)localScaling[i];
	const unsigned long long meshHash = model->meshHash();
	unsigned long long hash = fnv1a(kTreeCacheMagic, 8, 14695981039346656037ULL);
	hash = fnv1a(&meshHash, sizeof(meshHash), hash);
	return fnv1a(pose, sizeof(pose), hash);
}

bool readBlock(FILE* f, std::vector<float>& out, size_t count, size_t padding)
{
	out.assign(count + padding, 0.0f);
	return count == 0 || fread(&out[0], sizeof(float), count, f) == count;
}

// Fills the tree's blocks and the Embree image from the cache file when it carries this key and
// this mesh's counts; false leaves the caller to build the blocks itself.
bool readTreeCacheFile(const char* path, unsigned long long key, TinyRender::Model* model, MeshTree& tree, std::vector<char>& image)
{
	FILE* f = fopen(path, "rb");
	if (!f)
		return false;
	TreeCacheHeader header;
	bool ok = fread(&header, sizeof(header), 1, f) == 1 && memcmp(header.m_magic, kTreeCacheMagic, 8) == 0 &&
			  header.m_key == key && header.m_numVertices == (unsigned)model->nverts() && header.m_numTriangles == (unsigned)model->nfaces() &&
			  header.m_hasNormals == (unsigned)(model->nnormals() > 0) && header.m_treeBytes > 0;
	if (ok)
	{
		const size_t numVertices = header.m_numVertices, numTriangles = header.m_numTriangles;
		tree.m_indices.assign(numTriangles * 3, 0);
		image.resize((size_t)header.m_treeBytes);
		ok = readBlock(f, tree.m_vertices, numVertices * 3, kVertexPadding) &&
			 fread(&tree.m_indices[0], sizeof(unsigned), numTriangles * 3, f) == numTriangles * 3 &&
			 readBlock(f, tree.m_normals, header.m_hasNormals ? numVertices * 3 : 0, 0) &&
			 readBlock(f, tree.m_uvs, numVertices * 2, 0) &&
			 fread(&image[0], 1, image.size(), f) == image.size();
	}
	fclose(f);
	if (!ok)
		image.clear();
	return ok;
}

// Writes the tree's blocks and Embree's image of its committed scene; a scene whose tree the image
// format does not cover writes nothing. The file lands through a rename so a reader never sees a half.
void writeTreeCacheFile(const char* path, unsigned long long key, const MeshTree& tree)
{
	// A built tree takes about 70 bytes per triangle; a roomy first guess saves a second walk.
	std::vector<char> image((tree.m_indices.size() / 3) * 96 + 4096);
	size_t treeBytes = rtcSwarmSaveTree(tree.m_scene, &image[0], image.size());
	if (treeBytes > image.size())
	{
		image.resize(treeBytes);
		treeBytes = rtcSwarmSaveTree(tree.m_scene, &image[0], image.size());
	}
	if (treeBytes == 0 || treeBytes > image.size())
		return;
	image.resize(treeBytes);
	TreeCacheHeader header;
	memset(&header, 0, sizeof(header));
	memcpy(header.m_magic, kTreeCacheMagic, 8);
	header.m_key = key;
	header.m_numVertices = (unsigned)((tree.m_vertices.size() - kVertexPadding) / 3);
	header.m_numTriangles = (unsigned)(tree.m_indices.size() / 3);
	header.m_hasNormals = tree.m_normals.empty() ? 0 : 1;
	header.m_treeBytes = treeBytes;
	char tmpPath[1200];
	snprintf(tmpPath, sizeof(tmpPath), "%s.%d.tmp", path, (int)getpid());
	FILE* f = fopen(tmpPath, "wb");
	if (!f)
		return;
	const size_t numFloats = (size_t)header.m_numVertices * 3;
	bool ok = fwrite(&header, sizeof(header), 1, f) == 1 &&
			  fwrite(&tree.m_vertices[0], sizeof(float), numFloats, f) == numFloats &&
			  fwrite(&tree.m_indices[0], sizeof(unsigned), tree.m_indices.size(), f) == tree.m_indices.size() &&
			  (tree.m_normals.empty() || fwrite(&tree.m_normals[0], sizeof(float), tree.m_normals.size(), f) == tree.m_normals.size()) &&
			  fwrite(&tree.m_uvs[0], sizeof(float), tree.m_uvs.size(), f) == tree.m_uvs.size() &&
			  fwrite(&image[0], 1, image.size(), f) == image.size();
	ok = (fclose(f) == 0) && ok;
	if (!ok || rename(tmpPath, path) != 0)
		remove(tmpPath);
}

// 4x4 helpers in double precision: the camera setup runs once per frame, exactness matters more than speed.
void glToRows(const float gl[16], double m[4][4])
{
	for (int r = 0; r < 4; r++)
		for (int c = 0; c < 4; c++)
			m[r][c] = gl[c * 4 + r];
}

void multiply(const double a[4][4], const double b[4][4], double out[4][4])
{
	for (int r = 0; r < 4; r++)
		for (int c = 0; c < 4; c++)
		{
			double sum = 0.0;
			for (int k = 0; k < 4; k++)
				sum += a[r][k] * b[k][c];
			out[r][c] = sum;
		}
}

bool invert(const double in[4][4], double out[4][4])
{
	double a[4][8];
	for (int r = 0; r < 4; r++)
	{
		for (int c = 0; c < 4; c++)
			a[r][c] = in[r][c];
		for (int c = 0; c < 4; c++)
			a[r][4 + c] = (r == c) ? 1.0 : 0.0;
	}
	for (int col = 0; col < 4; col++)
	{
		int pivot = col;
		for (int r = col + 1; r < 4; r++)
			if (fabs(a[r][col]) > fabs(a[pivot][col]))
				pivot = r;
		if (a[pivot][col] == 0.0)
			return false;
		if (pivot != col)
			for (int k = 0; k < 8; k++)
			{
				double tmp = a[pivot][k];
				a[pivot][k] = a[col][k];
				a[col][k] = tmp;
			}
		const double divisor = a[col][col];
		for (int k = 0; k < 8; k++)
			a[col][k] /= divisor;
		for (int r = 0; r < 4; r++)
		{
			if (r == col)
				continue;
			const double factor = a[r][col];
			for (int k = 0; k < 8; k++)
				a[r][k] -= factor * a[col][k];
		}
	}
	for (int r = 0; r < 4; r++)
		for (int c = 0; c < 4; c++)
			out[r][c] = a[r][4 + c];
	return true;
}

struct Camera
{
	// The near and far planes map affinely from pixel position to world space, so each is an origin
	// plus two axis vectors; the double math runs once per frame instead of once per pixel.
	double m_near[3][3];
	double m_far[3][3];
	// Projection times view, for putting a world point back onto the frame in the edge pass.
	double m_viewProj[4][4];
	float m_origin[3];
	float m_viewRow2[4];
	float m_p22;
	float m_p23;
};

// World point on the plane ndc_z for the pixel position (ndc_x, ndc_y).
void unproject(const double invViewProj[4][4], double ndcX, double ndcY, double ndcZ, double out[3])
{
	double p[4];
	for (int r = 0; r < 4; r++)
		p[r] = invViewProj[r][0] * ndcX + invViewProj[r][1] * ndcY + invViewProj[r][2] * ndcZ + invViewProj[r][3];
	for (int i = 0; i < 3; i++)
		out[i] = p[i] / p[3];
}

void planeBasis(const double invViewProj[4][4], double ndcZ, double basis[3][3])
{
	double x[3], y[3];
	unproject(invViewProj, 0.0, 0.0, ndcZ, basis[0]);
	unproject(invViewProj, 1.0, 0.0, ndcZ, x);
	unproject(invViewProj, 0.0, 1.0, ndcZ, y);
	for (int i = 0; i < 3; i++)
	{
		basis[1][i] = x[i] - basis[0][i];
		basis[2][i] = y[i] - basis[0][i];
	}
}

void planePoint(const double basis[3][3], double ndcX, double ndcY, float out[3])
{
	for (int i = 0; i < 3; i++)
		out[i] = (float)(basis[0][i] + basis[1][i] * ndcX + basis[2][i] * ndcY);
}

bool setupCamera(const float viewMat[16], const float projMat[16], Camera& cam)
{
	double view[4][4], proj[4][4], invView[4][4], invProj[4][4], invViewProj[4][4];
	glToRows(viewMat, view);
	glToRows(projMat, proj);
	if (!invert(view, invView) || !invert(proj, invProj))
		return false;
	multiply(invView, invProj, invViewProj);
	multiply(proj, view, cam.m_viewProj);
	planeBasis(invViewProj, -1.0, cam.m_near);
	planeBasis(invViewProj, 1.0, cam.m_far);
	for (int i = 0; i < 3; i++)
		cam.m_origin[i] = (float)(invView[i][3] / invView[3][3]);
	for (int c = 0; c < 4; c++)
		cam.m_viewRow2[c] = viewMat[c * 4 + 2];
	cam.m_p22 = projMat[10];
	cam.m_p23 = projMat[14];
	return true;
}

// Where a lens's first rays landed on its last frame, in world space (x, y, z per pixel, NaN for a miss). The next frame
// from the same lens puts them back onto its own pixels to know about how far each ray has to search.
struct HitMemory
{
	int m_width;
	int m_height;
	float m_proj[16];
	std::vector<float> m_points;
	unsigned long long m_lastUse;
	// Frame reuse: the key of the lens's last request and, once a request repeated the one before, that frame's buffers
	// before the camera chain, the scene revision and move they were drawn at, and the region they depend on: the
	// pyramid from the eye to the depth its rays reached, swept towards the light when hits cast shadows.
	std::vector<unsigned char> m_key;
	bool m_kept;
	unsigned long long m_sceneRevision;
	unsigned long long m_movedFrom;
	double m_apex[3];
	double m_base[4][3];
	double m_light[3];
	bool m_sweep;
	std::vector<float> m_depth;
	std::vector<int> m_seg;
	std::vector<unsigned char> m_rgb;
	std::vector<float> m_radiance;
};

// Frame reuse: a kept frame is dropped once this many mover boxes wait to be tested against it.
const size_t kReuseMaxBoxes = 1 << 16;

// True unless an axis separates the box (lo, hi) from the region a kept frame depends on: the pyramid of its apex and
// base corners, swept along the light without end when the frame casts shadows. The axes are the faces and edge
// crossings of both shapes, so a box the region misses is told apart; a NaN anywhere answers true.
bool reachesFrame(const HitMemory& frame, const double lo[3], const double hi[3])
{
	double edges[7][3];
	for (int k = 0; k < 4; k++)
		for (int i = 0; i < 3; i++)
			edges[k][i] = frame.m_base[k][i] - frame.m_apex[i];
	for (int i = 0; i < 3; i++)
	{
		edges[4][i] = frame.m_base[1][i] - frame.m_base[0][i];
		edges[5][i] = frame.m_base[3][i] - frame.m_base[0][i];
		edges[6][i] = frame.m_light[i];
	}
	const int numEdges = frame.m_sweep ? 7 : 6;
	const double unit[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
	double axes[48][3];
	int numAxes = 0;
	for (int k = 0; k < 3; k++)
		memcpy(axes[numAxes++], unit[k], sizeof(axes[0]));
	const int faces[5][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}};
	for (int f = 0; f < 5; f++)
	{
		const double* a = edges[faces[f][0]];
		const double* b = edges[faces[f][1]];
		double* n = axes[numAxes++];
		n[0] = a[1] * b[2] - a[2] * b[1];
		n[1] = a[2] * b[0] - a[0] * b[2];
		n[2] = a[0] * b[1] - a[1] * b[0];
	}
	for (int e = 0; e < numEdges; e++)
		for (int k = 0; k < 3 + (frame.m_sweep && e < 6 ? 1 : 0); k++)
		{
			const double* a = k < 3 ? unit[k] : edges[6];
			const double* b = edges[e];
			double* n = axes[numAxes++];
			n[0] = a[1] * b[2] - a[2] * b[1];
			n[1] = a[2] * b[0] - a[0] * b[2];
			n[2] = a[0] * b[1] - a[1] * b[0];
		}
	double centre[3], half[3], size = 0.0;
	for (int i = 0; i < 3; i++)
	{
		centre[i] = 0.5 * (lo[i] + hi[i]);
		half[i] = 0.5 * fabs(hi[i] - lo[i]);
		size = fabs(lo[i]) > size ? fabs(lo[i]) : size;
		size = fabs(hi[i]) > size ? fabs(hi[i]) : size;
		size = fabs(frame.m_apex[i]) > size ? fabs(frame.m_apex[i]) : size;
	}
	// Wider than the rounding of the rays, the boxes and these sums, as the mover grid allows for.
	const double margin = 0.05 + size * 1e-4;
	for (int a = 0; a < numAxes; a++)
	{
		double* n = axes[a];
		const double length = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
		if (!(length > 1e-12))
			continue;
		for (int i = 0; i < 3; i++)
			n[i] /= length;
		double low = n[0] * frame.m_apex[0] + n[1] * frame.m_apex[1] + n[2] * frame.m_apex[2], high = low;
		for (int k = 0; k < 4; k++)
		{
			const double d = n[0] * frame.m_base[k][0] + n[1] * frame.m_base[k][1] + n[2] * frame.m_base[k][2];
			low = d < low ? d : low;
			high = d > high ? d : high;
		}
		if (frame.m_sweep)
		{
			// An axis across the light keeps the pyramid's extent; any other is open on the light's side.
			const double along = n[0] * frame.m_light[0] + n[1] * frame.m_light[1] + n[2] * frame.m_light[2];
			if (along > 1e-9)
				high = INFINITY;
			else if (along < -1e-9)
				low = -INFINITY;
		}
		const double mid = n[0] * centre[0] + n[1] * centre[1] + n[2] * centre[2];
		const double reach = fabs(n[0]) * half[0] + fabs(n[1]) * half[1] + fabs(n[2]) * half[2] + margin;
		if (mid + reach < low || mid - reach > high)
			return false;
	}
	return true;
}

// The movers seen from the light: a grid across the light's direction whose cells are set where a mover's box, widened
// for rounding, lies. A shadow ray runs along the light, so one leaving a point under a clear cell cannot meet a mover.
struct MoverShade
{
	float m_axisU[3];
	float m_axisV[3];
	float m_u0;
	float m_v0;
	float m_perCell;
	int m_cols;
	int m_rows;
	std::vector<unsigned char> m_cells;
};

// Lenses remembered at once (the wide feed, the zoom, the thermal camera), and the smallest frame that keeps a memory:
// the laser's few pixels gain nothing from one.
const int kHitMemories = 4;
const size_t kHintMinPixels = 64 * 64;
}  // namespace

struct SwarmRaycast::Data
{
	RTCDevice m_device;
	RTCScene m_top;
	RTCScene m_static;
	RTCScene m_staticShadows;  // Allocated only when a flagged placement needs the shadow map.
	bool m_staticShadowsDirty;
	std::map<unsigned long long, MeshTree*> m_sharedTrees;
	// The forest: every batch's instance array, reached from the top and shadow scenes through one instance.
	RTCScene m_forest;
	RTCGeometry m_forestInstance;
	unsigned m_forestId;
	bool m_forestDirty;
	std::vector<Batch*> m_batches;
	// Only the mover instances, so a shadow ray from a lit hit never walks the static tree.
	RTCScene m_movers;
	RTCGeometry m_staticInstance;
	unsigned m_staticInstanceId;
	bool m_staticBuilt;
	bool m_topDirty;
	bool m_moversDirty;
	std::vector<StaticMember*> m_members;
	std::map<const void*, MeshTree*> m_trees;
	std::vector<Instance*> m_byGeomId;
	std::vector<unsigned> m_freeGeomIds;
	std::map<TinyRenderObjectData*, ObjectState> m_objects;
	// SWARM_BVH_CACHE_DIR; empty when world trees are never written or read.
	std::string m_cacheDir;
	ShadowMap m_shadowMap;
	// A second, finer grid over the square of m_coreRadius about the world origin, for ER_SWARM_DAYLIGHT.
	ShadowMap m_shadowCore;
	float m_coreRadius;
	// Static members whose shadow changed since the map was cast: retired, hidden or shown again.
	std::vector<StaticMember*> m_shadowDirty;
	// ER_SWARM_THERMAL: the static bodies seen from straight above, so a point knows how much open sky it has.
	ShadowMap m_shelterMap;
	std::vector<StaticMember*> m_shelterDirty;
	// Mover instances attached to the mover scene, so a frame with none skips their shadow ray.
	int m_moverCount;
	HitMemory m_hitMemories[kHitMemories];
	unsigned long long m_frameCount;
	// Per pixel of the current frame: the depth along the camera's axis of the farthest remembered hit landing on it, -1
	// where none does.
	std::vector<float> m_hintFar;
	// The same for the share of the remembered hits each further render thread puts onto the frame.
	std::vector<std::vector<float> > m_hintFrames;
	MoverShade m_moverShade;
	// Frame reuse: counts every change to the scene or its shadow grids but a mover's move; each move instead leaves the
	// mover's world box before and after it (lo, hi), m_movedBase counting the boxes dropped from the front.
	unsigned long long m_sceneRevision;
	std::vector<double> m_moved;
	unsigned long long m_movedBase;

	// Notes a mover's move for the kept frames: the box of its tree under the old transform and under the new one. Boxes
	// every kept frame has already seen are dropped; too many waiting drop the kept frames instead.
	void moverMoved(const MeshTree* tree, const float before[16], const float after[16])
	{
		size_t from = m_moved.size() / 6;
		bool kept = false;
		for (int i = 0; i < kHitMemories; i++)
		{
			const HitMemory& memory = m_hitMemories[i];
			if (!memory.m_kept || memory.m_sceneRevision != m_sceneRevision || memory.m_movedFrom < m_movedBase)
				continue;
			kept = true;
			from = memory.m_movedFrom - m_movedBase < from ? (size_t)(memory.m_movedFrom - m_movedBase) : from;
		}
		m_moved.erase(m_moved.begin(), m_moved.begin() + from * 6);
		m_movedBase += from;
		if (kept && m_moved.size() / 6 + 2 > kReuseMaxBoxes)
		{
			m_sceneRevision++;
			m_movedBase += m_moved.size() / 6;
			m_moved.clear();
			kept = false;
		}
		if (!kept)
			return;
		RTCBounds bounds;
		rtcGetSceneBounds(tree->m_scene, &bounds);
		const float* transforms[2] = {before, after};
		for (int t = 0; t < 2; t++)
		{
			const float* m = transforms[t];
			double lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
			for (int k = 0; k < 8; k++)
			{
				const double corner[3] = {k & 1 ? bounds.upper_x : bounds.lower_x, k & 2 ? bounds.upper_y : bounds.lower_y, k & 4 ? bounds.upper_z : bounds.lower_z};
				for (int i = 0; i < 3; i++)
				{
					const double w = (double)m[i] * corner[0] + (double)m[4 + i] * corner[1] + (double)m[8 + i] * corner[2] + (double)m[12 + i];
					// A NaN corner leaves its box NaN, which every kept frame counts as reached.
					lo[i] = w < lo[i] || w != w ? w : lo[i];
					hi[i] = w > hi[i] || w != w ? w : hi[i];
				}
			}
			m_moved.insert(m_moved.end(), lo, lo + 3);
			m_moved.insert(m_moved.end(), hi, hi + 3);
		}
	}

	// The kept frame of this memory, put into the camera's buffers, when the request and the scene since allow it.
	bool reuseFrame(HitMemory& memory, const std::vector<unsigned char>& key, const SwarmRaycast::Target& target, size_t numPixels)
	{
		if (!memory.m_kept || memory.m_key != key || memory.m_sceneRevision != m_sceneRevision || memory.m_movedFrom < m_movedBase)
			return false;
		for (size_t i = (size_t)(memory.m_movedFrom - m_movedBase) * 6; i < m_moved.size(); i += 6)
			if (reachesFrame(memory, &m_moved[i], &m_moved[i + 3]))
			{
				memory.m_kept = false;
				return false;
			}
		memcpy(target.m_depth, &memory.m_depth[0], numPixels * sizeof(float));
		if (target.m_seg)
			memcpy(target.m_seg, &memory.m_seg[0], numPixels * sizeof(int));
		if (target.m_rgb)
			memcpy(target.m_rgb, &memory.m_rgb[0], numPixels * 3);
		return true;
	}

	// Takes this request's key, and keeps the frame it drew when the request repeated the lens's last one: a camera that
	// stays still is asked for it again, a moving one never is. farthest is how far along their rays its rays went.
	void keepFrame(HitMemory& memory, std::vector<unsigned char>& key, const Camera& cam, const SwarmRaycast::Target& target,
				   const std::vector<float>& radiance, float farthest, const SwarmRaycastShading* shading, int width, int height)
	{
		const size_t numPixels = (size_t)width * height;
		const bool repeat = memory.m_key == key;
		memory.m_key.swap(key);
		memory.m_kept = false;
		if (!repeat || !(farthest >= 0.0f && farthest < INFINITY))
			return;
		// Every ray point lies within its distance along the ray of the eye, so no deeper than the farthest one went.
		// The sides lie a pixel out: an edge probe on the left column or the bottom row samples up to half a pixel past ndc -1.
		double corners[4][3], centre[3];
		const double sideX = 1.0 + 2.0 / width, sideY = 1.0 + 2.0 / height;
		const double ndc[4][2] = {{-sideX, -sideY}, {sideX, -sideY}, {sideX, sideY}, {-sideX, sideY}};
		for (int i = 0; i < 3; i++)
		{
			memory.m_apex[i] = cam.m_origin[i];
			centre[i] = cam.m_far[0][i];
			for (int k = 0; k < 4; k++)
				corners[k][i] = cam.m_far[0][i] + cam.m_far[1][i] * ndc[k][0] + cam.m_far[2][i] * ndc[k][1];
		}
		const double farDepth = -(cam.m_viewRow2[0] * centre[0] + cam.m_viewRow2[1] * centre[1] + cam.m_viewRow2[2] * centre[2] + cam.m_viewRow2[3]);
		double share = ((double)farthest * 1.001 + 0.05) / farDepth;
		share = share < 1.0 ? share : 1.0;
		bool finite = farDepth > 0.0 && share > 0.0;
		for (int k = 0; k < 4; k++)
			for (int i = 0; i < 3; i++)
			{
				memory.m_base[k][i] = memory.m_apex[i] + (corners[k][i] - memory.m_apex[i]) * share;
				finite = finite && fabs(memory.m_base[k][i]) < INFINITY;
			}
		memory.m_sweep = shading && shading->m_shadow;
		for (int i = 0; i < 3; i++)
		{
			memory.m_light[i] = shading ? shading->m_lightDir[i] : 0.0f;
			finite = finite && fabs(memory.m_apex[i]) < INFINITY && fabs(memory.m_light[i]) < INFINITY;
		}
		if (!finite)
			return;
		memory.m_depth.assign(target.m_depth, target.m_depth + numPixels);
		if (target.m_seg)
			memory.m_seg.assign(target.m_seg, target.m_seg + numPixels);
		if (target.m_rgb)
			memory.m_rgb.assign(target.m_rgb, target.m_rgb + numPixels * 3);
		memory.m_radiance = radiance;
		memory.m_sceneRevision = m_sceneRevision;
		memory.m_movedFrom = m_movedBase + m_moved.size() / 6;
		memory.m_kept = true;
	}

	// The memory of the lens with this frame size and projection, or the one used longest ago, emptied for it.
	HitMemory* hitMemory(int width, int height, const float projMat[16])
	{
		HitMemory* oldest = &m_hitMemories[0];
		for (int i = 0; i < kHitMemories; i++)
		{
			HitMemory* memory = &m_hitMemories[i];
			if (!memory->m_points.empty() && memory->m_width == width && memory->m_height == height &&
				memcmp(memory->m_proj, projMat, sizeof(memory->m_proj)) == 0)
			{
				memory->m_lastUse = ++m_frameCount;
				return memory;
			}
			if (memory->m_lastUse < oldest->m_lastUse)
				oldest = memory;
		}
		oldest->m_points.clear();
		oldest->m_key.clear();
		oldest->m_kept = false;
		oldest->m_width = width;
		oldest->m_height = height;
		memcpy(oldest->m_proj, projMat, sizeof(oldest->m_proj));
		oldest->m_lastUse = ++m_frameCount;
		return oldest;
	}

	void shadowChanged(StaticMember* member)
	{
		if (m_shadowMap.m_built)
			m_shadowDirty.push_back(member);
		if (m_shelterMap.m_built)
			m_shelterDirty.push_back(member);
	}

	// Points a map at the tree its blocks are cast against and at what the hit filter reads, as they are now.
	void bindShadowMap(ShadowMap& map)
	{
		map.m_scene = m_staticShadows ? m_staticShadows : m_static;
		map.m_instances = &m_byGeomId;
		map.m_batches = &m_batches;
		map.m_forestId = m_forestId;
	}

	// Lays the grid over `bounds` for this light, at most kShadowMapMaxCells a side; frames cast its blocks as they read them.
	void buildShadowMap(ShadowMap& map, const RTCBounds& bounds, const float lightDir[3], bool alphaCutout, bool leafNoShadow)
	{
		m_sceneRevision++;
		map.m_built = false;
		map.m_alphaCutout = alphaCutout;
		map.m_leafNoShadow = leafNoShadow;
		map.m_depth.reset();
		map.m_blocks.reset();
		if (!(bounds.lower_x <= bounds.upper_x) || !(bounds.lower_y <= bounds.upper_y) || !(bounds.lower_z <= bounds.upper_z))
			return;
		for (int i = 0; i < 3; i++)
			map.m_lightDir[i] = lightDir[i];
		// The grid axes come from the world axis furthest from the light, so they are never degenerate.
		const bool steep = fabsf(lightDir[2]) >= 0.9f;
		const float helper[3] = {steep ? 1.0f : 0.0f, 0.0f, steep ? 0.0f : 1.0f};
		cross3(helper, lightDir, map.m_axisU);
		normalize3(map.m_axisU);
		cross3(lightDir, map.m_axisU, map.m_axisV);
		normalize3(map.m_axisV);
		if (dot3(map.m_axisU, map.m_axisU) == 0.0f || dot3(map.m_axisV, map.m_axisV) == 0.0f)
			return;
		// Extent of the eight corners of the bounds along the two axes, and the far edge along the light.
		float uMin = INFINITY, uMax = -INFINITY, vMin = INFINITY, vMax = -INFINITY, lMax = -INFINITY;
		for (int c = 0; c < 8; c++)
		{
			const float corner[3] = {(c & 1) ? bounds.upper_x : bounds.lower_x,
									 (c & 2) ? bounds.upper_y : bounds.lower_y,
									 (c & 4) ? bounds.upper_z : bounds.lower_z};
			const float u = dot3(corner, map.m_axisU), v = dot3(corner, map.m_axisV), l = dot3(corner, lightDir);
			uMin = u < uMin ? u : uMin;
			uMax = u > uMax ? u : uMax;
			vMin = v < vMin ? v : vMin;
			vMax = v > vMax ? v : vMax;
			lMax = l > lMax ? l : lMax;
		}
		const float extent = (uMax - uMin) > (vMax - vMin) ? (uMax - uMin) : (vMax - vMin);
		float cell = extent / (float)kShadowMapMaxCells;
		if (cell < kShadowMapMinCell)
			cell = kShadowMapMinCell;
		map.m_cell = cell;
		// One spare cell of margin on every side; the rays start one cell beyond the light-side edge.
		map.m_cols = (int)((uMax - uMin) / cell) + 3;
		map.m_rows = (int)((vMax - vMin) / cell) + 3;
		for (int i = 0; i < 3; i++)
			map.m_origin[i] = (map.m_axisU[i] * (uMin - cell) + map.m_axisV[i] * (vMin - cell)) + lightDir[i] * (lMax + cell);
		map.m_depth.reset(new float[(size_t)map.m_cols * map.m_rows]);
		map.m_blockCols = (map.m_cols + kShadowBlock - 1) / kShadowBlock;
		map.m_blocks.reset(new std::atomic<unsigned char>[(size_t)map.m_blockCols * ((map.m_rows + kShadowBlock - 1) / kShadowBlock)]());
		bindShadowMap(map);
		map.m_built = true;
	}

	// Recasts the cells of one map under one member's vertices, one cell wider on every side. Only blocks already cast
	// are recast; the others are cast as the tree is now when a frame first reads them.
	void recastMember(ShadowMap& map, const StaticMember* member, int threads)
	{
		if (!map.m_built)
			return;
		float uMin = INFINITY, uMax = -INFINITY, vMin = INFINITY, vMax = -INFINITY;
		const size_t numVerts = (member->m_vertices.size() - kVertexPadding) / 3;
		for (size_t i = 0; i < numVerts; i++)
		{
			float rel[3];
			for (int k = 0; k < 3; k++)
				rel[k] = member->m_vertices[i * 3 + k] - map.m_origin[k];
			const float u = dot3(rel, map.m_axisU) / map.m_cell, v = dot3(rel, map.m_axisV) / map.m_cell;
			uMin = u < uMin ? u : uMin;
			uMax = u > uMax ? u : uMax;
			vMin = v < vMin ? v : vMin;
			vMax = v > vMax ? v : vMax;
		}
		if (!(uMin <= uMax) || !(vMin <= vMax))
			return;
		const int col0 = uMin < 1.0f ? 0 : (int)uMin - 1;
		const int row0 = vMin < 1.0f ? 0 : (int)vMin - 1;
		const int col1 = uMax + 2.0f >= (float)map.m_cols ? map.m_cols : (int)uMax + 2;
		const int row1 = vMax + 2.0f >= (float)map.m_rows ? map.m_rows : (int)vMax + 2;
		if (!(col0 < col1 && row0 < row1))
			return;
#pragma omp parallel for num_threads(threads) schedule(static)
		for (int row = row0; row < row1; row++)
			for (int blockCol = col0 / kShadowBlock; blockCol <= (col1 - 1) / kShadowBlock; blockCol++)
				if (map.m_blocks[(size_t)(row / kShadowBlock) * map.m_blockCols + blockCol].load(std::memory_order_relaxed) == 2)
				{
					const int c0 = blockCol * kShadowBlock > col0 ? blockCol * kShadowBlock : col0;
					const int c1 = (blockCol + 1) * kShadowBlock < col1 ? (blockCol + 1) * kShadowBlock : col1;
					castShadowCells(map, c0, c1, row, row + 1);
				}
	}

	// Recasts both maps for a new light, core or cut-out setting, otherwise only the cells under the members that changed.
	void prepareShadowMap(const float lightDir[3], bool alphaCutout, bool leafNoShadow, float coreRadius, int upAxis, int threads)
	{
		const bool stale = !m_shadowMap.m_built || m_shadowMap.m_alphaCutout != alphaCutout || m_shadowMap.m_leafNoShadow != leafNoShadow ||
						   memcmp(m_shadowMap.m_lightDir, lightDir, sizeof(m_shadowMap.m_lightDir)) != 0;
		const bool coreStale = stale || coreRadius != m_coreRadius || (coreRadius > 0.0f) != m_shadowCore.m_built;
		RTCBounds bounds;
		rtcGetSceneBounds(m_staticShadows ? m_staticShadows : m_static, &bounds);
		bindShadowMap(m_shadowMap);
		bindShadowMap(m_shadowCore);
		if (stale)
			buildShadowMap(m_shadowMap, bounds, lightDir, alphaCutout, leafNoShadow);
		else
			for (size_t i = 0; i < m_shadowDirty.size(); i++)
			{
				recastMember(m_shadowMap, m_shadowDirty[i], threads);
				if (!coreStale)
					recastMember(m_shadowCore, m_shadowDirty[i], threads);
			}
		m_shadowDirty.clear();
		if (!coreStale)
			return;
		m_sceneRevision++;
		m_coreRadius = coreRadius;
		m_shadowCore.m_built = false;
		m_shadowCore.m_depth.reset();
		m_shadowCore.m_blocks.reset();
		if (coreRadius > 0.0f && bounds.lower_x <= bounds.upper_x)
		{
			// The core is the square about the origin across the two level axes; the up axis keeps the scene's full height.
			float lower[3] = {bounds.lower_x, bounds.lower_y, bounds.lower_z};
			float upper[3] = {bounds.upper_x, bounds.upper_y, bounds.upper_z};
			for (int axis = 0; axis < 3; axis++)
			{
				if (axis == upAxis)
					continue;
				lower[axis] = lower[axis] > -coreRadius ? lower[axis] : -coreRadius;
				upper[axis] = upper[axis] < coreRadius ? upper[axis] : coreRadius;
			}
			RTCBounds core;
			core.lower_x = lower[0];
			core.lower_y = lower[1];
			core.lower_z = lower[2];
			core.upper_x = upper[0];
			core.upper_y = upper[1];
			core.upper_z = upper[2];
			buildShadowMap(m_shadowCore, core, lightDir, alphaCutout, leafNoShadow);
		}
	}

	// The shelter map is the shadow map of a light straight overhead; it is cast once per world and then only under
	// the members that changed, since the sky does not move. Leaf cards shelter what is under them.
	void prepareShelterMap(int upAxis, bool alphaCutout, int threads)
	{
		float up[3] = {0.0f, 0.0f, 0.0f};
		up[upAxis] = 1.0f;
		bindShadowMap(m_shelterMap);
		if (!m_shelterMap.m_built || m_shelterMap.m_alphaCutout != alphaCutout || memcmp(m_shelterMap.m_lightDir, up, sizeof(up)) != 0)
		{
			RTCBounds bounds;
			rtcGetSceneBounds(m_staticShadows ? m_staticShadows : m_static, &bounds);
			buildShadowMap(m_shelterMap, bounds, up, alphaCutout, false);
		}
		else
			for (size_t i = 0; i < m_shelterDirty.size(); i++)
				recastMember(m_shelterMap, m_shelterDirty[i], threads);
		m_shelterDirty.clear();
	}

	void createStaticScene()
	{
		m_static = rtcNewScene(m_device);
		rtcSetSceneFlags(m_static, RTC_SCENE_FLAG_ROBUST);
		rtcSetSceneBuildQuality(m_static, RTC_BUILD_QUALITY_MEDIUM);
		m_staticInstance = rtcNewGeometry(m_device, RTC_GEOMETRY_TYPE_INSTANCE);
		rtcSetGeometryInstancedScene(m_staticInstance, m_static);
		const float identity[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
		rtcSetGeometryTransform(m_staticInstance, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, identity);
		rtcCommitGeometry(m_staticInstance);
		m_staticInstanceId = allocateGeomId(0);
		rtcAttachGeometryByID(m_top, m_staticInstance, m_staticInstanceId);
		m_staticBuilt = false;
		m_topDirty = true;
	}

	unsigned allocateGeomId(Instance* inst)
	{
		unsigned id;
		if (m_freeGeomIds.empty())
		{
			id = (unsigned)m_byGeomId.size();
			m_byGeomId.push_back(inst);
		}
		else
		{
			id = m_freeGeomIds.back();
			m_freeGeomIds.pop_back();
			m_byGeomId[id] = inst;
		}
		return id;
	}

	StaticMember* addMember(TinyRenderObjectData* obj, const btTransform& worldTransform, const btVector3& localScaling, const float transform[16])
	{
		StaticMember* member = new StaticMember;
		member->m_obj = obj;
		member->m_meshKey = obj->m_model->meshKey();
		member->m_textureRevision = obj->m_textureRevision;
		memcpy(member->m_transform, transform, sizeof(member->m_transform));
		copyRotation(worldTransform, member->m_rotation);
		member->m_retired = false;
		copyWorldVertices(obj->m_model, worldTransform, localScaling, member->m_vertices);
		copyIndices(obj->m_model, member->m_indices);
		// The model's uv array is already indexed by vertex for every mesh the plugin builds, so it is read, not copied.
		const bool borrowUvs = obj->m_model->uvArray() != 0 && obj->m_model->uvIndexedByVertex();
		copyAttributes(obj->m_model, member->m_indices, member->m_rotation, member->m_normals, member->m_uvsOwned, !borrowUvs);
		member->m_uvs = borrowUvs ? obj->m_model->uvArray() : (member->m_uvsOwned.empty() ? 0 : &member->m_uvsOwned[0]);
		member->m_geometry = newTriangles(m_device, member->m_vertices, member->m_indices);
		rtcSetGeometryUserData(member->m_geometry, member);
		rtcCommitGeometry(member->m_geometry);
		member->m_geomId = (unsigned)m_members.size();
		rtcAttachGeometryByID(m_static, member->m_geometry, member->m_geomId);
		m_members.push_back(member);
		return member;
	}

	MeshTree* acquireTree(TinyRender::Model* model, const void* key)
	{
		std::map<const void*, MeshTree*>::iterator found = m_trees.find(key);
		if (found != m_trees.end())
		{
			found->second->m_refs++;
			return found->second;
		}
		MeshTree* tree = new MeshTree;
		tree->m_refs = 1;
		tree->m_dirty = false;
		tree->m_world = false;
		tree->m_refitting = false;
		copyLocalVertices(model, tree->m_vertices);
		copyIndices(model, tree->m_indices);
		copyAttributes(model, tree->m_indices, 0, tree->m_normals, tree->m_uvs, true);
		buildTree(*tree, 0);
		m_trees[key] = tree;
		return tree;
	}

	// Commits the tree's scene over its blocks; a loaded Embree image replaces the build. A mesh tree is taken from the
	// process cache, so a mesh another world already drew costs no build.
	void buildTree(MeshTree& tree, const std::vector<char>* image)
	{
		tree.m_cached = 0;
		if (!tree.m_world)
		{
			tree.m_cached = acquireCachedTree(m_device, tree.m_vertices, tree.m_indices);
			tree.m_scene = tree.m_cached->m_scene;
			tree.m_geometry = tree.m_cached->m_geometry;
			return;
		}
		tree.m_scene = rtcNewScene(m_device);
		rtcSetSceneFlags(tree.m_scene, RTC_SCENE_FLAG_ROBUST);
		rtcSetSceneBuildQuality(tree.m_scene, RTC_BUILD_QUALITY_MEDIUM);
		tree.m_geometry = newTriangles(m_device, tree.m_vertices, tree.m_indices);
		rtcCommitGeometry(tree.m_geometry);
		rtcAttachGeometry(tree.m_scene, tree.m_geometry);
		if (image && !image->empty())
			rtcSwarmLoadTree(tree.m_scene, &(*image)[0], image->size());
		rtcCommitScene(tree.m_scene);
	}

	// A tree of one static body's triangles in world space, read from the cache folder when a file
	// carries this mesh at this pose, otherwise built here and written for the next process.
	MeshTree* acquireWorldTree(TinyRender::Model* model, const btTransform& worldTransform, const btVector3& localScaling, const float transform[16])
	{
		MeshTree* tree = new MeshTree;
		tree->m_refs = 1;
		tree->m_dirty = false;
		tree->m_world = true;
		tree->m_refitting = false;
		memcpy(tree->m_pose, transform, sizeof(tree->m_pose));
		const unsigned long long key = treeCacheKey(model, worldTransform, localScaling);
		char path[1024];
		snprintf(path, sizeof(path), "%s/%016llx.rtree", m_cacheDir.c_str(), key);
		std::vector<char> image;
		const bool loaded = readTreeCacheFile(path, key, model, *tree, image);
		if (!loaded)
		{
			float rotation[9];
			copyRotation(worldTransform, rotation);
			copyWorldVertices(model, worldTransform, localScaling, tree->m_vertices);
			copyIndices(model, tree->m_indices);
			copyAttributes(model, tree->m_indices, rotation, tree->m_normals, tree->m_uvs, true);
		}
		buildTree(*tree, &image);
		if (!loaded)
			writeTreeCacheFile(path, key, *tree);
		return tree;
	}

	// Content hashes share canonical trees even when model storage sharing is disabled.
	MeshTree* acquireSharedTree(TinyRender::Model* model, bool deformed)
	{
		const unsigned long long hash = deformed ? 0 : model->meshHash();
		std::map<unsigned long long, MeshTree*>::iterator found = m_sharedTrees.find(hash);
		// A hash match must also match in size, so a collision builds its own tree instead of borrowing one.
		if (hash && found != m_sharedTrees.end() && found->second->m_indices.size() == (size_t)model->nfaces() * 3 &&
			found->second->m_vertices.size() == (size_t)model->nverts() * 3 + kVertexPadding)
		{
			found->second->m_refs++;
			return found->second;
		}
		MeshTree* tree = new MeshTree;
		tree->m_refs = 1;
		tree->m_dirty = false;
		tree->m_world = false;
		tree->m_refitting = false;
		copyLocalVertices(model, tree->m_vertices);
		copyIndices(model, tree->m_indices);
		copyAttributes(model, tree->m_indices, 0, tree->m_normals, tree->m_uvs, true, false);
		buildTree(*tree, 0);
		if (hash)
			m_sharedTrees[hash] = tree;
		return tree;
	}

	void releaseTree(MeshTree* tree)
	{
		if (!tree || --tree->m_refs > 0)
			return;
		for (std::map<const void*, MeshTree*>::iterator it = m_trees.begin(); it != m_trees.end(); ++it)
			if (it->second == tree)
			{
				m_trees.erase(it);
				break;
			}
		for (std::map<unsigned long long, MeshTree*>::iterator it = m_sharedTrees.begin(); it != m_sharedTrees.end(); ++it)
			if (it->second == tree)
			{
				m_sharedTrees.erase(it);
				break;
			}
		if (tree->m_cached)
			releaseCachedTree(tree->m_cached);
		else
		{
			rtcReleaseGeometry(tree->m_geometry);
			rtcReleaseScene(tree->m_scene);
		}
		delete tree;
	}

	void refitTree(TinyRender::Model* model, MeshTree& tree)
	{
		if ((size_t)model->nverts() * 3 + kVertexPadding != tree.m_vertices.size())
			return;
		copyLocalVertices(model, tree.m_vertices);
		copyAttributes(model, tree.m_indices, 0, tree.m_normals, tree.m_uvs, true);
		// A medium-quality scene ignores the geometry's refit quality and rebuilds its whole tree on every commit,
		// so the first rewrite moves the mesh into a low-quality scene, whose two-level builder refits it in place.
		if (!tree.m_refitting)
		{
			// A cached tree is never rewritten: the mesh leaves it for a scene of its own.
			if (tree.m_cached)
				releaseCachedTree(tree.m_cached);
			else
			{
				rtcReleaseGeometry(tree.m_geometry);
				rtcReleaseScene(tree.m_scene);
			}
			tree.m_cached = 0;
			tree.m_scene = rtcNewScene(m_device);
			rtcSetSceneFlags(tree.m_scene, RTC_SCENE_FLAG_ROBUST | RTC_SCENE_FLAG_DYNAMIC);
			rtcSetSceneBuildQuality(tree.m_scene, RTC_BUILD_QUALITY_LOW);
			tree.m_geometry = newTriangles(m_device, tree.m_vertices, tree.m_indices);
			rtcSetGeometryBuildQuality(tree.m_geometry, RTC_BUILD_QUALITY_REFIT);
			rtcAttachGeometry(tree.m_scene, tree.m_geometry);
			tree.m_refitting = true;
		}
		else
			rtcUpdateGeometryBuffer(tree.m_geometry, RTC_BUFFER_TYPE_VERTEX, 0);
		rtcCommitGeometry(tree.m_geometry);
		rtcCommitScene(tree.m_scene);
		tree.m_dirty = false;
	}

	// worldTree asks for a fresh instance to draw its world tree from disk; the instance stays on
	// that tree while the body keeps its pose and mesh, and carries on as a plain mover otherwise.
	// True when the instance at most moved, and the move is noted for the kept frames.
	bool syncInstance(ObjectState& state, TinyRenderObjectData* obj, const btTransform& worldTransform, const btVector3& localScaling, const float transform[16], bool enabled, bool doubleSided, bool hasAlpha, bool glass, int segmentation, bool worldTree)
	{
		TinyRender::Model* model = obj->m_model;
		Instance* inst = state.m_instance;
		bool fresh = false;
		if (!inst)
		{
			inst = new Instance;
			memset(inst, 0, sizeof(Instance));
			inst->m_obj = obj;
			inst->m_enabled = true;
			inst->m_facingSign = 1.0f;
			inst->m_geometry = rtcNewGeometry(m_device, RTC_GEOMETRY_TYPE_INSTANCE);
			inst->m_shadowGeometry = rtcNewGeometry(m_device, RTC_GEOMETRY_TYPE_INSTANCE);
			state.m_instance = inst;
			fresh = true;
		}
		bool changed = fresh;
		const void* key = model->meshKey();
		if (inst->m_tree && inst->m_tree->m_world && (inst->m_meshKey != key || inst->m_tree->m_dirty || memcmp(transform, inst->m_tree->m_pose, sizeof(inst->m_tree->m_pose)) != 0))
		{
			releaseTree(inst->m_tree);
			inst->m_tree = 0;
		}
		if (!inst->m_tree || inst->m_meshKey != key)
		{
			releaseTree(inst->m_tree);
			inst->m_tree = fresh && worldTree ? acquireWorldTree(model, worldTransform, localScaling, transform) : acquireTree(model, key);
			inst->m_meshKey = key;
			rtcSetGeometryInstancedScene(inst->m_geometry, inst->m_tree->m_scene);
			rtcSetGeometryInstancedScene(inst->m_shadowGeometry, inst->m_tree->m_scene);
			changed = true;
		}
		else if (inst->m_tree->m_dirty)
		{
			refitTree(model, *inst->m_tree);
			rtcSetGeometryInstancedScene(inst->m_geometry, inst->m_tree->m_scene);
			rtcSetGeometryInstancedScene(inst->m_shadowGeometry, inst->m_tree->m_scene);
			changed = true;
		}
		bool onlyMoved = !changed;
		// A world tree already sits in world space, so its instance carries the identity.
		static const float identity[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
		const float* instanceTransform = inst->m_tree->m_world ? identity : transform;
		if (fresh || memcmp(instanceTransform, inst->m_transform, 16 * sizeof(float)) != 0)
		{
			if (onlyMoved)
				moverMoved(inst->m_tree, inst->m_transform, instanceTransform);
			memcpy(inst->m_transform, instanceTransform, 16 * sizeof(float));
			if (inst->m_tree->m_world)
				copyRotation(btTransform::getIdentity(), inst->m_rotation);
			else
				copyRotation(worldTransform, inst->m_rotation);
			rtcSetGeometryTransform(inst->m_geometry, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, instanceTransform);
			rtcSetGeometryTransform(inst->m_shadowGeometry, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, instanceTransform);
			const float* t = instanceTransform;
			const float det = t[0] * (t[5] * t[10] - t[9] * t[6]) - t[4] * (t[1] * t[10] - t[9] * t[2]) + t[8] * (t[1] * t[6] - t[5] * t[2]);
			inst->m_facingSign = (det < 0.0f) ? -1.0f : 1.0f;
			changed = true;
		}
		if (enabled != inst->m_enabled)
		{
			if (enabled)
			{
				rtcEnableGeometry(inst->m_geometry);
				rtcEnableGeometry(inst->m_shadowGeometry);
			}
			else
			{
				rtcDisableGeometry(inst->m_geometry);
				rtcDisableGeometry(inst->m_shadowGeometry);
			}
			inst->m_enabled = enabled;
			changed = true;
			onlyMoved = false;
		}
		onlyMoved = onlyMoved && inst->m_doubleSided == doubleSided && inst->m_hasAlpha == hasAlpha && inst->m_glass == glass &&
					inst->m_segmentation == segmentation;
		inst->m_doubleSided = doubleSided;
		inst->m_hasAlpha = hasAlpha;
		inst->m_glass = glass;
		inst->m_segmentation = segmentation;
		if (changed)
		{
			rtcCommitGeometry(inst->m_geometry);
			rtcCommitGeometry(inst->m_shadowGeometry);
			m_topDirty = true;
			m_moversDirty = true;
		}
		if (fresh)
		{
			inst->m_geomId = allocateGeomId(inst);
			rtcAttachGeometryByID(m_top, inst->m_geometry, inst->m_geomId);
			rtcAttachGeometryByID(m_movers, inst->m_shadowGeometry, inst->m_geomId);
			m_moverCount++;
		}
		return onlyMoved;
	}

	// Flagged placements remain in the map scene when moved, without changing the mover path.
	void syncSharedInstance(ObjectState& state, TinyRenderObjectData* obj, const float bodyTransform[16], bool enabled, int segmentation)
	{
		float transform[16];
		for (int r = 0; r < 4; r++)
			for (int c = 0; c < 4; c++)
			{
				float sum = 0.0f;
				for (int k = 0; k < 4; k++)
					sum += bodyTransform[k * 4 + r] * obj->m_meshTransform[k][c];
				transform[c * 4 + r] = sum;
			}
		Instance* inst = state.m_instance;
		const bool fresh = !inst;
		if (fresh)
		{
			if (!m_staticShadows)
			{
				m_staticShadows = rtcNewScene(m_device);
				rtcSetSceneFlags(m_staticShadows, RTC_SCENE_FLAG_ROBUST);
				rtcSetSceneBuildQuality(m_staticShadows, RTC_BUILD_QUALITY_MEDIUM);
				rtcAttachGeometryByID(m_staticShadows, m_staticInstance, m_staticInstanceId);
			}
			inst = new Instance;
			memset(inst, 0, sizeof(Instance));
			inst->m_obj = obj;
			inst->m_staticShared = true;
			inst->m_tree = acquireSharedTree(obj->m_model, state.m_deformed);
			inst->m_meshKey = obj->m_model->meshKey();
			inst->m_geometry = rtcNewGeometry(m_device, RTC_GEOMETRY_TYPE_INSTANCE);
			rtcSetGeometryInstancedScene(inst->m_geometry, inst->m_tree->m_scene);
			inst->m_geomId = allocateGeomId(inst);
			rtcAttachGeometryByID(m_top, inst->m_geometry, inst->m_geomId);
			rtcAttachGeometryByID(m_staticShadows, inst->m_geometry, inst->m_geomId);
			state.m_instance = inst;
		}
		const bool changed = fresh || memcmp(transform, inst->m_transform, sizeof(transform)) != 0 || inst->m_enabled != enabled;
		if (changed)
		{
			memcpy(inst->m_transform, transform, sizeof(transform));
			btMatrix3x3 basis(transform[0], transform[4], transform[8], transform[1], transform[5], transform[9], transform[2], transform[6], transform[10]);
			const btScalar det = basis.determinant();
			inst->m_facingSign = det < 0 ? -1.0f : 1.0f;
			// Inverse transpose keeps normals perpendicular under non-uniform and mirrored scales.
			const btMatrix3x3 normal = det != 0 ? basis.inverse().transpose() : btMatrix3x3::getIdentity();
			for (int r = 0; r < 3; r++)
				for (int c = 0; c < 3; c++)
					inst->m_rotation[r * 3 + c] = (float)normal[r][c];
			rtcSetGeometryTransform(inst->m_geometry, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, transform);
			if (enabled && det != 0)
				rtcEnableGeometry(inst->m_geometry);
			else
				rtcDisableGeometry(inst->m_geometry);
			rtcCommitGeometry(inst->m_geometry);
			m_topDirty = true;
			m_staticShadowsDirty = true;
		}
		// Texture contents can change without changing the mesh or its pose.
		if (changed || inst->m_doubleSided != obj->m_doubleSided || inst->m_textureRevision != obj->m_textureRevision)
			m_shadowMap.m_built = m_shelterMap.m_built = false;
		inst->m_textureRevision = obj->m_textureRevision;
		inst->m_enabled = enabled;
		inst->m_doubleSided = obj->m_doubleSided;
		inst->m_hasAlpha = obj->m_model->hasAlpha();
		inst->m_glass = obj->m_glass;
		inst->m_segmentation = segmentation;
	}

	// The forest scene and its one instance, made when the first batch arrives; the shadow scene then holds it too.
	void ensureForest()
	{
		if (m_forest)
			return;
		m_forest = rtcNewScene(m_device);
		rtcSetSceneFlags(m_forest, RTC_SCENE_FLAG_ROBUST);
		rtcSetSceneBuildQuality(m_forest, RTC_BUILD_QUALITY_MEDIUM);
		m_forestInstance = rtcNewGeometry(m_device, RTC_GEOMETRY_TYPE_INSTANCE);
		rtcSetGeometryInstancedScene(m_forestInstance, m_forest);
		const float identity[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
		rtcSetGeometryTransform(m_forestInstance, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, identity);
		m_forestId = allocateGeomId(0);
		rtcAttachGeometryByID(m_top, m_forestInstance, m_forestId);
		if (!m_staticShadows)
		{
			m_staticShadows = rtcNewScene(m_device);
			rtcSetSceneFlags(m_staticShadows, RTC_SCENE_FLAG_ROBUST);
			rtcSetSceneBuildQuality(m_staticShadows, RTC_BUILD_QUALITY_MEDIUM);
			rtcAttachGeometryByID(m_staticShadows, m_staticInstance, m_staticInstanceId);
		}
		rtcAttachGeometryByID(m_staticShadows, m_forestInstance, m_forestId);
		m_forestDirty = true;
	}

	// One batch per forest render object: its placements carried into world space by the body, the visual
	// frame and the mesh scale. They are written again only when the body moves.
	void syncBatch(ObjectState& state, TinyRenderObjectData* obj, const float bodyTransform[16], bool enabled, int segmentation)
	{
		const size_t count = obj->m_placements.size() / 12;
		if (count == 0)
			return;
		Batch* batch = state.m_batch;
		const bool fresh = !batch;
		if (fresh)
		{
			ensureForest();
			batch = new Batch;
			batch->m_obj = obj;
			batch->m_tree = acquireSharedTree(obj->m_model, false);
			batch->m_transforms.assign(count * 12, 0.0f);
			batch->m_normalRotations.assign(count * 9, 0.0f);
			batch->m_geometry = rtcNewGeometry(m_device, RTC_GEOMETRY_TYPE_INSTANCE_ARRAY);
			rtcSetGeometryInstancedScene(batch->m_geometry, batch->m_tree->m_scene);
			rtcSetSharedGeometryBuffer(batch->m_geometry, RTC_BUFFER_TYPE_TRANSFORM, 0, RTC_FORMAT_FLOAT3X4_COLUMN_MAJOR,
									   &batch->m_transforms[0], 0, 12 * sizeof(float), count);
			batch->m_enabled = true;
			batch->m_textureRevision = obj->m_textureRevision;
			memset(batch->m_bodyTransform, 0, sizeof(batch->m_bodyTransform));
			state.m_batch = batch;
		}
		bool changed = fresh;
		if (fresh || memcmp(bodyTransform, batch->m_bodyTransform, sizeof(batch->m_bodyTransform)) != 0)
		{
			memcpy(batch->m_bodyTransform, bodyTransform, sizeof(batch->m_bodyTransform));
			float frame[16];
			for (int r = 0; r < 4; r++)
				for (int c = 0; c < 4; c++)
				{
					float sum = 0.0f;
					for (int k = 0; k < 4; k++)
						sum += bodyTransform[k * 4 + r] * obj->m_meshTransform[k][c];
					frame[c * 4 + r] = sum;
				}
			for (size_t p = 0; p < count; p++)
			{
				const float* local = &obj->m_placements[p * 12];
				float* world = &batch->m_transforms[p * 12];
				for (int c = 0; c < 4; c++)
					for (int r = 0; r < 3; r++)
					{
						float sum = c == 3 ? frame[12 + r] : 0.0f;
						for (int k = 0; k < 3; k++)
							sum += frame[k * 4 + r] * local[c * 3 + k];
						world[c * 3 + r] = sum;
					}
				// Inverse transpose keeps normals perpendicular under a non-uniform or mirrored placement.
				const btMatrix3x3 basis(world[0], world[3], world[6], world[1], world[4], world[7], world[2], world[5], world[8]);
				const btMatrix3x3 normal = basis.determinant() != 0 ? basis.inverse().transpose() : btMatrix3x3::getIdentity();
				float* rotation = &batch->m_normalRotations[p * 9];
				for (int r = 0; r < 3; r++)
					for (int c = 0; c < 3; c++)
						rotation[r * 3 + c] = (float)normal[r][c];
			}
			if (!fresh)
				rtcUpdateGeometryBuffer(batch->m_geometry, RTC_BUFFER_TYPE_TRANSFORM, 0);
			changed = true;
		}
		if (enabled != batch->m_enabled)
		{
			if (enabled)
				rtcEnableGeometry(batch->m_geometry);
			else
				rtcDisableGeometry(batch->m_geometry);
			batch->m_enabled = enabled;
			changed = true;
		}
		if (changed || batch->m_doubleSided != obj->m_doubleSided || batch->m_textureRevision != obj->m_textureRevision)
			m_shadowMap.m_built = m_shelterMap.m_built = false;
		batch->m_doubleSided = obj->m_doubleSided;
		batch->m_hasAlpha = obj->m_model->hasAlpha();
		batch->m_glass = obj->m_glass;
		batch->m_textureRevision = obj->m_textureRevision;
		batch->m_segmentation = segmentation;
		if (changed)
		{
			rtcCommitGeometry(batch->m_geometry);
			m_forestDirty = true;
		}
		if (fresh)
		{
			batch->m_geomId = (unsigned)m_batches.size();
			m_batches.push_back(batch);
			rtcAttachGeometryByID(m_forest, batch->m_geometry, batch->m_geomId);
		}
	}

	void dropBatch(Batch* batch)
	{
		rtcDetachGeometry(m_forest, batch->m_geomId);
		rtcReleaseGeometry(batch->m_geometry);
		releaseTree(batch->m_tree);
		m_batches[batch->m_geomId] = 0;
		m_forestDirty = true;
		m_shadowMap.m_built = m_shelterMap.m_built = false;
		delete batch;
	}

	// Drops every batch and the forest scene itself, before the shadow scene that holds its instance goes.
	void releaseForest()
	{
		for (size_t i = 0; i < m_batches.size(); i++)
			if (m_batches[i])
			{
				rtcReleaseGeometry(m_batches[i]->m_geometry);
				releaseTree(m_batches[i]->m_tree);
				delete m_batches[i];
			}
		m_batches.clear();
		if (m_forest)
		{
			rtcDetachGeometry(m_top, m_forestId);
			rtcReleaseGeometry(m_forestInstance);
			rtcReleaseScene(m_forest);
		}
		m_forest = 0;
		m_forestInstance = 0;
		m_forestId = RTC_INVALID_GEOMETRY_ID;
		m_forestDirty = false;
	}

	void dropInstance(Instance* inst)
	{
		rtcDetachGeometry(m_top, inst->m_geomId);
		if (inst->m_staticShared)
		{
			rtcDetachGeometry(m_staticShadows, inst->m_geomId);
			m_staticShadowsDirty = true;
			m_shadowMap.m_built = m_shelterMap.m_built = false;
		}
		else
		{
			rtcDetachGeometry(m_movers, inst->m_geomId);
			m_moverCount--;
		}
		rtcReleaseGeometry(inst->m_geometry);
		if (inst->m_shadowGeometry)
			rtcReleaseGeometry(inst->m_shadowGeometry);
		releaseTree(inst->m_tree);
		m_byGeomId[inst->m_geomId] = 0;
		m_freeGeomIds.push_back(inst->m_geomId);
		delete inst;
		m_topDirty = true;
		m_moversDirty = true;
	}

	void releaseStatic()
	{
		if (m_staticShadows)
			rtcReleaseScene(m_staticShadows);
		m_staticShadows = 0;
		m_staticShadowsDirty = false;
		m_shadowMap.m_built = false;
		m_shadowMap.m_depth.reset();
		m_shadowMap.m_blocks.reset();
		m_shadowCore.m_built = false;
		m_shadowCore.m_depth.reset();
		m_shadowCore.m_blocks.reset();
		m_shadowDirty.clear();
		m_shelterMap.m_built = false;
		m_shelterMap.m_depth.reset();
		m_shelterMap.m_blocks.reset();
		m_shelterDirty.clear();
		for (size_t i = 0; i < m_members.size(); i++)
		{
			rtcReleaseGeometry(m_members[i]->m_geometry);
			delete m_members[i];
		}
		m_members.clear();
		if (m_staticInstance)
			rtcReleaseGeometry(m_staticInstance);
		if (m_static)
			rtcReleaseScene(m_static);
		m_staticInstance = 0;
		m_static = 0;
	}
};

SwarmRaycast::SwarmRaycast()
{
	m_data = new Data;
	m_data->m_top = 0;
	m_data->m_static = 0;
	m_data->m_staticShadows = 0;
	m_data->m_staticShadowsDirty = false;
	m_data->m_forest = 0;
	m_data->m_forestInstance = 0;
	m_data->m_forestId = RTC_INVALID_GEOMETRY_ID;
	m_data->m_forestDirty = false;
	m_data->m_movers = 0;
	m_data->m_staticInstance = 0;
	m_data->m_staticInstanceId = 0;
	m_data->m_staticBuilt = false;
	m_data->m_topDirty = false;
	m_data->m_moversDirty = true;
	m_data->m_shadowMap.m_built = false;
	m_data->m_shadowCore.m_built = false;
	m_data->m_shelterMap.m_built = false;
	m_data->m_coreRadius = 0.0f;
	m_data->m_moverCount = 0;
	m_data->m_frameCount = 0;
	m_data->m_sceneRevision = 0;
	m_data->m_movedBase = 0;
	for (int i = 0; i < kHitMemories; i++)
	{
		m_data->m_hitMemories[i].m_lastUse = 0;
		m_data->m_hitMemories[i].m_kept = false;
	}
	const char* cacheDir = getenv("SWARM_BVH_CACHE_DIR");
	m_data->m_cacheDir = cacheDir ? cacheDir : "";
	m_data->m_device = processDevice();
	if (!m_data->m_device)
	{
		b3Warning("SwarmRaycast: cannot create the Embree device (a CPU with AVX2 and FMA is required)");
		return;
	}
	rtcSetDeviceErrorFunction(m_data->m_device, reportError, 0);
	m_data->m_top = rtcNewScene(m_data->m_device);
	rtcSetSceneFlags(m_data->m_top, (RTCSceneFlags)(RTC_SCENE_FLAG_ROBUST | RTC_SCENE_FLAG_DYNAMIC));
	rtcSetSceneBuildQuality(m_data->m_top, RTC_BUILD_QUALITY_MEDIUM);
	m_data->m_movers = rtcNewScene(m_data->m_device);
	rtcSetSceneFlags(m_data->m_movers, (RTCSceneFlags)(RTC_SCENE_FLAG_ROBUST | RTC_SCENE_FLAG_DYNAMIC));
	rtcSetSceneBuildQuality(m_data->m_movers, RTC_BUILD_QUALITY_MEDIUM);
	m_data->createStaticScene();
}

SwarmRaycast::~SwarmRaycast()
{
	removeAll();
	if (m_data->m_top)
	{
		m_data->releaseStatic();
		rtcReleaseScene(m_data->m_movers);
		rtcReleaseScene(m_data->m_top);
	}
	delete m_data;
}

void SwarmRaycast::syncObject(TinyRenderObjectData* renderObj, const btTransform& worldTransform, const btVector3& localScaling)
{
	if (!m_data->m_top)
		return;
	TinyRender::Model* model = renderObj->m_model;
	if (!model || model->nfaces() == 0 || model->nverts() == 0)
		return;

	float transform[16];
	composeTransform(worldTransform, localScaling, transform);
	// TinyRenderer skips a fully transparent object; both trees hide it the same way.
	const bool visible = model->getColorRGBA()[3] != 0.0f;
	const bool doubleSided = renderObj->m_doubleSided;
	const bool hasAlpha = model->hasAlpha();
	const bool glass = renderObj->m_glass;
	const int segmentation = renderObj->m_objectIndex + ((renderObj->m_linkIndex + 1) << 24);

	// A body flagged for the disk cache keeps its own world tree instead of joining the static tree.
	const bool worldTree = renderObj->m_renderTreeCache && !m_data->m_cacheDir.empty() && model->meshHash() != 0;

	ObjectState& state = m_data->m_objects[renderObj];
	if (!renderObj->m_placements.empty())
	{
		m_data->m_sceneRevision++;
		m_data->syncBatch(state, renderObj, transform, visible, segmentation);
		return;
	}
	if (renderObj->m_renderInstanced)
	{
		m_data->m_sceneRevision++;
		m_data->syncSharedInstance(state, renderObj, transform, visible, segmentation);
		return;
	}
	StaticMember* member = state.m_member;
	if (member && !member->m_retired)
	{
		m_data->m_sceneRevision++;
		const bool moved = memcmp(transform, member->m_transform, sizeof(transform)) != 0 || member->m_meshKey != model->meshKey();
		if (!moved)
		{
			// A new texture or face setting changes what the member lets through, as hiding it does.
			if (member->m_visible != visible || member->m_doubleSided != doubleSided || member->m_hasAlpha != hasAlpha ||
				member->m_textureRevision != renderObj->m_textureRevision)
				m_data->shadowChanged(member);
			member->m_textureRevision = renderObj->m_textureRevision;
			member->m_visible = visible;
			member->m_doubleSided = doubleSided;
			member->m_hasAlpha = hasAlpha;
			member->m_glass = glass;
			member->m_segmentation = segmentation;
			return;
		}
		// The static tree stays as built; the hit filter ignores this member from now on.
		member->m_retired = true;
		m_data->shadowChanged(member);
	}
	else if (!member && !m_data->m_staticBuilt && !worldTree)
	{
		m_data->m_sceneRevision++;
		member = m_data->addMember(renderObj, worldTransform, localScaling, transform);
		member->m_visible = visible;
		member->m_doubleSided = doubleSided;
		member->m_hasAlpha = hasAlpha;
		member->m_glass = glass;
		member->m_segmentation = segmentation;
		state.m_member = member;
		state.m_instance = 0;
		return;
	}
	if (!member)
		state.m_member = 0;
	if (!m_data->syncInstance(state, renderObj, worldTransform, localScaling, transform, visible, doubleSided, hasAlpha, glass, segmentation, worldTree))
		m_data->m_sceneRevision++;
}

bool SwarmRaycast::meshChanged(TinyRenderObjectData* renderObj)
{
	m_data->m_sceneRevision++;
	// A batch's mesh tree is shared by every placement and never rewritten.
	if (!renderObj->m_placements.empty())
		return false;
	if (renderObj->m_renderInstanced)
		m_data->m_objects[renderObj].m_deformed = true;
	std::map<TinyRenderObjectData*, ObjectState>::iterator found = m_data->m_objects.find(renderObj);
	if (found == m_data->m_objects.end())
		return false;
	if (renderObj->m_renderInstanced)
	{
		// A rewritten mesh must never refit the immutable tree used by its other placements.
		if (found->second.m_instance)
			m_data->dropInstance(found->second.m_instance);
		found->second.m_instance = 0;
		found->second.m_deformed = true;
		return false;
	}
	// A rewritten static member counts as moved at the next sync; a mover refits its tree.
	if (found->second.m_member && !found->second.m_member->m_retired)
		found->second.m_member->m_transform[15] = -1.0f;
	if (!found->second.m_instance || !found->second.m_instance->m_tree)
		return false;
	found->second.m_instance->m_tree->m_dirty = true;
	return found->second.m_instance->m_tree->m_refs > 1;
}

void SwarmRaycast::removeObject(TinyRenderObjectData* renderObj)
{
	m_data->m_sceneRevision++;
	std::map<TinyRenderObjectData*, ObjectState>::iterator found = m_data->m_objects.find(renderObj);
	if (found == m_data->m_objects.end())
		return;
	ObjectState state = found->second;
	m_data->m_objects.erase(found);
	if (state.m_member)
	{
		state.m_member->m_retired = true;
		m_data->shadowChanged(state.m_member);
	}
	if (state.m_instance)
		m_data->dropInstance(state.m_instance);
	if (state.m_batch)
		m_data->dropBatch(state.m_batch);
}

void SwarmRaycast::removeAll()
{
	if (!m_data->m_top)
		return;
	for (std::map<TinyRenderObjectData*, ObjectState>::iterator it = m_data->m_objects.begin(); it != m_data->m_objects.end(); ++it)
		if (it->second.m_instance)
			m_data->dropInstance(it->second.m_instance);
	m_data->m_objects.clear();
	m_data->releaseForest();
	// Retired members keep their triangles until the world is cleared, which is what happens here.
	rtcDetachGeometry(m_data->m_top, m_data->m_staticInstanceId);
	m_data->releaseStatic();
	m_data->m_byGeomId.clear();
	m_data->m_freeGeomIds.clear();
	m_data->createStaticScene();
	m_data->m_sceneRevision++;
	for (int i = 0; i < kHitMemories; i++)
	{
		m_data->m_hitMemories[i].m_points.clear();
		m_data->m_hitMemories[i].m_kept = false;
	}
}

void SwarmRaycast::commit(bool moverShadows)
{
	if (!m_data->m_top)
		return;
	if (!m_data->m_staticBuilt)
	{
		joinCommit(m_data->m_static);
		m_data->m_staticBuilt = true;
		m_data->m_topDirty = true;
	}
	// The forest is committed before the instance that reaches it, and both parents after it.
	if (m_data->m_forestDirty)
	{
		joinCommit(m_data->m_forest);
		rtcCommitGeometry(m_data->m_forestInstance);
		m_data->m_forestDirty = false;
		m_data->m_topDirty = true;
		m_data->m_staticShadowsDirty = m_data->m_staticShadows != 0;
	}
	if (m_data->m_staticShadowsDirty)
	{
		rtcCommitScene(m_data->m_staticShadows);
		m_data->m_staticShadowsDirty = false;
	}
	if (m_data->m_topDirty)
	{
		rtcCommitScene(m_data->m_top);
		m_data->m_topDirty = false;
	}
	if (m_data->m_moversDirty && moverShadows)
	{
		rtcCommitScene(m_data->m_movers);
		m_data->m_moversDirty = false;
	}
}

namespace
{
// The body a ray hit and the hit triangle: its corners in world space, and its corner attributes
// through the per-vertex blocks the body carries. m_rotation is set only for a mover, whose normals
// are stored in its own frame; a static member's normals are already in world space.
struct HitSurface
{
	TinyRender::Model* m_model;
	const float* m_rotation;
	const float* m_normals;
	const float* m_uvs;
	unsigned m_vertexIds[3];
	float m_cornerNormals[9];
	bool m_transformedNormals;
	float m_corners[3][3];
	bool m_doubleSided;
	bool m_hasAlpha;
	bool m_glass;
	bool m_glassBacked;
	const TinyRenderThermal* m_thermal;
};

// Shadow rays start this far off the surface, along the face normal, so a surface never shades itself.
const float kShadowBias = 1e-3f;

// Whether the light's view of the static tree has a surface in front of `point`. A face turned away
// from the light is in its own shadow, as the shadow ray finds when it crosses the face it left. On a
// face turned towards it, the point is moved one cell along its normal and the stored depth gets one
// cell of slack, so a lit surface never shadows itself through the coarseness of the grid. Outside
// the grid, where the ray met nothing, or with no map at all, the point is lit.
inline bool shadowMapBlocked(const ShadowMap& map, const float point[3], const float unitNormal[3])
{
	if (!map.m_built)
		return false;
	if (dot3(unitNormal, map.m_lightDir) <= 0.0f)
		return true;
	float rel[3];
	for (int i = 0; i < 3; i++)
		rel[i] = (point[i] + unitNormal[i] * map.m_cell) - map.m_origin[i];
	const float u = dot3(rel, map.m_axisU) / map.m_cell;
	const float v = dot3(rel, map.m_axisV) / map.m_cell;
	if (!(u >= 0.0f) || !(v >= 0.0f) || u >= (float)map.m_cols || v >= (float)map.m_rows)
		return false;
	// Distance from the start plane along the ray direction, which is -light.
	const float depth = -dot3(rel, map.m_lightDir);
	ensureShadowCells(map, (int)u, (int)u, (int)v, (int)v);
	return depth > map.m_depth[(size_t)(int)v * map.m_cols + (int)u] + map.m_cell;
}

// Lit share of a point from the nine cells around it, a tent two cells wide centred on the point itself, so the
// edge slides with the point inside a cell instead of jumping cell by cell into a staircase.
inline float shadowMapLit(const ShadowMap& map, const float point[3], const float unitNormal[3])
{
	if (!map.m_built)
		return 1.0f;
	if (dot3(unitNormal, map.m_lightDir) <= 0.0f)
		return 0.0f;
	float rel[3];
	for (int i = 0; i < 3; i++)
		rel[i] = (point[i] + unitNormal[i] * map.m_cell) - map.m_origin[i];
	const float u = dot3(rel, map.m_axisU) / map.m_cell;
	const float v = dot3(rel, map.m_axisV) / map.m_cell;
	if (!(u >= 0.0f) || !(v >= 0.0f) || u >= (float)map.m_cols || v >= (float)map.m_rows)
		return 1.0f;
	const float depth = -dot3(rel, map.m_lightDir) - map.m_cell;
	const float su = u - 1.0f, sv = v - 1.0f;
	const int iu = (int)floorf(su), iv = (int)floorf(sv);
	const float fu = su - (float)iu, fv = sv - (float)iv;
	const float weightU[3] = {0.5f * (1.0f - fu), 0.5f, 0.5f * fu};
	const float weightV[3] = {0.5f * (1.0f - fv), 0.5f, 0.5f * fv};
	ensureShadowCells(map, iu, iu + 2, iv, iv + 2);
	float lit = 0.0f;
	// Inside the grid the same nine cells are read in the same order without a bounds check each.
	if (iu >= 0 && iv >= 0 && iu + 2 < map.m_cols && iv + 2 < map.m_rows)
	{
		for (int dv = 0; dv < 3; dv++)
		{
			const float* cells = &map.m_depth[(size_t)(iv + dv) * map.m_cols + iu];
			for (int du = 0; du < 3; du++)
				if (!(depth > cells[du]))
					lit += weightV[dv] * weightU[du];
		}
		return lit;
	}
	for (int dv = 0; dv < 3; dv++)
	{
		const int row = iv + dv;
		for (int du = 0; du < 3; du++)
		{
			const int col = iu + du;
			if (row < 0 || col < 0 || row >= map.m_rows || col >= map.m_cols || !(depth > map.m_depth[(size_t)row * map.m_cols + col]))
				lit += weightV[dv] * weightU[du];
		}
	}
	return lit;
}

// Whether a point lies over the cells of a map, so the fine core grid can answer for it.
inline bool shadowMapCovers(const ShadowMap& map, const float point[3])
{
	if (!map.m_built)
		return false;
	float rel[3];
	for (int i = 0; i < 3; i++)
		rel[i] = point[i] - map.m_origin[i];
	const float u = dot3(rel, map.m_axisU) / map.m_cell;
	const float v = dot3(rel, map.m_axisV) / map.m_cell;
	return u >= 1.0f && v >= 1.0f && u < (float)(map.m_cols - 1) && v < (float)(map.m_rows - 1);
}

// Column-major 4x4 times (v, 1), the same accumulation order as copyWorldVertices.
inline void transformPoint(const float m[16], const float v[3], float out[3])
{
	for (int r = 0; r < 3; r++)
	{
		float acc = 0.0f + m[12 + r] * 1.0f;
		acc = acc + m[8 + r] * v[2];
		acc = acc + m[4 + r] * v[1];
		acc = acc + m[r] * v[0];
		out[r] = acc;
	}
}

// x to an integer power by squaring: the specular exponent is 2 without a specular map and a
// texel value with one, so the maths library never enters the per-pixel path.
inline float powInt(float x, int e)
{
	float result = 1.0f;
	while (e > 0)
	{
		if (e & 1)
			result *= x;
		x *= x;
		e >>= 1;
	}
	return result;
}

// Near-infrared albedo from a linear visible colour: at least its luminance and its red, since skin, soil and dyes
// that reflect red reflect the near infrared too, raised towards a leaf's by how much green outweighs red and blue.
inline float nearInfraredAlbedo(const float base[3])
{
	const float luminance = (0.2126f * base[0] + 0.7152f * base[1]) + 0.0722f * base[2];
	const float floor = luminance > base[0] ? luminance : base[0];
	const float other = base[0] > base[2] ? base[0] : base[2];
	float green = base[1] > other ? (base[1] - other) / base[1] : 0.0f;
	green = green * 2.0f < 1.0f ? green * 2.0f : 1.0f;
	return floor + (kLeafNearInfrared - floor) * green;
}

// Irradiance the spot light puts on a point facing `normal`: its intensity over the squared distance, times the
// cosine at the surface, the smooth cone and the smooth end of its range.
inline float spotIrradiance(const SwarmRaycastShading& shading, const float point[3], const float normal[3])
{
	float toLamp[3];
	for (int i = 0; i < 3; i++)
		toLamp[i] = shading.m_spotPosition[i] - point[i];
	const float distance2 = dot3(toLamp, toLamp);
	if (!(distance2 > 0.0f))
		return 0.0f;
	const float distance = sqrtf(distance2);
	const float reach = distance / shading.m_spotRange;
	if (!(reach < 1.0f))
		return 0.0f;
	for (int i = 0; i < 3; i++)
		toLamp[i] /= distance;
	const float facing = dot3(normal, toLamp);
	const float axis = -dot3(shading.m_spotDirection, toLamp);
	if (!(facing > 0.0f) || !(axis > shading.m_spotCosOuter))
		return 0.0f;
	float cone = (axis - shading.m_spotCosOuter) / (shading.m_spotCosInner - shading.m_spotCosOuter);
	cone = cone < 1.0f ? cone : 1.0f;
	cone = cone * cone * (3.0f - 2.0f * cone);
	const float reach2 = reach * reach;
	const float window = 1.0f - reach2 * reach2;
	return shading.m_spotIntensity * facing * cone * (window * window) / distance2;
}

// Segmentation id of a hit, plus the surface when asked for; false for a hit the scene does not know.
// `cornersOnly` leaves out the corner normals, for a caller that only needs where the triangle is.
bool resolveHit(const RTCHit& hit, unsigned staticId, const std::vector<StaticMember*>& members,
				const std::vector<Instance*>& instances, const std::vector<Batch*>& batches, unsigned forestId,
				int& segmentation, HitSurface* surface, bool cornersOnly = false)
{
	const float* vertices;
	const unsigned* indices;
	const float* transform = 0;
	const float* cornerRotation = 0;
	float placement[16];
	if (surface)
		surface->m_transformedNormals = false;
	if (forestId != RTC_INVALID_GEOMETRY_ID && hit.instID[0] == forestId)
	{
		if (hit.instID[1] >= batches.size() || !batches[hit.instID[1]])
			return false;
		const Batch* batch = batches[hit.instID[1]];
		segmentation = batch->m_segmentation;
		if (!surface)
			return true;
		const float* t = &batch->m_transforms[(size_t)hit.instPrimID[1] * 12];
		for (int c = 0; c < 4; c++)
		{
			for (int r = 0; r < 3; r++)
				placement[c * 4 + r] = t[c * 3 + r];
			placement[c * 4 + 3] = c == 3 ? 1.0f : 0.0f;
		}
		surface->m_model = batch->m_obj->m_model;
		surface->m_doubleSided = batch->m_doubleSided;
		surface->m_hasAlpha = batch->m_hasAlpha;
		surface->m_glass = batch->m_glass;
		surface->m_glassBacked = batch->m_obj->m_glassBacked;
		surface->m_thermal = &batch->m_obj->m_thermal;
		surface->m_rotation = 0;
		surface->m_normals = batch->m_tree->m_normals.empty() ? 0 : &batch->m_tree->m_normals[0];
		surface->m_uvs = batch->m_tree->m_uvs.data();
		vertices = &batch->m_tree->m_vertices[0];
		indices = &batch->m_tree->m_indices[0];
		transform = placement;
		cornerRotation = &batch->m_normalRotations[(size_t)hit.instPrimID[1] * 9];
		surface->m_transformedNormals = surface->m_normals != 0;
	}
	else if (hit.instID[0] == staticId)
	{
		if (hit.geomID >= members.size())
			return false;
		const StaticMember* member = members[hit.geomID];
		segmentation = member->m_segmentation;
		if (!surface)
			return true;
		surface->m_model = member->m_obj->m_model;
		surface->m_doubleSided = member->m_doubleSided;
		surface->m_hasAlpha = member->m_hasAlpha;
		surface->m_glass = member->m_glass;
		surface->m_glassBacked = member->m_obj->m_glassBacked;
		surface->m_thermal = &member->m_obj->m_thermal;
		surface->m_rotation = 0;
		surface->m_normals = member->m_normals.empty() ? 0 : &member->m_normals[0];
		surface->m_uvs = member->m_uvs;
		vertices = &member->m_vertices[0];
		indices = &member->m_indices[0];
	}
	else
	{
		if (hit.instID[0] >= instances.size() || !instances[hit.instID[0]])
			return false;
		const Instance* inst = instances[hit.instID[0]];
		segmentation = inst->m_segmentation;
		if (!surface)
			return true;
		surface->m_model = inst->m_obj->m_model;
		surface->m_doubleSided = inst->m_doubleSided;
		surface->m_hasAlpha = inst->m_hasAlpha;
		surface->m_glass = inst->m_glass;
		surface->m_glassBacked = inst->m_obj->m_glassBacked;
		surface->m_thermal = &inst->m_obj->m_thermal;
		surface->m_rotation = inst->m_rotation;
		surface->m_normals = inst->m_tree->m_normals.empty() ? 0 : &inst->m_tree->m_normals[0];
		surface->m_uvs = inst->m_tree->m_uvs.data();
		vertices = &inst->m_tree->m_vertices[0];
		indices = &inst->m_tree->m_indices[0];
		transform = inst->m_transform;
		cornerRotation = inst->m_rotation;
		surface->m_transformedNormals = inst->m_staticShared && surface->m_normals;
	}
	for (int j = 0; j < 3; j++)
	{
		const unsigned id = indices[(size_t)hit.primID * 3 + j];
		surface->m_vertexIds[j] = id;
		if (surface->m_transformedNormals && !cornersOnly)
		{
			const float* n = surface->m_normals + (size_t)id * 3;
			float* out = surface->m_cornerNormals + j * 3;
			for (int r = 0; r < 3; r++)
				out[r] = dot3(cornerRotation + r * 3, n);
			const float length = sqrtf(dot3(out, out));
			for (int r = 0; r < 3; r++)
				out[r] = length > 0.0f ? out[r] / length : 0.0f;
		}
		const float* v = vertices + (size_t)id * 3;
		if (transform)
			transformPoint(transform, v, surface->m_corners[j]);
		else
			for (int i = 0; i < 3; i++)
				surface->m_corners[j][i] = v[i];
	}
	return true;
}

// TinyRenderer's fragment shader, term for term: interpolated normal and uv, the texture times the
// object colour, ambient plus the shadowed diffuse and specular terms, truncated to bytes. A mesh
// without vertex normals is lit by faceNormal, the triangle's own normal turned towards the camera.
// With the glint on, the lit colour then blends towards the sky mirrored about the normal from viewDir.
// With linear light the same terms run on the decoded value of the texture-times-colour byte, the
// glint blends in linear too, and the byte written is the sRGB encoding of the result; the spot light
// adds to it there, and in near infrared the colour becomes the near-infrared albedo.
void shadeDaylight(const SwarmRaycastShading& shading, const HitSurface& surface, const RTCHit& hit, const float faceNormal[3],
				   const float viewDir[3], float shadow, bool filtered, const float duvdx[2], const float duvdy[2],
				   float distance, unsigned char out[3]);

void shadeHit(const SwarmRaycastShading& shading, const HitSurface& surface, const RTCHit& hit, const float faceNormal[3],
			  const float viewDir[3], float shadow, bool filtered, const float duvdx[2], const float duvdy[2],
			  float distance, const float point[3], unsigned char out[3])
{
	if (shading.m_daylight)
	{
		shadeDaylight(shading, surface, hit, faceNormal, viewDir, shadow, filtered, duvdx, duvdy, distance, out);
		return;
	}
	TinyRender::Model* model = surface.m_model;
	const float weights[3] = {1.0f - hit.u - hit.v, hit.u, hit.v};
	float normal[3] = {0.0f, 0.0f, 0.0f};
	TinyRender::Vec2f uv(0.0f, 0.0f);
	for (int j = 0; j < 3; j++)
	{
		const float* uvj = surface.m_uvs + (size_t)surface.m_vertexIds[j] * 2;
		uv.x += uvj[0] * weights[j];
		uv.y += uvj[1] * weights[j];
		if (!surface.m_normals)
			continue;
		const float* nj = surface.m_transformedNormals ? surface.m_cornerNormals + j * 3 : surface.m_normals + (size_t)surface.m_vertexIds[j] * 3;
		for (int r = 0; r < 3; r++)
			normal[r] += (surface.m_rotation && !surface.m_transformedNormals ? dot3(surface.m_rotation + r * 3, nj) : nj[r]) * weights[j];
	}
	if (!surface.m_normals)
		for (int i = 0; i < 3; i++)
			normal[i] = faceNormal[i];
	normalize3(normal);

	const float nDotL = dot3(normal, shading.m_lightDir);
	float reflection[3];
	for (int i = 0; i < 3; i++)
		reflection[i] = normal[i] * (nDotL * 2.0f) - shading.m_lightDir[i];
	normalize3(reflection);
	const float specular = powInt(reflection[2] > 0.0f ? reflection[2] : 0.0f, (int)model->specular(uv));
	const float diffuse = nDotL > 0.0f ? nDotL : 0.0f;

	TGAColor color = filtered
									 ? model->diffuseFiltered(uv, TinyRender::Vec2f(duvdx[0], duvdx[1]), TinyRender::Vec2f(duvdy[0], duvdy[1]))
									 : model->diffuse(uv);
	const TinyRender::Vec4f& rgba = model->getColorRGBA();
	const float toCamera[3] = {-viewDir[0], -viewDir[1], -viewDir[2]};
	float lit[3];
	if (shading.m_linearLight)
	{
		float base[3];
		for (int i = 0; i < 3; i++)
			base[i] = kSwarmSrgbToLinear[(unsigned char)(color[i] * rgba[i])];
		if (shading.m_nearInfrared)
			base[0] = base[1] = base[2] = nearInfraredAlbedo(base);
		for (int i = 0; i < 3; i++)
			lit[i] = shading.m_ambientCoeff * base[i] * shading.m_ambientColor[i] + shadow * (shading.m_diffuseCoeff * diffuse + shading.m_specularCoeff * specular) * base[i] * shading.m_lightColor[i];
		if (shading.m_spot)
		{
			// The lamp lights the side the camera sees, whichever way a double-sided face is wound.
			float facing[3] = {normal[0], normal[1], normal[2]};
			if (dot3(facing, faceNormal) < 0.0f)
				for (int i = 0; i < 3; i++)
					facing[i] = -facing[i];
			const float lamp = spotIrradiance(shading, point, facing);
			for (int i = 0; i < 3; i++)
				lit[i] += lamp * base[i];
		}
		if (shading.m_glint.m_enabled)
			shading.m_glint.applyLinear(normal, toCamera, &model->getSpecularColor()[0], lit);
		for (int i = 0; i < 3; i++)
			out[i] = swarmLinearToSrgb(lit[i]);
		return;
	}
	for (int i = 0; i < 3; i++)
	{
		const unsigned char base = (unsigned char)(color[i] * rgba[i]);
		const float value = (shading.m_ambientCoeff * base * shading.m_ambientColor[i] + shadow * (shading.m_diffuseCoeff * diffuse + shading.m_specularCoeff * specular) * base * shading.m_lightColor[i]);
		// The rasteriser truncates the lit colour to a byte before anything else reads it.
		lit[i] = (float)(int)(value == value ? (value < 0.0f ? 0.0f : (value > 255.0f ? 255.0f : value)) : 0.0f);
	}
	if (shading.m_glint.m_enabled)
		shading.m_glint.apply(normal, toCamera, &model->getSpecularColor()[0], lit);
	for (int i = 0; i < 3; i++)
	{
		int value = 0;
		if (lit[i] == lit[i])
			value = (int)lit[i];
		out[i] = (unsigned char)(value < 0 ? 0 : (value > 255 ? 255 : value));
	}
}
// The surface at a hit: its shading normal turned to the camera and the linear tint, texture times object colour.
void surfaceAt(const HitSurface& surface, const RTCHit& hit, const float faceNormal[3], bool filtered, const float duvdx[2], const float duvdy[2],
			   float normal[3], float base[3], TinyRender::Vec2f* uvOut = 0)
{
	TinyRender::Model* model = surface.m_model;
	const float weights[3] = {1.0f - hit.u - hit.v, hit.u, hit.v};
	TinyRender::Vec2f uv(0.0f, 0.0f);
	for (int i = 0; i < 3; i++)
		normal[i] = 0.0f;
	for (int j = 0; j < 3; j++)
	{
		const float* uvj = surface.m_uvs + (size_t)surface.m_vertexIds[j] * 2;
		uv.x += uvj[0] * weights[j];
		uv.y += uvj[1] * weights[j];
		if (!surface.m_normals)
			continue;
		const float* nj = surface.m_transformedNormals ? surface.m_cornerNormals + j * 3 : surface.m_normals + (size_t)surface.m_vertexIds[j] * 3;
		for (int r = 0; r < 3; r++)
			normal[r] += (surface.m_rotation && !surface.m_transformedNormals ? dot3(surface.m_rotation + r * 3, nj) : nj[r]) * weights[j];
	}
	if (!surface.m_normals)
		for (int i = 0; i < 3; i++)
			normal[i] = faceNormal[i];
	normalize3(normal);
	// faceNormal already faces the camera; a vertex normal of a face seen from behind does not.
	if (dot3(normal, faceNormal) < 0.0f)
		for (int i = 0; i < 3; i++)
			normal[i] = -normal[i];

	TGAColor color = filtered
						 ? model->diffuseFiltered(uv, TinyRender::Vec2f(duvdx[0], duvdx[1]), TinyRender::Vec2f(duvdy[0], duvdy[1]), 4)
						 : model->diffuse(uv);
	const TinyRender::Vec4f& rgba = model->getColorRGBA();
	for (int i = 0; i < 3; i++)
		base[i] = kSwarmSrgbToLinear[(unsigned char)(color[i] * rgba[i])];
	if (uvOut)
		*uvOut = uv;
}

// Daylight on a surface, in linear light: the sky in the direction it faces plus the sun, and the glint of the sky on glass.
void daylightLight(const SwarmRaycastShading& shading, const HitSurface& surface, const float normal[3], const float base[3],
				   const float viewDir[3], float shadow, float lit[3])
{
	const float nDotL = dot3(normal, shading.m_lightDir);
	float direct = nDotL > 0.0f ? nDotL : 0.0f;
	if (surface.m_doubleSided && surface.m_hasAlpha && nDotL < 0.0f)
		direct = -nDotL * kLeafTransmit;

	float skyLight[3];
	if (shading.m_sky)
		shading.m_sky->irradiance(normal, skyLight);
	else
		for (int i = 0; i < 3; i++)
			skyLight[i] = shading.m_ambientColor[i];
	for (int i = 0; i < 3; i++)
		lit[i] = base[i] * (shading.m_ambientCoeff * skyLight[i] + shadow * shading.m_diffuseCoeff * direct * shading.m_lightColor[i]);

	const float toCamera[3] = {-viewDir[0], -viewDir[1], -viewDir[2]};
	const float* specular = &surface.m_model->getSpecularColor()[0];
	if (shading.m_glint.m_enabled && (specular[0] > 0.0f || specular[1] > 0.0f || specular[2] > 0.0f))
	{
		float nDotV = dot3(normal, toCamera);
		nDotV = nDotV < 0.0f ? 0.0f : (nDotV > 1.0f ? 1.0f : nDotV);
		float rise = (kGlassFlatUntilCos - nDotV) / (kGlassFlatUntilCos - kGlassMirrorFromCos);
		rise = rise < 0.0f ? 0.0f : (rise > 1.0f ? 1.0f : rise);
		const float fresnel = kGlassFlat + (1.0f - kGlassFlat) * rise * rise * (3.0f - 2.0f * rise);
		float mirror[3], sky[3];
		for (int i = 0; i < 3; i++)
			mirror[i] = normal[i] * (2.0f * nDotV) - toCamera[i];
		if (shading.m_sky)
			shading.m_sky->radiance(mirror[0], mirror[1], mirror[2], sky);
		else
		{
			const float up = mirror[shading.m_glint.m_upAxis] > 0.0f ? mirror[shading.m_glint.m_upAxis] : 0.0f;
			for (int i = 0; i < 3; i++)
				sky[i] = swarmUnitToLinear(shading.m_glint.m_skyHorizon[i] + (shading.m_glint.m_skyZenith[i] - shading.m_glint.m_skyHorizon[i]) * up);
		}
		for (int i = 0; i < 3; i++)
		{
			const float w = specular[i] * fresnel;
			lit[i] = lit[i] + (sky[i] - lit[i]) * w;
		}
	}
}

// A thin pane: the sky mirrored about it, and the share that passes through by the Fresnel of its two faces (about 8 % mirror head-on, all mirror when grazing).
float paneLight(const SwarmRaycastShading& shading, const float normal[3], const float viewDir[3], float sky[3])
{
	const float toCamera[3] = {-viewDir[0], -viewDir[1], -viewDir[2]};
	float nDotV = dot3(normal, toCamera);
	nDotV = nDotV < 0.0f ? 0.0f : (nDotV > 1.0f ? 1.0f : nDotV);
	const float away = 1.0f - nDotV;
	const float away2 = away * away;
	const float face = kPaneF0 + (1.0f - kPaneF0) * away2 * away2 * away;
	const float reflect = (2.0f * face) / (1.0f + face);
	float mirror[3];
	for (int i = 0; i < 3; i++)
		mirror[i] = normal[i] * (2.0f * nDotV) - toCamera[i];
	if (shading.m_sky)
		shading.m_sky->radiance(mirror[0], mirror[1], mirror[2], sky);
	else
	{
		const float up = mirror[shading.m_glint.m_upAxis] > 0.0f ? mirror[shading.m_glint.m_upAxis] : 0.0f;
		for (int i = 0; i < 3; i++)
			sky[i] = swarmUnitToLinear(shading.m_glint.m_skyHorizon[i] + (shading.m_glint.m_skyZenith[i] - shading.m_glint.m_skyHorizon[i]) * up);
	}
	return 1.0f - reflect;
}

// Linear light to the byte: haze by distance towards the horizon colour, exposure, the film curve.
void daylightWrite(const SwarmRaycastShading& shading, const float litIn[3], const float viewDir[3], float distance, unsigned char out[3])
{
	float lit[3] = {litIn[0], litIn[1], litIn[2]};
	if (shading.m_hazeDistance > 0.0f)
	{
		const float haze = 1.0f - (float)swarmExp(-(double)distance / (double)shading.m_hazeDistance);
		float horizonColour[3];
		if (shading.m_sky)
		{
			// The haze takes the sky's colour just above the horizon in the direction of view.
			float level[3] = {viewDir[0], viewDir[1], viewDir[2]};
			level[shading.m_glint.m_upAxis] = 0.02f;
			shading.m_sky->radiance(level[0], level[1], level[2], horizonColour);
		}
		else
			for (int i = 0; i < 3; i++)
				horizonColour[i] = swarmUnitToLinear(shading.m_glint.m_skyHorizon[i]);
		for (int i = 0; i < 3; i++)
			lit[i] = lit[i] + (horizonColour[i] - lit[i]) * haze;
	}

	float exposed[3], display[3];
	for (int i = 0; i < 3; i++)
		exposed[i] = lit[i] * shading.m_exposure;
	SwarmAgx::apply(exposed, display);
	for (int i = 0; i < 3; i++)
		out[i] = SwarmAgx::toByte(display[i]);
}

// Daylight shading of one hit, in linear light: sky by direction plus sun, glass reflecting the sky, haze by distance, the film curve on the write.
void shadeDaylight(const SwarmRaycastShading& shading, const HitSurface& surface, const RTCHit& hit, const float faceNormal[3],
				   const float viewDir[3], float shadow, bool filtered, const float duvdx[2], const float duvdy[2],
				   float distance, unsigned char out[3])
{
	float normal[3], base[3], lit[3];
	surfaceAt(surface, hit, faceNormal, filtered, duvdx, duvdy, normal, base);
	daylightLight(shading, surface, normal, base, viewDir, shadow, lit);
	daylightWrite(shading, lit, viewDir, distance, out);
}

// One camera resolved for the frame: its rays, plus the constant step to the pixel to the right and
// the one above, which the texture footprint needs.
struct CameraSetup
{
	Camera m_cam;
	float m_stepX[3];
	float m_stepY[3];
	bool m_valid;
};

// Everything a thread needs to trace one tile; all of it is read-only during the frame.
struct TileJob
{
	RTCScene m_top;
	const std::vector<Instance*>* m_instances;
	const std::vector<StaticMember*>* m_members;
	const std::vector<Batch*>* m_batches;
	unsigned m_forestId;
	const SwarmRaycastShading* m_shading;
	// The light's view of the static tree when the shadow comes from the map; null for shadow rays.
	const ShadowMap* m_shadowMap;
	// The tree of the mover instances when a lit hit also asks them for shadow; null otherwise.
	RTCScene m_movers;
	// Under ER_SWARM_DAYLIGHT: the fine grid about the origin, read before the map for a point it covers.
	const ShadowMap* m_shadowCore;
	// Under ER_SWARM_THERMAL: the static bodies seen from straight above, for the open sky a point has, and the sky.
	const ShadowMap* m_shelterMap;
	SwarmThermal::Sky m_sky;
	unsigned m_staticId;
	int m_width;
	int m_height;
	bool m_filtered;
	// Angle one pixel spans, for the cut-out footprint; 0 unless textures are filtered under daylight.
	float m_pixelSpread;
	// The depth hint, when the lens remembers its last frame: per pixel the farthest remembered hit landing on it.
	const float* m_hintFar;
	// Where every first ray lands this frame, kept for the next; each pixel writes only its own.
	float* m_hitPoints;
	// Where the movers can shade, when the movers answer the shadow rays; null otherwise.
	const MoverShade* m_moverShade;
};

// The triangle a ray landed on, enough to find its corners again.
struct HitId
{
	unsigned m_inst;
	unsigned m_geom;
	unsigned m_prim;
	// Inside the forest: the batch and the placement within it.
	unsigned m_inst1;
	unsigned m_instPrim1;
};

// What one ray brings back: the clip depth and its reciprocal eye depth, the segmentation id and
// triangle of its hit, and the shaded colour when the job carries a light and the body is known.
struct Sample
{
	float m_depth;
	float m_inverseEyeDepth;
	int m_segmentation;
	HitId m_hit;
	bool m_shaded;
	unsigned char m_rgb[3];
	// ER_SWARM_THERMAL: the in-band radiance reaching the camera along the ray, the sky's on a miss.
	float m_radiance;
};

// Per-camera scratch for the edge pass, all of it written by pass one and only read by pass two: the
// id, triangle and 1/zEye of every pixel (-1 and an invalid primitive for a miss), the colour buffer
// before any ray, which is the sky or the clear colour, and the colour buffer after pass one.
struct EdgeScratch
{
	std::vector<int> m_ids;
	std::vector<HitId> m_hits;
	std::vector<float> m_inverseEyeDepth;
	std::vector<unsigned char> m_background;
	std::vector<unsigned char> m_rgb1;
};

// Relative slack on the 1/depth line test; float rounding on a plane sits three orders below it.
const float kEdgeTolerance = 1e-3f;
// ER_SWARM_EDGE_OUTLINE's slack: a neighbour more than a quarter farther away (a fifth, seen from the far side) is an
// edge, so the leaves of one crown, a few per cent apart in depth, are not, while a crown against trees well behind it is.
const float kOutlineTolerance = 0.2f;
// Coverage below this is left to the colour already in the pixel: it is under one colour step.
const double kCoverageEpsilon = 1.0 / 512.0;

// Cell of the mover grid under a point, or -1 off the grid.
inline int moverCell(const MoverShade& shade, const float point[3])
{
	const float u = (dot3(point, shade.m_axisU) - shade.m_u0) * shade.m_perCell;
	const float v = (dot3(point, shade.m_axisV) - shade.m_v0) * shade.m_perCell;
	if (!(u >= 0.0f) || !(v >= 0.0f) || u >= (float)shade.m_cols || v >= (float)shade.m_rows)
		return -1;
	return (int)v * shade.m_cols + (int)u;
}

// Lays the grid over the enabled movers for a light direction: each mover's tree bounds, carried into world space by its
// instance, put across the light and widened well past the rounding of these sums and of the ray's own. False, and no
// grid, when a mover or the light leaves the finite range: every lit point then asks the movers, as without the grid.
bool castMoverShade(const std::vector<Instance*>& instances, const float lightDir[3], MoverShade& shade)
{
	const float ax = fabsf(lightDir[0]), ay = fabsf(lightDir[1]), az = fabsf(lightDir[2]);
	float helper[3] = {0.0f, 0.0f, 0.0f};
	helper[ax <= ay && ax <= az ? 0 : (ay <= az ? 1 : 2)] = 1.0f;
	cross3(lightDir, helper, shade.m_axisU);
	normalize3(shade.m_axisU);
	cross3(lightDir, shade.m_axisU, shade.m_axisV);
	normalize3(shade.m_axisV);
	std::vector<float> rects;
	float lowU = INFINITY, highU = -INFINITY, lowV = INFINITY, highV = -INFINITY;
	for (size_t i = 0; i < instances.size(); i++)
	{
		const Instance* inst = instances[i];
		if (!inst || inst->m_staticShared || !inst->m_enabled)
			continue;
		RTCBounds bounds;
		rtcGetSceneBounds(inst->m_tree->m_scene, &bounds);
		if (!(bounds.lower_x <= bounds.upper_x && bounds.lower_y <= bounds.upper_y && bounds.lower_z <= bounds.upper_z))
			continue;
		float rect[4] = {INFINITY, -INFINITY, INFINITY, -INFINITY}, size = 0.0f;
		for (int k = 0; k < 8; k++)
		{
			const float corner[3] = {k & 1 ? bounds.upper_x : bounds.lower_x, k & 2 ? bounds.upper_y : bounds.lower_y, k & 4 ? bounds.upper_z : bounds.lower_z};
			float world[3];
			transformPoint(inst->m_transform, corner, world);
			const float u = dot3(world, shade.m_axisU), v = dot3(world, shade.m_axisV);
			rect[0] = u < rect[0] ? u : rect[0];
			rect[1] = u > rect[1] ? u : rect[1];
			rect[2] = v < rect[2] ? v : rect[2];
			rect[3] = v > rect[3] ? v : rect[3];
			for (int j = 0; j < 3; j++)
				size = fabsf(world[j]) > size ? fabsf(world[j]) : size;
		}
		const float margin = 0.05f + size * 1e-4f;
		for (int j = 0; j < 4; j++)
			if (!(fabsf(rect[j]) + margin < INFINITY))
				return false;
		rects.push_back(rect[0] - margin);
		rects.push_back(rect[1] + margin);
		rects.push_back(rect[2] - margin);
		rects.push_back(rect[3] + margin);
		lowU = rects[rects.size() - 4] < lowU ? rects[rects.size() - 4] : lowU;
		highU = rects[rects.size() - 3] > highU ? rects[rects.size() - 3] : highU;
		lowV = rects[rects.size() - 2] < lowV ? rects[rects.size() - 2] : lowV;
		highV = rects[rects.size() - 1] > highV ? rects[rects.size() - 1] : highV;
	}
	shade.m_cols = shade.m_rows = 0;
	shade.m_cells.clear();
	if (rects.empty())
		return true;
	// At most 512 cells a side, none under 25 cm: a few movers far apart still leave most cells clear.
	const float span = highU - lowU > highV - lowV ? highU - lowU : highV - lowV;
	if (!(span < INFINITY))
		return false;
	const float cell = span / 512.0f > 0.25f ? span / 512.0f : 0.25f;
	shade.m_u0 = lowU;
	shade.m_v0 = lowV;
	shade.m_perCell = 1.0f / cell;
	shade.m_cols = (int)((highU - lowU) * shade.m_perCell) + 1;
	shade.m_rows = (int)((highV - lowV) * shade.m_perCell) + 1;
	shade.m_cells.assign((size_t)shade.m_cols * shade.m_rows, 0);
	for (size_t r = 0; r < rects.size(); r += 4)
	{
		// Cell indices grow with u and v, so the cells under a rectangle's corners bound every cell under it.
		const int col0 = (int)((rects[r] - lowU) * shade.m_perCell), col1 = (int)((rects[r + 1] - lowU) * shade.m_perCell);
		const int row0 = (int)((rects[r + 2] - lowV) * shade.m_perCell), row1 = (int)((rects[r + 3] - lowV) * shade.m_perCell);
		for (int row = row0; row <= row1 && row < shade.m_rows; row++)
			for (int col = col0; col <= col1 && col < shade.m_cols; col++)
				shade.m_cells[(size_t)row * shade.m_cols + col] = 1;
	}
	return true;
}

// The share of the sun a point keeps: the map answers for the static bodies and the ray for the rest, softly under daylight; faceNormal need not be unit.
// A leaf card with leaf shadows off asks from its sunward side, since its back-light is not its own shadow.
float shadowAt(const TileJob& job, const float point[3], const float faceNormal[3], bool leaf, RTCOccludedArguments* shadowArgs)
{
	const SwarmRaycastShading* shading = job.m_shading;
	float unitNormal[3] = {faceNormal[0], faceNormal[1], faceNormal[2]};
	normalize3(unitNormal);
	if (leaf && shading->m_leafNoShadow && dot3(unitNormal, shading->m_lightDir) < 0.0f)
		for (int i = 0; i < 3; i++)
			unitNormal[i] = -unitNormal[i];
	float litShare = 1.0f;
	bool blocked = false;
	if (job.m_shadowMap && shading->m_daylight)
	{
		const ShadowMap* map = (job.m_shadowCore && shadowMapCovers(*job.m_shadowCore, point)) ? job.m_shadowCore : job.m_shadowMap;
		litShare = shadowMapLit(*map, point, unitNormal);
		blocked = litShare <= 0.0f;
	}
	else
		blocked = job.m_shadowMap && shadowMapBlocked(*job.m_shadowMap, point, unitNormal);
	const RTCScene occluders = job.m_shadowMap ? job.m_movers : job.m_top;
	const float origin[3] = {point[0] + unitNormal[0] * kShadowBias, point[1] + unitNormal[1] * kShadowBias, point[2] + unitNormal[2] * kShadowBias};
	bool reachable = true;
	if (!blocked && occluders == job.m_movers && job.m_moverShade)
	{
		const int cell = moverCell(*job.m_moverShade, origin);
		reachable = cell >= 0 && job.m_moverShade->m_cells[(size_t)cell];
	}
	if (!blocked && occluders && reachable)
	{
		RTCRay ray;
		ray.org_x = origin[0];
		ray.org_y = origin[1];
		ray.org_z = origin[2];
		ray.dir_x = shading->m_lightDir[0];
		ray.dir_y = shading->m_lightDir[1];
		ray.dir_z = shading->m_lightDir[2];
		ray.tnear = 0.0f;
		ray.tfar = INFINITY;
		ray.time = 0.0f;
		ray.mask = (unsigned)-1;
		ray.id = 0;
		ray.flags = 0;
		rtcOccluded1(occluders, &ray, shadowArgs);
		blocked = ray.tfar < 0.0f;
	}
	if (shading->m_daylight)
		return blocked ? shading->m_shadowLightCoeff : shading->m_shadowLightCoeff + (1.0f - shading->m_shadowLightCoeff) * litShare;
	return blocked ? shading->m_shadowLightCoeff : 1.0f;
}

// The texture footprint of a hit, measured as TinyRenderer does at the pixel to the right and the one above: the
// barycentric weights of corners 1 and 2 where each neighbour's ray meets the triangle's plane, skipped when that ray
// runs along the plane. woundNormal must be the triangle's normal as wound, since the weights carry its sign.
void footprintAt(const CameraSetup& setup, const float rawDir[3], const HitSurface& surface, const float woundNormal[3], const RTCHit& hit,
				 float duvdx[2], float duvdy[2])
{
	const float weights[2] = {hit.u, hit.v};
	const float* steps[2] = {setup.m_stepX, setup.m_stepY};
	float* out[2] = {duvdx, duvdy};
	const float* uv0 = surface.m_uvs + (size_t)surface.m_vertexIds[0] * 2;
	const float* uv1 = surface.m_uvs + (size_t)surface.m_vertexIds[1] * 2;
	const float* uv2 = surface.m_uvs + (size_t)surface.m_vertexIds[2] * 2;
	const float* origin = setup.m_cam.m_origin;
	const float(*corners)[3] = surface.m_corners;
	float e1[3], e2[3], toCorner[3];
	for (int i = 0; i < 3; i++)
	{
		e1[i] = corners[1][i] - corners[0][i];
		e2[i] = corners[2][i] - corners[0][i];
		toCorner[i] = corners[0][i] - origin[i];
	}
	const float nn = dot3(woundNormal, woundNormal);
	const float reach = dot3(toCorner, woundNormal);
	for (int k = 0; k < 2; k++)
	{
		float neighbourDir[3], w[3], t1[3], t2[3];
		for (int i = 0; i < 3; i++)
			neighbourDir[i] = rawDir[i] + steps[k][i];
		const float denom = dot3(neighbourDir, woundNormal);
		if (denom == 0.0f || nn == 0.0f)
			continue;
		const float s = reach / denom;
		for (int i = 0; i < 3; i++)
			w[i] = (origin[i] + neighbourDir[i] * s) - corners[0][i];
		cross3(w, e2, t1);
		cross3(e1, w, t2);
		const float u = dot3(t1, woundNormal) / nn;
		const float v = dot3(t2, woundNormal) / nn;
		out[k][0] = (uv1[0] - uv0[0]) * (u - weights[0]) + (uv2[0] - uv0[0]) * (v - weights[1]);
		out[k][1] = (uv1[1] - uv0[1]) * (u - weights[0]) + (uv2[1] - uv0[1]) * (v - weights[1]);
	}
}

// ER_SWARM_THERMAL surfaces. Emissivity when none is set; an open surface cools by kSkyCooling of the air-to-zenith
// gap and a black one facing a full sun warms by kSunHeating degrees. A texel brighter than its texture's mean is
// warmer, by kPassiveDetail degrees per unit of linear luminance on a passive surface (soil, grass, moisture) and by
// kSetDetail on one whose temperature is given (folds and seams), within kDetailLimit either way.
const float kThermalEmissivity = 0.95f;
const float kThermalGlassEmissivity = 0.84f;
const float kSkyCooling = 0.1f;
const float kSunHeating = 22.0f;
const float kPassiveDetail = 25.0f;
const float kSetDetail = 2.0f;
const float kDetailLimit = 0.2f;

// Temperature a heat map gives at uv: its red byte from low to high, with the colour texture's wrap and nearest texel.
float heatAt(const TinyRenderThermal& thermal, const TinyRender::Vec2f& uv)
{
	double whole;
	float u = (float)modf((double)uv.x, &whole), v = (float)modf((double)uv.y, &whole);
	u = u < 0.0f ? u + 1.0f : u;
	v = v < 0.0f ? v + 1.0f : v;
	const int w = thermal.m_heatWidth, h = thermal.m_heatHeight;
	int x = (int)(u * w), y = (int)(v * h);
	x = x < 0 ? 0 : (x >= w ? w - 1 : x);
	y = y < 0 ? 0 : (y >= h ? h - 1 : y);
	// The colour texture is stored bottom row first, so its row y is row h - 1 - y of the texels as loaded.
	const unsigned char byte = thermal.m_heatTexels[((size_t)(h - 1 - y) * w + x) * 3];
	return thermal.m_heatLow + (thermal.m_heatHigh - thermal.m_heatLow) * ((float)byte * (1.0f / 255.0f));
}

// Linear luminance of a texel times the object colour.
float texelLuminance(TGAColor texel, const TinyRender::Vec4f& rgba)
{
	return (0.2126f * kSwarmSrgbToLinear[(unsigned char)(texel[0] * rgba[0])] + 0.7152f * kSwarmSrgbToLinear[(unsigned char)(texel[1] * rgba[1])]) +
		   0.0722f * kSwarmSrgbToLinear[(unsigned char)(texel[2] * rgba[2])];
}

// Share of the pixel a cut-out covers, from its alpha averaged over the pixel's footprint; averaged says the footprint
// spans more than one texel, the only case where a hit is a veil rather than a solid texel.
float veilCoverage(const HitSurface& surface, const RTCHit& hit, const float duvdx[2], const float duvdy[2], bool* averaged)
{
	const float weights[3] = {1.0f - hit.u - hit.v, hit.u, hit.v};
	TinyRender::Vec2f uv(0.0f, 0.0f);
	for (int j = 0; j < 3; j++)
	{
		const float* uvj = surface.m_uvs + (size_t)surface.m_vertexIds[j] * 2;
		uv.x += uvj[0] * weights[j];
		uv.y += uvj[1] * weights[j];
	}
	const float rx2 = duvdx[0] * duvdx[0] + duvdx[1] * duvdx[1], ry2 = duvdy[0] * duvdy[0] + duvdy[1] * duvdy[1];
	return surface.m_model->alphaFiltered(uv, rx2 > ry2 ? rx2 : ry2, averaged) / 255.0f;
}

// In-band radiance leaving a hit towards the camera: the surface's emission at its temperature, plus the sky or the
// surroundings its emissivity leaves it to reflect, mirrored about its normal when it is glossy.
float thermalHit(const TileJob& job, const float dir[3], const HitSurface& surface, const RTCHit& hit, const float faceNormal[3],
				 const float point[3], const float duvdx[2], const float duvdy[2])
{
	const SwarmRaycastShading& shading = *job.m_shading;
	const int upAxis = shading.m_glint.m_upAxis;
	const float air = shading.m_airTemperature, zenith = shading.m_skyTemperature;
	float normal[3], base[3];
	TinyRender::Vec2f uv(0.0f, 0.0f);
	surfaceAt(surface, hit, faceNormal, true, duvdx, duvdy, normal, base, &uv);
	const float albedo = (0.2126f * base[0] + 0.7152f * base[1]) + 0.0722f * base[2];
	TGAColor mean = surface.m_model->diffuseMean(uv);
	float detail = albedo - texelLuminance(mean, surface.m_model->getColorRGBA());
	detail = detail < -kDetailLimit ? -kDetailLimit : (detail > kDetailLimit ? kDetailLimit : detail);

	// Open sky: the shelter map read as if lit from straight overhead, times the share of the sky the surface's tilt
	// faces. A leaf or grass card is a thin blade open on both sides, so it takes the open sky whole.
	float up[3] = {0.0f, 0.0f, 0.0f};
	up[upAxis] = 1.0f;
	const float open = job.m_shelterMap ? shadowMapLit(*job.m_shelterMap, point, up) : 1.0f;
	const float skyView = leafCard(surface.m_doubleSided, surface.m_hasAlpha) ? open : open * 0.5f * (1.0f + normal[upAxis]);

	const TinyRenderThermal& thermal = *surface.m_thermal;
	float temperature;
	if (thermal.m_heatTexels)
		temperature = heatAt(thermal, uv) + kSetDetail * detail;
	else if (thermal.m_hasTemperature)
		temperature = thermal.m_temperature + kSetDetail * detail;
	else
	{
		temperature = air - kSkyCooling * (air - zenith) * skyView + kPassiveDetail * detail;
		const float nDotL = dot3(normal, shading.m_lightDir);
		if (shading.m_lightDir[upAxis] > 0.0f && nDotL > 0.0f)
		{
			const float lit = job.m_shadowMap ? shadowMapLit(*job.m_shadowMap, point, normal) : 1.0f;
			const float sun = (0.2126f * shading.m_lightColor[0] + 0.7152f * shading.m_lightColor[1]) + 0.0722f * shading.m_lightColor[2];
			temperature += kSunHeating * sun * (1.0f - albedo) * nDotL * lit;
		}
	}

	const float emissivity = thermal.m_emissivity >= 0.0f ? thermal.m_emissivity : (surface.m_glass ? kThermalGlassEmissivity : kThermalEmissivity);
	float reflectance = 1.0f - emissivity;
	float environment;
	const float* specular = &surface.m_model->getSpecularColor()[0];
	if (surface.m_glass || specular[0] > 0.0f || specular[1] > 0.0f || specular[2] > 0.0f)
	{
		// Glossy: the Fresnel rise towards a mirror as the view grazes, and the sky in the mirrored direction.
		const float toCamera[3] = {-dir[0], -dir[1], -dir[2]};
		float nDotV = dot3(normal, toCamera);
		nDotV = nDotV < 0.0f ? 0.0f : (nDotV > 1.0f ? 1.0f : nDotV);
		const float away = 1.0f - nDotV, away2 = away * away;
		reflectance += (1.0f - reflectance) * (away2 * away2 * away);
		// Mirrored below the horizon it is the open ground, cooled as a passive surface is.
		const float mirrorUp = normal[upAxis] * (2.0f * nDotV) - toCamera[upAxis];
		environment = mirrorUp > 0.0f ? SwarmThermal::skyRadiance(job.m_sky, mirrorUp) : SwarmThermal::radiance(air - kSkyCooling * (air - zenith));
	}
	else
		environment = skyView * job.m_sky.m_hemisphere + (1.0f - skyView) * job.m_sky.m_air;
	return (1.0f - reflectance) * SwarmThermal::radiance(temperature) + reflectance * environment;
}

// A hinted ray searches this far past the farthest known hit around its pixel, as a share and in metres, for the step
// from one pixel to the next on a slanted surface; a surface found farther costs a second, full search.
const float kHintSlack = 1.0f / 32.0f;
const float kHintSlackMetres = 0.25f;

// The depth along the camera's axis the ray of pixel (row, col) has to search to: past the farthest remembered hit on it
// and its eight neighbours; no limit where none of them remembers one.
inline float hintReach(const float* hintFar, int width, int height, int row, int col)
{
	float farthest = -1.0f;
	for (int r = row > 0 ? row - 1 : row; r <= row + 1 && r < height; r++)
		for (int c = col > 0 ? col - 1 : col; c <= col + 1 && c < width; c++)
		{
			const float far = hintFar[(size_t)r * width + c];
			farthest = far > farthest ? far : farthest;
		}
	return farthest < 0.0f ? INFINITY : farthest + farthest * kHintSlack + kHintSlackMetres;
}

// A remembered hit put onto this frame's pixels: the pixel it lands on keeps the farthest depth along the camera's axis.
inline void hintPoint(const float viewProj[4][4], const float* q, int width, int height, float* hintFar)
{
	const float w = ((viewProj[3][0] * q[0] + viewProj[3][1] * q[1]) + viewProj[3][2] * q[2]) + viewProj[3][3];
	if (!(w > 1e-6f))
		return;
	const float ndcX = (((viewProj[0][0] * q[0] + viewProj[0][1] * q[1]) + viewProj[0][2] * q[2]) + viewProj[0][3]) / w;
	const float ndcY = (((viewProj[1][0] * q[0] + viewProj[1][1] * q[1]) + viewProj[1][2] * q[2]) + viewProj[1][3]) / w;
	const float x = (ndcX + 1.0f) * 0.5f * (float)width + 0.5f;
	const float y = (1.0f - ndcY) * 0.5f * (float)height - 0.5f;
	if (!(x >= 0.0f && y >= 0.0f && x < (float)width && y < (float)height))
		return;
	float& far = hintFar[(size_t)(int)y * width + (size_t)(int)x];
	far = w > far ? w : far;
}

// The same for an edge pixel's probe ray, from this frame's first rays: their 1/zEye, the far plane's for a miss.
inline float probeReach(const float* inverseEyeDepth, int width, int height, int row, int col)
{
	float nearest = -INFINITY;
	for (int r = row > 0 ? row - 1 : row; r <= row + 1 && r < height; r++)
		for (int c = col > 0 ? col - 1 : col; c <= col + 1 && c < width; c++)
		{
			const float w = inverseEyeDepth[(size_t)r * width + c];
			nearest = w > nearest ? w : nearest;
		}
	if (!(nearest < 0.0f))
		return INFINITY;
	const float farthest = -1.0f / nearest;
	return farthest + farthest * kHintSlack + kHintSlackMetres;
}

// The first hit of a camera ray, searched first only up to `reach` and two windows past it when that falls short of the
// ray's end; the window is what Embree's rounding can move a box or a hit by: about 2^-24 of each coordinate over the
// direction component it is divided by, and of the distance, with eight times that allowed. The hinted search meets
// every surface within its reach (none is cut by the end) and around the nearest (see HintedSearch), so its nearest hit
// is the full search's when no second hit is within a window of it. A miss, a hit past `reach`, a second hit that close
// or a face seen nearly edge-on searches again over the whole ray, so a wrong hint costs time and never changes the hit.
void firstHit(RTCScene scene, RTCRayHit& rayhit, float reach, RTCIntersectArguments* args)
{
	const float end = rayhit.ray.tfar;
	if (reach < end)
	{
		const float org[3] = {rayhit.ray.org_x, rayhit.ray.org_y, rayhit.ray.org_z};
		const float dir[3] = {rayhit.ray.dir_x, rayhit.ray.dir_y, rayhit.ray.dir_z};
		float largest = 0.0f, spread = 0.0f;
		for (int k = 0; k < 3; k++)
		{
			const float size = fabsf(org[k]) + reach;
			largest = size > largest ? size : largest;
			const float step = size / fabsf(dir[k]);
			spread = step > spread ? step : spread;
		}
		HintedSearch search;
		for (int k = 0; k < 3; k++)
			search.m_dir[k] = dir[k];
		search.m_largest = largest;
		search.m_window = (spread + largest / kHintMinFacing + reach) * (8.0f / 16777216.0f);
		search.m_limit = reach + 2.0f * search.m_window;
		if (search.m_limit < end)
		{
			search.m_nearest = search.m_next = INFINITY;
			search.m_oblique = false;
			rayhit.ray.tfar = search.m_limit;
			t_hintedSearch = &search;
			rtcIntersect1(scene, &rayhit, args);
			t_hintedSearch = 0;
			if (search.m_nearest <= reach && !search.m_oblique && search.m_next - search.m_nearest > search.m_window)
			{
				rayhit.ray.tfar = search.m_nearest;
				rayhit.hit = search.m_hit;
				return;
			}
			rayhit.ray.tfar = end;
			rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
			rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
		}
	}
	rtcIntersect1(scene, &rayhit, args);
}

// Notes where a camera ray ended, its hit or its far end, for the region a kept frame depends on; a NaN sticks.
inline void reached(RTCIntersectArguments* args, float t)
{
	QueryContext* ctx = (QueryContext*)args->context;
	if (ctx->m_farthest == ctx->m_farthest && !(t <= ctx->m_farthest))
		ctx->m_farthest = t;
}

// One ray of a camera through the frame position (ndcX, ndcY). False on a miss; a hit fills `out`.
// `reach`, when finite, is the depth hint for this ray along the camera's axis; `landed`, when given, takes where the ray
// landed (NaN on a miss).
bool traceRay(const TileJob& job, const CameraSetup& setup, double ndcX, double ndcY,
			  RTCIntersectArguments* args, RTCOccludedArguments* shadowArgs, Sample& out, float reach = INFINITY, float* landed = 0)
{
	const Camera& cam = setup.m_cam;
	const SwarmRaycastShading* shading = job.m_shading;
	const bool filtered = job.m_filtered;
	if (landed)
		landed[0] = landed[1] = landed[2] = NAN;

	float nearPoint[3], farPoint[3];
	planePoint(cam.m_near, ndcX, ndcY, nearPoint);
	planePoint(cam.m_far, ndcX, ndcY, farPoint);
	float dir[3], rawDir[3];
	for (int i = 0; i < 3; i++)
		dir[i] = rawDir[i] = farPoint[i] - nearPoint[i];
	const float length = sqrtf(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
	if (!(length > 0.0f))
		return false;
	const float invLength = 1.0f / length;
	float toNear[3];
	for (int i = 0; i < 3; i++)
	{
		dir[i] *= invLength;
		toNear[i] = nearPoint[i] - cam.m_origin[i];
	}
	const float tNear = sqrtf(toNear[0] * toNear[0] + toNear[1] * toNear[1] + toNear[2] * toNear[2]);
	// The sky along the ray, only for a ray that ends on it: a miss, an unknown body, or what a thermal veil leaves.
	const bool thermalSky = shading && shading->m_thermal;

	RTCRayHit rayhit;
	rayhit.ray.org_x = cam.m_origin[0];
	rayhit.ray.org_y = cam.m_origin[1];
	rayhit.ray.org_z = cam.m_origin[2];
	rayhit.ray.dir_x = dir[0];
	rayhit.ray.dir_y = dir[1];
	rayhit.ray.dir_z = dir[2];
	// Near and far are measured along the ray, so the two clip planes act exactly as TinyRenderer's.
	rayhit.ray.tnear = tNear;
	rayhit.ray.tfar = tNear + length;
	rayhit.ray.time = 0.0f;
	rayhit.ray.mask = (unsigned)-1;
	rayhit.ray.id = 0;
	rayhit.ray.flags = 0;
	rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
	rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
	// Along this ray the hint lies at its depth over the ray's share of the camera's axis.
	const float facing = -dot3(cam.m_viewRow2, dir);
	firstHit(job.m_top, rayhit, facing > 0.0f ? reach / facing : INFINITY, args);
	reached(args, rayhit.ray.tfar);
	if (rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID)
	{
		if (thermalSky)
			out.m_radiance = SwarmThermal::skyRadiance(job.m_sky, dir[shading->m_glint.m_upAxis]);
		return false;
	}

	const float t = rayhit.ray.tfar;
	const float hx = cam.m_origin[0] + dir[0] * t;
	const float hy = cam.m_origin[1] + dir[1] * t;
	const float hz = cam.m_origin[2] + dir[2] * t;
	if (landed)
	{
		landed[0] = hx;
		landed[1] = hy;
		landed[2] = hz;
	}
	const float zEye = ((cam.m_viewRow2[0] * hx + cam.m_viewRow2[1] * hy) + cam.m_viewRow2[2] * hz) + cam.m_viewRow2[3];
	out.m_depth = -(cam.m_p22 * zEye + cam.m_p23);
	out.m_inverseEyeDepth = 1.0f / zEye;

	int segmentation = -1;
	HitSurface surface;
	const bool known = resolveHit(rayhit.hit, job.m_staticId, *job.m_members, *job.m_instances, *job.m_batches, job.m_forestId, segmentation, shading ? &surface : 0);
	out.m_segmentation = segmentation;
	out.m_hit.m_inst = rayhit.hit.instID[0];
	out.m_hit.m_geom = rayhit.hit.geomID;
	out.m_hit.m_prim = rayhit.hit.primID;
	out.m_hit.m_inst1 = rayhit.hit.instID[1];
	out.m_hit.m_instPrim1 = rayhit.hit.instPrimID[1];
	out.m_shaded = shading && known;
	if (!out.m_shaded)
	{
		if (thermalSky)
			out.m_radiance = SwarmThermal::skyRadiance(job.m_sky, dir[shading->m_glint.m_upAxis]);
		return true;
	}

	// The triangle's own normal, kept as wound for the barycentric solve, and a copy turned
	// towards the camera for shading and for the side the shadow ray leaves from.
	float woundNormal[3], faceNormal[3], e1[3], e2[3];
	for (int i = 0; i < 3; i++)
	{
		e1[i] = surface.m_corners[1][i] - surface.m_corners[0][i];
		e2[i] = surface.m_corners[2][i] - surface.m_corners[0][i];
	}
	cross3(e1, e2, woundNormal);
	const bool awayFromCamera = dot3(woundNormal, dir) > 0.0f;
	for (int i = 0; i < 3; i++)
		faceNormal[i] = awayFromCamera ? -woundNormal[i] : woundNormal[i];

	float shadow = 1.0f;
	if (shading->m_shadow && !shading->m_thermal)
	{
		const float point[3] = {hx, hy, hz};
		shadow = shadowAt(job, point, faceNormal, leafCard(surface.m_doubleSided, surface.m_hasAlpha), shadowArgs);
	}

	float duvdx[2] = {0.0f, 0.0f}, duvdy[2] = {0.0f, 0.0f};
	if (filtered)
		footprintAt(setup, rawDir, surface, woundNormal, rayhit.hit, duvdx, duvdy);

	if (shading->m_thermal)
	{
		const float point[3] = {hx, hy, hz};
		float radiance = thermalHit(job, dir, surface, rayhit.hit, faceNormal, point, duvdx, duvdy);
		bool averaged = false;
		const float coverage = job.m_pixelSpread > 0.0f && surface.m_hasAlpha ? veilCoverage(surface, rayhit.hit, duvdx, duvdy, &averaged) : 1.0f;
		if (averaged && coverage < 0.97f)
		{
			// A veil, such as a far fence or a thin crown: its share of the pixel, and the rest from what the ray meets next.
			float behind;
			RTCRayHit next = rayhit;
			next.ray.tnear = t + kPaneBias;
			next.ray.tfar = tNear + length;
			next.hit.geomID = RTC_INVALID_GEOMETRY_ID;
			next.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
			rtcIntersect1(job.m_top, &next, args);
			reached(args, next.ray.tfar);
			int backSegmentation = -1;
			HitSurface back;
			if (next.hit.geomID != RTC_INVALID_GEOMETRY_ID &&
				resolveHit(next.hit, job.m_staticId, *job.m_members, *job.m_instances, *job.m_batches, job.m_forestId, backSegmentation, &back))
			{
				float backWound[3], backFace[3], f1[3], f2[3];
				for (int i = 0; i < 3; i++)
				{
					f1[i] = back.m_corners[1][i] - back.m_corners[0][i];
					f2[i] = back.m_corners[2][i] - back.m_corners[0][i];
				}
				cross3(f1, f2, backWound);
				const bool backAway = dot3(backWound, dir) > 0.0f;
				for (int i = 0; i < 3; i++)
					backFace[i] = backAway ? -backWound[i] : backWound[i];
				float duv2x[2] = {0.0f, 0.0f}, duv2y[2] = {0.0f, 0.0f};
				footprintAt(setup, rawDir, back, backWound, next.hit, duv2x, duv2y);
				const float t2 = next.ray.tfar;
				const float point2[3] = {cam.m_origin[0] + dir[0] * t2, cam.m_origin[1] + dir[1] * t2, cam.m_origin[2] + dir[2] * t2};
				behind = thermalHit(job, dir, back, next.hit, backFace, point2, duv2x, duv2y);
			}
			else
				behind = SwarmThermal::skyRadiance(job.m_sky, dir[shading->m_glint.m_upAxis]);
			radiance = coverage * radiance + (1.0f - coverage) * behind;
		}
		out.m_radiance = radiance;
		return true;
	}

	// A cut-out seen over many of its texels is a veil, such as a far chain-link fence: its coverage of the pixel is lit as
	// the surface and the rest is the next surface along the ray, so thin wire fades as it does to a lens.
	if (shading->m_daylight && filtered && job.m_pixelSpread > 0.0f && surface.m_hasAlpha && !surface.m_glass)
	{
		const float weights[3] = {1.0f - rayhit.hit.u - rayhit.hit.v, rayhit.hit.u, rayhit.hit.v};
		TinyRender::Vec2f uv(0.0f, 0.0f);
		for (int j = 0; j < 3; j++)
		{
			const float* uvj = surface.m_uvs + (size_t)surface.m_vertexIds[j] * 2;
			uv.x += uvj[0] * weights[j];
			uv.y += uvj[1] * weights[j];
		}
		const float rx2 = duvdx[0] * duvdx[0] + duvdx[1] * duvdx[1], ry2 = duvdy[0] * duvdy[0] + duvdy[1] * duvdy[1];
		bool averaged = false;
		const float coverage = surface.m_model->alphaFiltered(uv, rx2 > ry2 ? rx2 : ry2, &averaged) / 255.0f;
		if (averaged && coverage < 0.97f)
		{
			float normal[3], base[3], lit[3], behind[3];
			surfaceAt(surface, rayhit.hit, faceNormal, filtered, duvdx, duvdy, normal, base);
			daylightLight(*shading, surface, normal, base, dir, shadow, lit);
			RTCRayHit next = rayhit;
			next.ray.tnear = t + kPaneBias;
			next.ray.tfar = tNear + length;
			next.hit.geomID = RTC_INVALID_GEOMETRY_ID;
			next.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
			rtcIntersect1(job.m_top, &next, args);
			reached(args, next.ray.tfar);
			int backSegmentation = -1;
			HitSurface back;
			if (next.hit.geomID == RTC_INVALID_GEOMETRY_ID || !resolveHit(next.hit, job.m_staticId, *job.m_members, *job.m_instances, *job.m_batches, job.m_forestId, backSegmentation, &back))
			{
				if (shading->m_sky)
					shading->m_sky->radiance(dir[0], dir[1], dir[2], behind);
				else
					for (int i = 0; i < 3; i++)
						behind[i] = shading->m_ambientColor[i];
			}
			else
			{
				float backWound[3], backFace[3], f1[3], f2[3];
				for (int i = 0; i < 3; i++)
				{
					f1[i] = back.m_corners[1][i] - back.m_corners[0][i];
					f2[i] = back.m_corners[2][i] - back.m_corners[0][i];
				}
				cross3(f1, f2, backWound);
				const bool backAway = dot3(backWound, dir) > 0.0f;
				for (int i = 0; i < 3; i++)
					backFace[i] = backAway ? -backWound[i] : backWound[i];
				float duv2x[2] = {0.0f, 0.0f}, duv2y[2] = {0.0f, 0.0f};
				footprintAt(setup, rawDir, back, backWound, next.hit, duv2x, duv2y);
				const float t2 = next.ray.tfar;
				const float point2[3] = {cam.m_origin[0] + dir[0] * t2, cam.m_origin[1] + dir[1] * t2, cam.m_origin[2] + dir[2] * t2};
				const float shadow2 = shading->m_shadow ? shadowAt(job, point2, backFace, leafCard(back.m_doubleSided, back.m_hasAlpha), shadowArgs) : 1.0f;
				float backNormal[3], backBase[3];
				surfaceAt(back, next.hit, backFace, filtered, duv2x, duv2y, backNormal, backBase);
				daylightLight(*shading, back, backNormal, backBase, dir, shadow2, behind);
			}
			for (int i = 0; i < 3; i++)
				lit[i] = coverage * lit[i] + (1.0f - coverage) * behind[i];
			daylightWrite(*shading, lit, dir, t, out.m_rgb);
			return true;
		}
	}

	if (shading->m_daylight && surface.m_glass)
	{
		// A pane: the ray carries on through up to kPaneDepth panes, each adding its share of the sky and dimming what follows by its tint.
		float weight[3] = {1.0f, 1.0f, 1.0f};
		float lit[3] = {0.0f, 0.0f, 0.0f};
		RTCRayHit current = rayhit;
		HitSurface pane = surface;
		float paneFace[3] = {faceNormal[0], faceNormal[1], faceNormal[2]};
		float paneDuvdx[2] = {duvdx[0], duvdx[1]}, paneDuvdy[2] = {duvdy[0], duvdy[1]};
		if (surface.m_glassBacked)
		{
			// A module: its cells over a white backsheet in the pane's own colour, lit as the pane is and shaded by its
			// shadow, which is what the ray behind it would meet, so no ray goes behind it.
			float normal[3], base[3], sky[3], skyLight[3];
			surfaceAt(surface, rayhit.hit, faceNormal, filtered, duvdx, duvdy, normal, base);
			const float through = paneLight(*shading, normal, dir, sky);
			if (shading->m_sky)
				shading->m_sky->irradiance(normal, skyLight);
			else
				for (int i = 0; i < 3; i++)
					skyLight[i] = shading->m_ambientColor[i];
			const float nDotL = dot3(normal, shading->m_lightDir);
			const float direct = nDotL > 0.0f ? nDotL : 0.0f;
			const TinyRender::Vec4f& rgba = surface.m_model->getColorRGBA();
			for (int i = 0; i < 3; i++)
			{
				const float backing = kSwarmSrgbToLinear[(unsigned char)(kPaneBacking * rgba[i])];
				lit[i] = (1.0f - through) * sky[i] + through * base[i] * backing * (shading->m_ambientCoeff * skyLight[i] + shadow * shading->m_diffuseCoeff * direct * shading->m_lightColor[i]);
			}
			daylightWrite(*shading, lit, dir, t, out.m_rgb);
			return true;
		}
		for (int depth = 0; depth < kPaneDepth; depth++)
		{
			float normal[3], base[3], sky[3];
			surfaceAt(pane, current.hit, paneFace, filtered, paneDuvdx, paneDuvdy, normal, base);
			const float through = paneLight(*shading, normal, dir, sky);
			for (int i = 0; i < 3; i++)
			{
				lit[i] += weight[i] * (1.0f - through) * sky[i];
				weight[i] *= through * base[i];
			}
			RTCRayHit next = current;
			next.ray.tnear = current.ray.tfar + kPaneBias;
			next.ray.tfar = tNear + length;
			next.hit.geomID = RTC_INVALID_GEOMETRY_ID;
			next.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
			rtcIntersect1(job.m_top, &next, args);
			reached(args, next.ray.tfar);
			int backSegmentation = -1;
			HitSurface back;
			if (next.hit.geomID == RTC_INVALID_GEOMETRY_ID || !resolveHit(next.hit, job.m_staticId, *job.m_members, *job.m_instances, *job.m_batches, job.m_forestId, backSegmentation, &back))
			{
				float behind[3];
				if (shading->m_sky)
					shading->m_sky->radiance(dir[0], dir[1], dir[2], behind);
				else
					for (int i = 0; i < 3; i++)
						behind[i] = shading->m_ambientColor[i];
				for (int i = 0; i < 3; i++)
					lit[i] += weight[i] * behind[i];
				break;
			}
			float backWound[3], backFace[3], f1[3], f2[3];
			for (int i = 0; i < 3; i++)
			{
				f1[i] = back.m_corners[1][i] - back.m_corners[0][i];
				f2[i] = back.m_corners[2][i] - back.m_corners[0][i];
			}
			cross3(f1, f2, backWound);
			const bool backAway = dot3(backWound, dir) > 0.0f;
			for (int i = 0; i < 3; i++)
				backFace[i] = backAway ? -backWound[i] : backWound[i];
			float duv2x[2] = {0.0f, 0.0f}, duv2y[2] = {0.0f, 0.0f};
			if (filtered)
				footprintAt(setup, rawDir, back, backWound, next.hit, duv2x, duv2y);
			if (back.m_glass && depth + 1 < kPaneDepth)
			{
				current = next;
				pane = back;
				for (int i = 0; i < 3; i++)
					paneFace[i] = backFace[i];
				paneDuvdx[0] = duv2x[0];
				paneDuvdx[1] = duv2x[1];
				paneDuvdy[0] = duv2y[0];
				paneDuvdy[1] = duv2y[1];
				continue;
			}
			const float t2 = next.ray.tfar;
			const float point2[3] = {cam.m_origin[0] + dir[0] * t2, cam.m_origin[1] + dir[1] * t2, cam.m_origin[2] + dir[2] * t2};
			const float shadow2 = shading->m_shadow ? shadowAt(job, point2, backFace, leafCard(back.m_doubleSided, back.m_hasAlpha), shadowArgs) : 1.0f;
			float backNormal[3], backBase[3], behind[3];
			surfaceAt(back, next.hit, backFace, filtered, duv2x, duv2y, backNormal, backBase);
			daylightLight(*shading, back, backNormal, backBase, dir, shadow2, behind);
			for (int i = 0; i < 3; i++)
				lit[i] += weight[i] * behind[i];
			break;
		}
		daylightWrite(*shading, lit, dir, t, out.m_rgb);
		return true;
	}

	const float point[3] = {hx, hy, hz};
	shadeHit(*shading, surface, rayhit.hit, faceNormal, dir, shadow, filtered, duvdx, duvdy, t, point, out.m_rgb);
	return true;
}

// Output row `row` is TinyRenderer's raster row height - 1 - row, sampled at the integer pixel corner.
inline double pixelNdcX(int col, int width)
{
	return (2.0 * col) / (double)width - 1.0;
}

inline double pixelNdcY(int row, int height)
{
	return 1.0 - (2.0 * row + 2.0) / (double)height;
}

// 1/zEye of a pixel the first ray missed, from the clear value in the depth buffer. A plane is linear
// in 1/zEye across the frame, so a flat surface at any angle passes the line test in isEdge and a step
// or crease fails it.
inline float inverseEyeDepth(const Camera& cam, float depth)
{
	return 1.0f / (-(depth + cam.m_p23) / cam.m_p22);
}

// Traces the pixels [col0, col1) x [row0, row1) of one camera into its buffers. Every pixel is
// written by exactly one call, so the tile order and the thread that runs it cannot change the bytes.
// `scratch`, when given, records the id and triangle of every hit for the edge pass. `radiance`, under ER_SWARM_THERMAL,
// takes every pixel's in-band radiance, hit or miss, in place of a colour.
void renderTile(const TileJob& job, const CameraSetup& setup, const SwarmRaycast::Target& target, EdgeScratch* scratch, float* radiance,
				int row0, int row1, int col0, int col1,
				RTCIntersectArguments* args, RTCOccludedArguments* shadowArgs)
{
	const int width = job.m_width;
	for (int row = row0; row < row1; row++)
	{
		const double ndcY = pixelNdcY(row, job.m_height);
		for (int col = col0; col < col1; col++)
		{
			Sample sample;
			sample.m_radiance = 0.0f;
			const size_t offset = (size_t)row * width + col;
			const float reach = job.m_hintFar ? hintReach(job.m_hintFar, width, job.m_height, row, col) : INFINITY;
			const bool hit = traceRay(job, setup, pixelNdcX(col, width), ndcY, args, shadowArgs, sample, reach,
									  job.m_hitPoints ? job.m_hitPoints + offset * 3 : 0);
			if (radiance)
				radiance[offset] = sample.m_radiance;
			if (target.m_background && !(hit && sample.m_shaded))
				target.m_background->pixel(row, col, &target.m_rgb[offset * 3]);
			if (!hit)
			{
				if (scratch)
					scratch->m_inverseEyeDepth[offset] = inverseEyeDepth(setup.m_cam, target.m_depth[offset]);
				continue;
			}
			target.m_depth[offset] = sample.m_depth;
			if (target.m_seg)
				target.m_seg[offset] = sample.m_segmentation;
			if (scratch)
			{
				scratch->m_ids[offset] = sample.m_segmentation;
				scratch->m_hits[offset] = sample.m_hit;
				scratch->m_inverseEyeDepth[offset] = sample.m_inverseEyeDepth;
			}
			if (sample.m_shaded && !radiance)
				for (int i = 0; i < 3; i++)
					target.m_rgb[offset * 3 + i] = sample.m_rgb[i];
		}
	}
}

// A pixel is an edge when one of its four neighbours landed on another body, or when its 1/zEye is
// off the straight line through its two neighbours on either axis by more than `tolerance` of its own.
bool isEdge(const int* ids, const float* w, int width, int height, int row, int col, float tolerance)
{
	const size_t offset = (size_t)row * width + col;
	const int id = ids[offset];
	const bool hasLeft = col > 0, hasRight = col + 1 < width, hasUp = row > 0, hasDown = row + 1 < height;
	if ((hasLeft && ids[offset - 1] != id) || (hasRight && ids[offset + 1] != id) ||
		(hasUp && ids[offset - width] != id) || (hasDown && ids[offset + width] != id))
		return true;
	const float limit = tolerance * fabsf(w[offset]);
	if (hasLeft && hasRight && fabsf((w[offset - 1] + w[offset + 1]) - 2.0f * w[offset]) > limit)
		return true;
	if (hasUp && hasDown && fabsf((w[offset - width] + w[offset + width]) - 2.0f * w[offset]) > limit)
		return true;
	return false;
}

// A convex polygon on the frame, in ndc; a pixel square clipped by three edges has at most seven corners.
struct Polygon
{
	double m_x[8];
	double m_y[8];
	int m_n;
};

// The triangle a first ray landed on, put back onto the frame: false when it is unknown to the scene
// or reaches behind the camera, where its projection is not a triangle.
bool projectTriangle(const TileJob& job, const Camera& cam, const HitId& hit, double x[3], double y[3])
{
	RTCHit rtcHit;
	rtcHit.instID[0] = hit.m_inst;
	rtcHit.geomID = hit.m_geom;
	rtcHit.primID = hit.m_prim;
	rtcHit.instID[1] = hit.m_inst1;
	rtcHit.instPrimID[0] = 0;
	rtcHit.instPrimID[1] = hit.m_instPrim1;
	int segmentation;
	HitSurface surface;
	if (!resolveHit(rtcHit, job.m_staticId, *job.m_members, *job.m_instances, *job.m_batches, job.m_forestId, segmentation, &surface, true))
		return false;
	for (int j = 0; j < 3; j++)
	{
		const float* c = surface.m_corners[j];
		double clip[4];
		for (int r = 0; r < 4; r++)
			clip[r] = ((cam.m_viewProj[r][0] * c[0] + cam.m_viewProj[r][1] * c[1]) + cam.m_viewProj[r][2] * c[2]) + cam.m_viewProj[r][3];
		if (!(clip[3] > 1e-9))
			return false;
		x[j] = clip[0] / clip[3];
		y[j] = clip[1] / clip[3];
	}
	return true;
}

inline bool sameTriangle(const HitId& a, const HitId& b)
{
	return a.m_inst == b.m_inst && a.m_geom == b.m_geom && a.m_prim == b.m_prim && a.m_inst1 == b.m_inst1 && a.m_instPrim1 == b.m_instPrim1;
}

// The triangles a thread put onto the frame, one slot per hash of the triangle, kept for the whole
// frame: an edge runs through several pixels that all ask for the same few triangles, so each is
// projected about once instead of once per pixel.
const int kProjectionSlots = 256;
struct ProjectionCache
{
	HitId m_key[kProjectionSlots];
	double m_x[kProjectionSlots][3];
	double m_y[kProjectionSlots][3];
	bool m_ok[kProjectionSlots];
	bool m_used[kProjectionSlots];
};

bool projectCached(ProjectionCache& cache, const TileJob& job, const Camera& cam, const HitId& hit, const double*& x, const double*& y)
{
	const unsigned mix = (hit.m_prim * 0x9E3779B1u) ^ (hit.m_instPrim1 * 0x85EBCA77u) ^ (hit.m_inst * 0xC2B2AE3Du) ^ (hit.m_geom * 0x27D4EB2Fu) ^ hit.m_inst1;
	const int slot = (int)(mix >> 24);
	x = cache.m_x[slot];
	y = cache.m_y[slot];
	if (!cache.m_used[slot] || !sameTriangle(cache.m_key[slot], hit))
	{
		cache.m_used[slot] = true;
		cache.m_key[slot] = hit;
		cache.m_ok[slot] = projectTriangle(job, cam, hit, cache.m_x[slot], cache.m_y[slot]);
	}
	return cache.m_ok[slot];
}

// Keeps the part of `poly` on the inner side of the directed line a -> b, where inside is the side
// `sign` says the triangle's third corner lies on.
void clipByEdge(const Polygon& poly, double ax, double ay, double bx, double by, double sign, Polygon& out)
{
	out.m_n = 0;
	const double ex = bx - ax, ey = by - ay;
	for (int i = 0; i < poly.m_n; i++)
	{
		const int k = (i + 1) % poly.m_n;
		const double px = poly.m_x[i], py = poly.m_y[i], qx = poly.m_x[k], qy = poly.m_y[k];
		const double dp = (ex * (py - ay) - ey * (px - ax)) * sign;
		const double dq = (ex * (qy - ay) - ey * (qx - ax)) * sign;
		if (dp >= 0.0)
		{
			out.m_x[out.m_n] = px;
			out.m_y[out.m_n] = py;
			out.m_n++;
		}
		if ((dp >= 0.0) != (dq >= 0.0))
		{
			const double t = dp / (dp - dq);
			out.m_x[out.m_n] = px + (qx - px) * t;
			out.m_y[out.m_n] = py + (qy - py) * t;
			out.m_n++;
		}
	}
}

// Area of the polygon and its centroid; a polygon too thin to have an area reports zero and its first corner.
double polygonArea(const Polygon& poly, double& cx, double& cy)
{
	double twice = 0.0, sx = 0.0, sy = 0.0;
	for (int i = 0; i < poly.m_n; i++)
	{
		const int k = (i + 1) % poly.m_n;
		const double cross = poly.m_x[i] * poly.m_y[k] - poly.m_x[k] * poly.m_y[i];
		twice += cross;
		sx += (poly.m_x[i] + poly.m_x[k]) * cross;
		sy += (poly.m_y[i] + poly.m_y[k]) * cross;
	}
	if (fabs(twice) < 1e-300 || poly.m_n < 3)
	{
		cx = poly.m_n ? poly.m_x[0] : 0.0;
		cy = poly.m_n ? poly.m_y[0] : 0.0;
		return 0.0;
	}
	cx = sx / (3.0 * twice);
	cy = sy / (3.0 * twice);
	return fabs(twice) * 0.5;
}

// Share of the pixel square that the triangle covers, and the centroid of that part.
double coverage(const Polygon& square, double squareArea, const double x[3], const double y[3], double& cx, double& cy)
{
	const double sign = (x[1] - x[0]) * (y[2] - y[0]) - (y[1] - y[0]) * (x[2] - x[0]) >= 0.0 ? 1.0 : -1.0;
	Polygon a, b, c;
	clipByEdge(square, x[0], y[0], x[1], y[1], sign, a);
	clipByEdge(a, x[1], y[1], x[2], y[2], sign, b);
	clipByEdge(b, x[2], y[2], x[0], y[0], sign, c);
	const double part = polygonArea(c, cx, cy);
	const double share = part / squareArea;
	return share > 1.0 ? 1.0 : share;
}

// One surface that may cover part of a pixel: the triangle a first ray landed on, the colour that ray
// shaded, and the depth it sits at.
struct Candidate
{
	HitId m_hit;
	const unsigned char* m_rgb;
	float m_depth;
};

// ER_SWARM_CREASE_FILL: a mover's box put onto the frame, kMoverMargin pixels wider, and the farthest eye depth it
// reaches. A crease nearer than that inside it may show the mover through a gap narrower than a pixel, so it keeps its probe.
struct MoverRect
{
	int m_col0;
	int m_col1;
	int m_row0;
	int m_row1;
	float m_depth;
};
const int kMoverMargin = 2;
// A box reaching nearer than kMoverNear guards the whole frame, unless it lies within kAircraftReach of the eye: those are
// the aircraft's own parts, in front of everything it films.
const float kMoverNear = 1.0f;
const float kAircraftReach = 2.0f;

// The frame rectangles of every enabled mover, the same bodies the mover shadow grid covers.
void moverRects(const std::vector<Instance*>& instances, const Camera& cam, int width, int height, std::vector<MoverRect>& out)
{
	out.clear();
	for (size_t i = 0; i < instances.size(); i++)
	{
		const Instance* inst = instances[i];
		if (!inst || inst->m_staticShared || !inst->m_enabled)
			continue;
		RTCBounds bounds;
		rtcGetSceneBounds(inst->m_tree->m_scene, &bounds);
		if (!(bounds.lower_x <= bounds.upper_x && bounds.lower_y <= bounds.upper_y && bounds.lower_z <= bounds.upper_z))
			continue;
		float nearest = INFINITY, farthest = -INFINITY, low[3] = {INFINITY, INFINITY, INFINITY}, high[3] = {-INFINITY, -INFINITY, -INFINITY};
		double col0 = INFINITY, col1 = -INFINITY, row0 = INFINITY, row1 = -INFINITY;
		for (int k = 0; k < 8; k++)
		{
			const float corner[3] = {k & 1 ? bounds.upper_x : bounds.lower_x, k & 2 ? bounds.upper_y : bounds.lower_y, k & 4 ? bounds.upper_z : bounds.lower_z};
			float world[3];
			transformPoint(inst->m_transform, corner, world);
			for (int j = 0; j < 3; j++)
			{
				low[j] = world[j] < low[j] ? world[j] : low[j];
				high[j] = world[j] > high[j] ? world[j] : high[j];
			}
			const float depth = -(((cam.m_viewRow2[0] * world[0] + cam.m_viewRow2[1] * world[1]) + cam.m_viewRow2[2] * world[2]) + cam.m_viewRow2[3]);
			nearest = depth < nearest ? depth : nearest;
			farthest = depth > farthest ? depth : farthest;
			double clip[4];
			for (int r = 0; r < 4; r++)
				clip[r] = ((cam.m_viewProj[r][0] * world[0] + cam.m_viewProj[r][1] * world[1]) + cam.m_viewProj[r][2] * world[2]) + cam.m_viewProj[r][3];
			if (!(clip[3] > 0.0))
				continue;
			// The inverse of pixelNdcX and pixelNdcY.
			const double col = (clip[0] / clip[3] + 1.0) * 0.5 * width;
			const double row = ((1.0 - clip[1] / clip[3]) * height - 2.0) * 0.5;
			col0 = col < col0 ? col : col0;
			col1 = col > col1 ? col : col1;
			row0 = row < row0 ? row : row0;
			row1 = row > row1 ? row : row1;
		}
		// Wholly behind the eye, it cannot show on the frame.
		if (farthest <= 0.0f)
			continue;
		MoverRect rect;
		if (!(nearest >= kMoverNear))
		{
			float gap2 = 0.0f;
			for (int j = 0; j < 3; j++)
			{
				const float below = low[j] - cam.m_origin[j], above = cam.m_origin[j] - high[j];
				const float gap = below > above ? below : above;
				gap2 += gap > 0.0f ? gap * gap : 0.0f;
			}
			if (gap2 <= kAircraftReach * kAircraftReach)
				continue;
			rect.m_col0 = rect.m_row0 = 0;
			rect.m_col1 = width - 1;
			rect.m_row1 = height - 1;
			rect.m_depth = INFINITY;
			out.push_back(rect);
			continue;
		}
		if (!(col1 >= -kMoverMargin && col0 <= width + kMoverMargin && row1 >= -kMoverMargin && row0 <= height + kMoverMargin))
			continue;
		// Held to the frame before the conversion, so a corner just in front of the eye cannot overflow an int.
		rect.m_col0 = (int)floor(col0 > -1.0 ? col0 : -1.0) - kMoverMargin;
		rect.m_col1 = (int)ceil(col1 < width ? col1 : (double)width) + kMoverMargin;
		rect.m_row0 = (int)floor(row0 > -1.0 ? row0 : -1.0) - kMoverMargin;
		rect.m_row1 = (int)ceil(row1 < height ? row1 : (double)height) + kMoverMargin;
		rect.m_depth = farthest;
		out.push_back(rect);
	}
}

// Second pass over one tile: the exact anti-aliasing a ray caster can afford. For every edge pixel
// the triangles its own ray and its four neighbours' rays landed on are put back onto the frame and
// the pixel square is clipped against each, front to back, so each surface gets exactly the share of
// the pixel it covers, with the colour pass one already shaded for it. A body in front of the pixel's
// own hit is composited over it at its true width, so a thin pole or cable stops breaking into dots.
// Whatever share is left uncovered is asked with one probe ray at its centre, which is the only ray
// this pass casts. Depth and segmentation stay as the first ray wrote them. With linear light every
// byte that enters the blend is decoded first and the blend is encoded once on the write.
void refineTile(const TileJob& job, const CameraSetup& setup, const SwarmRaycast::Target& target, const EdgeScratch& scratch,
				int row0, int row1, int col0, int col1,
				RTCIntersectArguments* args, RTCOccludedArguments* shadowArgs, ProjectionCache& cache, const std::vector<MoverRect>& movers)
{
	const Camera& cam = setup.m_cam;
	const int width = job.m_width;
	const int height = job.m_height;
	const int* ids = &scratch.m_ids[0];
	const HitId* hits = &scratch.m_hits[0];
	const unsigned char* rgb1 = &scratch.m_rgb1[0];
	const bool linear = job.m_shading->m_linearLight || job.m_shading->m_daylight;
	const double halfX = 1.0 / (double)width;
	const double halfY = 1.0 / (double)height;
	const double squareArea = 4.0 * halfX * halfY;
	const float tolerance = job.m_shading->m_edgeOutline ? kOutlineTolerance : kEdgeTolerance;
	for (int row = row0; row < row1; row++)
	{
		const double ndcY = pixelNdcY(row, height);
		for (int col = col0; col < col1; col++)
		{
			if (!isEdge(ids, &scratch.m_inverseEyeDepth[0], width, height, row, col, tolerance))
				continue;
			const size_t offset = (size_t)row * width + col;
			const double ndcX = pixelNdcX(col, width);
			// The square is the box filter around the sample: half a pixel each way.
			Polygon square;
			square.m_n = 4;
			square.m_x[0] = ndcX - halfX; square.m_y[0] = ndcY - halfY;
			square.m_x[1] = ndcX + halfX; square.m_y[1] = ndcY - halfY;
			square.m_x[2] = ndcX + halfX; square.m_y[2] = ndcY + halfY;
			square.m_x[3] = ndcX - halfX; square.m_y[3] = ndcY + halfY;

			// Candidates, each triangle once: bodies in front of this pixel's hit from the four neighbours,
			// then the pixel's own triangle, then the neighbours on the same body (a crease or a facet).
			// A body behind is never assumed: what shows past the pixel's own surface is asked with a ray.
			const int id = ids[offset];
			const bool hasOwn = hits[offset].m_prim != RTC_INVALID_GEOMETRY_ID;
			const float ownDepth = target.m_depth[offset];
			const size_t neighbours[4] = {col > 0 ? offset - 1 : offset, col + 1 < width ? offset + 1 : offset,
										  row > 0 ? offset - width : offset, row + 1 < height ? offset + width : offset};
			Candidate candidates[9];
			int numCandidates = 0;
			for (int pass = 0; pass < 3; pass++)
			{
				if (pass == 1)
				{
					if (!hasOwn)
						continue;
					candidates[numCandidates].m_hit = hits[offset];
					candidates[numCandidates].m_rgb = rgb1 + offset * 3;
					candidates[numCandidates].m_depth = ownDepth;
					numCandidates++;
					continue;
				}
				for (int n = 0; n < 4; n++)
				{
					const size_t at = neighbours[n];
					if (at == offset || hits[at].m_prim == RTC_INVALID_GEOMETRY_ID)
						continue;
					// Clip depth grows towards the camera, so a larger value is a body in front.
					const bool inFront = !hasOwn || target.m_depth[at] > ownDepth;
					const bool wanted = pass == 0 ? (ids[at] != id && inFront) : (ids[at] == id);
					if (!wanted)
						continue;
					bool seen = false;
					for (int c = 0; c < numCandidates && !seen; c++)
						seen = sameTriangle(candidates[c].m_hit, hits[at]);
					if (seen)
						continue;
					candidates[numCandidates].m_hit = hits[at];
					candidates[numCandidates].m_rgb = rgb1 + at * 3;
					candidates[numCandidates].m_depth = target.m_depth[at];
					numCandidates++;
				}
			}

			double covered = 0.0, colour[3] = {0.0, 0.0, 0.0}, coveredCx = 0.0, coveredCy = 0.0;
			for (int c = 0; c < numCandidates && covered < 1.0 - kCoverageEpsilon; c++)
			{
				const double *x, *y;
				double cx, cy;
				if (!projectCached(cache, job, cam, candidates[c].m_hit, x, y))
					continue;
				double share = coverage(square, squareArea, x, y, cx, cy);
				if (share > 1.0 - covered)
					share = 1.0 - covered;
				if (share <= 0.0)
					continue;
				for (int i = 0; i < 3; i++)
					colour[i] += share * (linear ? kSwarmSrgbToLinear[candidates[c].m_rgb[i]] : (double)candidates[c].m_rgb[i]);
				coveredCx += share * cx;
				coveredCy += share * cy;
				covered += share;
			}

			const double rest = 1.0 - covered;
			bool crease = job.m_shading->m_creaseFill && hasOwn && covered > 0.0;
			for (int n = 0; n < 4 && crease; n++)
				crease = neighbours[n] != offset && ids[neighbours[n]] == id && hits[neighbours[n]].m_prim != RTC_INVALID_GEOMETRY_ID;
			// One body can span far depths, as the whole forest does: a step that passes the outline rule keeps its probe.
			crease = crease && !isEdge(ids, &scratch.m_inverseEyeDepth[0], width, height, row, col, kOutlineTolerance);
			const float ownEyeDepth = -1.0f / scratch.m_inverseEyeDepth[offset];
			for (size_t m = 0; m < movers.size() && crease; m++)
				crease = !(col >= movers[m].m_col0 && col <= movers[m].m_col1 && row >= movers[m].m_row0 && row <= movers[m].m_row1 &&
						   ownEyeDepth < movers[m].m_depth);
			if (rest > kCoverageEpsilon && !crease)
			{
				const unsigned char* restColour = rgb1 + offset * 3;
				Sample probe;
				unsigned char background[3];
				if (hasOwn)
				{
					// The uncovered part is what lies beyond the pixel's own surface: ask it with one ray at
					// that part's centre, and take the sky or clear colour when the ray meets nothing.
					// The square reaches half a pixel past the frame's sides on the left column and bottom row; keepFrame widens its region by one pixel for this.
					double px = (ndcX - coveredCx) / rest, py = (ndcY - coveredCy) / rest;
					px = px < square.m_x[0] ? square.m_x[0] : (px > square.m_x[1] ? square.m_x[1] : px);
					py = py < square.m_y[0] ? square.m_y[0] : (py > square.m_y[2] ? square.m_y[2] : py);
					const float reach = probeReach(&scratch.m_inverseEyeDepth[0], width, height, row, col);
					if (traceRay(job, setup, px, py, args, shadowArgs, probe, reach) && probe.m_shaded)
						restColour = probe.m_rgb;
					else if (target.m_background)
					{
						target.m_background->pixel(row, col, background);
						restColour = background;
					}
					else
						restColour = &scratch.m_background[offset * 3];
				}
				for (int i = 0; i < 3; i++)
					colour[i] += rest * (linear ? kSwarmSrgbToLinear[restColour[i]] : (double)restColour[i]);
			}
			else if (covered < 1.0)
				// Also a crease under ER_SWARM_CREASE_FILL: what shows through between its neighbours' triangles is
				// taken to look like them, which spares a probe ray on nearly every pixel of a crown at full smoothing.
				for (int i = 0; i < 3; i++)
					colour[i] /= covered;

			unsigned char* pixel = target.m_rgb + offset * 3;
			for (int i = 0; i < 3; i++)
			{
				if (linear)
				{
					pixel[i] = swarmLinearToSrgb((float)colour[i]);
					continue;
				}
				const int value = (int)(colour[i] + 0.5);
				pixel[i] = (unsigned char)(value < 0 ? 0 : (value > 255 ? 255 : value));
			}
		}
	}
}

// Every pixel of a camera that traces no ray takes its background, when it has one.
void fillBackground(const SwarmRaycast::Target& target, int width, int height)
{
	if (!target.m_background || !target.m_rgb)
		return;
	for (int row = 0; row < height; row++)
		for (int col = 0; col < width; col++)
			target.m_background->pixel(row, col, &target.m_rgb[((size_t)row * width + col) * 3]);
}

// The camera chain on one camera's finished frame: thermal develops the radiance into the bytes; the low-light camera
// and the grey of the near infrared work on the colour, sky and edges included.
void developCamera(const SwarmRaycastShading* shading, const float* radiance, int width, int height, int threads, unsigned char* rgb)
{
	if (!shading || !rgb)
		return;
	if (shading->m_thermal)
	{
		SwarmThermal::develop(radiance, width, height, shading->m_thermalSeed, threads, rgb);
		return;
	}
	if (!shading->m_lowLight && !shading->m_nearInfrared)
		return;
	SwarmLowLight::Settings camera;
	camera.m_lowLight = shading->m_lowLight;
	camera.m_grey = shading->m_nearInfrared;
	camera.m_photons = shading->m_sensorPhotons;
	camera.m_readNoise = shading->m_sensorReadNoise;
	camera.m_gainCap = shading->m_sensorGainCap;
	camera.m_seed = shading->m_sensorSeed;
	for (int k = 0; k < 3; k++)
		camera.m_whiteBalance[k] = shading->m_whiteBalance[k];
	// Every pass splits its rows over the render threads; the bytes are the same at any count.
	SwarmLowLight::develop(rgb, width, height, camera, threads);
}

// Everything a lone camera's frame reads but the scene: the caller's bytes, the frame size, projection and view, which
// buffers it writes, cut-outs, and the shading but its grain seeds, which only the camera chain reads.
void frameKey(std::vector<unsigned char>& key, const void* reuseKey, size_t reuseKeyBytes, const SwarmRaycast::Target& target,
			  const float projMat[16], int width, int height, const SwarmRaycastShading* shading, bool alphaCutout)
{
	const int layout[7] = {width, height, target.m_seg != 0, target.m_rgb != 0, target.m_background != 0, shading != 0, alphaCutout};
	const unsigned char* parts[4] = {(const unsigned char*)reuseKey, (const unsigned char*)layout, (const unsigned char*)projMat, (const unsigned char*)target.m_view};
	const size_t sizes[4] = {reuseKeyBytes, sizeof(layout), 16 * sizeof(float), 16 * sizeof(float)};
	key.clear();
	for (int i = 0; i < 4; i++)
		key.insert(key.end(), parts[i], parts[i] + sizes[i]);
	if (!shading)
		return;
	SwarmRaycastShading seedless = *shading;
	seedless.m_thermalSeed = 0;
	seedless.m_sensorSeed = 0;
	key.insert(key.end(), (const unsigned char*)&seedless, (const unsigned char*)&seedless + sizeof(seedless));
}
}  // namespace

void SwarmRaycast::render(const Target* targets, int numTargets, const float projMat[16], int width, int height,
						  const SwarmRaycastShading* shading, int threads, bool alphaCutout, const void* reuseKey, size_t reuseKeyBytes) const
{
	if (!m_data->m_top || m_data->m_objects.empty() || width <= 0 || height <= 0 || numTargets <= 0)
	{
		for (int i = 0; i < numTargets && width > 0 && height > 0; i++)
			fillBackground(targets[i], width, height);
		return;
	}
	if (threads < 1)
		threads = 1;

	TileJob job;
	job.m_top = m_data->m_top;
	job.m_instances = &m_data->m_byGeomId;
	job.m_members = &m_data->m_members;
	job.m_batches = &m_data->m_batches;
	job.m_forestId = m_data->m_forestId;
	job.m_shading = shading;
	job.m_staticId = m_data->m_staticInstanceId;
	job.m_width = width;
	job.m_height = height;
	const bool thermal = shading && shading->m_thermal;
	// Thermal reads every texture at its footprint, for the detail it takes from the mip chain.
	job.m_filtered = shading && (shading->m_textureFilter || thermal);
	// The map is laid out here; the pixel loop casts each block of it the first time a pixel reads one.
	job.m_shadowMap = 0;
	job.m_shadowCore = 0;
	job.m_shelterMap = 0;
	job.m_movers = 0;
	if (thermal)
	{
		// The shelter map every frame, the sun's map only while the sun is up; movers leave no warm print of their shade.
		const int upAxis = shading->m_glint.m_upAxis;
		job.m_sky = SwarmThermal::sky(shading->m_airTemperature, shading->m_skyTemperature);
		m_data->prepareShelterMap(upAxis, alphaCutout, threads);
		job.m_shelterMap = &m_data->m_shelterMap;
		if (shading->m_lightDir[upAxis] > 0.0f)
		{
			m_data->prepareShadowMap(shading->m_lightDir, alphaCutout, shading->m_leafNoShadow, m_data->m_coreRadius, upAxis, threads);
			job.m_shadowMap = &m_data->m_shadowMap;
		}
	}
	else if (shading && shading->m_shadow && shading->m_shadowMap)
	{
		m_data->prepareShadowMap(shading->m_lightDir, alphaCutout, shading->m_leafNoShadow, shading->m_daylight ? shading->m_shadowCoreRadius : 0.0f, shading->m_glint.m_upAxis, threads);
		job.m_shadowMap = &m_data->m_shadowMap;
		if (shading->m_daylight && m_data->m_shadowCore.m_built)
			job.m_shadowCore = &m_data->m_shadowCore;
		// With no mover attached the mover tree is empty, and a ray into it can only say lit.
		if (shading->m_moverShadow && m_data->m_moverCount > 0)
			job.m_movers = m_data->m_movers;
	}
	job.m_moverShade = 0;
	if (job.m_movers && castMoverShade(m_data->m_byGeomId, shading->m_lightDir, m_data->m_moverShade))
		job.m_moverShade = &m_data->m_moverShade;
	if (job.m_filtered)
	{
		// Mip chains are built once here, on one thread, so the pixel loop below only reads them.
		for (std::map<TinyRenderObjectData*, ObjectState>::const_iterator it = m_data->m_objects.begin(); it != m_data->m_objects.end(); ++it)
			it->first->m_model->buildMipmaps();
	}

	// The edge pass runs only for a camera that is both shaded and given a depth buffer to test on.
	const size_t numPixels = (size_t)width * height;
	std::vector<CameraSetup> setups((size_t)numTargets);
	for (int i = 0; i < numTargets; i++)
	{
		CameraSetup& setup = setups[(size_t)i];
		setup.m_valid = setupCamera(targets[i].m_view, projMat, setup.m_cam);
		if (!setup.m_valid)
		{
			fillBackground(targets[i], width, height);
			continue;
		}
		// The unnormalised ray direction is affine in the pixel position, so the direction one pixel to
		// the right or one row up is the pixel's own direction plus a constant step.
		for (int k = 0; k < 3; k++)
		{
			setup.m_stepX[k] = (float)((setup.m_cam.m_far[1][k] - setup.m_cam.m_near[1][k]) * (2.0 / (double)width));
			setup.m_stepY[k] = (float)((setup.m_cam.m_far[2][k] - setup.m_cam.m_near[2][k]) * (2.0 / (double)height));
		}
	}
	job.m_pixelSpread = 0.0f;
	if (job.m_filtered && (shading->m_daylight || thermal) && numTargets > 0 && setups[0].m_valid)
	{
		double step = 0.0, centre = 0.0;
		for (int k = 0; k < 3; k++)
		{
			step += (double)setups[0].m_stepX[k] * setups[0].m_stepX[k];
			const double axis = setups[0].m_cam.m_far[0][k] - setups[0].m_cam.m_near[0][k];
			centre += axis * axis;
		}
		job.m_pixelSpread = centre > 0.0 ? (float)sqrt(step / centre) : 0.0f;
	}

	// A lone camera with a frame big enough keeps a memory of its lens; asked again for its kept frame, it gives it back.
	HitMemory* memory = numTargets == 1 && setups[0].m_valid && numPixels >= kHintMinPixels ? m_data->hitMemory(width, height, projMat) : 0;
	std::vector<unsigned char> key;
	if (memory && reuseKey)
	{
		frameKey(key, reuseKey, reuseKeyBytes, targets[0], projMat, width, height, shading, alphaCutout);
		if (m_data->reuseFrame(*memory, key, targets[0], numPixels))
		{
			developCamera(shading, memory->m_radiance.empty() ? 0 : &memory->m_radiance[0], width, height, threads, targets[0].m_rgb);
			return;
		}
	}

	// The depth hint: the last frame of the same lens, its hits put onto this frame's pixels, tells each ray about how
	// far it has to search.
	job.m_hintFar = 0;
	job.m_hitPoints = 0;
	float viewProj[4][4];
	std::vector<float*> hintFrames;
	const float* lastPoints = 0;
	if (memory)
	{
		const Camera& cam = setups[0].m_cam;
		if (memory->m_points.size() == numPixels * 3)
		{
			for (int r = 0; r < 4; r++)
				for (int c = 0; c < 4; c++)
					viewProj[r][c] = (float)cam.m_viewProj[r][c];
			// One frame of farthest hits per render thread, the first being the hint itself; the threads fill them inside
			// the pixel loop's parallel region.
			m_data->m_hintFar.resize(numPixels);
			m_data->m_hintFrames.resize((size_t)threads - 1);
			hintFrames.push_back(&m_data->m_hintFar[0]);
			for (int i = 0; i < threads - 1; i++)
			{
				m_data->m_hintFrames[(size_t)i].resize(numPixels);
				hintFrames.push_back(&m_data->m_hintFrames[(size_t)i][0]);
			}
			lastPoints = &memory->m_points[0];
			job.m_hintFar = &m_data->m_hintFar[0];
		}
		memory->m_points.resize(numPixels * 3);
		job.m_hitPoints = &memory->m_points[0];
	}

	std::vector<EdgeScratch> scratch((shading && shading->m_edgeAntialias && !thermal) ? (size_t)numTargets : 0);
	std::vector<float> radiance(thermal ? numPixels * (size_t)numTargets : 0);
	std::vector<std::vector<MoverRect> > movers((size_t)numTargets);
	for (size_t i = 0; i < scratch.size() && shading->m_creaseFill; i++)
		if (setups[i].m_valid)
			moverRects(m_data->m_byGeomId, setups[i].m_cam, width, height, movers[i]);
	for (size_t i = 0; i < scratch.size(); i++)
	{
		if (!setups[i].m_valid || !targets[i].m_rgb || !targets[i].m_depth)
			continue;
		HitId none;
		none.m_inst = none.m_geom = none.m_prim = none.m_inst1 = none.m_instPrim1 = RTC_INVALID_GEOMETRY_ID;
		scratch[i].m_ids.assign(numPixels, -1);
		scratch[i].m_hits.assign(numPixels, none);
		scratch[i].m_inverseEyeDepth.assign(numPixels, 0.0f);
		if (!targets[i].m_background)
			scratch[i].m_background.assign(targets[i].m_rgb, targets[i].m_rgb + numPixels * 3);
	}

	// The tile grid is fixed by the frame size alone: tile k covers the same pixels of the same camera
	// whatever the thread count. Each thread takes the next untraced tile, so none idles while another is left
	// with the slow ones; whichever thread traces a tile, its pixels come out the same.
	const int tilesX = (width + kTileSize - 1) / kTileSize;
	const int tilesY = (height + kTileSize - 1) / kTileSize;
	const int tilesPerCamera = tilesX * tilesY;
	const int numTiles = tilesPerCamera * numTargets;
	std::atomic<int> nextTile[2];
	nextTile[0] = 0;
	nextTile[1] = 0;
	float farthest = 0.0f;

#pragma omp parallel num_threads(threads)
	{
		QueryContext ctx;
		rtcInitRayQueryContext(&ctx.m_context);
		ctx.m_instances = job.m_instances;
		ctx.m_batches = job.m_batches;
		ctx.m_forestId = job.m_forestId;
		ctx.m_anyWinding = false;
		ctx.m_alphaCutout = alphaCutout;
		ctx.m_leafNoShadow = shading && shading->m_leafNoShadow;
		ctx.m_pixelSpread = job.m_pixelSpread;
		ctx.m_farthest = 0.0f;
		RTCIntersectArguments args;
		rtcInitIntersectArguments(&args);
		args.context = &ctx.m_context;
		RTCOccludedArguments shadowArgs;
		rtcInitOccludedArguments(&shadowArgs);
		shadowArgs.context = &ctx.m_context;

		if (lastPoints)
		{
			// Each thread puts its share of the remembered hits into its own frame, then each pixel takes the farthest of
			// the frames: a maximum, so the hint is the same whichever thread took which hit.
#ifdef _OPENMP
			const int team = omp_get_num_threads(), member = omp_get_thread_num();
#else
			const int team = 1, member = 0;
#endif
			float* own = hintFrames[(size_t)member];
			for (size_t i = 0; i < numPixels; i++)
				own[i] = -1.0f;
#pragma omp for schedule(static)
			for (long long i = 0; i < (long long)numPixels; i++)
				hintPoint(viewProj, lastPoints + i * 3, width, height, own);
#pragma omp for schedule(static)
			for (long long i = 0; i < (long long)numPixels; i++)
			{
				float far = hintFrames[0][i];
				for (int k = 1; k < team; k++)
					far = hintFrames[(size_t)k][i] > far ? hintFrames[(size_t)k][i] : far;
				hintFrames[0][i] = far;
			}
		}

		// Pass 2 reads the neighbours pass 1 wrote and the colours pass 1 shaded, so every thread
		// finishes pass 1, and the pass-1 colours are copied aside, before any thread starts pass 2.
		const int passes = scratch.empty() ? 1 : 2;
		ProjectionCache cache;
		int cacheCamera = -1;
		for (int pass = 0; pass < passes; pass++)
		{
			if (pass == 1)
			{
#pragma omp barrier
#pragma omp single
				for (size_t i = 0; i < scratch.size(); i++)
					if (!scratch[i].m_ids.empty())
						scratch[i].m_rgb1.assign(targets[i].m_rgb, targets[i].m_rgb + numPixels * 3);
			}
			for (int tile = nextTile[pass]++; tile < numTiles; tile = nextTile[pass]++)
			{
				const int camIndex = tile / tilesPerCamera;
				if (!setups[(size_t)camIndex].m_valid)
					continue;
				const int local = tile - camIndex * tilesPerCamera;
				const int row0 = (local / tilesX) * kTileSize;
				const int col0 = (local % tilesX) * kTileSize;
				const int row1 = row0 + kTileSize < height ? row0 + kTileSize : height;
				const int col1 = col0 + kTileSize < width ? col0 + kTileSize : width;
				EdgeScratch* edge = (!scratch.empty() && !scratch[(size_t)camIndex].m_ids.empty()) ? &scratch[(size_t)camIndex] : 0;
				float* cameraRadiance = thermal ? &radiance[(size_t)camIndex * numPixels] : 0;
				if (pass == 0)
					renderTile(job, setups[(size_t)camIndex], targets[camIndex], edge, cameraRadiance, row0, row1, col0, col1, &args, &shadowArgs);
				else if (edge)
				{
					// A projection belongs to one camera, so the cache starts over when this thread moves to another.
					if (camIndex != cacheCamera)
					{
						memset(cache.m_used, 0, sizeof(cache.m_used));
						cacheCamera = camIndex;
					}
					refineTile(job, setups[(size_t)camIndex], targets[camIndex], *edge, row0, row1, col0, col1, &args, &shadowArgs, cache,
							   movers[(size_t)camIndex]);
				}
			}
		}
#pragma omp critical
		if (farthest == farthest && !(ctx.m_farthest <= farthest))
			farthest = ctx.m_farthest;
	}
	if (memory && reuseKey)
		m_data->keepFrame(*memory, key, setups[0].m_cam, targets[0], radiance, farthest, shading, width, height);

	// The camera chain needs the whole frame, so it runs once every ray has landed.
	for (int i = 0; i < numTargets; i++)
		if (setups[(size_t)i].m_valid)
			developCamera(shading, thermal ? &radiance[(size_t)i * numPixels] : 0, width, height, threads, targets[i].m_rgb);
}
