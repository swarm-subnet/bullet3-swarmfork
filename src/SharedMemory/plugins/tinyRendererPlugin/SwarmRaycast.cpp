#include "SwarmRaycast.h"

#include <embree4/rtcore.h>
#include <math.h>
#include <algorithm>
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
#if defined(__AVX2__)
#include <immintrin.h>
#endif
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
#include "Bullet3Common/b3DiskCache.h"
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

// ER_SWARM_RASTER: a run of consecutive triangles of one mesh and their box, in the mesh's own frame, so a frame drops
// what lies outside its view before projecting a single corner.
struct RasterChunk
{
	float m_lo[3];
	float m_hi[3];
	unsigned m_first;
	unsigned m_count;
};

const unsigned kRasterChunkTriangles = 64;

// Cuts a mesh into chunks of kRasterChunkTriangles triangles in file order, each with the box of its corners, and
// writes the box of them all into lo and hi.
void buildRasterChunks(const std::vector<float>& vertices, const std::vector<unsigned>& indices, std::vector<RasterChunk>& out,
					   float lo[3], float hi[3])
{
	out.clear();
	for (int i = 0; i < 3; i++)
	{
		lo[i] = INFINITY;
		hi[i] = -INFINITY;
	}
	const size_t numTriangles = indices.size() / 3;
	for (size_t first = 0; first < numTriangles; first += kRasterChunkTriangles)
	{
		RasterChunk chunk;
		chunk.m_first = (unsigned)first;
		chunk.m_count = (unsigned)(numTriangles - first < kRasterChunkTriangles ? numTriangles - first : kRasterChunkTriangles);
		for (int i = 0; i < 3; i++)
		{
			chunk.m_lo[i] = INFINITY;
			chunk.m_hi[i] = -INFINITY;
		}
		for (size_t k = first * 3; k < (first + chunk.m_count) * 3; k++)
		{
			const float* v = &vertices[(size_t)indices[k] * 3];
			for (int i = 0; i < 3; i++)
			{
				chunk.m_lo[i] = v[i] < chunk.m_lo[i] ? v[i] : chunk.m_lo[i];
				chunk.m_hi[i] = v[i] > chunk.m_hi[i] ? v[i] : chunk.m_hi[i];
			}
		}
		for (int i = 0; i < 3; i++)
		{
			lo[i] = chunk.m_lo[i] < lo[i] ? chunk.m_lo[i] : lo[i];
			hi[i] = chunk.m_hi[i] > hi[i] ? chunk.m_hi[i] : hi[i];
		}
		out.push_back(chunk);
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
	// ER_SWARM_RASTER: the member's triangles in world-space chunks, cut the first time a frame paints it, and their box.
	std::vector<RasterChunk> m_chunks;
	float m_chunksLo[3];
	float m_chunksHi[3];
};

struct CachedTree;
// The process's mesh trees by vertex and triangle count, and the idle ones, least recently used first.
typedef std::multimap<std::pair<size_t, size_t>, CachedTree*> TreeCache;
typedef std::list<CachedTree*> IdleTrees;

// A mesh tree over its own copy of the arrays, kept for the process: a later world with equal arrays gets the same tree.
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
	// ER_SWARM_RASTER: the mesh's chunks in its own frame, cut the first time a frame paints it and again after a rewrite,
	// and their box.
	std::vector<RasterChunk> m_chunks;
	float m_chunksLo[3];
	float m_chunksHi[3];
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
	// ER_SWARM_RASTER: the forest tree a ray cast straight into its mesh is in, since such a hit names no instance.
	const Batch* m_directBatch;
	unsigned m_directPlacement;
};

// Casts the shadow-map cells [col0, col1) x [row0, row1) along -light; each ray depends only on the map and the tree.
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
	ctx.m_directBatch = 0;
	ctx.m_directPlacement = 0;
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

// Casts every uncast block with a cell in [col0, col1] x [row0, row1]; the first thread to claim one casts it, others wait.
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
	if (ctx->m_directBatch)
	{
		placement = ctx->m_directPlacement;
		return ctx->m_directBatch;
	}
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

// A hinted search's nearest and next kept hit, searched a window past the nearest so box rounding cannot reorder them.
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
// Below this ray-face cosine, at the window's coordinate size, the face's own distance rounds beyond the window.
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

// The ray-face cosine in world space, scaled down by how far the hit frame's coordinates outgrow the window's.
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

// One never-released device, since cached trees outlive their world; threads=1 so builds run only on committing threads.
RTCDevice processDevice()
{
	static const RTCDevice device = rtcNewDevice("threads=1,set_affinity=0");
	return device;
}

// Commits a scene on up to two render threads; Embree's build tasks, and so the tree, ignore the thread count.
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

// Weight of idle trees kept, oldest dropped first; a tree weighs its triangles plus a fixed share, so tiny ones count.
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
	// A tree another process of this machine built over the same arrays is loaded instead of built; the load checks the
	// image against the geometry and builds when they differ.
	std::string diskPath;
	std::vector<char> image;
	unsigned long long diskKey = b3DiskCacheHash("SWCTREE1", 8);
	diskKey = b3DiskCacheHash(&vertices[0], numVertices * 3 * sizeof(float), diskKey);
	diskKey = b3DiskCacheHash(&indices[0], indices.size() * sizeof(unsigned), diskKey);
	const bool onDisk = b3DiskCachePath(diskKey, "ctree", diskPath);
	if (onDisk && b3DiskCacheRead(diskPath, image) && !image.empty())
		rtcSwarmLoadTree(tree->m_scene, &image[0], image.size());
	joinCommit(tree->m_scene);
	if (onDisk && image.empty())
	{
		image.resize(rtcSwarmSaveTree(tree->m_scene, 0, 0));
		if (!image.empty() && rtcSwarmSaveTree(tree->m_scene, &image[0], image.size()) == image.size())
			b3DiskCacheWrite(diskPath, &image[0], image.size());
	}
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

// Where a lens's first rays landed last frame (world x, y, z per pixel, NaN on a miss), the next frame's depth hint.
struct HitMemory
{
	int m_width;
	int m_height;
	float m_proj[16];
	std::vector<float> m_points;
	unsigned long long m_lastUse;
	// Frame reuse: the last request's key and, after a repeat, its frame before the camera chain and the region it reads.
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

// True unless an axis separates the box (lo, hi) from a kept frame's pyramid, swept along the light; NaN answers true.
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

// The movers' boxes seen from the light as a grid; a shadow ray from a point under a clear cell cannot meet a mover.
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

// Lenses remembered at once (wide, zoom, thermal), and the smallest frame worth a memory, so the laser keeps none.
const int kHitMemories = 4;
const size_t kHintMinPixels = 64 * 64;

// ER_SWARM_RASTER. What a cut-out test needs of a painted body: its texture and mesh as the hit filter reads them, and
// for a mover the inverse of its transform's 3x3 part, which takes a camera ray into the mesh's frame as Embree does.
struct RasterSource
{
	TinyRender::Model* m_model;
	const float* m_vertices;
	const float* m_uvs;
	const std::vector<unsigned>* m_indices;
	bool m_objectSpace;
	float m_inverse[9];
};

// A painted triangle in fixed point: three edge functions over the frame's samples, already biased by the fill rule so
// a sample on an edge two triangles share belongs to exactly one, the box of samples it may cover, the solve that gives a
// camera ray's distance and barycentrics on the world triangle, and the ids a ray hit on it would carry.
struct RasterTri
{
	long long m_edge[3];
	long long m_stepX[3];
	long long m_stepY[3];
	int m_col0;
	int m_col1;
	int m_row0;
	int m_row1;
	float m_det[3];
	float m_u[3];
	float m_v[3];
	float m_t;
	unsigned m_inst;
	unsigned m_geom;
	unsigned m_prim;
	unsigned long long m_key;
	// The body whose cut-outs a covered sample is tested on, or null when every sample is solid.
	const RasterSource* m_source;
};

// A world box only a ray can draw, a forest tree or a chunk with more triangles than its screen box has room to paint:
// the samples it may cover, its least distance from the eye, and the box itself, which a sample's ray must enter nearer
// than what was painted there for the sample to be searched.
struct RayRect
{
	int m_col0;
	int m_col1;
	int m_row0;
	int m_row1;
	float m_distance;
	float m_lo[3];
	float m_hi[3];
	// The forest tree the box holds, its batch and placement; kNoTree for a dense chunk, which only a full search draws.
	unsigned m_batch;
	unsigned m_placement;
};
const unsigned kNoTree = ~0u;

// A forest tree a sample's ray enters before the painted hit, and where it enters the tree's box.
struct TreeCandidate
{
	float m_enter;
	unsigned m_batch;
	unsigned m_placement;
};
// Trees a sample keeps before it falls back to the full search.
const int kTreeCandidates = 32;

// One render thread's share of a painted frame: its triangles and rects, and per tile the ones that reach it, a rect's
// index marked by kRectBit.
struct RasterLane
{
	std::vector<RasterTri> m_tris;
	std::vector<RayRect> m_rects;
	std::vector<std::vector<unsigned> > m_bins;
};
const unsigned kRectBit = 0x80000000u;

// One chunk to paint, of a static member or of a mover's mesh, and the body its cut-outs are tested on.
struct RasterJob
{
	const RasterChunk* m_chunk;
	const StaticMember* m_member;
	const Instance* m_instance;
	const RasterSource* m_source;
};

// The forest's trees by square cell across the ground, each tree's world box in cell order with its batch and placement.
struct ForestGrid
{
	bool m_built;
	std::vector<float> m_cellBoxes;
	std::vector<unsigned> m_cellStart;
	std::vector<float> m_treeBoxes;
	std::vector<unsigned> m_treeBatch;
	std::vector<unsigned> m_treePlacement;
};
const float kForestCell = 16.0f;
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
	// Per pixel: the camera-axis depth of the farthest remembered hit landing on it, -1 where none does.
	std::vector<float> m_hintFar;
	// The same for the share of the remembered hits each further render thread puts onto the frame.
	std::vector<std::vector<float> > m_hintFrames;
	MoverShade m_moverShade;
	// Frame reuse: scene changes but mover moves, which leave old and new boxes; m_movedBase counts boxes dropped.
	unsigned long long m_sceneRevision;
	std::vector<double> m_moved;
	unsigned long long m_movedBase;
	// ER_SWARM_RASTER: per render thread scratch, the frame's chunks and cut-out sources, and the forest's grid.
	std::vector<RasterLane> m_rasterLanes;
	std::vector<RasterJob> m_rasterJobs;
	std::vector<RasterSource> m_rasterSources;
	ForestGrid m_forestGrid;

	// Notes a mover's old and new boxes for kept frames, dropping boxes all have seen, or the frames if too many wait.
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

	// Takes the request's key and keeps its frame when it repeated the last one; farthest is how far its rays went.
	void keepFrame(HitMemory& memory, std::vector<unsigned char>& key, const Camera& cam, const SwarmRaycast::Target& target,
				   const std::vector<float>& radiance, float farthest, const SwarmRaycastShading* shading, int width, int height)
	{
		const size_t numPixels = (size_t)width * height;
		const bool repeat = memory.m_key == key;
		memory.m_key.swap(key);
		memory.m_kept = false;
		if (!repeat || !(farthest >= 0.0f && farthest < INFINITY))
			return;
		// A pixel wider each side, since edge probes on the left column and bottom row sample half a pixel past ndc -1.
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

	// Recasts one map's cast blocks under a member's vertices, a cell wider each side; the rest cast when first read.
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

	// Commits the tree's scene, or loads its Embree image; a mesh tree another world already built comes from the cache.
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
		tree.m_chunks.clear();
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
		if (changed)
			m_forestGrid.m_built = false;
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
		m_forestGrid.m_built = false;
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
		m_forestGrid.m_built = false;
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
	m_data->m_forestGrid.m_built = false;
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

// The fragment shader's interpolated normal, normalised but not turned to the camera, and its texture coordinates, from
// a hit's barycentric (u, v): shadeHit up to its texture read.
void shadeHitFrame(const HitSurface& surface, float u, float v, const float faceNormal[3], float normal[3], TinyRender::Vec2f& uv)
{
	const float weights[3] = {1.0f - u - v, u, v};
	for (int i = 0; i < 3; i++)
		normal[i] = 0.0f;
	uv = TinyRender::Vec2f(0.0f, 0.0f);
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
}

// The fragment shader from its texel on: the lit colour of the hit and its bytes.
void shadeHitFinish(const SwarmRaycastShading& shading, const HitSurface& surface, const float normal[3], TinyRender::Vec2f uv, TGAColor color,
					const float faceNormal[3], const float viewDir[3], float shadow, const float point[3], unsigned char out[3])
{
	TinyRender::Model* model = surface.m_model;
	const float nDotL = dot3(normal, shading.m_lightDir);
	float reflection[3];
	for (int i = 0; i < 3; i++)
		reflection[i] = normal[i] * (nDotL * 2.0f) - shading.m_lightDir[i];
	normalize3(reflection);
	const float specular = powInt(reflection[2] > 0.0f ? reflection[2] : 0.0f, (int)model->specular(uv));
	const float diffuse = nDotL > 0.0f ? nDotL : 0.0f;

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

void shadeHit(const SwarmRaycastShading& shading, const HitSurface& surface, const RTCHit& hit, const float faceNormal[3],
			  const float viewDir[3], float shadow, bool filtered, const float duvdx[2], const float duvdy[2],
			  float distance, const float point[3], unsigned char out[3])
{
	if (shading.m_daylight)
	{
		shadeDaylight(shading, surface, hit, faceNormal, viewDir, shadow, filtered, duvdx, duvdy, distance, out);
		return;
	}
	float normal[3];
	TinyRender::Vec2f uv;
	shadeHitFrame(surface, hit.u, hit.v, faceNormal, normal, uv);
	TGAColor color = filtered
						 ? surface.m_model->diffuseFiltered(uv, TinyRender::Vec2f(duvdx[0], duvdx[1]), TinyRender::Vec2f(duvdy[0], duvdy[1]))
						 : surface.m_model->diffuse(uv);
	shadeHitFinish(shading, surface, normal, uv, color, faceNormal, viewDir, shadow, point, out);
}

// A hit's shading normal turned to the camera and its texture coordinates, from its barycentric (u, v): the surface
// up to its texture read.
void surfaceFrame(const HitSurface& surface, float u, float v, const float faceNormal[3], float normal[3], TinyRender::Vec2f& uv)
{
	const float weights[3] = {1.0f - u - v, u, v};
	uv = TinyRender::Vec2f(0.0f, 0.0f);
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
}

// The linear tint of a surface from the texel read at its hit: texture times object colour.
void surfaceTint(const HitSurface& surface, TGAColor color, float base[3])
{
	const TinyRender::Vec4f& rgba = surface.m_model->getColorRGBA();
	for (int i = 0; i < 3; i++)
		base[i] = kSwarmSrgbToLinear[(unsigned char)(color[i] * rgba[i])];
}

// The surface at a hit: its shading normal turned to the camera and the linear tint, texture times object colour.
void surfaceAt(const HitSurface& surface, const RTCHit& hit, const float faceNormal[3], bool filtered, const float duvdx[2], const float duvdy[2],
			   float normal[3], float base[3], TinyRender::Vec2f* uvOut = 0)
{
	TinyRender::Vec2f uv;
	surfaceFrame(surface, hit.u, hit.v, faceNormal, normal, uv);
	TGAColor color = filtered
						 ? surface.m_model->diffuseFiltered(uv, TinyRender::Vec2f(duvdx[0], duvdx[1]), TinyRender::Vec2f(duvdy[0], duvdy[1]), 4)
						 : surface.m_model->diffuse(uv);
	surfaceTint(surface, color, base);
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

// daylightLight for count surfaces under a sky: eight at a time in vector lanes where the compiler has them, each lane
// daylightLight's steps in their order, the sky's light and glint looked up eight at a time, so every colour is the
// one daylightLight gives.
void daylightLightMany(const SwarmRaycastShading& shading, const HitSurface* const* surfaces, const float (*normals)[3], const float (*bases)[3],
					   const float (*viewDirs)[3], const float* shadows, int count, float (*lit)[3])
{
	int k = 0;
#if defined(__GNUC__)
	typedef float Lanes __attribute__((vector_size(32)));
	typedef int Ints __attribute__((vector_size(32)));
	const Lanes zero = {0, 0, 0, 0, 0, 0, 0, 0}, one = zero + 1.0f;
	for (; shading.m_sky && k + 8 <= count; k += 8)
	{
		Lanes n[3], base[3], toCamera[3], shadow, specular[3];
		Ints leaf;
		for (int l = 0; l < 8; l++)
		{
			const HitSurface& surface = *surfaces[k + l];
			const float* spec = &surface.m_model->getSpecularColor()[0];
			for (int i = 0; i < 3; i++)
			{
				n[i][l] = normals[k + l][i];
				base[i][l] = bases[k + l][i];
				toCamera[i][l] = -viewDirs[k + l][i];
				specular[i][l] = spec[i];
			}
			shadow[l] = shadows[k + l];
			leaf[l] = surface.m_doubleSided && surface.m_hasAlpha ? -1 : 0;
		}
		const Lanes nDotL = (n[0] * shading.m_lightDir[0] + n[1] * shading.m_lightDir[1]) + n[2] * shading.m_lightDir[2];
		Lanes direct = nDotL > zero ? nDotL : zero;
		direct = (leaf & (nDotL < zero)) ? -nDotL * kLeafTransmit : direct;
		float nx[8], ny[8], nz[8], skyLight[8][3];
		memcpy(nx, &n[0], sizeof(nx));
		memcpy(ny, &n[1], sizeof(ny));
		memcpy(nz, &n[2], sizeof(nz));
		shading.m_sky->irradianceMany(nx, ny, nz, 8, skyLight);
		Lanes out[3];
		for (int i = 0; i < 3; i++)
		{
			Lanes sky;
			for (int l = 0; l < 8; l++)
				sky[l] = skyLight[l][i];
			out[i] = base[i] * (shading.m_ambientCoeff * sky + shadow * shading.m_diffuseCoeff * direct * shading.m_lightColor[i]);
		}
		const Ints glint = (specular[0] > zero) | (specular[1] > zero) | (specular[2] > zero);
		bool anyGlint = false;
		for (int l = 0; l < 8; l++)
			anyGlint = anyGlint || glint[l];
		if (shading.m_glint.m_enabled && anyGlint)
		{
			Lanes nDotV = (n[0] * toCamera[0] + n[1] * toCamera[1]) + n[2] * toCamera[2];
			nDotV = nDotV < zero ? zero : (nDotV > one ? one : nDotV);
			Lanes rise = (kGlassFlatUntilCos - nDotV) / (kGlassFlatUntilCos - kGlassMirrorFromCos);
			rise = rise < zero ? zero : (rise > one ? one : rise);
			const Lanes fresnel = kGlassFlat + (1.0f - kGlassFlat) * rise * rise * (3.0f - 2.0f * rise);
			float mirror[3][8], sky[8][3];
			for (int i = 0; i < 3; i++)
			{
				const Lanes m = n[i] * (2.0f * nDotV) - toCamera[i];
				memcpy(mirror[i], &m, sizeof(m));
			}
			shading.m_sky->radianceMany(mirror[0], mirror[1], mirror[2], 8, sky);
			for (int i = 0; i < 3; i++)
			{
				Lanes s;
				for (int l = 0; l < 8; l++)
					s[l] = sky[l][i];
				const Lanes w = specular[i] * fresnel;
				out[i] = glint ? out[i] + (s - out[i]) * w : out[i];
			}
		}
		for (int l = 0; l < 8; l++)
			for (int i = 0; i < 3; i++)
				lit[k + l][i] = out[i][l];
	}
#endif
	for (; k < count; k++)
		daylightLight(shading, *surfaces[k], normals[k], bases[k], viewDirs[k], shadows[k], lit[k]);
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

// A glass-backed module seen in daylight, in linear light: the sky its pane mirrors, and through the pane its backsheet
// in the pane's colour, lit and shadowed as the pane, standing in for a ray behind it.
void moduleLight(const SwarmRaycastShading& shading, const HitSurface& surface, const float normal[3], const float base[3],
				 const float viewDir[3], float shadow, float lit[3])
{
	float sky[3], skyLight[3];
	const float through = paneLight(shading, normal, viewDir, sky);
	if (shading.m_sky)
		shading.m_sky->irradiance(normal, skyLight);
	else
		for (int i = 0; i < 3; i++)
			skyLight[i] = shading.m_ambientColor[i];
	const float nDotL = dot3(normal, shading.m_lightDir);
	const float direct = nDotL > 0.0f ? nDotL : 0.0f;
	const TinyRender::Vec4f& rgba = surface.m_model->getColorRGBA();
	for (int i = 0; i < 3; i++)
	{
		const float backing = kSwarmSrgbToLinear[(unsigned char)(kPaneBacking * rgba[i])];
		lit[i] = (1.0f - through) * sky[i] + through * base[i] * backing * (shading.m_ambientCoeff * skyLight[i] + shadow * shading.m_diffuseCoeff * direct * shading.m_lightColor[i]);
	}
}

// A daylight colour on its way to the byte: its linear light, the horizon colour its haze blends towards and its distance.
struct DaylightColour
{
	float m_lit[3];
	float m_horizon[3];
	float m_distance;
};

// Everything the byte of a lit colour needs from the scene: the horizon colour is only looked up under haze. With
// skyLater, a sky's horizon colour is left for the caller to look up with others.
void daylightPrepare(const SwarmRaycastShading& shading, const float lit[3], const float viewDir[3], float distance, DaylightColour& colour,
					 bool skyLater = false)
{
	for (int i = 0; i < 3; i++)
		colour.m_lit[i] = lit[i];
	colour.m_distance = distance;
	if (!(shading.m_hazeDistance > 0.0f) || (skyLater && shading.m_sky))
		return;
	if (shading.m_sky)
	{
		// The haze takes the sky's colour just above the horizon in the direction of view.
		float level[3] = {viewDir[0], viewDir[1], viewDir[2]};
		level[shading.m_glint.m_upAxis] = 0.02f;
		shading.m_sky->radiance(level[0], level[1], level[2], colour.m_horizon);
	}
	else
		for (int i = 0; i < 3; i++)
			colour.m_horizon[i] = swarmUnitToLinear(shading.m_glint.m_skyHorizon[i]);
}

// Linear light to the byte: haze by distance towards the horizon colour, exposure, the film curve.
void daylightFinish(const SwarmRaycastShading& shading, const DaylightColour& colour, unsigned char out[3])
{
	float lit[3] = {colour.m_lit[0], colour.m_lit[1], colour.m_lit[2]};
	if (shading.m_hazeDistance > 0.0f)
	{
		const float haze = 1.0f - (float)swarmExp(-(double)colour.m_distance / (double)shading.m_hazeDistance);
		for (int i = 0; i < 3; i++)
			lit[i] = lit[i] + (colour.m_horizon[i] - lit[i]) * haze;
	}

	float exposed[3], display[3];
	for (int i = 0; i < 3; i++)
		exposed[i] = lit[i] * shading.m_exposure;
	SwarmAgx::apply(exposed, display);
	for (int i = 0; i < 3; i++)
		out[i] = SwarmAgx::toByte(display[i]);
}

// The byte of a lit colour in one go.
void daylightWrite(const SwarmRaycastShading& shading, const float lit[3], const float viewDir[3], float distance, unsigned char out[3])
{
	DaylightColour colour;
	daylightPrepare(shading, lit, viewDir, distance, colour);
	daylightFinish(shading, colour, out);
}

// daylightFinish for many colours: eight at a time in vector lanes where the compiler has them, each lane the steps of
// one colour in their order, so every byte is the one daylightFinish writes.
void daylightFinishAll(const SwarmRaycastShading& shading, const DaylightColour* colours, int count, unsigned char* const* outs)
{
	int k = 0;
#if defined(__GNUC__)
	const bool haze = shading.m_hazeDistance > 0.0f;
	for (; k + 8 <= count; k += 8)
	{
		SwarmLanes8 lit[3], display[3];
		for (int i = 0; i < 3; i++)
			for (int l = 0; l < 8; l++)
				lit[i][l] = colours[k + l].m_lit[i];
		if (haze)
		{
			SwarmDoubles4 low, high;
			for (int l = 0; l < 4; l++)
			{
				low[l] = -(double)colours[k + l].m_distance;
				high[l] = -(double)colours[k + 4 + l].m_distance;
			}
			low = swarmExp(low / (double)shading.m_hazeDistance);
			high = swarmExp(high / (double)shading.m_hazeDistance);
			SwarmLanes8 haze8;
			for (int l = 0; l < 4; l++)
			{
				haze8[l] = 1.0f - (float)low[l];
				haze8[4 + l] = 1.0f - (float)high[l];
			}
			for (int i = 0; i < 3; i++)
			{
				SwarmLanes8 horizon;
				for (int l = 0; l < 8; l++)
					horizon[l] = colours[k + l].m_horizon[i];
				lit[i] = lit[i] + (horizon - lit[i]) * haze8;
			}
		}
		for (int i = 0; i < 3; i++)
			lit[i] = lit[i] * shading.m_exposure;
		SwarmAgx::apply8(lit, display);
		for (int i = 0; i < 3; i++)
		{
			const SwarmLaneMask8 bytes = __builtin_convertvector(display[i] * 255.0f + 0.5f, SwarmLaneMask8);
			for (int l = 0; l < 8; l++)
				outs[k + l][i] = (unsigned char)bytes[l];
		}
	}
#endif
	for (; k < count; k++)
		daylightFinish(shading, colours[k], outs[k]);
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

struct RasterFrame;

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
	// ER_SWARM_RASTER: the frame's painted triangles and rects, when the camera paints instead of searching; null otherwise.
	const RasterFrame* m_raster;
	bool m_alphaCutout;
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
// What a daylight hit's shading needs once its tile reads the textures: the surface, where on its triangle the hit
// lies, the face normal, the texture footprint, the view direction, the shadow and the distance.
struct ShadeWait
{
	const HitSurface* m_surface;
	float m_u;
	float m_v;
	float m_faceNormal[3];
	float m_duvdx[2];
	float m_duvdy[2];
	float m_dir[3];
	float m_shadow;
	float m_distance;
	// Which shading waits: an opaque daylight surface, a glass-backed module in daylight, or the fragment shader.
	enum Kind
	{
		kDaylight,
		kModule,
		kFragment
	} m_kind;
	// For the fragment shader: the hit point, which the spot light reads.
	float m_point[3];
	// The footprint is worked out with the tile's others when m_footprintLater; the camera ray and wound normal it needs.
	// The shadow likewise when m_shadowLater, from the hit point and face normal.
	bool m_footprintLater;
	bool m_shadowLater;
	float m_rawDir[3];
	float m_wound[3];
};

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
	// A daylight colour left for its tile's batched write instead of written to m_rgb, when the caller asks for that.
	bool m_deferred;
	DaylightColour m_colour;
	// A hit whose shading waits for its tile, so the tile reads its samples' textures together; it is written straight
	// into the tile's slot m_shade points at.
	bool m_shadeDeferred;
	ShadeWait* m_shade;
};

// Leaves a daylight hit's shading for its tile: what the shading needs is kept in the sample.
inline void waitForTile(Sample& out, const HitSurface& surface, const RTCHit& hit, const float faceNormal[3], const float duvdx[2],
						const float duvdy[2], const float dir[3], float shadow, float distance, ShadeWait::Kind kind, const float point[3],
						bool footprintLater, const float rawDir[3], const float wound[3], bool shadowLater)
{
	ShadeWait& wait = *out.m_shade;
	wait.m_surface = &surface;
	wait.m_u = hit.u;
	wait.m_v = hit.v;
	for (int i = 0; i < 3; i++)
	{
		wait.m_faceNormal[i] = faceNormal[i];
		wait.m_dir[i] = dir[i];
	}
	for (int i = 0; i < 2; i++)
	{
		wait.m_duvdx[i] = duvdx[i];
		wait.m_duvdy[i] = duvdy[i];
	}
	wait.m_shadow = shadow;
	wait.m_distance = distance;
	wait.m_kind = kind;
	wait.m_footprintLater = footprintLater;
	wait.m_shadowLater = shadowLater;
	for (int i = 0; i < 3; i++)
	{
		wait.m_point[i] = point[i];
		wait.m_rawDir[i] = rawDir[i];
		wait.m_wound[i] = wound[i];
	}
	out.m_shadeDeferred = true;
}

// A lit daylight colour's byte now, or the colour kept in the sample for its tile's batched write.
inline void daylightOut(const SwarmRaycastShading& shading, const float lit[3], const float viewDir[3], float distance, Sample& out, bool defer)
{
	if (!defer)
	{
		daylightWrite(shading, lit, viewDir, distance, out.m_rgb);
		return;
	}
	daylightPrepare(shading, lit, viewDir, distance, out.m_colour);
	out.m_deferred = true;
}

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
// ER_SWARM_EDGE_OUTLINE: a neighbour a quarter farther (a fifth from the far side) is an edge; leaves of one crown are not.
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

// Lays the grid over the enabled movers' widened boxes across the light; false, so every ray fires, if any is not finite.
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
// The normal a shadow is looked up with: the face normal made unit, turned to the light for a leaf card when leaf
// shadows are off, so the light through it is not cancelled by the map's rule that a face turned from the sun is dark.
inline void shadowNormal(const SwarmRaycastShading* shading, const float faceNormal[3], bool leaf, float unitNormal[3])
{
	for (int i = 0; i < 3; i++)
		unitNormal[i] = faceNormal[i];
	normalize3(unitNormal);
	if (leaf && shading->m_leafNoShadow && dot3(unitNormal, shading->m_lightDir) < 0.0f)
		for (int i = 0; i < 3; i++)
			unitNormal[i] = -unitNormal[i];
}

// The map a daylight point is looked up in: the fine core grid where it covers the point, else the main map.
inline const ShadowMap* shadowMapFor(const TileJob& job, const float point[3])
{
	return (job.m_shadowCore && shadowMapCovers(*job.m_shadowCore, point)) ? job.m_shadowCore : job.m_shadowMap;
}

// The light a point keeps once the map has answered: a shadow ray into the movers or the whole scene where the map did
// not already block it, then the shadow coefficient.
float shadowFinish(const TileJob& job, const float point[3], const float unitNormal[3], float litShare, bool blocked, RTCOccludedArguments* shadowArgs)
{
	const SwarmRaycastShading* shading = job.m_shading;
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

float shadowAt(const TileJob& job, const float point[3], const float faceNormal[3], bool leaf, RTCOccludedArguments* shadowArgs)
{
	const SwarmRaycastShading* shading = job.m_shading;
	float unitNormal[3];
	shadowNormal(shading, faceNormal, leaf, unitNormal);
	float litShare = 1.0f;
	bool blocked = false;
	if (job.m_shadowMap && shading->m_daylight)
	{
		litShare = shadowMapLit(*shadowMapFor(job, point), point, unitNormal);
		blocked = litShare <= 0.0f;
	}
	else
		blocked = job.m_shadowMap && shadowMapBlocked(*job.m_shadowMap, point, unitNormal);
	return shadowFinish(job, point, unitNormal, litShare, blocked, shadowArgs);
}

// shadowMapBlocked for count points of one map at once: eight at a time in vector lanes where the build has AVX2, each
// lane shadowMapBlocked's steps on its one cell; the cells a lane reads are cast first, as shadowMapBlocked casts them.
void shadowBlockedMany(const ShadowMap& map, const float (*points)[3], const float (*normals)[3], int count, bool* blocked)
{
	int k = 0;
#if defined(__AVX2__)
	typedef float Lanes __attribute__((vector_size(32)));
	typedef int Ints __attribute__((vector_size(32)));
	const Lanes zero = {0, 0, 0, 0, 0, 0, 0, 0};
	for (; map.m_built && k + 8 <= count; k += 8)
	{
		Lanes p[3], n[3];
		for (int l = 0; l < 8; l++)
			for (int i = 0; i < 3; i++)
			{
				p[i][l] = points[k + l][i];
				n[i][l] = normals[k + l][i];
			}
		const Lanes facing = (n[0] * map.m_lightDir[0] + n[1] * map.m_lightDir[1]) + n[2] * map.m_lightDir[2];
		Lanes rel[3];
		for (int i = 0; i < 3; i++)
			rel[i] = (p[i] + n[i] * map.m_cell) - map.m_origin[i];
		const Lanes u = ((rel[0] * map.m_axisU[0] + rel[1] * map.m_axisU[1]) + rel[2] * map.m_axisU[2]) / map.m_cell;
		const Lanes v = ((rel[0] * map.m_axisV[0] + rel[1] * map.m_axisV[1]) + rel[2] * map.m_axisV[2]) / map.m_cell;
		const Ints dark = facing <= zero;
		const Ints read = ~dark & (u >= zero) & (v >= zero) & (u < (float)map.m_cols) & (v < (float)map.m_rows);
		const Ints iu = __builtin_convertvector(u, Ints), iv = __builtin_convertvector(v, Ints);
		for (int l = 0; l < 8; l++)
			if (read[l])
				ensureShadowCells(map, iu[l], iu[l], iv[l], iv[l]);
		const Lanes depth = -((rel[0] * map.m_lightDir[0] + rel[1] * map.m_lightDir[1]) + rel[2] * map.m_lightDir[2]);
		const Lanes cell = (Lanes)_mm256_i32gather_ps(map.m_depth.get(), (__m256i)(read & (iv * map.m_cols + iu)), 4);
		const Ints behind = read & (depth > cell + map.m_cell);
		for (int l = 0; l < 8; l++)
			blocked[k + l] = dark[l] || behind[l];
	}
#endif
	for (; k < count; k++)
		blocked[k] = shadowMapBlocked(map, points[k], normals[k]);
}

// shadowMapLit for count points at once, each with its own map and unit normal: eight at a time in vector lanes where
// the build has AVX2 and all eight read the inside of one map, each lane shadowMapLit's steps and its nine cells added
// in the same order; any other group point by point.
void shadowLitMany(const ShadowMap* const* maps, const float (*points)[3], const float (*normals)[3], int count, float* lit)
{
	int k = 0;
#if defined(__AVX2__)
	typedef float Lanes __attribute__((vector_size(32)));
	typedef int Ints __attribute__((vector_size(32)));
	const Lanes zero = {0, 0, 0, 0, 0, 0, 0, 0}, one = zero + 1.0f, half = zero + 0.5f;
	for (; k + 8 <= count; k += 8)
	{
		const ShadowMap& map = *maps[k];
		bool same = map.m_built;
		for (int l = 1; l < 8 && same; l++)
			same = maps[k + l] == &map;
		if (!same)
		{
			for (int l = 0; l < 8; l++)
				lit[k + l] = shadowMapLit(*maps[k + l], points[k + l], normals[k + l]);
			continue;
		}
		Lanes p[3], n[3];
		for (int l = 0; l < 8; l++)
			for (int i = 0; i < 3; i++)
			{
				p[i][l] = points[k + l][i];
				n[i][l] = normals[k + l][i];
			}
		const Lanes facing = (n[0] * map.m_lightDir[0] + n[1] * map.m_lightDir[1]) + n[2] * map.m_lightDir[2];
		Lanes rel[3];
		for (int i = 0; i < 3; i++)
			rel[i] = (p[i] + n[i] * map.m_cell) - map.m_origin[i];
		const Lanes u = ((rel[0] * map.m_axisU[0] + rel[1] * map.m_axisU[1]) + rel[2] * map.m_axisU[2]) / map.m_cell;
		const Lanes v = ((rel[0] * map.m_axisV[0] + rel[1] * map.m_axisV[1]) + rel[2] * map.m_axisV[2]) / map.m_cell;
		const Ints outside = ~(u >= zero) | ~(v >= zero) | (u >= (float)map.m_cols) | (v >= (float)map.m_rows);
		const Lanes depth = -((rel[0] * map.m_lightDir[0] + rel[1] * map.m_lightDir[1]) + rel[2] * map.m_lightDir[2]) - map.m_cell;
		const Lanes su = u - 1.0f, sv = v - 1.0f;
		const Lanes flU = (Lanes)_mm256_round_ps((__m256)su, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
		const Lanes flV = (Lanes)_mm256_round_ps((__m256)sv, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
		const Ints iu = __builtin_convertvector(flU, Ints), iv = __builtin_convertvector(flV, Ints);
		const Lanes fu = su - __builtin_convertvector(iu, Lanes), fv = sv - __builtin_convertvector(iv, Lanes);
		const Ints dark = ~(facing > zero);
		// Lanes the vector path reads: facing the light, on the map, and with all nine cells inside it.
		const Ints read = ~dark & ~outside & (iu >= 0) & (iv >= 0) & (iu + 2 < map.m_cols) & (iv + 2 < map.m_rows);
		// A lane on the map but too near its border for the inside path makes the whole group go point by point.
		bool border = false;
		for (int l = 0; l < 8; l++)
			border |= !dark[l] && !outside[l] && !read[l];
		if (border)
		{
			for (int l = 0; l < 8; l++)
				lit[k + l] = shadowMapLit(map, points[k + l], normals[k + l]);
			continue;
		}
		for (int l = 0; l < 8; l++)
			if (read[l])
				ensureShadowCells(map, iu[l], iu[l] + 2, iv[l], iv[l] + 2);
		const Lanes weightU[3] = {half * (one - fu), half, half * fu};
		const Lanes weightV[3] = {half * (one - fv), half, half * fv};
		const Ints first = (read & (iv * map.m_cols + iu));
		Lanes sum = zero;
		for (int dv = 0; dv < 3; dv++)
			for (int du = 0; du < 3; du++)
			{
				const Ints at = read & (first + dv * map.m_cols + du);
				const Lanes cell = (Lanes)_mm256_i32gather_ps(map.m_depth.get(), (__m256i)at, 4);
				sum += ~(depth > cell) & read ? weightV[dv] * weightU[du] : zero;
			}
		const Lanes result = dark ? zero : (outside ? one : sum);
		for (int l = 0; l < 8; l++)
			lit[k + l] = result[l];
	}
#endif
	for (; k < count; k++)
		lit[k] = shadowMapLit(*maps[k], points[k], normals[k]);
}

// A hit's texture footprint from the right and upper neighbours' rays on its plane; woundNormal keeps the weights' sign.
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

// footprintAt for count hits at once, each given by its camera ray, triangle corners, wound normal, barycentric (u, v)
// and corner uvs: eight at a time in vector lanes where the compiler has them, each lane footprintAt's steps in order.
// duvdx and duvdy must start at zero, as footprintAt's callers start them.
void footprintMany(const CameraSetup& setup, const float (*rawDir)[3], const float (*corners)[3][3], const float (*wound)[3], const float* hitU,
				   const float* hitV, const float (*uvs)[6], int count, float (*duvdx)[2], float (*duvdy)[2])
{
	int k = 0;
#if defined(__GNUC__)
	typedef float Lanes __attribute__((vector_size(32)));
	typedef int Ints __attribute__((vector_size(32)));
	const Lanes zero = {0, 0, 0, 0, 0, 0, 0, 0};
	for (; k + 8 <= count; k += 8)
	{
		Lanes c0[3], e1[3], e2[3], to[3], n[3], dir[3], u0, u1, u2, v0, v1, v2, wu, wv;
		for (int l = 0; l < 8; l++)
		{
			const int at = k + l;
			for (int i = 0; i < 3; i++)
			{
				c0[i][l] = corners[at][0][i];
				e1[i][l] = corners[at][1][i] - corners[at][0][i];
				e2[i][l] = corners[at][2][i] - corners[at][0][i];
				to[i][l] = corners[at][0][i] - setup.m_cam.m_origin[i];
				n[i][l] = wound[at][i];
				dir[i][l] = rawDir[at][i];
			}
			u0[l] = uvs[at][0], v0[l] = uvs[at][1], u1[l] = uvs[at][2], v1[l] = uvs[at][3], u2[l] = uvs[at][4], v2[l] = uvs[at][5];
			wu[l] = hitU[at], wv[l] = hitV[at];
		}
		const Lanes nn = (n[0] * n[0] + n[1] * n[1]) + n[2] * n[2];
		const Lanes reach = (to[0] * n[0] + to[1] * n[1]) + to[2] * n[2];
		for (int side = 0; side < 2; side++)
		{
			const float* step = side == 0 ? setup.m_stepX : setup.m_stepY;
			Lanes nd[3], w[3];
			for (int i = 0; i < 3; i++)
				nd[i] = dir[i] + step[i];
			const Lanes denom = (nd[0] * n[0] + nd[1] * n[1]) + nd[2] * n[2];
			const Ints skip = (denom == zero) | (nn == zero);
			const Lanes sd = reach / denom;
			for (int i = 0; i < 3; i++)
				w[i] = (setup.m_cam.m_origin[i] + nd[i] * sd) - c0[i];
			const Lanes t1[3] = {w[1] * e2[2] - w[2] * e2[1], w[2] * e2[0] - w[0] * e2[2], w[0] * e2[1] - w[1] * e2[0]};
			const Lanes t2[3] = {e1[1] * w[2] - e1[2] * w[1], e1[2] * w[0] - e1[0] * w[2], e1[0] * w[1] - e1[1] * w[0]};
			const Lanes bu = ((t1[0] * n[0] + t1[1] * n[1]) + t1[2] * n[2]) / nn;
			const Lanes bv = ((t2[0] * n[0] + t2[1] * n[1]) + t2[2] * n[2]) / nn;
			const Lanes outU = (u1 - u0) * (bu - wu) + (u2 - u0) * (bv - wv);
			const Lanes outV = (v1 - v0) * (bu - wu) + (v2 - v0) * (bv - wv);
			float (*out)[2] = side == 0 ? duvdx : duvdy;
			for (int l = 0; l < 8; l++)
				if (!skip[l])
				{
					out[k + l][0] = outU[l];
					out[k + l][1] = outV[l];
				}
		}
	}
#endif
	for (; k < count; k++)
	{
		HitSurface surface;
		surface.m_uvs = uvs[k];
		for (int j = 0; j < 3; j++)
		{
			surface.m_vertexIds[j] = (unsigned)j;
			for (int i = 0; i < 3; i++)
				surface.m_corners[j][i] = corners[k][j][i];
		}
		RTCHit hit;
		hit.u = hitU[k];
		hit.v = hitV[k];
		footprintAt(setup, rawDir[k], surface, wound[k], hit, duvdx[k], duvdy[k]);
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

// How far past the farthest known hit around its pixel a hinted ray searches, for the step across a slanted surface.
const float kHintSlack = 1.0f / 32.0f;
const float kHintSlackMetres = 0.25f;

// The camera-axis depth pixel (row, col) searches to: past the farthest remembered hit around it, else no limit.
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

// A camera ray's first hit, tried to `reach` plus a rounding window; any doubt reruns the full ray, so hits never change.
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

// A camera's ray through the frame position (ndcX, ndcY): its unit and raw direction, how far along it the near plane
// lies and how much further the far plane; false when the near and far points coincide.
bool pixelRay(const Camera& cam, double ndcX, double ndcY, float dir[3], float rawDir[3], float& tNear, float& length)
{
	float nearPoint[3], farPoint[3];
	planePoint(cam.m_near, ndcX, ndcY, nearPoint);
	planePoint(cam.m_far, ndcX, ndcY, farPoint);
	for (int i = 0; i < 3; i++)
		dir[i] = rawDir[i] = farPoint[i] - nearPoint[i];
	length = sqrtf(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
	if (!(length > 0.0f))
		return false;
	const float invLength = 1.0f / length;
	float toNear[3];
	for (int i = 0; i < 3; i++)
	{
		dir[i] *= invLength;
		toNear[i] = nearPoint[i] - cam.m_origin[i];
	}
	tNear = sqrtf(toNear[0] * toNear[0] + toNear[1] * toNear[1] + toNear[2] * toNear[2]);
	return true;
}

// What is already known of a ray's first hit: nothing, so it is searched with an along-ray depth hint; a hit to shade as
// it stands; a miss; or the few forest trees a nearer hit can only be in.
enum FoundKind
{
	kFoundSearch,
	kFoundHit,
	kFoundMiss,
	kFoundTrees
};

// For kFoundTrees: the painted hit, when there is one, and the forest trees the ray enters before it, which are the
// only places a nearer hit can be.
struct FoundHit
{
	FoundKind m_kind;
	float m_reach;
	bool m_painted;
	float m_t;
	RTCHit m_hit;
	const TreeCandidate* m_trees;
	int m_numTrees;
	// The sample's camera ray as painting worked it out, so it is not worked out again: false when there is none.
	bool m_ray;
	const float* m_dir;
	const float* m_rawDir;
	float m_tNear;
	float m_length;
};

// Every triangle a tile's rays landed on and what resolving it gave, so each is resolved once per tile and the tile's
// waiting hits point at it instead of carrying a copy. A tile resolves at most one triangle per sample.
struct SurfaceMemo
{
	struct Entry
	{
		unsigned m_key[5];
		bool m_known;
		int m_segmentation;
		HitSurface m_surface;
		float m_woundNormal[3];
	};
	static const int kSlots = 512;
	Entry m_entries[kTileSize * kTileSize];
	short m_slot[kSlots];
	int m_count;
	int m_last;

	// Empty, for a new tile.
	void clear()
	{
		m_count = 0;
		m_last = 0;
		memset(m_slot, 0, sizeof(m_slot));
	}

	// The entry for this triangle, made empty for the caller to fill when the tile has not met it; `added` says which.
	Entry& find(const unsigned key[5], bool& added)
	{
		added = false;
		if (m_count && memcmp(m_entries[m_last].m_key, key, sizeof(m_entries[0].m_key)) == 0)
			return m_entries[m_last];
		const unsigned mix = (key[0] * 0x9E3779B1u) ^ (key[1] * 0x85EBCA77u) ^ (key[2] * 0xC2B2AE3Du) ^ (key[3] * 0x27D4EB2Fu) ^ (key[4] * 0x165667B1u);
		for (unsigned slot = mix >> 23;; slot = (slot + 1) & (kSlots - 1))
		{
			const int at = m_slot[slot] - 1;
			if (at < 0)
			{
				m_last = m_count++;
				m_slot[slot] = (short)(m_last + 1);
				memcpy(m_entries[m_last].m_key, key, sizeof(m_entries[0].m_key));
				added = true;
				return m_entries[m_last];
			}
			if (memcmp(m_entries[at].m_key, key, sizeof(m_entries[0].m_key)) == 0)
			{
				m_last = at;
				return m_entries[at];
			}
		}
	}
};

// Nearer entry first, then the lesser batch and placement, so the order never depends on the order the trees were filed.
inline bool enteredBefore(const TreeCandidate& a, const TreeCandidate& b)
{
	if (a.m_enter != b.m_enter)
		return a.m_enter < b.m_enter;
	return a.m_batch != b.m_batch ? a.m_batch < b.m_batch : a.m_placement < b.m_placement;
}

// The first hit of a ray that can only meet something nearer than its painted hit inside a few forest trees: each
// tree's mesh is searched in the tree's own frame, nearest box first, until the next box starts past the nearest hit.
// The hit then names the forest, the batch and the placement, as a hit found through the forest's instances does.
void treesHit(const TileJob& job, RTCRayHit& rayhit, const FoundHit& found, RTCIntersectArguments* args)
{
	TreeCandidate order[kTreeCandidates];
	for (int i = 0; i < found.m_numTrees; i++)
	{
		int j = i;
		while (j > 0 && enteredBefore(found.m_trees[i], order[j - 1]))
		{
			order[j] = order[j - 1];
			j--;
		}
		order[j] = found.m_trees[i];
	}
	QueryContext* ctx = (QueryContext*)args->context;
	float nearest = found.m_painted ? found.m_t : rayhit.ray.tfar;
	bool inTree = false;
	for (int i = 0; i < found.m_numTrees && order[i].m_enter <= nearest; i++)
	{
		const Batch* batch = order[i].m_batch < job.m_batches->size() ? (*job.m_batches)[order[i].m_batch] : 0;
		if (!batch)
			continue;
		// The inverse of the placement's 3x3 part is the transpose of the inverse transpose the batch keeps for normals.
		const float* inverse = &batch->m_normalRotations[(size_t)order[i].m_placement * 9];
		const float* placement = &batch->m_transforms[(size_t)order[i].m_placement * 12];
		const float org[3] = {rayhit.ray.org_x - placement[9], rayhit.ray.org_y - placement[10], rayhit.ray.org_z - placement[11]};
		const float dir[3] = {rayhit.ray.dir_x, rayhit.ray.dir_y, rayhit.ray.dir_z};
		float localOrg[3], localDir[3];
		for (int r = 0; r < 3; r++)
		{
			localOrg[r] = (inverse[r] * org[0] + inverse[3 + r] * org[1]) + inverse[6 + r] * org[2];
			localDir[r] = (inverse[r] * dir[0] + inverse[3 + r] * dir[1]) + inverse[6 + r] * dir[2];
		}
		RTCRayHit local;
		local.ray.org_x = localOrg[0];
		local.ray.org_y = localOrg[1];
		local.ray.org_z = localOrg[2];
		local.ray.dir_x = localDir[0];
		local.ray.dir_y = localDir[1];
		local.ray.dir_z = localDir[2];
		local.ray.tnear = rayhit.ray.tnear;
		local.ray.tfar = nearest;
		local.ray.time = 0.0f;
		local.ray.mask = (unsigned)-1;
		local.ray.id = 0;
		local.ray.flags = 0;
		local.hit.geomID = RTC_INVALID_GEOMETRY_ID;
		local.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
		ctx->m_directBatch = batch;
		ctx->m_directPlacement = order[i].m_placement;
		rtcIntersect1(batch->m_tree->m_scene, &local, args);
		ctx->m_directBatch = 0;
		if (local.hit.geomID == RTC_INVALID_GEOMETRY_ID)
			continue;
		nearest = local.ray.tfar;
		inTree = true;
		rayhit.hit = local.hit;
		for (int l = 0; l < RTC_MAX_INSTANCE_LEVEL_COUNT; l++)
		{
			rayhit.hit.instID[l] = RTC_INVALID_GEOMETRY_ID;
			rayhit.hit.instPrimID[l] = RTC_INVALID_GEOMETRY_ID;
		}
		rayhit.hit.instID[0] = job.m_forestId;
		rayhit.hit.instPrimID[0] = 0;
		rayhit.hit.instID[1] = order[i].m_batch;
		rayhit.hit.instPrimID[1] = order[i].m_placement;
	}
	if (inTree)
		rayhit.ray.tfar = nearest;
	else if (found.m_painted)
	{
		rayhit.ray.tfar = found.m_t;
		rayhit.hit = found.m_hit;
	}
}

// One ray of a camera through the frame position (ndcX, ndcY). False on a miss; a hit fills `out`.
// `reach` is the ray's depth hint, and `landed`, when given, takes its hit point or NaN on a miss. `found`, when given,
// replaces the search for the first hit: shaded as it stands, a miss, or searched with its own along-ray reach.
bool traceRay(const TileJob& job, const CameraSetup& setup, double ndcX, double ndcY,
			  RTCIntersectArguments* args, RTCOccludedArguments* shadowArgs, Sample& out, float reach = INFINITY, float* landed = 0,
			  const FoundHit* found = 0, SurfaceMemo* memo = 0, bool deferColour = false)
{
	const Camera& cam = setup.m_cam;
	const SwarmRaycastShading* shading = job.m_shading;
	const bool filtered = job.m_filtered;
	out.m_deferred = false;
	out.m_shadeDeferred = false;
	if (landed)
		landed[0] = landed[1] = landed[2] = NAN;

	float dir[3], rawDir[3], tNear, length;
	if (found)
	{
		if (!found->m_ray)
			return false;
		for (int i = 0; i < 3; i++)
		{
			dir[i] = found->m_dir[i];
			rawDir[i] = found->m_rawDir[i];
		}
		tNear = found->m_tNear;
		length = found->m_length;
	}
	else if (!pixelRay(cam, ndcX, ndcY, dir, rawDir, tNear, length))
		return false;
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
	if (found && found->m_kind == kFoundHit)
	{
		rayhit.ray.tfar = found->m_t;
		rayhit.hit = found->m_hit;
	}
	else if (found && found->m_kind == kFoundTrees)
		treesHit(job, rayhit, *found, args);
	else if (!found || found->m_kind == kFoundSearch)
		firstHit(job.m_top, rayhit, found ? found->m_reach : (facing > 0.0f ? reach / facing : INFINITY), args);
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
	HitSurface ownSurface;
	float ownWound[3];
	HitSurface* surfaceSlot = &ownSurface;
	float* woundNormal = ownWound;
	bool known;
	const unsigned key[5] = {rayhit.hit.instID[0], rayhit.hit.geomID, rayhit.hit.primID, rayhit.hit.instID[1], rayhit.hit.instPrimID[1]};
	bool resolve = true;
	SurfaceMemo::Entry* entry = 0;
	if (memo && shading)
	{
		entry = &memo->find(key, resolve);
		surfaceSlot = &entry->m_surface;
		woundNormal = entry->m_woundNormal;
	}
	HitSurface& surface = *surfaceSlot;
	if (!resolve)
	{
		known = entry->m_known;
		segmentation = entry->m_segmentation;
	}
	else
	{
		known = resolveHit(rayhit.hit, job.m_staticId, *job.m_members, *job.m_instances, *job.m_batches, job.m_forestId, segmentation, shading ? &surface : 0);
		// The triangle's own normal, kept as wound for the barycentric solve.
		if (shading && known)
		{
			float e1[3], e2[3];
			for (int i = 0; i < 3; i++)
			{
				e1[i] = surface.m_corners[1][i] - surface.m_corners[0][i];
				e2[i] = surface.m_corners[2][i] - surface.m_corners[0][i];
			}
			cross3(e1, e2, woundNormal);
		}
		if (entry)
		{
			entry->m_known = known;
			entry->m_segmentation = segmentation;
		}
	}
	out.m_segmentation = segmentation;
	out.m_hit.m_inst = rayhit.hit.instID[0];
	out.m_hit.m_geom = rayhit.hit.geomID;
	out.m_hit.m_prim = rayhit.hit.primID;
	out.m_hit.m_inst1 = rayhit.hit.instID[1];
	out.m_hit.m_instPrim1 = rayhit.hit.instPrimID[1];
	// A painted frame mixes painted and searched hits, so outside the forest neither names a second instance level.
	if (job.m_raster && rayhit.hit.instID[0] != job.m_forestId)
		out.m_hit.m_inst1 = out.m_hit.m_instPrim1 = RTC_INVALID_GEOMETRY_ID;
	out.m_shaded = shading && known;
	if (!out.m_shaded)
	{
		if (thermalSky)
			out.m_radiance = SwarmThermal::skyRadiance(job.m_sky, dir[shading->m_glint.m_upAxis]);
		return true;
	}

	// The wound normal turned towards the camera for shading and for the side the shadow ray leaves from.
	float faceNormal[3];
	const bool awayFromCamera = dot3(woundNormal, dir) > 0.0f;
	for (int i = 0; i < 3; i++)
		faceNormal[i] = awayFromCamera ? -woundNormal[i] : woundNormal[i];

	// A daylight hit whose shading waits for its tile leaves its shadow-map lookup to the tile's batch too, unless a step
	// before the wait reads the shadow: a cut-out that may be a veil, or a pane the ray passes through.
	// A waiting hit points at its triangle in the tile's memo, so only a hit with one may wait.
	const bool waitable = deferColour && memo && !shading->m_thermal && !(shading->m_daylight && filtered && job.m_pixelSpread > 0.0f && surface.m_hasAlpha && !surface.m_glass) &&
						  !(shading->m_daylight && surface.m_glass && !surface.m_glassBacked);
	const bool shadowLater = waitable && shading->m_shadow && job.m_shadowMap;
	float shadow = 1.0f;
	if (shading->m_shadow && !shading->m_thermal && !shadowLater)
	{
		const float point[3] = {hx, hy, hz};
		shadow = shadowAt(job, point, faceNormal, leafCard(surface.m_doubleSided, surface.m_hasAlpha), shadowArgs);
	}

	// A hit whose shading waits for its tile leaves its footprint to be worked out with the tile's others, unless a step
	// before the wait reads it: a thermal frame, a cut-out that may be a veil, or a pane the ray passes through.
	const bool footprintLater = waitable && filtered;
	float duvdx[2] = {0.0f, 0.0f}, duvdy[2] = {0.0f, 0.0f};
	if (filtered && !footprintLater)
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
			daylightOut(*shading, lit, dir, t, out, deferColour);
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
			if (deferColour && memo)
			{
				const float point[3] = {hx, hy, hz};
				waitForTile(out, surface, rayhit.hit, faceNormal, duvdx, duvdy, dir, shadow, t, ShadeWait::kModule, point, footprintLater, rawDir,
							woundNormal, shadowLater);
				return true;
			}
			float normal[3], base[3];
			surfaceAt(surface, rayhit.hit, faceNormal, filtered, duvdx, duvdy, normal, base);
			moduleLight(*shading, surface, normal, base, dir, shadow, lit);
			daylightOut(*shading, lit, dir, t, out, false);
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
		daylightOut(*shading, lit, dir, t, out, deferColour);
		return true;
	}

	const float point[3] = {hx, hy, hz};
	if (deferColour && memo)
	{
		// The shading is left for the tile, which reads the textures of its samples together.
		waitForTile(out, surface, rayhit.hit, faceNormal, duvdx, duvdy, dir, shadow, t, shading->m_daylight ? ShadeWait::kDaylight : ShadeWait::kFragment,
					point, footprintLater, rawDir, woundNormal, shadowLater);
		return true;
	}
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

// ER_SWARM_RASTER. The frame a lone camera paints: its projection in double, the six planes of its view, the eye, the
// frame size and tile grid, and the static tree's instance id a painted static hit carries.
struct RasterView
{
	double m_viewProj[4][4];
	double m_planes[6][4];
	float m_origin[3];
	int m_width;
	int m_height;
	int m_tilesX;
	unsigned m_staticId;
};

// What the tiles read of a painted frame: every thread's lane, and the footprint angle the cut-out test reads at.
struct RasterFrame
{
	const std::vector<RasterLane>* m_lanes;
	int m_tilesX;
	float m_spread;
};

// Sub-pixel steps of the fixed-point frame; how many frame half-widths a triangle may reach past the frame before it is
// clipped; and how many triangles per sample of its screen box make a chunk cheaper to search than to paint.
const int kSubPixel = 256;
const double kGuardBand = 16.0;
const float kDenseTriangles = 4.0f;
// Sign of Embree's geometric normal against the cross product of a triangle's edges from its first corner.
const float kEmbreeNormalSign = 1.0f;

// The view's planes from projection times view, each a combination of its rows that is positive inside.
void setupRasterView(const Camera& cam, int width, int height, unsigned staticId, RasterView& view)
{
	memcpy(view.m_viewProj, cam.m_viewProj, sizeof(view.m_viewProj));
	const double(*m)[4] = cam.m_viewProj;
	for (int c = 0; c < 4; c++)
	{
		view.m_planes[0][c] = m[3][c] + m[0][c];
		view.m_planes[1][c] = m[3][c] - m[0][c];
		view.m_planes[2][c] = m[3][c] + m[1][c];
		view.m_planes[3][c] = m[3][c] - m[1][c];
		view.m_planes[4][c] = m[3][c] + m[2][c];
		view.m_planes[5][c] = m[3][c] - m[2][c];
	}
	for (int i = 0; i < 3; i++)
		view.m_origin[i] = cam.m_origin[i];
	view.m_width = width;
	view.m_height = height;
	view.m_tilesX = (width + kTileSize - 1) / kTileSize;
	view.m_staticId = staticId;
}

// False only when the box lies wholly outside one plane of the view; a NaN box counts as seen.
bool boxVisible(const RasterView& view, const float lo[3], const float hi[3])
{
	for (int k = 0; k < 6; k++)
	{
		const double* p = view.m_planes[k];
		const double d = p[0] * (p[0] >= 0.0 ? hi[0] : lo[0]) + p[1] * (p[1] >= 0.0 ? hi[1] : lo[1]) + p[2] * (p[2] >= 0.0 ? hi[2] : lo[2]) + p[3];
		if (d < 0.0)
			return false;
	}
	return true;
}

// The box around a mesh box carried by a column-major transform, a hair wider for the rounding of its corners.
void worldBox(const float m[16], const float lo[3], const float hi[3], float outLo[3], float outHi[3])
{
	for (int i = 0; i < 3; i++)
	{
		outLo[i] = INFINITY;
		outHi[i] = -INFINITY;
	}
	for (int k = 0; k < 8; k++)
	{
		const float corner[3] = {k & 1 ? hi[0] : lo[0], k & 2 ? hi[1] : lo[1], k & 4 ? hi[2] : lo[2]};
		float world[3];
		transformPoint(m, corner, world);
		for (int i = 0; i < 3; i++)
		{
			outLo[i] = world[i] < outLo[i] ? world[i] : outLo[i];
			outHi[i] = world[i] > outHi[i] ? world[i] : outHi[i];
		}
	}
	for (int i = 0; i < 3; i++)
	{
		const float margin = 1e-4f + 1e-6f * (fabsf(outLo[i]) + fabsf(outHi[i]));
		outLo[i] -= margin;
		outHi[i] += margin;
	}
}

// The samples a box may cover, one more each side, or the whole frame when a corner lies at or behind the eye.
bool boxRect(const RasterView& view, const float lo[3], const float hi[3], RayRect& rect)
{
	double x0 = INFINITY, x1 = -INFINITY, y0 = INFINITY, y1 = -INFINITY;
	bool whole = false;
	for (int k = 0; k < 8 && !whole; k++)
	{
		const double c[3] = {k & 1 ? hi[0] : lo[0], k & 2 ? hi[1] : lo[1], k & 4 ? hi[2] : lo[2]};
		double clip[4];
		for (int r = 0; r < 4; r++)
			clip[r] = view.m_viewProj[r][0] * c[0] + view.m_viewProj[r][1] * c[1] + view.m_viewProj[r][2] * c[2] + view.m_viewProj[r][3];
		const double x = (clip[0] / clip[3] + 1.0) * 0.5 * view.m_width;
		const double y = (1.0 - clip[1] / clip[3]) * 0.5 * view.m_height - 1.0;
		if (!(clip[3] > 1e-6) || !(x == x) || !(y == y))
		{
			whole = true;
			break;
		}
		x0 = x < x0 ? x : x0;
		x1 = x > x1 ? x : x1;
		y0 = y < y0 ? y : y0;
		y1 = y > y1 ? y : y1;
	}
	if (whole)
	{
		rect.m_col0 = rect.m_row0 = 0;
		rect.m_col1 = view.m_width - 1;
		rect.m_row1 = view.m_height - 1;
		return true;
	}
	// Held to the frame before the conversion, so a corner just in front of the eye cannot overflow an int.
	const double right = view.m_width + 1.0, bottom = view.m_height + 1.0;
	x0 = x0 < -2.0 ? -2.0 : (x0 > right ? right : x0);
	x1 = x1 < -2.0 ? -2.0 : (x1 > right ? right : x1);
	y0 = y0 < -2.0 ? -2.0 : (y0 > bottom ? bottom : y0);
	y1 = y1 < -2.0 ? -2.0 : (y1 > bottom ? bottom : y1);
	rect.m_col0 = (int)floor(x0) - 1;
	rect.m_col1 = (int)ceil(x1) + 1;
	rect.m_row0 = (int)floor(y0) - 1;
	rect.m_row1 = (int)ceil(y1) + 1;
	rect.m_col0 = rect.m_col0 < 0 ? 0 : rect.m_col0;
	rect.m_row0 = rect.m_row0 < 0 ? 0 : rect.m_row0;
	rect.m_col1 = rect.m_col1 >= view.m_width ? view.m_width - 1 : rect.m_col1;
	rect.m_row1 = rect.m_row1 >= view.m_height ? view.m_height - 1 : rect.m_row1;
	return rect.m_col0 <= rect.m_col1 && rect.m_row0 <= rect.m_row1;
}

// The least distance from the eye to the box, a hair short, so no ray's rounded distance into the box falls under it.
float boxDistance(const float origin[3], const float lo[3], const float hi[3])
{
	float sum = 0.0f;
	for (int i = 0; i < 3; i++)
	{
		const float below = lo[i] - origin[i], above = origin[i] - hi[i];
		const float gap = below > above ? below : above;
		sum += gap > 0.0f ? gap * gap : 0.0f;
	}
	return sqrtf(sum) * (1.0f - 1e-5f) - 1e-4f;
}

// Files an item under every tile its samples reach.
void binInto(RasterLane& lane, const RasterView& view, int col0, int col1, int row0, int row1, unsigned index)
{
	for (int ty = row0 / kTileSize; ty <= row1 / kTileSize; ty++)
		for (int tx = col0 / kTileSize; tx <= col1 / kTileSize; tx++)
			lane.m_bins[(size_t)ty * view.m_tilesX + tx].push_back(index);
}

// Keeps a ray rect in the lane and files it under its tiles.
void addRayRect(RasterLane& lane, const RasterView& view, const RayRect& rect)
{
	binInto(lane, view, rect.m_col0, rect.m_col1, rect.m_row0, rect.m_row1, (unsigned)lane.m_rects.size() | kRectBit);
	lane.m_rects.push_back(rect);
}

struct ClipVertex
{
	double m_p[4];
};

// How far inside a plane a clip-space point lies: the near plane, then the four sides of the guard band.
inline double guardDistance(const ClipVertex& v, int plane)
{
	switch (plane)
	{
		case 0:
			return v.m_p[2] + v.m_p[3];
		case 1:
			return kGuardBand * v.m_p[3] - v.m_p[0];
		case 2:
			return kGuardBand * v.m_p[3] + v.m_p[0];
		case 3:
			return kGuardBand * v.m_p[3] - v.m_p[1];
		default:
			return kGuardBand * v.m_p[3] + v.m_p[1];
	}
}

inline bool clipBefore(const ClipVertex& a, const ClipVertex& b)
{
	for (int k = 0; k < 4; k++)
		if (a.m_p[k] != b.m_p[k])
			return a.m_p[k] < b.m_p[k];
	return false;
}

// Keeps the part of a polygon inside one plane. A cut edge is interpolated from its lesser end, so an edge two triangles
// share is cut at the same point by both and the painted surface keeps no crack along it.
int clipPolygon(const ClipVertex* in, int n, int plane, ClipVertex* out)
{
	int kept = 0;
	for (int i = 0; i < n; i++)
	{
		const ClipVertex& p = in[i];
		const ClipVertex& q = in[(i + 1) % n];
		const double dp = guardDistance(p, plane), dq = guardDistance(q, plane);
		if (dp >= 0.0)
			out[kept++] = p;
		if ((dp >= 0.0) != (dq >= 0.0))
		{
			const bool flip = clipBefore(q, p);
			const ClipVertex& a = flip ? q : p;
			const ClipVertex& b = flip ? p : q;
			const double da = flip ? dq : dp, db = flip ? dp : dq;
			const double t = da / (da - db);
			for (int k = 0; k < 4; k++)
				out[kept].m_p[k] = a.m_p[k] + (b.m_p[k] - a.m_p[k]) * t;
			kept++;
		}
	}
	return kept;
}

inline long long floorDiv(long long a, long long b)
{
	return a >= 0 ? a / b : -((-a + b - 1) / b);
}

// Puts one clipped piece of a triangle on the frame in fixed point; a piece with no area or no sample adds nothing.
void addScreenTriangle(RasterLane& lane, const RasterView& view, const ClipVertex* const corners[3], const RasterTri& world)
{
	long long x[3], y[3];
	for (int j = 0; j < 3; j++)
	{
		const double* p = corners[j]->m_p;
		x[j] = (long long)floor((p[0] / p[3] + 1.0) * 0.5 * view.m_width * kSubPixel + 0.5);
		y[j] = (long long)floor(((1.0 - p[1] / p[3]) * 0.5 * view.m_height - 1.0) * kSubPixel + 0.5);
	}
	const long long area = (x[1] - x[0]) * (y[2] - y[0]) - (y[1] - y[0]) * (x[2] - x[0]);
	if (area == 0)
		return;
	if (area < 0)
	{
		std::swap(x[1], x[2]);
		std::swap(y[1], y[2]);
	}
	const long long minX = std::min(x[0], std::min(x[1], x[2])), maxX = std::max(x[0], std::max(x[1], x[2]));
	const long long minY = std::min(y[0], std::min(y[1], y[2])), maxY = std::max(y[0], std::max(y[1], y[2]));
	const int col0 = (int)std::max(-floorDiv(-minX, kSubPixel), 0LL);
	const int col1 = (int)std::min(floorDiv(maxX, kSubPixel), (long long)view.m_width - 1);
	const int row0 = (int)std::max(-floorDiv(-minY, kSubPixel), 0LL);
	const int row1 = (int)std::min(floorDiv(maxY, kSubPixel), (long long)view.m_height - 1);
	if (col0 > col1 || row0 > row1)
		return;
	RasterTri tri = world;
	for (int i = 0; i < 3; i++)
	{
		const int k = (i + 1) % 3;
		const long long dx = x[k] - x[i], dy = y[k] - y[i];
		// A sample exactly on an edge belongs to the one triangle of the two sharing it that owns the edge.
		const bool owns = dy > 0 || (dy == 0 && dx < 0);
		tri.m_edge[i] = dy * x[i] - dx * y[i] - (owns ? 0 : 1);
		tri.m_stepX[i] = -dy * kSubPixel;
		tri.m_stepY[i] = dx * kSubPixel;
	}
	tri.m_col0 = col0;
	tri.m_col1 = col1;
	tri.m_row0 = row0;
	tri.m_row1 = row1;
	binInto(lane, view, col0, col1, row0, row1, (unsigned)lane.m_tris.size());
	lane.m_tris.push_back(tri);
}

// Puts one world triangle on the frame: dropped when single-sided and turned away from the eye, as the camera ray's
// filter drops it, or when wholly outside one side of the view; clipped to the near plane and the guard band otherwise.
void paintTriangle(RasterLane& lane, const RasterView& view, const float c[3][3], bool doubleSided, unsigned inst, unsigned geom,
				   unsigned prim, const RasterSource* source)
{
	float e1[3], e2[3], toEye[3], normal[3];
	for (int i = 0; i < 3; i++)
	{
		e1[i] = c[1][i] - c[0][i];
		e2[i] = c[2][i] - c[0][i];
		toEye[i] = view.m_origin[i] - c[0][i];
	}
	cross3(e1, e2, normal);
	// Every camera ray onto the plane meets the normal at the sign it has against the first corner seen from the eye.
	if (!doubleSided && -kEmbreeNormalSign * dot3(normal, toEye) >= 0.0f)
		return;
	ClipVertex v[3];
	for (int j = 0; j < 3; j++)
		for (int r = 0; r < 4; r++)
			v[j].m_p[r] = view.m_viewProj[r][0] * c[j][0] + view.m_viewProj[r][1] * c[j][1] + view.m_viewProj[r][2] * c[j][2] + view.m_viewProj[r][3];
	for (int axis = 0; axis < 3; axis++)
		for (int side = -1; side <= 1; side += 2)
		{
			bool out = true;
			for (int j = 0; j < 3 && out; j++)
				out = !(v[j].m_p[3] + side * v[j].m_p[axis] >= 0.0);
			if (out)
				return;
		}
	// The Moller-Trumbore solve with the eye as every ray's origin: its determinant and the barycentric and distance
	// numerators are each a constant vector dotted with the ray's direction.
	RasterTri world;
	cross3(e2, e1, world.m_det);
	cross3(e2, toEye, world.m_u);
	cross3(toEye, e1, world.m_v);
	world.m_t = dot3(e2, world.m_v);
	world.m_inst = inst;
	world.m_geom = geom;
	world.m_prim = prim;
	world.m_key = ((unsigned long long)inst << 48) ^ ((unsigned long long)geom << 32) ^ prim;
	world.m_source = source;
	bool inside = true;
	for (int plane = 0; plane < 5 && inside; plane++)
		for (int j = 0; j < 3 && inside; j++)
			inside = guardDistance(v[j], plane) >= 0.0;
	if (inside)
	{
		const ClipVertex* const corners[3] = {&v[0], &v[1], &v[2]};
		addScreenTriangle(lane, view, corners, world);
		return;
	}
	ClipVertex polygon[2][10];
	int n = 3;
	for (int j = 0; j < 3; j++)
		polygon[0][j] = v[j];
	int current = 0;
	for (int plane = 0; plane < 5 && n >= 3; plane++)
	{
		n = clipPolygon(polygon[current], n, plane, polygon[1 - current]);
		current = 1 - current;
	}
	for (int j = 1; j + 1 < n; j++)
	{
		const ClipVertex* const corners[3] = {&polygon[current][0], &polygon[current][j], &polygon[current][j + 1]};
		addScreenTriangle(lane, view, corners, world);
	}
}

// Paints one chunk; one with more triangles than its screen box has samples to spare leaves its box to rays instead.
void paintJob(RasterLane& lane, const RasterView& view, const RasterJob& job)
{
	const RasterChunk& chunk = *job.m_chunk;
	const Instance* inst = job.m_instance;
	float lo[3], hi[3];
	if (inst)
		worldBox(inst->m_transform, chunk.m_lo, chunk.m_hi, lo, hi);
	else
		for (int i = 0; i < 3; i++)
		{
			lo[i] = chunk.m_lo[i];
			hi[i] = chunk.m_hi[i];
		}
	RayRect rect;
	if (!boxVisible(view, lo, hi) || !boxRect(view, lo, hi, rect))
		return;
	const float samples = (float)(rect.m_col1 - rect.m_col0 + 1) * (float)(rect.m_row1 - rect.m_row0 + 1);
	if ((float)chunk.m_count > kDenseTriangles * samples)
	{
		rect.m_distance = boxDistance(view.m_origin, lo, hi);
		memcpy(rect.m_lo, lo, sizeof(rect.m_lo));
		memcpy(rect.m_hi, hi, sizeof(rect.m_hi));
		rect.m_batch = kNoTree;
		rect.m_placement = 0;
		addRayRect(lane, view, rect);
		return;
	}
	const float* vertices = inst ? &inst->m_tree->m_vertices[0] : &job.m_member->m_vertices[0];
	const unsigned* indices = inst ? &inst->m_tree->m_indices[0] : &job.m_member->m_indices[0];
	const bool doubleSided = inst ? inst->m_doubleSided : job.m_member->m_doubleSided;
	const unsigned instId = inst ? inst->m_geomId : view.m_staticId;
	const unsigned geomId = inst ? 0u : job.m_member->m_geomId;
	for (unsigned t = chunk.m_first; t < chunk.m_first + chunk.m_count; t++)
	{
		// Corners land on the floats resolveHit gives the same triangle, so painting and shading agree on where it is.
		float c[3][3];
		for (int j = 0; j < 3; j++)
		{
			const float* p = vertices + (size_t)indices[(size_t)t * 3 + j] * 3;
			if (inst)
				transformPoint(inst->m_transform, p, c[j]);
			else
				for (int i = 0; i < 3; i++)
					c[j][i] = p[i];
		}
		paintTriangle(lane, view, c, doubleSided, instId, geomId, t, job.m_source);
	}
}

// Leaves a ray rect for every enabled tree of a cell in view.
void paintForestCell(RasterLane& lane, const RasterView& view, const ForestGrid& grid, size_t cell, const std::vector<Batch*>& batches)
{
	if (!boxVisible(view, &grid.m_cellBoxes[cell * 6], &grid.m_cellBoxes[cell * 6 + 3]))
		return;
	for (unsigned k = grid.m_cellStart[cell]; k < grid.m_cellStart[cell + 1]; k++)
	{
		const Batch* batch = batches[grid.m_treeBatch[k]];
		if (!batch || !batch->m_enabled)
			continue;
		const float* lo = &grid.m_treeBoxes[(size_t)k * 6];
		const float* hi = lo + 3;
		RayRect rect;
		if (!boxVisible(view, lo, hi) || !boxRect(view, lo, hi, rect))
			continue;
		rect.m_distance = boxDistance(view.m_origin, lo, hi);
		memcpy(rect.m_lo, lo, sizeof(rect.m_lo));
		memcpy(rect.m_hi, hi, sizeof(rect.m_hi));
		rect.m_batch = grid.m_treeBatch[k];
		rect.m_placement = grid.m_treePlacement[k];
		addRayRect(lane, view, rect);
	}
}

// Files every tree of every batch under the square cell its box's centre falls in, with its world box a hair wider
// than its mesh, and gives every cell the box around its trees.
void buildForestGrid(ForestGrid& grid, const std::vector<Batch*>& batches)
{
	struct Tree
	{
		long long m_x;
		long long m_y;
		unsigned m_batch;
		unsigned m_placement;
		float m_box[6];
		bool operator<(const Tree& other) const { return m_x != other.m_x ? m_x < other.m_x : m_y < other.m_y; }
	};
	std::vector<Tree> trees;
	for (size_t b = 0; b < batches.size(); b++)
	{
		const Batch* batch = batches[b];
		if (!batch)
			continue;
		const std::vector<float>& vertices = batch->m_tree->m_vertices;
		float lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
		for (size_t i = 0; i + kVertexPadding < vertices.size(); i += 3)
			for (int k = 0; k < 3; k++)
			{
				lo[k] = vertices[i + k] < lo[k] ? vertices[i + k] : lo[k];
				hi[k] = vertices[i + k] > hi[k] ? vertices[i + k] : hi[k];
			}
		if (!(lo[0] <= hi[0] && lo[1] <= hi[1] && lo[2] <= hi[2]))
			continue;
		for (size_t p = 0; (p + 1) * 12 <= batch->m_transforms.size(); p++)
		{
			const float* t = &batch->m_transforms[p * 12];
			Tree tree;
			tree.m_batch = (unsigned)b;
			tree.m_placement = (unsigned)p;
			for (int r = 0; r < 3; r++)
			{
				tree.m_box[r] = INFINITY;
				tree.m_box[3 + r] = -INFINITY;
			}
			for (int k = 0; k < 8; k++)
			{
				const double c[3] = {k & 1 ? hi[0] : lo[0], k & 2 ? hi[1] : lo[1], k & 4 ? hi[2] : lo[2]};
				for (int r = 0; r < 3; r++)
				{
					const float w = (float)((double)t[9 + r] + (double)t[r] * c[0] + (double)t[3 + r] * c[1] + (double)t[6 + r] * c[2]);
					tree.m_box[r] = w < tree.m_box[r] ? w : tree.m_box[r];
					tree.m_box[3 + r] = w > tree.m_box[3 + r] ? w : tree.m_box[3 + r];
				}
			}
			for (int r = 0; r < 3; r++)
			{
				const float margin = 1e-3f + 1e-5f * (fabsf(tree.m_box[r]) + fabsf(tree.m_box[3 + r]));
				tree.m_box[r] -= margin;
				tree.m_box[3 + r] += margin;
			}
			const double midX = 0.5 * ((double)tree.m_box[0] + tree.m_box[3]), midY = 0.5 * ((double)tree.m_box[1] + tree.m_box[4]);
			tree.m_x = fabs(midX) < 1e12 ? (long long)floor(midX / kForestCell) : 0;
			tree.m_y = fabs(midY) < 1e12 ? (long long)floor(midY / kForestCell) : 0;
			trees.push_back(tree);
		}
	}
	std::stable_sort(trees.begin(), trees.end());
	grid.m_cellBoxes.clear();
	grid.m_cellStart.clear();
	grid.m_treeBoxes.resize(trees.size() * 6);
	grid.m_treeBatch.resize(trees.size());
	grid.m_treePlacement.resize(trees.size());
	for (size_t i = 0; i < trees.size(); i++)
	{
		if (i == 0 || trees[i].m_x != trees[i - 1].m_x || trees[i].m_y != trees[i - 1].m_y)
		{
			grid.m_cellStart.push_back((unsigned)i);
			for (int r = 0; r < 3; r++)
				grid.m_cellBoxes.push_back(INFINITY);
			for (int r = 0; r < 3; r++)
				grid.m_cellBoxes.push_back(-INFINITY);
		}
		float* cell = &grid.m_cellBoxes[grid.m_cellBoxes.size() - 6];
		for (int r = 0; r < 3; r++)
		{
			cell[r] = trees[i].m_box[r] < cell[r] ? trees[i].m_box[r] : cell[r];
			cell[3 + r] = trees[i].m_box[3 + r] > cell[3 + r] ? trees[i].m_box[3 + r] : cell[3 + r];
		}
		memcpy(&grid.m_treeBoxes[i * 6], trees[i].m_box, sizeof(trees[i].m_box));
		grid.m_treeBatch[i] = trees[i].m_batch;
		grid.m_treePlacement[i] = trees[i].m_placement;
	}
	grid.m_cellStart.push_back((unsigned)trees.size());
	grid.m_built = true;
}

// A tile's painted result per sample: the nearest painted hit, where on its triangle, and the least distance at which a
// ray rect asks for a search.
struct TilePaint
{
	float m_t[kTileSize * kTileSize];
	float m_u[kTileSize * kTileSize];
	float m_v[kTileSize * kTileSize];
	const RasterTri* m_tri[kTileSize * kTileSize];
	float m_rayNear[kTileSize * kTileSize];
	// The forest trees each sample's ray enters before the painted hit, and whether only the full search will do: a
	// dense chunk lies on the way, or more trees than kTreeCandidates.
	TreeCandidate m_trees[kTileSize * kTileSize][kTreeCandidates];
	int m_numTrees[kTileSize * kTileSize];
	bool m_wide[kTileSize * kTileSize];
	// Each sample's camera ray, worked out once for painting and shading both; m_ray is false where it has none.
	bool m_ray[kTileSize * kTileSize];
	float m_dir[kTileSize * kTileSize][3];
	float m_rawDir[kTileSize * kTileSize][3];
	float m_tNear[kTileSize * kTileSize];
	float m_length[kTileSize * kTileSize];
};

// Where a ray from origin along dir enters a box no later than limit, a hair early for rounding; INFINITY when it misses
// the box or enters it only past limit. A zero direction component keeps its slab only when the origin lies inside it.
inline float rayEnters(const float origin[3], const float dir[3], const float lo[3], const float hi[3], float limit)
{
	float enter = 0.0f, leave = limit + limit * 1e-5f + 1e-4f;
	for (int i = 0; i < 3; i++)
	{
		if (dir[i] == 0.0f)
		{
			if (!(origin[i] >= lo[i] && origin[i] <= hi[i]))
				return INFINITY;
			continue;
		}
		const float inv = 1.0f / dir[i];
		float near = (lo[i] - origin[i]) * inv, far = (hi[i] - origin[i]) * inv;
		if (near > far)
			std::swap(near, far);
		enter = near > enter ? near : enter;
		leave = far < leave ? far : leave;
	}
	if (!(enter <= leave))
		return INFINITY;
	return enter * (1.0f - 1e-5f) - 1e-4f;
}

// True when a covered sample lands on a texel the hit filter would call see-through: the same test on the same mesh,
// with the camera ray carried into the mesh's frame for a mover as Embree carries it.
bool paintedCutOut(const RasterFrame& frame, const RasterTri& tri, const float dir[3], float t, float u, float v)
{
	const RasterSource& source = *tri.m_source;
	const unsigned* ids = &(*source.m_indices)[(size_t)tri.m_prim * 3];
	const float* p0 = source.m_vertices + (size_t)ids[0] * 3;
	const float* p1 = source.m_vertices + (size_t)ids[1] * 3;
	const float* p2 = source.m_vertices + (size_t)ids[2] * 3;
	float e1[3], e2[3], normal[3];
	for (int i = 0; i < 3; i++)
	{
		e1[i] = p1[i] - p0[i];
		e2[i] = p2[i] - p0[i];
	}
	cross3(e1, e2, normal);
	RTCHit hit;
	hit.primID = tri.m_prim;
	hit.u = u;
	hit.v = v;
	hit.Ng_x = normal[0];
	hit.Ng_y = normal[1];
	hit.Ng_z = normal[2];
	float local[3] = {dir[0], dir[1], dir[2]};
	if (source.m_objectSpace)
		for (int r = 0; r < 3; r++)
			local[r] = dot3(source.m_inverse + r * 3, dir);
	RTCRay ray;
	ray.dir_x = local[0];
	ray.dir_y = local[1];
	ray.dir_z = local[2];
	ray.tfar = t;
	return cutOutAt(source.m_model, source.m_vertices, source.m_uvs, *source.m_indices, &hit, &ray, frame.m_spread);
}

// Paints one tile: every lane's triangles reaching it keep, per sample, the nearest hit between the clip planes whose
// texel is solid, nearer ties going to the lesser key, so the order the triangles come in never changes the result.
void paintTile(const RasterFrame& frame, const Camera& cam, int width, int height, int row0, int row1, int col0, int col1, TilePaint& paint)
{
	const int tile = (row0 / kTileSize) * frame.m_tilesX + col0 / kTileSize;
	float(*dir)[3] = paint.m_dir;
	float* tNear = paint.m_tNear;
	float tFar[kTileSize * kTileSize];
	for (int row = row0; row < row1; row++)
	{
		const double ndcY = pixelNdcY(row, height);
		for (int col = col0; col < col1; col++)
		{
			const int k = (row - row0) * kTileSize + (col - col0);
			paint.m_t[k] = INFINITY;
			paint.m_tri[k] = 0;
			paint.m_rayNear[k] = INFINITY;
			paint.m_numTrees[k] = 0;
			paint.m_wide[k] = false;
			paint.m_ray[k] = pixelRay(cam, pixelNdcX(col, width), ndcY, dir[k], paint.m_rawDir[k], tNear[k], paint.m_length[k]);
			if (paint.m_ray[k])
				tFar[k] = tNear[k] + paint.m_length[k];
			else
			{
				dir[k][0] = dir[k][1] = dir[k][2] = 0.0f;
				tNear[k] = INFINITY;
				tFar[k] = -INFINITY;
			}
		}
	}
	const std::vector<RasterLane>& lanes = *frame.m_lanes;
	for (size_t l = 0; l < lanes.size(); l++)
	{
		const std::vector<unsigned>& bin = lanes[l].m_bins[(size_t)tile];
		for (size_t b = 0; b < bin.size(); b++)
		{
			if (bin[b] & kRectBit)
				continue;
			const RasterTri& tri = lanes[l].m_tris[bin[b]];
			const int r0 = std::max(tri.m_row0, row0), r1 = std::min(tri.m_row1, row1 - 1);
			const int c0 = std::max(tri.m_col0, col0), c1 = std::min(tri.m_col1, col1 - 1);
			// An edge function is linear over the rectangle, so its largest value there is at a corner, and along a row at
			// one end: where an edge is negative even there, no sample is covered and the rectangle or row is skipped.
			long long rowGain[3];
			bool empty = false;
			for (int i = 0; i < 3; i++)
			{
				rowGain[i] = std::max(0LL, tri.m_stepX[i] * (c1 - c0));
				const long long best = tri.m_edge[i] + tri.m_stepX[i] * c0 + rowGain[i] + std::max(tri.m_stepY[i] * r0, tri.m_stepY[i] * r1);
				empty = empty || best < 0;
			}
			if (empty)
				continue;
			for (int row = r0; row <= r1; row++)
			{
				long long e0 = tri.m_edge[0] + tri.m_stepX[0] * c0 + tri.m_stepY[0] * row;
				long long e1 = tri.m_edge[1] + tri.m_stepX[1] * c0 + tri.m_stepY[1] * row;
				long long e2 = tri.m_edge[2] + tri.m_stepX[2] * c0 + tri.m_stepY[2] * row;
				if (e0 + rowGain[0] < 0 || e1 + rowGain[1] < 0 || e2 + rowGain[2] < 0)
					continue;
				for (int col = c0; col <= c1; col++, e0 += tri.m_stepX[0], e1 += tri.m_stepX[1], e2 += tri.m_stepX[2])
				{
					if ((e0 | e1 | e2) < 0)
						continue;
					const int k = (row - row0) * kTileSize + (col - col0);
					const float* d = dir[k];
					const float det = dot3(d, tri.m_det);
					if (det == 0.0f)
						continue;
					const float inv = 1.0f / det;
					const float t = tri.m_t * inv;
					if (!(t >= tNear[k] && t <= tFar[k]))
						continue;
					if (t > paint.m_t[k] || (t == paint.m_t[k] && tri.m_key >= paint.m_tri[k]->m_key))
						continue;
					float u = dot3(d, tri.m_u) * inv, v = dot3(d, tri.m_v) * inv;
					u = u < 0.0f ? 0.0f : (u > 1.0f ? 1.0f : u);
					v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
					if (u + v > 1.0f)
					{
						const float scale = 1.0f / (u + v);
						u *= scale;
						v *= scale;
					}
					if (tri.m_source && paintedCutOut(frame, tri, d, t, u, v))
						continue;
					paint.m_t[k] = t;
					paint.m_u[k] = u;
					paint.m_v[k] = v;
					paint.m_tri[k] = &tri;
				}
			}
		}
	}
	// Once every triangle is in, a sample is searched only where its own ray enters a rect's box before the painted hit,
	// and only in the trees it enters while no dense chunk lies on its way and the trees fit the sample's list.
	for (size_t l = 0; l < lanes.size(); l++)
	{
		const std::vector<unsigned>& bin = lanes[l].m_bins[(size_t)tile];
		for (size_t b = 0; b < bin.size(); b++)
		{
			if (!(bin[b] & kRectBit))
				continue;
			const RayRect& rect = lanes[l].m_rects[bin[b] & ~kRectBit];
			const int r0 = std::max(rect.m_row0, row0), r1 = std::min(rect.m_row1, row1 - 1);
			const int c0 = std::max(rect.m_col0, col0), c1 = std::min(rect.m_col1, col1 - 1);
			for (int row = r0; row <= r1; row++)
				for (int col = c0; col <= c1; col++)
				{
					const int k = (row - row0) * kTileSize + (col - col0);
					const float limit = paint.m_t[k] < tFar[k] ? paint.m_t[k] : tFar[k];
					if (paint.m_wide[k] || !(rect.m_distance <= limit))
						continue;
					const float enter = rayEnters(cam.m_origin, dir[k], rect.m_lo, rect.m_hi, limit);
					if (!(enter < INFINITY))
						continue;
					paint.m_rayNear[k] = enter < paint.m_rayNear[k] ? enter : paint.m_rayNear[k];
					if (rect.m_batch == kNoTree || paint.m_numTrees[k] == kTreeCandidates)
					{
						paint.m_wide[k] = true;
						continue;
					}
					TreeCandidate& tree = paint.m_trees[k][paint.m_numTrees[k]++];
					tree.m_enter = enter;
					tree.m_batch = rect.m_batch;
					tree.m_placement = rect.m_placement;
				}
		}
	}
}

// What the painting says of one sample: shade the painted hit; search only the forest trees the ray enters before it;
// search everything, no further than just past the painted hit, where a dense chunk or too many trees lie on the way;
// or a miss.
void paintedFound(const TilePaint& paint, int k, FoundHit& found)
{
	found.m_ray = paint.m_ray[k];
	found.m_dir = paint.m_dir[k];
	found.m_rawDir = paint.m_rawDir[k];
	found.m_tNear = paint.m_tNear[k];
	found.m_length = paint.m_length[k];
	const RasterTri* tri = paint.m_tri[k];
	const bool search = paint.m_rayNear[k] < INFINITY && paint.m_rayNear[k] <= paint.m_t[k];
	found.m_painted = tri != 0;
	if (search && paint.m_wide[k])
	{
		found.m_kind = kFoundSearch;
		found.m_reach = tri ? paint.m_t[k] + paint.m_t[k] * (1.0f / 1024.0f) + 0.01f : INFINITY;
		return;
	}
	found.m_kind = search ? kFoundTrees : (tri ? kFoundHit : kFoundMiss);
	found.m_trees = paint.m_trees[k];
	found.m_numTrees = paint.m_numTrees[k];
	if (!tri)
		return;
	found.m_t = paint.m_t[k];
	RTCHit& hit = found.m_hit;
	hit.Ng_x = hit.Ng_y = hit.Ng_z = 0.0f;
	hit.u = paint.m_u[k];
	hit.v = paint.m_v[k];
	hit.primID = tri->m_prim;
	hit.geomID = tri->m_geom;
	for (int l = 0; l < RTC_MAX_INSTANCE_LEVEL_COUNT; l++)
	{
		hit.instID[l] = RTC_INVALID_GEOMETRY_ID;
		hit.instPrimID[l] = RTC_INVALID_GEOMETRY_ID;
	}
	hit.instID[0] = tri->m_inst;
	hit.instPrimID[0] = 0;
}

// The tile's waiting hits shaded together: their footprints, shadows, normals and textures, then their light, which joins
// the colours waiting in `waiting`; then every waiting colour is finished and written.
void finishTileShading(const TileJob& job, const CameraSetup& setup, RTCOccludedArguments* shadowArgs, ShadeWait* shades,
					   unsigned char* const* shadeOut, int numShades, DaylightColour* waiting, unsigned char** waitingOut, int numWaiting)
{
	if (numShades)
	{
		// The waiting hits' normals and texture coordinates, their textures read together, then their light, which
		// joins the colours waiting for the write.
		// Daylight surfaces read up to four texels along a slanted footprint, the fragment shader one; each kind's reads
		// go together, in the order the samples came.
		TinyRender::Vec2f uvs[kTileSize * kTileSize], duvdx[kTileSize * kTileSize], duvdy[kTileSize * kTileSize];
		TGAColor texels[kTileSize * kTileSize], kindTexels[kTileSize * kTileSize];
		float normals[kTileSize * kTileSize][3];
		int order[kTileSize * kTileSize], numDaylight = 0;
		{
			// The footprints left for the tile, worked out together.
			int late[kTileSize * kTileSize], numLate = 0;
			float rawDirs[kTileSize * kTileSize][3], corners[kTileSize * kTileSize][3][3], wounds[kTileSize * kTileSize][3];
			float hitU[kTileSize * kTileSize], hitV[kTileSize * kTileSize], cornerUvs[kTileSize * kTileSize][6];
			float lateX[kTileSize * kTileSize][2], lateY[kTileSize * kTileSize][2];
			for (int i = 0; i < numShades; i++)
			{
				const ShadeWait& wait = shades[i];
				if (!wait.m_footprintLater)
					continue;
				const int n = numLate++;
				late[n] = i;
				for (int c = 0; c < 3; c++)
				{
					rawDirs[n][c] = wait.m_rawDir[c];
					wounds[n][c] = wait.m_wound[c];
					for (int j = 0; j < 3; j++)
						corners[n][j][c] = wait.m_surface->m_corners[j][c];
				}
				for (int j = 0; j < 3; j++)
				{
					const float* uv = wait.m_surface->m_uvs + (size_t)wait.m_surface->m_vertexIds[j] * 2;
					cornerUvs[n][2 * j] = uv[0];
					cornerUvs[n][2 * j + 1] = uv[1];
				}
				hitU[n] = wait.m_u;
				hitV[n] = wait.m_v;
				lateX[n][0] = lateX[n][1] = lateY[n][0] = lateY[n][1] = 0.0f;
			}
			if (numLate)
				footprintMany(setup, rawDirs, corners, wounds, hitU, hitV, cornerUvs, numLate, lateX, lateY);
			for (int n = 0; n < numLate; n++)
				for (int c = 0; c < 2; c++)
				{
					shades[late[n]].m_duvdx[c] = lateX[n][c];
					shades[late[n]].m_duvdy[c] = lateY[n][c];
				}
			// The shadows left for the tile: the maps' lit shares together, then each point's remaining shadow ray.
			const ShadowMap* maps[kTileSize * kTileSize];
			float points[kTileSize * kTileSize][3], normals[kTileSize * kTileSize][3], lit[kTileSize * kTileSize];
			numLate = 0;
			for (int i = 0; i < numShades; i++)
			{
				const ShadeWait& wait = shades[i];
				if (!wait.m_shadowLater)
					continue;
				const int n = numLate++;
				late[n] = i;
				shadowNormal(job.m_shading, wait.m_faceNormal, leafCard(wait.m_surface->m_doubleSided, wait.m_surface->m_hasAlpha), normals[n]);
				for (int c = 0; c < 3; c++)
					points[n][c] = wait.m_point[c];
				maps[n] = shadowMapFor(job, points[n]);
			}
			if (numLate && job.m_shading->m_daylight)
			{
				shadowLitMany(maps, points, normals, numLate, lit);
				for (int n = 0; n < numLate; n++)
					shades[late[n]].m_shadow = shadowFinish(job, points[n], normals[n], lit[n], lit[n] <= 0.0f, shadowArgs);
			}
			else if (numLate)
			{
				bool blocked[kTileSize * kTileSize];
				shadowBlockedMany(*job.m_shadowMap, points, normals, numLate, blocked);
				for (int n = 0; n < numLate; n++)
					shades[late[n]].m_shadow = shadowFinish(job, points[n], normals[n], 1.0f, blocked[n], shadowArgs);
			}
		}
		for (int i = 0; i < numShades; i++)
		{
			const ShadeWait& wait = shades[i];
			if (wait.m_kind == ShadeWait::kFragment)
				shadeHitFrame(*wait.m_surface, wait.m_u, wait.m_v, wait.m_faceNormal, normals[i], uvs[i]);
			else
			{
				surfaceFrame(*wait.m_surface, wait.m_u, wait.m_v, wait.m_faceNormal, normals[i], uvs[i]);
				order[numDaylight++] = i;
			}
		}
		for (int i = 0, fragment = numDaylight; i < numShades; i++)
			if (shades[i].m_kind == ShadeWait::kFragment)
				order[fragment++] = i;
		TinyRender::Model* kindModels[kTileSize * kTileSize];
		for (int n = 0; n < numShades; n++)
		{
			const ShadeWait& wait = shades[order[n]];
			kindModels[n] = wait.m_surface->m_model;
			duvdx[n] = TinyRender::Vec2f(wait.m_duvdx[0], wait.m_duvdx[1]);
			duvdy[n] = TinyRender::Vec2f(wait.m_duvdy[0], wait.m_duvdy[1]);
		}
		TinyRender::Vec2f kindUvs[kTileSize * kTileSize];
		for (int n = 0; n < numShades; n++)
			kindUvs[n] = uvs[order[n]];
		if (job.m_filtered)
		{
			TinyRender::Model::diffuseFilteredMany(kindModels, kindUvs, duvdx, duvdy, numDaylight, 4, kindTexels);
			TinyRender::Model::diffuseFilteredMany(kindModels + numDaylight, kindUvs + numDaylight, duvdx + numDaylight, duvdy + numDaylight,
												   numShades - numDaylight, 1, kindTexels + numDaylight);
		}
		else
			for (int n = 0; n < numShades; n++)
				kindTexels[n] = kindModels[n]->diffuse(kindUvs[n]);
		for (int n = 0; n < numShades; n++)
			texels[order[n]] = kindTexels[n];
		// The daylight surfaces' light, worked out together.
		const HitSurface* daySurfaces[kTileSize * kTileSize];
		float dayNormals[kTileSize * kTileSize][3], dayBases[kTileSize * kTileSize][3], dayDirs[kTileSize * kTileSize][3];
		float dayShadows[kTileSize * kTileSize], dayLit[kTileSize * kTileSize][3];
		int numDay = 0;
		for (int i = 0; i < numShades; i++)
		{
			const ShadeWait& wait = shades[i];
			if (wait.m_kind != ShadeWait::kDaylight)
				continue;
			daySurfaces[numDay] = wait.m_surface;
			surfaceTint(*wait.m_surface, texels[i], dayBases[numDay]);
			for (int c = 0; c < 3; c++)
			{
				dayNormals[numDay][c] = normals[i][c];
				dayDirs[numDay][c] = wait.m_dir[c];
			}
			dayShadows[numDay++] = wait.m_shadow;
		}
		daylightLightMany(*job.m_shading, daySurfaces, dayNormals, dayBases, dayDirs, dayShadows, numDay, dayLit);
		int skyFor[kTileSize * kTileSize], numSky = 0;
		const int firstSky = numWaiting;
		for (int i = 0, day = 0; i < numShades; i++)
		{
			const ShadeWait& wait = shades[i];
			if (wait.m_kind == ShadeWait::kFragment)
			{
				shadeHitFinish(*job.m_shading, *wait.m_surface, normals[i], uvs[i], texels[i], wait.m_faceNormal, wait.m_dir, wait.m_shadow,
							   wait.m_point, shadeOut[i]);
				continue;
			}
			float base[3], moduleLit[3];
			const float* lit = moduleLit;
			if (wait.m_kind == ShadeWait::kModule)
			{
				surfaceTint(*wait.m_surface, texels[i], base);
				moduleLight(*job.m_shading, *wait.m_surface, normals[i], base, wait.m_dir, wait.m_shadow, moduleLit);
			}
			else
				lit = dayLit[day++];
			daylightPrepare(*job.m_shading, lit, wait.m_dir, wait.m_distance, waiting[numWaiting], true);
			skyFor[numSky++] = i;
			waitingOut[numWaiting++] = shadeOut[i];
		}
		// Their horizon colours under haze, looked up together: the sky's colour just above the horizon each way.
		const SwarmRaycastShading& shading = *job.m_shading;
		if (shading.m_hazeDistance > 0.0f && shading.m_sky && numSky)
		{
			float x[kTileSize * kTileSize], y[kTileSize * kTileSize], z[kTileSize * kTileSize];
			float horizon[kTileSize * kTileSize][3];
			for (int n = 0; n < numSky; n++)
			{
				float level[3] = {shades[skyFor[n]].m_dir[0], shades[skyFor[n]].m_dir[1], shades[skyFor[n]].m_dir[2]};
				level[shading.m_glint.m_upAxis] = 0.02f;
				x[n] = level[0];
				y[n] = level[1];
				z[n] = level[2];
			}
			shading.m_sky->radianceMany(x, y, z, numSky, horizon);
			for (int n = 0; n < numSky; n++)
				for (int c = 0; c < 3; c++)
					waiting[firstSky + n].m_horizon[c] = horizon[n][c];
		}
	}
	if (numWaiting)
		daylightFinishAll(*job.m_shading, waiting, numWaiting, waitingOut);
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
	TilePaint paint;
	if (job.m_raster)
		paintTile(*job.m_raster, setup.m_cam, width, job.m_height, row0, row1, col0, col1, paint);
	SurfaceMemo memo;
	memo.clear();
	// Shading and daylight colours wait here and are done together once the tile's samples are in.
	const bool defer = job.m_shading && !radiance;
	DaylightColour waiting[kTileSize * kTileSize];
	unsigned char* waitingOut[kTileSize * kTileSize];
	int numWaiting = 0;
	ShadeWait shades[kTileSize * kTileSize];
	unsigned char* shadeOut[kTileSize * kTileSize];
	int numShades = 0;
	for (int row = row0; row < row1; row++)
	{
		const double ndcY = pixelNdcY(row, job.m_height);
		for (int col = col0; col < col1; col++)
		{
			Sample sample;
			sample.m_radiance = 0.0f;
			sample.m_shade = &shades[numShades];
			const size_t offset = (size_t)row * width + col;
			const float reach = job.m_hintFar ? hintReach(job.m_hintFar, width, job.m_height, row, col) : INFINITY;
			FoundHit found;
			if (job.m_raster)
				paintedFound(paint, (row - row0) * kTileSize + (col - col0), found);
			const bool hit = traceRay(job, setup, pixelNdcX(col, width), ndcY, args, shadowArgs, sample, reach,
									  job.m_hitPoints ? job.m_hitPoints + offset * 3 : 0, job.m_raster ? &found : 0, &memo, defer);
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
			if (sample.m_shaded && !radiance && sample.m_shadeDeferred)
				shadeOut[numShades++] = &target.m_rgb[offset * 3];
			else if (sample.m_shaded && !radiance && sample.m_deferred)
			{
				waiting[numWaiting] = sample.m_colour;
				waitingOut[numWaiting++] = &target.m_rgb[offset * 3];
			}
			else if (sample.m_shaded && !radiance)
				for (int i = 0; i < 3; i++)
					target.m_rgb[offset * 3 + i] = sample.m_rgb[i];
		}
	}
	finishTileShading(job, setup, shadowArgs, shades, shadeOut, numShades, waiting, waitingOut, numWaiting);
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

// isEdge for columns col0 up to col1 of one row into `edge`, eight at a time where all four neighbours are inside the frame.
void edgeRow(const int* ids, const float* w, int width, int height, int row, int col0, int col1, float tolerance, bool* edge)
{
	int col = col0;
#if defined(__GNUC__)
	if (row > 0 && row + 1 < height)
	{
		if (col == 0 && col < col1)
		{
			edge[0] = isEdge(ids, w, width, height, row, 0, tolerance);
			col++;
		}
		const SwarmLaneMask8 magnitude = {0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff};
		for (; col + 8 <= col1 && col + 8 < width; col += 8)
		{
			const size_t offset = (size_t)row * width + col;
			SwarmLaneMask8 id, left, right, up, down;
			SwarmLanes8 wc, wl, wr, wu, wd;
			memcpy(&id, ids + offset, sizeof(id));
			memcpy(&left, ids + offset - 1, sizeof(left));
			memcpy(&right, ids + offset + 1, sizeof(right));
			memcpy(&up, ids + offset - width, sizeof(up));
			memcpy(&down, ids + offset + width, sizeof(down));
			memcpy(&wc, w + offset, sizeof(wc));
			memcpy(&wl, w + offset - 1, sizeof(wl));
			memcpy(&wr, w + offset + 1, sizeof(wr));
			memcpy(&wu, w + offset - width, sizeof(wu));
			memcpy(&wd, w + offset + width, sizeof(wd));
			// fabsf on each lane is the value with its sign bit cleared.
			const SwarmLanes8 limit = tolerance * (SwarmLanes8)((SwarmLaneMask8)wc & magnitude);
			const SwarmLanes8 across = (SwarmLanes8)((SwarmLaneMask8)((wl + wr) - 2.0f * wc) & magnitude);
			const SwarmLanes8 along = (SwarmLanes8)((SwarmLaneMask8)((wu + wd) - 2.0f * wc) & magnitude);
			const SwarmLaneMask8 hit = (left != id) | (right != id) | (up != id) | (down != id) | (across > limit) | (along > limit);
			for (int l = 0; l < 8; l++)
				edge[col - col0 + l] = hit[l] != 0;
		}
	}
#endif
	for (; col < col1; col++)
		edge[col - col0] = isEdge(ids, w, width, height, row, col, tolerance);
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

// A thread's projected triangles for the whole frame, one slot per hash, since an edge's pixels ask for the same few.
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
		const int k = i + 1 < poly.m_n ? i + 1 : 0;
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
		const int k = i + 1 < poly.m_n ? i + 1 : 0;
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

// ER_SWARM_CREASE_FILL: a mover's widened screen box and depth, where a nearer crease keeps its probe for sub-pixel gaps.
struct MoverRect
{
	int m_col0;
	int m_col1;
	int m_row0;
	int m_row1;
	float m_depth;
};
const int kMoverMargin = 2;
// A box nearer than kMoverNear guards the whole frame, unless within kAircraftReach: the aircraft's own parts.
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

// An edge pixel's blend written to the frame: encoded once from linear light, or rounded from byte values.
inline void writeBlend(unsigned char* pixel, const double colour[3], bool linear)
{
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

// An edge pixel whose probe hit waits for the tile's batched shading: its blend so far and the share the probe fills.
struct ProbeWait
{
	size_t m_offset;
	double m_colour[3];
	double m_rest;
	unsigned char m_rgb[3];
};

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
	// The probes' shading waits for the tile, as the first pass's does, and their pixels are blended once it is done.
	const bool defer = !job.m_shading->m_thermal;
	ShadeWait shades[kTileSize * kTileSize];
	unsigned char* shadeOut[kTileSize * kTileSize];
	DaylightColour waiting[kTileSize * kTileSize];
	unsigned char* waitingOut[kTileSize * kTileSize];
	ProbeWait probes[kTileSize * kTileSize];
	int numShades = 0, numWaiting = 0, numProbes = 0;
	SurfaceMemo memo;
	memo.clear();
	for (int row = row0; row < row1; row++)
	{
		const double ndcY = pixelNdcY(row, height);
		bool edge[kTileSize];
		edgeRow(ids, &scratch.m_inverseEyeDepth[0], width, height, row, col0, col1, tolerance, edge);
		for (int col = col0; col < col1; col++)
		{
			if (!edge[col - col0])
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
				probe.m_shade = &shades[numShades];
				unsigned char background[3];
				if (hasOwn)
				{
					// The uncovered part is what lies beyond the pixel's own surface: ask it with one ray at
					// that part's centre, and take the sky or clear colour when the ray meets nothing.
					// On the left column and bottom row this samples half a pixel outside, as keepFrame allows for.
					double px = (ndcX - coveredCx) / rest, py = (ndcY - coveredCy) / rest;
					px = px < square.m_x[0] ? square.m_x[0] : (px > square.m_x[1] ? square.m_x[1] : px);
					py = py < square.m_y[0] ? square.m_y[0] : (py > square.m_y[2] ? square.m_y[2] : py);
					const float reach = probeReach(&scratch.m_inverseEyeDepth[0], width, height, row, col);
					const bool hit = traceRay(job, setup, px, py, args, shadowArgs, probe, reach, 0, 0, &memo, defer) && probe.m_shaded;
					if (hit && (probe.m_shadeDeferred || probe.m_deferred))
					{
						ProbeWait& wait = probes[numProbes++];
						wait.m_offset = offset;
						for (int i = 0; i < 3; i++)
							wait.m_colour[i] = colour[i];
						wait.m_rest = rest;
						if (probe.m_shadeDeferred)
							shadeOut[numShades++] = wait.m_rgb;
						else
						{
							waiting[numWaiting] = probe.m_colour;
							waitingOut[numWaiting++] = wait.m_rgb;
						}
						continue;
					}
					if (hit)
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
				// A crease under ER_SWARM_CREASE_FILL lands here too, its gap taken to look like its neighbours' triangles.
				for (int i = 0; i < 3; i++)
					colour[i] /= covered;

			writeBlend(target.m_rgb + offset * 3, colour, linear);
		}
	}
	if (!numProbes)
		return;
	finishTileShading(job, setup, shadowArgs, shades, shadeOut, numShades, waiting, waitingOut, numWaiting);
	for (int n = 0; n < numProbes; n++)
	{
		ProbeWait& wait = probes[n];
		for (int i = 0; i < 3; i++)
			wait.m_colour[i] += wait.m_rest * (linear ? kSwarmSrgbToLinear[wait.m_rgb[i]] : (double)wait.m_rgb[i]);
		writeBlend(target.m_rgb + wait.m_offset * 3, wait.m_colour, linear);
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

// The camera chain on a finished frame: thermal develops the radiance; low light and near infrared work on the colour.
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

// Everything a lone camera's frame reads but the scene, leaving out the grain seeds that only the camera chain reads.
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

	// ER_SWARM_RASTER: a camera that remembers its lens paints its frame. Here the chunks in view are listed, cut the first
	// time they are painted; in the parallel region every thread paints its share before the first tile is traced.
	const bool raster = memory != 0 && shading && shading->m_raster;
	RasterView rasterView;
	RasterFrame rasterFrame;
	long long forestCells = 0;
	job.m_raster = 0;
	job.m_alphaCutout = alphaCutout;
	if (raster)
	{
		setupRasterView(setups[0].m_cam, width, height, m_data->m_staticInstanceId, rasterView);
		const size_t frameTiles = (size_t)rasterView.m_tilesX * (size_t)((height + kTileSize - 1) / kTileSize);
		m_data->m_rasterLanes.resize((size_t)threads);
		for (size_t i = 0; i < m_data->m_rasterLanes.size(); i++)
		{
			RasterLane& lane = m_data->m_rasterLanes[i];
			lane.m_tris.clear();
			lane.m_rects.clear();
			lane.m_bins.resize(frameTiles);
			for (size_t t = 0; t < frameTiles; t++)
				lane.m_bins[t].clear();
		}
		std::vector<RasterJob>& jobs = m_data->m_rasterJobs;
		std::vector<RasterSource>& sources = m_data->m_rasterSources;
		jobs.clear();
		sources.clear();
		// Reserved whole, so the sources the jobs point at never move.
		sources.reserve(m_data->m_members.size() + m_data->m_byGeomId.size());
		for (size_t i = 0; i < m_data->m_members.size(); i++)
		{
			StaticMember* member = m_data->m_members[i];
			if (member->m_retired || !member->m_visible)
				continue;
			if (member->m_chunks.empty())
				buildRasterChunks(member->m_vertices, member->m_indices, member->m_chunks, member->m_chunksLo, member->m_chunksHi);
			if (!boxVisible(rasterView, member->m_chunksLo, member->m_chunksHi))
				continue;
			const RasterSource* source = 0;
			if (alphaCutout && member->m_hasAlpha)
			{
				RasterSource s;
				s.m_model = member->m_obj->m_model;
				s.m_vertices = &member->m_vertices[0];
				s.m_uvs = member->m_uvs;
				s.m_indices = &member->m_indices;
				s.m_objectSpace = false;
				sources.push_back(s);
				source = &sources.back();
			}
			for (size_t c = 0; c < member->m_chunks.size(); c++)
			{
				const RasterJob unit = {&member->m_chunks[c], member, 0, source};
				jobs.push_back(unit);
			}
		}
		for (size_t i = 0; i < m_data->m_byGeomId.size(); i++)
		{
			const Instance* inst = m_data->m_byGeomId[i];
			if (!inst || !inst->m_enabled || !inst->m_tree)
				continue;
			const float* m = inst->m_transform;
			const double a[3][3] = {{m[0], m[4], m[8]}, {m[1], m[5], m[9]}, {m[2], m[6], m[10]}};
			const double cof[3][3] = {{a[1][1] * a[2][2] - a[1][2] * a[2][1], a[0][2] * a[2][1] - a[0][1] * a[2][2], a[0][1] * a[1][2] - a[0][2] * a[1][1]},
									  {a[1][2] * a[2][0] - a[1][0] * a[2][2], a[0][0] * a[2][2] - a[0][2] * a[2][0], a[0][2] * a[1][0] - a[0][0] * a[1][2]},
									  {a[1][0] * a[2][1] - a[1][1] * a[2][0], a[0][1] * a[2][0] - a[0][0] * a[2][1], a[0][0] * a[1][1] - a[0][1] * a[1][0]}};
			const double det = a[0][0] * cof[0][0] + a[0][1] * cof[1][0] + a[0][2] * cof[2][0];
			// A flattened placement is disabled in the tree, so nothing of it is drawn.
			if (!(det != 0.0))
				continue;
			MeshTree* tree = inst->m_tree;
			if (tree->m_chunks.empty())
				buildRasterChunks(tree->m_vertices, tree->m_indices, tree->m_chunks, tree->m_chunksLo, tree->m_chunksHi);
			float lo[3], hi[3];
			worldBox(m, tree->m_chunksLo, tree->m_chunksHi, lo, hi);
			if (!boxVisible(rasterView, lo, hi))
				continue;
			const RasterSource* source = 0;
			if (alphaCutout && inst->m_hasAlpha)
			{
				RasterSource s;
				s.m_model = inst->m_obj->m_model;
				s.m_vertices = &tree->m_vertices[0];
				s.m_uvs = tree->m_uvs.data();
				s.m_indices = &tree->m_indices;
				s.m_objectSpace = true;
				for (int r = 0; r < 3; r++)
					for (int c = 0; c < 3; c++)
						s.m_inverse[r * 3 + c] = (float)(cof[r][c] / det);
				sources.push_back(s);
				source = &sources.back();
			}
			for (size_t c = 0; c < tree->m_chunks.size(); c++)
			{
				const RasterJob unit = {&tree->m_chunks[c], 0, inst, source};
				jobs.push_back(unit);
			}
		}
		if (!m_data->m_forestGrid.m_built)
			buildForestGrid(m_data->m_forestGrid, m_data->m_batches);
		forestCells = (long long)m_data->m_forestGrid.m_cellStart.size() - 1;
		rasterFrame.m_lanes = &m_data->m_rasterLanes;
		rasterFrame.m_tilesX = rasterView.m_tilesX;
		rasterFrame.m_spread = job.m_pixelSpread;
		job.m_raster = &rasterFrame;
	}

	// The depth hint: the same lens's last hits, put onto this frame's pixels, tell each ray about how far to search.
	// A painted frame needs none: the painting bounds every ray it still casts.
	job.m_hintFar = 0;
	job.m_hitPoints = 0;
	float viewProj[4][4];
	std::vector<float*> hintFrames;
	const float* lastPoints = 0;
	if (memory)
	{
		const Camera& cam = setups[0].m_cam;
		if (memory->m_points.size() == numPixels * 3 && !raster)
		{
			for (int r = 0; r < 4; r++)
				for (int c = 0; c < 4; c++)
					viewProj[r][c] = (float)cam.m_viewProj[r][c];
			// One farthest-hit frame per render thread, the first being the hint, filled inside the pixel loop's region.
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
	// whatever the thread count; each thread takes the next untraced tile, whose pixels never depend on which.
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
		ctx.m_directBatch = 0;
		ctx.m_directPlacement = 0;
		RTCIntersectArguments args;
		rtcInitIntersectArguments(&args);
		args.context = &ctx.m_context;
		RTCOccludedArguments shadowArgs;
		rtcInitOccludedArguments(&shadowArgs);
		shadowArgs.context = &ctx.m_context;

		if (lastPoints)
		{
			// Each thread fills its own frame, then each pixel takes the maximum, the same whichever thread took which hit.
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

		// Every thread paints its share of the chunks and trees into its own lane; the loops' barriers then hand all the
		// lanes to whichever thread traces a tile.
		if (job.m_raster)
		{
#ifdef _OPENMP
			RasterLane& lane = m_data->m_rasterLanes[(size_t)omp_get_thread_num()];
#else
			RasterLane& lane = m_data->m_rasterLanes[0];
#endif
			const long long numJobs = (long long)m_data->m_rasterJobs.size();
#pragma omp for schedule(static)
			for (long long i = 0; i < numJobs; i++)
				paintJob(lane, rasterView, m_data->m_rasterJobs[(size_t)i]);
#pragma omp for schedule(static)
			for (long long i = 0; i < forestCells; i++)
				paintForestCell(lane, rasterView, m_data->m_forestGrid, (size_t)i, m_data->m_batches);
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
