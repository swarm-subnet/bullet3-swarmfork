#include "SwarmRaycast.h"

#include <embree4/rtcore.h>
#include <math.h>
#include <string.h>
#include <map>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "../../../TinyRenderer/TinyRenderer.h"
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
	std::vector<float> m_depth;  // INFINITY where the ray met nothing
	bool m_built;
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
	// Per-vertex unit normals already in world space, and uv pairs, indexed like m_vertices.
	std::vector<float> m_normals;
	std::vector<float> m_uvs;
	const void* m_meshKey;
	float m_transform[16];
	// Row-major rotation of the body: what TinyRenderer's inverse-transpose model matrix does to a normal.
	float m_rotation[9];
	bool m_doubleSided;
	bool m_visible;
	bool m_retired;
	int m_segmentation;
};

// One tree per distinct mesh, shared by every mover instance drawn from that mesh.
struct MeshTree
{
	RTCScene m_scene;
	RTCGeometry m_geometry;
	std::vector<float> m_vertices;
	std::vector<unsigned> m_indices;
	// Per-vertex unit normals in the mesh's own frame, and uv pairs, indexed like m_vertices.
	std::vector<float> m_normals;
	std::vector<float> m_uvs;
	int m_refs;
	bool m_dirty;
};

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
	// Sign of the instance determinant: a mirroring scale flips which side of a face is the front.
	float m_facingSign;
	int m_segmentation;
};

struct ObjectState
{
	StaticMember* m_member;
	Instance* m_instance;
};

struct QueryContext
{
	RTCRayQueryContext m_context;
	const std::vector<Instance*>* m_instances;
	// Set for the rays that build the shadow map: any drawn face stops them, like a shadow ray.
	bool m_anyWinding;
};

// TinyRenderer drops a single-sided face whose winding normal points away from the camera; the
// filter does the same in object space, where Embree hands over both the ray and the hit. Static
// members also drop out here when retired or fully transparent, so the static tree is never rebuilt.
void hitFilter(const RTCFilterFunctionNArguments* args)
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
		return;
	}
	const unsigned instId = hit->instID[0];
	if (instId == RTC_INVALID_GEOMETRY_ID || instId >= ctx->m_instances->size())
		return;
	const Instance* inst = (*ctx->m_instances)[instId];
	if (inst && !inst->m_doubleSided && facing * inst->m_facingSign >= 0.0f)
		args->valid[0] = 0;
}

// A shadow ray is stopped by any surface it meets, whichever way that surface is wound: a single-sided
// roof hides the sun from the ground even though the camera would see through its underside. Only the
// bodies that are not drawn at all, retired static members and fully transparent ones, let light past.
void shadowFilter(const RTCFilterFunctionNArguments* args)
{
	if (args->N != 1 || !args->geometryUserPtr)
		return;
	const StaticMember* member = (const StaticMember*)args->geometryUserPtr;
	if (member->m_retired || !member->m_visible)
		args->valid[0] = 0;
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
		std::vector<int> face = model->face(f);
		for (int j = 0; j < 3; j++)
			out[(size_t)f * 3 + j] = (unsigned)face[j];
	}
}

// Per-vertex uv and unit normal, gathered through the faces: every model the plugin builds names the
// same index for a corner's position, normal and uv, so each vertex slot is written with its own
// attributes. Normals are rotated into world space when a rotation is given. A model without normals
// leaves the normal block empty and shades with the face normal instead.
void copyAttributes(TinyRender::Model* model, const std::vector<unsigned>& indices, const float rotation[9],
					std::vector<float>& normals, std::vector<float>& uvs)
{
	const int numVerts = model->nverts();
	const bool hasNormals = model->nnormals() > 0;
	uvs.assign((size_t)numVerts * 2, 0.0f);
	normals.assign(hasNormals ? (size_t)numVerts * 3 : 0, 0.0f);
	for (size_t f = 0; f * 3 + 2 < indices.size(); f++)
		for (int j = 0; j < 3; j++)
		{
			const unsigned v = indices[f * 3 + j];
			if (v >= (unsigned)numVerts)
				continue;
			const TinyRender::Vec2f uv = model->uv((int)f, j);
			uvs[(size_t)v * 2] = uv.x;
			uvs[(size_t)v * 2 + 1] = uv.y;
			if (!hasNormals)
				continue;
			const TinyRender::Vec3f n = model->storedNormal((int)f, j);
			const float local[3] = {n.x, n.y, n.z};
			float out[3];
			for (int r = 0; r < 3; r++)
				out[r] = rotation ? (rotation[r * 3] * local[0] + rotation[r * 3 + 1] * local[1]) + rotation[r * 3 + 2] * local[2] : local[r];
			const float length = sqrtf((out[0] * out[0] + out[1] * out[1]) + out[2] * out[2]);
			for (int r = 0; r < 3; r++)
				normals[(size_t)v * 3 + r] = length > 0.0f ? out[r] / length : 0.0f;
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

// Rotation part of the body transform, row-major, for turning a model normal into world space.
void copyRotation(const btTransform& worldTransform, float out[9])
{
	const btMatrix3x3& basis = worldTransform.getBasis();
	for (int r = 0; r < 3; r++)
		for (int c = 0; c < 3; c++)
			out[r * 3 + c] = (float)basis[r][c];
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
}  // namespace

struct SwarmRaycast::Data
{
	RTCDevice m_device;
	RTCScene m_top;
	RTCScene m_static;
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
	ShadowMap m_shadowMap;
	// Static members whose shadow changed since the map was cast: retired, hidden or shown again.
	std::vector<StaticMember*> m_shadowDirty;

	void shadowChanged(StaticMember* member)
	{
		if (m_shadowMap.m_built)
			m_shadowDirty.push_back(member);
	}

	// Casts the cells [col0, col1) x [row0, row1) of the shadow map from the start plane along -light
	// into the static tree. Each cell is written by exactly one ray, so the thread count cannot change
	// the bytes.
	void castShadowCells(int col0, int col1, int row0, int row1, int threads)
	{
		ShadowMap& map = m_shadowMap;
		const float dir[3] = {-map.m_lightDir[0], -map.m_lightDir[1], -map.m_lightDir[2]};
#pragma omp parallel for num_threads(threads) schedule(static)
		for (int row = row0; row < row1; row++)
		{
			QueryContext ctx;
			rtcInitRayQueryContext(&ctx.m_context);
			ctx.m_instances = 0;
			ctx.m_anyWinding = true;
			RTCIntersectArguments args;
			rtcInitIntersectArguments(&args);
			args.context = &ctx.m_context;
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
				rtcIntersect1(m_static, &rayhit, &args);
				map.m_depth[(size_t)row * map.m_cols + col] = rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID ? INFINITY : rayhit.ray.tfar;
			}
		}
	}

	// Lays the grid over the bounds of the static tree for this light, then casts every cell.
	void buildShadowMap(const float lightDir[3], int threads)
	{
		ShadowMap& map = m_shadowMap;
		map.m_built = false;
		std::vector<float>().swap(map.m_depth);
		m_shadowDirty.clear();
		RTCBounds bounds;
		rtcGetSceneBounds(m_static, &bounds);
		if (!(bounds.lower_x <= bounds.upper_x))
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
		map.m_depth.assign((size_t)map.m_cols * map.m_rows, INFINITY);
		castShadowCells(0, map.m_cols, 0, map.m_rows, threads);
		map.m_built = true;
	}

	// Recasts the cells under one member's vertices, one cell wider on every side.
	void recastMember(const StaticMember* member, int threads)
	{
		const ShadowMap& map = m_shadowMap;
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
		if (col0 < col1 && row0 < row1)
			castShadowCells(col0, col1, row0, row1, threads);
	}

	// Brings the map up to date for this light: cast in full when there is none or the light moved,
	// otherwise only under the members that changed since.
	void prepareShadowMap(const float lightDir[3], int threads)
	{
		if (!m_shadowMap.m_built || memcmp(m_shadowMap.m_lightDir, lightDir, sizeof(m_shadowMap.m_lightDir)) != 0)
		{
			buildShadowMap(lightDir, threads);
			return;
		}
		for (size_t i = 0; i < m_shadowDirty.size(); i++)
			recastMember(m_shadowDirty[i], threads);
		m_shadowDirty.clear();
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
		memcpy(member->m_transform, transform, sizeof(member->m_transform));
		copyRotation(worldTransform, member->m_rotation);
		member->m_retired = false;
		copyWorldVertices(obj->m_model, worldTransform, localScaling, member->m_vertices);
		copyIndices(obj->m_model, member->m_indices);
		copyAttributes(obj->m_model, member->m_indices, member->m_rotation, member->m_normals, member->m_uvs);
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
		copyLocalVertices(model, tree->m_vertices);
		copyIndices(model, tree->m_indices);
		copyAttributes(model, tree->m_indices, 0, tree->m_normals, tree->m_uvs);
		tree->m_scene = rtcNewScene(m_device);
		rtcSetSceneFlags(tree->m_scene, RTC_SCENE_FLAG_ROBUST);
		rtcSetSceneBuildQuality(tree->m_scene, RTC_BUILD_QUALITY_MEDIUM);
		tree->m_geometry = newTriangles(m_device, tree->m_vertices, tree->m_indices);
		rtcCommitGeometry(tree->m_geometry);
		rtcAttachGeometry(tree->m_scene, tree->m_geometry);
		rtcCommitScene(tree->m_scene);
		m_trees[key] = tree;
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
		rtcReleaseGeometry(tree->m_geometry);
		rtcReleaseScene(tree->m_scene);
		delete tree;
	}

	void refitTree(TinyRender::Model* model, MeshTree& tree)
	{
		if ((size_t)model->nverts() * 3 + kVertexPadding != tree.m_vertices.size())
			return;
		copyLocalVertices(model, tree.m_vertices);
		copyAttributes(model, tree.m_indices, 0, tree.m_normals, tree.m_uvs);
		rtcSetGeometryBuildQuality(tree.m_geometry, RTC_BUILD_QUALITY_REFIT);
		rtcUpdateGeometryBuffer(tree.m_geometry, RTC_BUFFER_TYPE_VERTEX, 0);
		rtcCommitGeometry(tree.m_geometry);
		rtcCommitScene(tree.m_scene);
		tree.m_dirty = false;
	}

	void syncInstance(ObjectState& state, TinyRenderObjectData* obj, const btTransform& worldTransform, const float transform[16], bool enabled, bool doubleSided, int segmentation)
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
		if (!inst->m_tree || inst->m_meshKey != key)
		{
			releaseTree(inst->m_tree);
			inst->m_tree = acquireTree(model, key);
			inst->m_meshKey = key;
			rtcSetGeometryInstancedScene(inst->m_geometry, inst->m_tree->m_scene);
			rtcSetGeometryInstancedScene(inst->m_shadowGeometry, inst->m_tree->m_scene);
			changed = true;
		}
		else if (inst->m_tree->m_dirty)
		{
			refitTree(model, *inst->m_tree);
			changed = true;
		}
		if (fresh || memcmp(transform, inst->m_transform, 16 * sizeof(float)) != 0)
		{
			memcpy(inst->m_transform, transform, 16 * sizeof(float));
			copyRotation(worldTransform, inst->m_rotation);
			rtcSetGeometryTransform(inst->m_geometry, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, transform);
			rtcSetGeometryTransform(inst->m_shadowGeometry, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, transform);
			const float det = transform[0] * (transform[5] * transform[10] - transform[9] * transform[6]) - transform[4] * (transform[1] * transform[10] - transform[9] * transform[2]) + transform[8] * (transform[1] * transform[6] - transform[5] * transform[2]);
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
		}
		inst->m_doubleSided = doubleSided;
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
		}
	}

	void dropInstance(Instance* inst)
	{
		rtcDetachGeometry(m_top, inst->m_geomId);
		rtcDetachGeometry(m_movers, inst->m_geomId);
		rtcReleaseGeometry(inst->m_geometry);
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
		m_shadowMap.m_built = false;
		std::vector<float>().swap(m_shadowMap.m_depth);
		m_shadowDirty.clear();
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
	m_data->m_movers = 0;
	m_data->m_staticInstance = 0;
	m_data->m_staticInstanceId = 0;
	m_data->m_staticBuilt = false;
	m_data->m_topDirty = false;
	m_data->m_moversDirty = true;
	m_data->m_shadowMap.m_built = false;
	// threads=1 keeps every tree build on the calling thread, so the same input gives the same tree everywhere.
	m_data->m_device = rtcNewDevice("threads=1,set_affinity=0");
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
	if (m_data->m_device)
		rtcReleaseDevice(m_data->m_device);
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
	const int segmentation = renderObj->m_objectIndex + ((renderObj->m_linkIndex + 1) << 24);

	ObjectState& state = m_data->m_objects[renderObj];
	StaticMember* member = state.m_member;
	if (member && !member->m_retired)
	{
		const bool moved = memcmp(transform, member->m_transform, sizeof(transform)) != 0 || member->m_meshKey != model->meshKey();
		if (!moved)
		{
			if (member->m_visible != visible)
				m_data->shadowChanged(member);
			member->m_visible = visible;
			member->m_doubleSided = doubleSided;
			member->m_segmentation = segmentation;
			return;
		}
		// The static tree stays as built; the hit filter ignores this member from now on.
		member->m_retired = true;
		m_data->shadowChanged(member);
	}
	else if (!member && !m_data->m_staticBuilt)
	{
		member = m_data->addMember(renderObj, worldTransform, localScaling, transform);
		member->m_visible = visible;
		member->m_doubleSided = doubleSided;
		member->m_segmentation = segmentation;
		state.m_member = member;
		state.m_instance = 0;
		return;
	}
	if (!member)
		state.m_member = 0;
	m_data->syncInstance(state, renderObj, worldTransform, transform, visible, doubleSided, segmentation);
}

void SwarmRaycast::meshChanged(TinyRenderObjectData* renderObj)
{
	std::map<TinyRenderObjectData*, ObjectState>::iterator found = m_data->m_objects.find(renderObj);
	if (found == m_data->m_objects.end())
		return;
	// A rewritten static member counts as moved at the next sync; a mover refits its tree.
	if (found->second.m_member && !found->second.m_member->m_retired)
		found->second.m_member->m_transform[15] = -1.0f;
	if (found->second.m_instance && found->second.m_instance->m_tree)
		found->second.m_instance->m_tree->m_dirty = true;
}

void SwarmRaycast::removeObject(TinyRenderObjectData* renderObj)
{
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
}

void SwarmRaycast::removeAll()
{
	if (!m_data->m_top)
		return;
	for (std::map<TinyRenderObjectData*, ObjectState>::iterator it = m_data->m_objects.begin(); it != m_data->m_objects.end(); ++it)
		if (it->second.m_instance)
			m_data->dropInstance(it->second.m_instance);
	m_data->m_objects.clear();
	// Retired members keep their triangles until the world is cleared, which is what happens here.
	rtcDetachGeometry(m_data->m_top, m_data->m_staticInstanceId);
	m_data->releaseStatic();
	m_data->m_byGeomId.clear();
	m_data->m_freeGeomIds.clear();
	m_data->createStaticScene();
}

void SwarmRaycast::commit()
{
	if (!m_data->m_top)
		return;
	if (!m_data->m_staticBuilt)
	{
		rtcCommitScene(m_data->m_static);
		m_data->m_staticBuilt = true;
		m_data->m_topDirty = true;
	}
	if (m_data->m_topDirty)
	{
		rtcCommitScene(m_data->m_top);
		m_data->m_topDirty = false;
	}
	if (m_data->m_moversDirty)
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
	float m_corners[3][3];
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
	return depth > map.m_depth[(size_t)(int)v * map.m_cols + (int)u] + map.m_cell;
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

// Segmentation id of a hit, plus the surface when asked for; false for a hit the scene does not know.
bool resolveHit(const RTCHit& hit, unsigned staticId, const std::vector<StaticMember*>& members,
				const std::vector<Instance*>& instances, int& segmentation, HitSurface* surface)
{
	const float* vertices;
	const unsigned* indices;
	const float* transform = 0;
	if (hit.instID[0] == staticId)
	{
		if (hit.geomID >= members.size())
			return false;
		const StaticMember* member = members[hit.geomID];
		segmentation = member->m_segmentation;
		if (!surface)
			return true;
		surface->m_model = member->m_obj->m_model;
		surface->m_rotation = 0;
		surface->m_normals = member->m_normals.empty() ? 0 : &member->m_normals[0];
		surface->m_uvs = &member->m_uvs[0];
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
		surface->m_rotation = inst->m_rotation;
		surface->m_normals = inst->m_tree->m_normals.empty() ? 0 : &inst->m_tree->m_normals[0];
		surface->m_uvs = &inst->m_tree->m_uvs[0];
		vertices = &inst->m_tree->m_vertices[0];
		indices = &inst->m_tree->m_indices[0];
		transform = inst->m_transform;
	}
	for (int j = 0; j < 3; j++)
	{
		const unsigned id = indices[(size_t)hit.primID * 3 + j];
		surface->m_vertexIds[j] = id;
		const float* v = vertices + (size_t)id * 3;
		if (transform)
			transformPoint(transform, v, surface->m_corners[j]);
		else
			for (int i = 0; i < 3; i++)
				surface->m_corners[j][i] = v[i];
	}
	return true;
}

// Barycentric weights of corners 1 and 2 where the ray origin + s * dir meets the triangle's plane;
// false when the ray runs along the plane. Used for the texture footprint of the neighbouring pixels.
// `normal` must be the triangle's normal as wound, since the weights carry its sign.
bool planeBarycentric(const float origin[3], const float dir[3], const float corners[3][3], const float normal[3], float& u, float& v)
{
	float e1[3], e2[3], toCorner[3], w[3], t1[3], t2[3];
	for (int i = 0; i < 3; i++)
	{
		e1[i] = corners[1][i] - corners[0][i];
		e2[i] = corners[2][i] - corners[0][i];
		toCorner[i] = corners[0][i] - origin[i];
	}
	const float denom = dot3(dir, normal);
	const float nn = dot3(normal, normal);
	if (denom == 0.0f || nn == 0.0f)
		return false;
	const float s = dot3(toCorner, normal) / denom;
	for (int i = 0; i < 3; i++)
		w[i] = (origin[i] + dir[i] * s) - corners[0][i];
	cross3(w, e2, t1);
	cross3(e1, w, t2);
	u = dot3(t1, normal) / nn;
	v = dot3(t2, normal) / nn;
	return true;
}

// TinyRenderer's fragment shader, term for term: interpolated normal and uv, the texture times the
// object colour, ambient plus the shadowed diffuse and specular terms, truncated to bytes. A mesh
// without vertex normals is lit by faceNormal, the triangle's own normal turned towards the camera.
void shadeHit(const SwarmRaycastShading& shading, const HitSurface& surface, const RTCHit& hit, const float faceNormal[3],
			  float shadow, bool filtered, const float duvdx[2], const float duvdy[2], unsigned char out[3])
{
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
		const float* nj = surface.m_normals + (size_t)surface.m_vertexIds[j] * 3;
		for (int r = 0; r < 3; r++)
			normal[r] += (surface.m_rotation ? dot3(surface.m_rotation + r * 3, nj) : nj[r]) * weights[j];
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
	for (int i = 0; i < 3; i++)
	{
		const unsigned char base = (unsigned char)(color[i] * rgba[i]);
		const float lit = (shading.m_ambientCoeff * base + shadow * (shading.m_diffuseCoeff * diffuse + shading.m_specularCoeff * specular) * base * shading.m_lightColor[i]);
		int value = 0;
		if (lit == lit)
			value = (int)lit;
		out[i] = (unsigned char)(value < 0 ? 0 : (value > 255 ? 255 : value));
	}
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
	const SwarmRaycastShading* m_shading;
	// The light's view of the static tree when the shadow comes from the map; null for shadow rays.
	const ShadowMap* m_shadowMap;
	// The tree of the mover instances when a lit hit also asks them for shadow; null otherwise.
	RTCScene m_movers;
	unsigned m_staticId;
	int m_width;
	int m_height;
	bool m_filtered;
};

// The triangle a ray landed on, enough to find its corners again.
struct HitId
{
	unsigned m_inst;
	unsigned m_geom;
	unsigned m_prim;
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
// Coverage below this is left to the colour already in the pixel: it is under one colour step.
const double kCoverageEpsilon = 1.0 / 512.0;

// One ray of a camera through the frame position (ndcX, ndcY). False on a miss; a hit fills `out`.
bool traceRay(const TileJob& job, const CameraSetup& setup, double ndcX, double ndcY,
			  RTCIntersectArguments* args, RTCOccludedArguments* shadowArgs, Sample& out)
{
	const Camera& cam = setup.m_cam;
	const SwarmRaycastShading* shading = job.m_shading;
	const bool filtered = job.m_filtered;
	const float* stepX = setup.m_stepX;
	const float* stepY = setup.m_stepY;

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
	rtcIntersect1(job.m_top, &rayhit, args);
	if (rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID)
		return false;

	const float t = rayhit.ray.tfar;
	const float hx = cam.m_origin[0] + dir[0] * t;
	const float hy = cam.m_origin[1] + dir[1] * t;
	const float hz = cam.m_origin[2] + dir[2] * t;
	const float zEye = ((cam.m_viewRow2[0] * hx + cam.m_viewRow2[1] * hy) + cam.m_viewRow2[2] * hz) + cam.m_viewRow2[3];
	out.m_depth = -(cam.m_p22 * zEye + cam.m_p23);
	out.m_inverseEyeDepth = 1.0f / zEye;

	int segmentation = -1;
	HitSurface surface;
	const bool known = resolveHit(rayhit.hit, job.m_staticId, *job.m_members, *job.m_instances, segmentation, shading ? &surface : 0);
	out.m_segmentation = segmentation;
	out.m_hit.m_inst = rayhit.hit.instID[0];
	out.m_hit.m_geom = rayhit.hit.geomID;
	out.m_hit.m_prim = rayhit.hit.primID;
	out.m_shaded = shading && known;
	if (!out.m_shaded)
		return true;

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
	if (shading->m_shadow)
	{
		float unitNormal[3] = {faceNormal[0], faceNormal[1], faceNormal[2]};
		normalize3(unitNormal);
		const float point[3] = {hx, hy, hz};
		// The map answers for the static bodies; the ray then goes to the whole world without a
		// map, to the movers alone with one, or nowhere when the map already says blocked.
		bool blocked = job.m_shadowMap && shadowMapBlocked(*job.m_shadowMap, point, unitNormal);
		const RTCScene occluders = job.m_shadowMap ? job.m_movers : job.m_top;
		if (!blocked && occluders)
		{
			RTCRay ray;
			ray.org_x = hx + unitNormal[0] * kShadowBias;
			ray.org_y = hy + unitNormal[1] * kShadowBias;
			ray.org_z = hz + unitNormal[2] * kShadowBias;
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
		// The same 0.8 floor TinyRenderer's shader applies where its shadow buffer says blocked.
		shadow = (float)(0.8 + 0.2 * !blocked);
	}

	float duvdx[2] = {0.0f, 0.0f}, duvdy[2] = {0.0f, 0.0f};
	if (filtered)
	{
		// TinyRenderer measures the texture footprint at the pixel to the right and the one above.
		const float weights[2] = {rayhit.hit.u, rayhit.hit.v};
		const float* steps[2] = {stepX, stepY};
		float* out[2] = {duvdx, duvdy};
		const float* uv0 = surface.m_uvs + (size_t)surface.m_vertexIds[0] * 2;
		const float* uv1 = surface.m_uvs + (size_t)surface.m_vertexIds[1] * 2;
		const float* uv2 = surface.m_uvs + (size_t)surface.m_vertexIds[2] * 2;
		for (int k = 0; k < 2; k++)
		{
			float neighbourDir[3], u, v;
			for (int i = 0; i < 3; i++)
				neighbourDir[i] = rawDir[i] + steps[k][i];
			if (!planeBarycentric(cam.m_origin, neighbourDir, surface.m_corners, woundNormal, u, v))
				continue;
			out[k][0] = (uv1[0] - uv0[0]) * (u - weights[0]) + (uv2[0] - uv0[0]) * (v - weights[1]);
			out[k][1] = (uv1[1] - uv0[1]) * (u - weights[0]) + (uv2[1] - uv0[1]) * (v - weights[1]);
		}
	}

	shadeHit(*shading, surface, rayhit.hit, faceNormal, shadow, filtered, duvdx, duvdy, out.m_rgb);
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
// `scratch`, when given, records the id and triangle of every hit for the edge pass.
void renderTile(const TileJob& job, const CameraSetup& setup, const SwarmRaycast::Target& target, EdgeScratch* scratch,
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
			const size_t offset = (size_t)row * width + col;
			if (!traceRay(job, setup, pixelNdcX(col, width), ndcY, args, shadowArgs, sample))
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
			if (sample.m_shaded)
				for (int i = 0; i < 3; i++)
					target.m_rgb[offset * 3 + i] = sample.m_rgb[i];
		}
	}
}

// A pixel is an edge when one of its four neighbours landed on another body, or when its 1/zEye is
// off the straight line through its two neighbours on either axis.
bool isEdge(const int* ids, const float* w, int width, int height, int row, int col)
{
	const size_t offset = (size_t)row * width + col;
	const int id = ids[offset];
	const bool hasLeft = col > 0, hasRight = col + 1 < width, hasUp = row > 0, hasDown = row + 1 < height;
	if ((hasLeft && ids[offset - 1] != id) || (hasRight && ids[offset + 1] != id) ||
		(hasUp && ids[offset - width] != id) || (hasDown && ids[offset + width] != id))
		return true;
	const float limit = kEdgeTolerance * fabsf(w[offset]);
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
	int segmentation;
	HitSurface surface;
	if (!resolveHit(rtcHit, job.m_staticId, *job.m_members, *job.m_instances, segmentation, &surface))
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

// The last few triangles a thread put onto the frame: an edge runs through several pixels that all
// ask for the same triangle, so each is projected once per run instead of once per pixel.
struct ProjectionCache
{
	HitId m_key[8];
	double m_x[8][3];
	double m_y[8][3];
	bool m_ok[8];
	int m_next;
	int m_count;
};

bool projectCached(ProjectionCache& cache, const TileJob& job, const Camera& cam, const HitId& hit, const double*& x, const double*& y)
{
	for (int i = 0; i < cache.m_count; i++)
		if (cache.m_key[i].m_inst == hit.m_inst && cache.m_key[i].m_geom == hit.m_geom && cache.m_key[i].m_prim == hit.m_prim)
		{
			x = cache.m_x[i];
			y = cache.m_y[i];
			return cache.m_ok[i];
		}
	const int slot = cache.m_next;
	cache.m_next = (slot + 1) % 8;
	if (cache.m_count < 8)
		cache.m_count++;
	cache.m_key[slot] = hit;
	cache.m_ok[slot] = projectTriangle(job, cam, hit, cache.m_x[slot], cache.m_y[slot]);
	x = cache.m_x[slot];
	y = cache.m_y[slot];
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

inline bool sameTriangle(const HitId& a, const HitId& b)
{
	return a.m_inst == b.m_inst && a.m_geom == b.m_geom && a.m_prim == b.m_prim;
}

// Second pass over one tile: the exact anti-aliasing a ray caster can afford. For every edge pixel
// the triangles its own ray and its four neighbours' rays landed on are put back onto the frame and
// the pixel square is clipped against each, front to back, so each surface gets exactly the share of
// the pixel it covers, with the colour pass one already shaded for it. A body in front of the pixel's
// own hit is composited over it at its true width, so a thin pole or cable stops breaking into dots.
// Whatever share is left uncovered is asked with one probe ray at its centre, which is the only ray
// this pass casts. Depth and segmentation stay as the first ray wrote them.
void refineTile(const TileJob& job, const CameraSetup& setup, const SwarmRaycast::Target& target, const EdgeScratch& scratch,
				int row0, int row1, int col0, int col1,
				RTCIntersectArguments* args, RTCOccludedArguments* shadowArgs, ProjectionCache& cache)
{
	const Camera& cam = setup.m_cam;
	const int width = job.m_width;
	const int height = job.m_height;
	const int* ids = &scratch.m_ids[0];
	const HitId* hits = &scratch.m_hits[0];
	const unsigned char* rgb1 = &scratch.m_rgb1[0];
	const double halfX = 1.0 / (double)width;
	const double halfY = 1.0 / (double)height;
	const double squareArea = 4.0 * halfX * halfY;
	for (int row = row0; row < row1; row++)
	{
		const double ndcY = pixelNdcY(row, height);
		for (int col = col0; col < col1; col++)
		{
			if (!isEdge(ids, &scratch.m_inverseEyeDepth[0], width, height, row, col))
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
					colour[i] += share * candidates[c].m_rgb[i];
				coveredCx += share * cx;
				coveredCy += share * cy;
				covered += share;
			}

			const double rest = 1.0 - covered;
			if (rest > kCoverageEpsilon)
			{
				const unsigned char* restColour = rgb1 + offset * 3;
				Sample probe;
				if (hasOwn)
				{
					// The uncovered part is what lies beyond the pixel's own surface: ask it with one ray at
					// that part's centre, and take the sky or clear colour when the ray meets nothing.
					double px = (ndcX - coveredCx) / rest, py = (ndcY - coveredCy) / rest;
					px = px < square.m_x[0] ? square.m_x[0] : (px > square.m_x[1] ? square.m_x[1] : px);
					py = py < square.m_y[0] ? square.m_y[0] : (py > square.m_y[2] ? square.m_y[2] : py);
					restColour = (traceRay(job, setup, px, py, args, shadowArgs, probe) && probe.m_shaded)
									 ? probe.m_rgb
									 : &scratch.m_background[offset * 3];
				}
				for (int i = 0; i < 3; i++)
					colour[i] += rest * restColour[i];
			}
			else if (covered < 1.0)
				for (int i = 0; i < 3; i++)
					colour[i] /= covered;

			unsigned char* pixel = target.m_rgb + offset * 3;
			for (int i = 0; i < 3; i++)
			{
				const int value = (int)(colour[i] + 0.5);
				pixel[i] = (unsigned char)(value < 0 ? 0 : (value > 255 ? 255 : value));
			}
		}
	}
}
}  // namespace

void SwarmRaycast::render(const Target* targets, int numTargets, const float projMat[16], int width, int height,
						  const SwarmRaycastShading* shading, int threads) const
{
	if (!m_data->m_top || m_data->m_objects.empty() || width <= 0 || height <= 0 || numTargets <= 0)
		return;
	if (threads < 1)
		threads = 1;

	TileJob job;
	job.m_top = m_data->m_top;
	job.m_instances = &m_data->m_byGeomId;
	job.m_members = &m_data->m_members;
	job.m_shading = shading;
	job.m_staticId = m_data->m_staticInstanceId;
	job.m_width = width;
	job.m_height = height;
	job.m_filtered = shading && shading->m_textureFilter;
	// The map is cast on the calling thread's schedule before the pixel loop, which then only reads it.
	job.m_shadowMap = 0;
	job.m_movers = 0;
	if (shading && shading->m_shadow && shading->m_shadowMap)
	{
		m_data->prepareShadowMap(shading->m_lightDir, threads);
		job.m_shadowMap = &m_data->m_shadowMap;
		if (shading->m_moverShadow)
			job.m_movers = m_data->m_movers;
	}
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
			continue;
		// The unnormalised ray direction is affine in the pixel position, so the direction one pixel to
		// the right or one row up is the pixel's own direction plus a constant step.
		for (int k = 0; k < 3; k++)
		{
			setup.m_stepX[k] = (float)((setup.m_cam.m_far[1][k] - setup.m_cam.m_near[1][k]) * (2.0 / (double)width));
			setup.m_stepY[k] = (float)((setup.m_cam.m_far[2][k] - setup.m_cam.m_near[2][k]) * (2.0 / (double)height));
		}
	}

	std::vector<EdgeScratch> scratch((shading && shading->m_edgeAntialias) ? (size_t)numTargets : 0);
	for (size_t i = 0; i < scratch.size(); i++)
	{
		if (!setups[i].m_valid || !targets[i].m_rgb || !targets[i].m_depth)
			continue;
		HitId none;
		none.m_inst = none.m_geom = none.m_prim = RTC_INVALID_GEOMETRY_ID;
		scratch[i].m_ids.assign(numPixels, -1);
		scratch[i].m_hits.assign(numPixels, none);
		scratch[i].m_inverseEyeDepth.assign(numPixels, 0.0f);
		scratch[i].m_background.assign(targets[i].m_rgb, targets[i].m_rgb + numPixels * 3);
	}

	// The tile grid is fixed by the frame size alone: tile k covers the same pixels of the same camera
	// whatever the thread count, and thread t traces tiles t, t + threads, t + 2 threads, ...
	const int tilesX = (width + kTileSize - 1) / kTileSize;
	const int tilesY = (height + kTileSize - 1) / kTileSize;
	const int tilesPerCamera = tilesX * tilesY;
	const int numTiles = tilesPerCamera * numTargets;

#pragma omp parallel num_threads(threads)
	{
#ifdef _OPENMP
		const int tid = omp_get_thread_num();
		const int cnt = omp_get_num_threads();
#else
		const int tid = 0;
		const int cnt = 1;
#endif
		QueryContext ctx;
		rtcInitRayQueryContext(&ctx.m_context);
		ctx.m_instances = job.m_instances;
		ctx.m_anyWinding = false;
		RTCIntersectArguments args;
		rtcInitIntersectArguments(&args);
		args.context = &ctx.m_context;
		RTCOccludedArguments shadowArgs;
		rtcInitOccludedArguments(&shadowArgs);
		shadowArgs.context = &ctx.m_context;

		// Pass 2 reads the neighbours pass 1 wrote and the colours pass 1 shaded, so every thread
		// finishes pass 1, and the pass-1 colours are copied aside, before any thread starts pass 2.
		const int passes = scratch.empty() ? 1 : 2;
		ProjectionCache cache;
		cache.m_next = 0;
		cache.m_count = 0;
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
			for (int tile = tid; tile < numTiles; tile += cnt)
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
				if (pass == 0)
					renderTile(job, setups[(size_t)camIndex], targets[camIndex], edge, row0, row1, col0, col1, &args, &shadowArgs);
				else if (edge)
					refineTile(job, setups[(size_t)camIndex], targets[camIndex], *edge, row0, row1, col0, col1, &args, &shadowArgs, cache);
			}
		}
	}
}
