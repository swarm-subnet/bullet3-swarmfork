#include "SwarmRaycast.h"

#include <embree4/rtcore.h>
#include <math.h>
#include <string.h>
#include <map>
#include <vector>

#include "../../../TinyRenderer/TinyRenderer.h"
#include "Bullet3Common/b3Logging.h"
#include "LinearMath/btTransform.h"

namespace
{
// Embree reads vertices 16 bytes at a time, so every vertex block carries one spare slot.
const size_t kVertexPadding = 4;

// A body that never moved since the first frame: its triangles sit in world space inside the one
// static tree. When it moves later it is retired here and carries on as a mover instance.
struct StaticMember
{
	TinyRenderObjectData* m_obj;
	RTCGeometry m_geometry;
	unsigned m_geomId;
	std::vector<float> m_vertices;
	std::vector<unsigned> m_indices;
	const void* m_meshKey;
	float m_transform[16];
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
	int m_refs;
	bool m_dirty;
};

struct Instance
{
	TinyRenderObjectData* m_obj;
	RTCGeometry m_geometry;
	unsigned m_geomId;
	MeshTree* m_tree;
	const void* m_meshKey;
	float m_transform[16];
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
	const float facing = hit->Ng_x * ray->dir_x + hit->Ng_y * ray->dir_y + hit->Ng_z * ray->dir_z;
	if (args->geometryUserPtr)
	{
		const StaticMember* member = (const StaticMember*)args->geometryUserPtr;
		if (member->m_retired || !member->m_visible || (!member->m_doubleSided && facing >= 0.0f))
			args->valid[0] = 0;
		return;
	}
	const QueryContext* ctx = (const QueryContext*)args->context;
	const unsigned instId = hit->instID[0];
	if (instId == RTC_INVALID_GEOMETRY_ID || instId >= ctx->m_instances->size())
		return;
	const Instance* inst = (*ctx->m_instances)[instId];
	if (inst && !inst->m_doubleSided && facing * inst->m_facingSign >= 0.0f)
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

RTCGeometry newTriangles(RTCDevice device, std::vector<float>& vertices, std::vector<unsigned>& indices)
{
	RTCGeometry geometry = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
	rtcSetSharedGeometryBuffer(geometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
							   &vertices[0], 0, 3 * sizeof(float), (vertices.size() - kVertexPadding) / 3);
	rtcSetSharedGeometryBuffer(geometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
							   &indices[0], 0, 3 * sizeof(unsigned), indices.size() / 3);
	rtcSetGeometryIntersectFilterFunction(geometry, hitFilter);
	return geometry;
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
	RTCGeometry m_staticInstance;
	unsigned m_staticInstanceId;
	bool m_staticBuilt;
	bool m_topDirty;
	std::vector<StaticMember*> m_members;
	std::map<const void*, MeshTree*> m_trees;
	std::vector<Instance*> m_byGeomId;
	std::vector<unsigned> m_freeGeomIds;
	std::map<TinyRenderObjectData*, ObjectState> m_objects;

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
		member->m_retired = false;
		copyWorldVertices(obj->m_model, worldTransform, localScaling, member->m_vertices);
		copyIndices(obj->m_model, member->m_indices);
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
		rtcSetGeometryBuildQuality(tree.m_geometry, RTC_BUILD_QUALITY_REFIT);
		rtcUpdateGeometryBuffer(tree.m_geometry, RTC_BUFFER_TYPE_VERTEX, 0);
		rtcCommitGeometry(tree.m_geometry);
		rtcCommitScene(tree.m_scene);
		tree.m_dirty = false;
	}

	void syncInstance(ObjectState& state, TinyRenderObjectData* obj, const float transform[16], bool enabled, bool doubleSided, int segmentation)
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
			rtcSetGeometryTransform(inst->m_geometry, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, transform);
			const float det = transform[0] * (transform[5] * transform[10] - transform[9] * transform[6]) - transform[4] * (transform[1] * transform[10] - transform[9] * transform[2]) + transform[8] * (transform[1] * transform[6] - transform[5] * transform[2]);
			inst->m_facingSign = (det < 0.0f) ? -1.0f : 1.0f;
			changed = true;
		}
		if (enabled != inst->m_enabled)
		{
			if (enabled)
				rtcEnableGeometry(inst->m_geometry);
			else
				rtcDisableGeometry(inst->m_geometry);
			inst->m_enabled = enabled;
			changed = true;
		}
		inst->m_doubleSided = doubleSided;
		inst->m_segmentation = segmentation;
		if (changed)
		{
			rtcCommitGeometry(inst->m_geometry);
			m_topDirty = true;
		}
		if (fresh)
		{
			inst->m_geomId = allocateGeomId(inst);
			rtcAttachGeometryByID(m_top, inst->m_geometry, inst->m_geomId);
		}
	}

	void dropInstance(Instance* inst)
	{
		rtcDetachGeometry(m_top, inst->m_geomId);
		rtcReleaseGeometry(inst->m_geometry);
		releaseTree(inst->m_tree);
		m_byGeomId[inst->m_geomId] = 0;
		m_freeGeomIds.push_back(inst->m_geomId);
		delete inst;
		m_topDirty = true;
	}

	void releaseStatic()
	{
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
	m_data->m_staticInstance = 0;
	m_data->m_staticInstanceId = 0;
	m_data->m_staticBuilt = false;
	m_data->m_topDirty = false;
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
	m_data->createStaticScene();
}

SwarmRaycast::~SwarmRaycast()
{
	removeAll();
	if (m_data->m_top)
	{
		m_data->releaseStatic();
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
			member->m_visible = visible;
			member->m_doubleSided = doubleSided;
			member->m_segmentation = segmentation;
			return;
		}
		// The static tree stays as built; the hit filter ignores this member from now on.
		member->m_retired = true;
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
	m_data->syncInstance(state, renderObj, transform, visible, doubleSided, segmentation);
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
		state.m_member->m_retired = true;
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
}

void SwarmRaycast::renderDepth(const float viewMat[16], const float projMat[16], int width, int height,
							   float* depthOut, int* segOut, int threads) const
{
	if (!m_data->m_top || m_data->m_objects.empty() || width <= 0 || height <= 0)
		return;
	Camera cam;
	if (!setupCamera(viewMat, projMat, cam))
		return;
	if (threads < 1)
		threads = 1;

	const std::vector<Instance*>* instances = &m_data->m_byGeomId;
	const std::vector<StaticMember*>* members = &m_data->m_members;
	const unsigned staticId = m_data->m_staticInstanceId;
	RTCScene top = m_data->m_top;

#pragma omp parallel for schedule(static) num_threads(threads)
	for (int row = 0; row < height; row++)
	{
		QueryContext ctx;
		rtcInitRayQueryContext(&ctx.m_context);
		ctx.m_instances = instances;
		RTCIntersectArguments args;
		rtcInitIntersectArguments(&args);
		args.context = &ctx.m_context;

		// Output row `row` is TinyRenderer's raster row height - 1 - row, sampled at the integer pixel corner.
		const double ndcY = 1.0 - (2.0 * row + 2.0) / (double)height;
		for (int col = 0; col < width; col++)
		{
			const double ndcX = (2.0 * col) / (double)width - 1.0;
			float nearPoint[3], farPoint[3];
			planePoint(cam.m_near, ndcX, ndcY, nearPoint);
			planePoint(cam.m_far, ndcX, ndcY, farPoint);
			float dir[3];
			for (int i = 0; i < 3; i++)
				dir[i] = farPoint[i] - nearPoint[i];
			const float length = sqrtf(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
			if (!(length > 0.0f))
				continue;
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
			rtcIntersect1(top, &rayhit, &args);
			if (rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID)
				continue;

			const float t = rayhit.ray.tfar;
			const float hx = cam.m_origin[0] + dir[0] * t;
			const float hy = cam.m_origin[1] + dir[1] * t;
			const float hz = cam.m_origin[2] + dir[2] * t;
			const float zEye = ((cam.m_viewRow2[0] * hx + cam.m_viewRow2[1] * hy) + cam.m_viewRow2[2] * hz) + cam.m_viewRow2[3];
			const size_t offset = (size_t)row * width + col;
			depthOut[offset] = -(cam.m_p22 * zEye + cam.m_p23);
			if (segOut)
			{
				const unsigned instId = rayhit.hit.instID[0];
				int segmentation = -1;
				if (instId == staticId)
				{
					if (rayhit.hit.geomID < members->size())
						segmentation = (*members)[rayhit.hit.geomID]->m_segmentation;
				}
				else if (instId < instances->size() && (*instances)[instId])
				{
					segmentation = (*instances)[instId]->m_segmentation;
				}
				segOut[offset] = segmentation;
			}
		}
	}
}
