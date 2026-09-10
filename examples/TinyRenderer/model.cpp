
#include "model.h"
#include <stdlib.h>  // getenv
#include <string.h>  // memcpy
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include "Bullet3Common/b3Logging.h"

namespace TinyRender
{
struct SharedMesh
{
	std::vector<Vec3f> verts_;
	std::vector<std::vector<Vec3i> > faces_;  // attention, this Vec3i means vertex/uv/normal
	std::vector<Vec3f> norms_;
	std::vector<Vec2f> uv_;
	unsigned long long hash_;
	int refs_;
	bool registered_;
	bool hasAABB_;
	Vec3f aabbMin_;
	Vec3f aabbMax_;

	SharedMesh() : hash_(0), refs_(1), registered_(false), hasAABB_(false) {}
};

struct SharedTexture
{
	TGAImage img_;
	const unsigned char* source_;  // texels the image was built from; a lookup key, never dereferenced blindly
	int refs_;
	bool registered_;

	SharedTexture() : source_(0), refs_(1), registered_(false) {}
};

// Process-wide registries of the blocks currently alive; an entry leaves when its last owner does.
static std::vector<SharedMesh*> gSharedMeshes;
static std::vector<SharedTexture*> gSharedTextures;

// SWARM_SHARE_MESH=0 gives every Model private copies, for before/after measurements on one build.
static bool sharingEnabled()
{
	static int cached = -1;
	if (cached < 0)
	{
		const char* env = getenv("SWARM_SHARE_MESH");
		cached = (env && env[0] == '0') ? 0 : 1;
	}
	return cached != 0;
}

// FNV-1a over 32-bit words: every mesh value is a float or an int, so word steps keep it exact and fast.
static unsigned long long fnv1a(const void* data, size_t numWords, unsigned long long hash)
{
	const unsigned int* words = (const unsigned int*)data;
	for (size_t i = 0; i < numWords; i++)
	{
		hash ^= words[i];
		hash *= 1099511628211ULL;
	}
	return hash;
}

static const unsigned long long FNV_SEED = 14695981039346656037ULL;

static void releaseMesh(SharedMesh* mesh)
{
	if (!mesh || --mesh->refs_ > 0)
		return;
	if (mesh->registered_)
	{
		for (size_t i = 0; i < gSharedMeshes.size(); i++)
		{
			if (gSharedMeshes[i] == mesh)
			{
				gSharedMeshes[i] = gSharedMeshes.back();
				gSharedMeshes.pop_back();
				break;
			}
		}
	}
	delete mesh;
}

static void releaseTexture(SharedTexture* tex)
{
	if (!tex || --tex->refs_ > 0)
		return;
	if (tex->registered_)
	{
		for (size_t i = 0; i < gSharedTextures.size(); i++)
		{
			if (gSharedTextures[i] == tex)
			{
				gSharedTextures[i] = gSharedTextures.back();
				gSharedTextures.pop_back();
				break;
			}
		}
	}
	delete tex;
}

// Bitwise comparison of the stored arrays against the raw input, so a hash match alone never selects a block.
static bool meshMatchesArrays(const SharedMesh& mesh, const float* vertices, int numVertices, const int* indices, int numIndices)
{
	if ((int)mesh.verts_.size() != numVertices || (int)mesh.faces_.size() != numIndices / 3)
		return false;
	for (int i = 0; i < numVertices; i++)
	{
		const float* v = vertices + i * 9;
		if (memcmp(&mesh.verts_[i][0], v, 3 * sizeof(float)) != 0 ||
			memcmp(&mesh.norms_[i][0], v + 4, 3 * sizeof(float)) != 0 ||
			memcmp(&mesh.uv_[i][0], v + 7, 2 * sizeof(float)) != 0)
			return false;
	}
	for (int f = 0; f < numIndices / 3; f++)
	{
		for (int j = 0; j < 3; j++)
		{
			if (mesh.faces_[f][j][0] != indices[f * 3 + j])
				return false;
		}
	}
	return true;
}

Model::Model(const char *filename) : m_mesh(new SharedMesh), m_diffuse(0), normalmap_(), specularmap_()
{
	std::ifstream in;
	in.open(filename, std::ifstream::in);
	if (in.fail()) return;
	std::string line;
	while (!in.eof())
	{
		std::getline(in, line);
		std::istringstream iss(line.c_str());
		char trash;
		if (!line.compare(0, 2, "v "))
		{
			iss >> trash;
			Vec3f v;
			for (int i = 0; i < 3; i++) iss >> v[i];
			m_mesh->verts_.push_back(v);
		}
		else if (!line.compare(0, 3, "vn "))
		{
			iss >> trash >> trash;
			Vec3f n;
			for (int i = 0; i < 3; i++) iss >> n[i];
			m_mesh->norms_.push_back(n);
		}
		else if (!line.compare(0, 3, "vt "))
		{
			iss >> trash >> trash;
			Vec2f uv;
			for (int i = 0; i < 2; i++) iss >> uv[i];
			m_mesh->uv_.push_back(uv);
		}
		else if (!line.compare(0, 2, "f "))
		{
			std::vector<Vec3i> f;
			Vec3i tmp;
			iss >> trash;
			while (iss >> tmp[0] >> trash >> tmp[1] >> trash >> tmp[2])
			{
				for (int i = 0; i < 3; i++) tmp[i]--;  // in wavefront obj all indices start at 1, not zero
				f.push_back(tmp);
			}
			m_mesh->faces_.push_back(f);
		}
	}
	std::cerr << "# v# " << m_mesh->verts_.size() << " f# " << m_mesh->faces_.size() << " vt# " << m_mesh->uv_.size() << " vn# " << m_mesh->norms_.size() << std::endl;
	m_diffuse = new SharedTexture;
	load_texture(filename, "_diffuse.tga", m_diffuse->img_);
	load_texture(filename, "_nm_tangent.tga", normalmap_);
	load_texture(filename, "_spec.tga", specularmap_);
}

Model::Model() : m_mesh(new SharedMesh), m_diffuse(0), normalmap_(), specularmap_()
{
}

void Model::setDiffuseTextureFromData(unsigned char *textureImage, int textureWidth, int textureHeight)
{
	releaseTexture(m_diffuse);
	m_diffuse = 0;
	if (!textureImage)
		return;

	const int rowBytes = textureWidth * 3;
	if (sharingEnabled())
	{
		// Same source texels and size is the candidate; the row compare below makes it certain.
		for (size_t i = 0; i < gSharedTextures.size(); i++)
		{
			SharedTexture* tex = gSharedTextures[i];
			if (tex->source_ != textureImage || tex->img_.get_width() != textureWidth || tex->img_.get_height() != textureHeight)
				continue;
			// The stored image is flipped, so row y holds input row height-1-y.
			bool same = true;
			for (int y = 0; same && y < textureHeight; y++)
			{
				same = memcmp(tex->img_.buffer() + (size_t)y * rowBytes, textureImage + (size_t)(textureHeight - 1 - y) * rowBytes, rowBytes) == 0;
			}
			if (same)
			{
				tex->refs_++;
				m_diffuse = tex;
				return;
			}
		}
	}

	m_diffuse = new SharedTexture;
	m_diffuse->source_ = textureImage;
	{
		B3_PROFILE("new TGAImage");
		m_diffuse->img_ = TGAImage(textureWidth, textureHeight, TGAImage::RGB);
	}
	{
		B3_PROFILE("copy texels");
		memcpy(m_diffuse->img_.buffer(), textureImage, (size_t)rowBytes * textureHeight);
	}
	{
		B3_PROFILE("flip_vertically");
		m_diffuse->img_.flip_vertically();
	}
	if (sharingEnabled())
	{
		m_diffuse->registered_ = true;
		gSharedTextures.push_back(m_diffuse);
	}
}

void Model::loadDiffuseTexture(const char *relativeFileName)
{
	releaseTexture(m_diffuse);
	m_diffuse = new SharedTexture;
	m_diffuse->img_.read_tga_file(relativeFileName);
}

void Model::setMeshFromArrays(const float* vertices, int numVertices, const int* indices, int numIndices)
{
	unsigned long long hash = fnv1a(&numVertices, 1, FNV_SEED);
	hash = fnv1a(&numIndices, 1, hash);
	for (int i = 0; i < numVertices; i++)
	{
		const float* v = vertices + i * 9;
		hash = fnv1a(v, 3, hash);
		hash = fnv1a(v + 4, 5, hash);
	}
	hash = fnv1a(indices, (size_t)numIndices, hash);

	if (sharingEnabled())
	{
		for (size_t i = 0; i < gSharedMeshes.size(); i++)
		{
			SharedMesh* mesh = gSharedMeshes[i];
			if (mesh->hash_ == hash && meshMatchesArrays(*mesh, vertices, numVertices, indices, numIndices))
			{
				mesh->refs_++;
				releaseMesh(m_mesh);
				m_mesh = mesh;
				return;
			}
		}
	}

	{
		B3_PROFILE("reserveMemory");
		reserveMemory(numVertices, numIndices);
	}
	{
		B3_PROFILE("addVertex");
		for (int i = 0; i < numVertices; i++)
		{
			addVertex(vertices[i * 9],
					  vertices[i * 9 + 1],
					  vertices[i * 9 + 2],
					  vertices[i * 9 + 4],
					  vertices[i * 9 + 5],
					  vertices[i * 9 + 6],
					  vertices[i * 9 + 7],
					  vertices[i * 9 + 8]);
		}
	}
	{
		B3_PROFILE("addTriangle");
		for (int i = 0; i < numIndices; i += 3)
		{
			addTriangle(indices[i], indices[i], indices[i],
						indices[i + 1], indices[i + 1], indices[i + 1],
						indices[i + 2], indices[i + 2], indices[i + 2]);
		}
	}
	m_mesh->hash_ = hash;
	if (sharingEnabled())
	{
		m_mesh->registered_ = true;
		gSharedMeshes.push_back(m_mesh);
	}
}

// Any write goes to a private block: clone when shared, unregister when this is the only owner.
void Model::detachMesh()
{
	m_mesh->hasAABB_ = false;
	if (m_mesh->refs_ > 1)
	{
		SharedMesh* copy = new SharedMesh;
		copy->verts_ = m_mesh->verts_;
		copy->faces_ = m_mesh->faces_;
		copy->norms_ = m_mesh->norms_;
		copy->uv_ = m_mesh->uv_;
		releaseMesh(m_mesh);
		m_mesh = copy;
	}
	else if (m_mesh->registered_)
	{
		for (size_t i = 0; i < gSharedMeshes.size(); i++)
		{
			if (gSharedMeshes[i] == m_mesh)
			{
				gSharedMeshes[i] = gSharedMeshes.back();
				gSharedMeshes.pop_back();
				break;
			}
		}
		m_mesh->registered_ = false;
	}
}

void Model::reserveMemory(int numVertices, int numIndices)
{
	detachMesh();
	m_mesh->verts_.reserve(numVertices);
	m_mesh->norms_.reserve(numVertices);
	m_mesh->uv_.reserve(numVertices);
	m_mesh->faces_.reserve(numIndices);
}

void Model::addVertex(float x, float y, float z, float normalX, float normalY, float normalZ, float u, float v)
{
	detachMesh();
	m_mesh->verts_.push_back(Vec3f(x, y, z));
	m_mesh->norms_.push_back(Vec3f(normalX, normalY, normalZ));
	m_mesh->uv_.push_back(Vec2f(u, v));
}
void Model::addTriangle(int vertexposIndex0, int normalIndex0, int uvIndex0,
						int vertexposIndex1, int normalIndex1, int uvIndex1,
						int vertexposIndex2, int normalIndex2, int uvIndex2)
{
	detachMesh();
	std::vector<Vec3i> f;
	f.push_back(Vec3i(vertexposIndex0, normalIndex0, uvIndex0));
	f.push_back(Vec3i(vertexposIndex1, normalIndex1, uvIndex1));
	f.push_back(Vec3i(vertexposIndex2, normalIndex2, uvIndex2));
	m_mesh->faces_.push_back(f);
}

bool Model::getLocalAABB(Vec3f& aabbMin, Vec3f& aabbMax)
{
	if (m_mesh->verts_.empty())
		return false;
	if (!m_mesh->hasAABB_)
	{
		Vec3f mn = m_mesh->verts_[0];
		Vec3f mx = mn;
		for (size_t i = 1; i < m_mesh->verts_.size(); i++)
		{
			const Vec3f& v = m_mesh->verts_[i];
			for (int k = 0; k < 3; k++)
			{
				if (v[k] < mn[k]) mn[k] = v[k];
				if (v[k] > mx[k]) mx[k] = v[k];
			}
		}
		m_mesh->aabbMin_ = mn;
		m_mesh->aabbMax_ = mx;
		m_mesh->hasAABB_ = true;
	}
	aabbMin = m_mesh->aabbMin_;
	aabbMax = m_mesh->aabbMax_;
	return true;
}

Model::~Model()
{
	releaseMesh(m_mesh);
	releaseTexture(m_diffuse);
}

int Model::nverts()
{
	return (int)m_mesh->verts_.size();
}

int Model::nnormals()
{
	return (int)m_mesh->norms_.size();
}

int Model::nfaces()
{
	return (int)m_mesh->faces_.size();
}

std::vector<int> Model::face(int idx)
{
	std::vector<int> face;
	face.reserve((int)m_mesh->faces_[idx].size());
	for (int i = 0; i < (int)m_mesh->faces_[idx].size(); i++)
		face.push_back(m_mesh->faces_[idx][i][0]);
	return face;
}

Vec3f Model::vert(int i)
{
	return m_mesh->verts_[i];
}

Vec3f Model::vert(int iface, int nthvert)
{
	return m_mesh->verts_[m_mesh->faces_[iface][nthvert][0]];
}

Vec3f* Model::readWriteVertices()
{
	detachMesh();
	if (m_mesh->verts_.empty())
		return 0;
	return &m_mesh->verts_[0];
}

Vec3f* Model::readWriteNormals()
{
	detachMesh();
	if (m_mesh->norms_.empty())
		return 0;
	return &m_mesh->norms_[0];
}

void Model::load_texture(std::string filename, const char *suffix, TGAImage &img)
{
	std::string texfile(filename);
	size_t dot = texfile.find_last_of('.');
	if (dot != std::string::npos)
	{
		texfile = texfile.substr(0, dot) + std::string(suffix);
		std::cerr << "texture file " << texfile << " loading " << (img.read_tga_file(texfile.c_str()) ? "ok" : "failed") << std::endl;
		img.flip_vertically();
	}
}

TGAColor Model::diffuse(Vec2f uvf)
{
	if (m_diffuse && m_diffuse->img_.get_width() && m_diffuse->img_.get_height())
	{
		TGAImage& diffusemap_ = m_diffuse->img_;
		double val;
		//		bool repeat = true;
		//		if (repeat)
		{
			uvf[0] = std::modf(uvf[0], &val);
			if (uvf[0] < 0)
			{
				uvf[0] = uvf[0] + 1;
			}
			uvf[1] = std::modf(uvf[1], &val);
			if (uvf[1] < 0)
			{
				uvf[1] = uvf[1] + 1;
			}
		}
        	Vec2i uv(uvf[0] * diffusemap_.get_width(), uvf[1] * diffusemap_.get_height());
		return diffusemap_.get(uv[0], uv[1]);
	}
	return TGAColor(255, 255, 255, 255);
}


Vec3f Model::normal(Vec2f uvf)
{
	Vec2i uv(uvf[0] * normalmap_.get_width(), uvf[1] * normalmap_.get_height());
	TGAColor c = normalmap_.get(uv[0], uv[1]);
	Vec3f res;
	for (int i = 0; i < 3; i++)
		res[2 - i] = (float)c[i] / 255.f * 2.f - 1.f;
	return res;
}

Vec2f Model::uv(int iface, int nthvert)
{
	return m_mesh->uv_[m_mesh->faces_[iface][nthvert][1]];
}

float Model::specular(Vec2f uvf)
{
	if (specularmap_.get_width() && specularmap_.get_height())
	{
		Vec2i uv(uvf[0] * specularmap_.get_width(), uvf[1] * specularmap_.get_height());
		return specularmap_.get(uv[0], uv[1])[0] / 1.f;
	}
	return 2.0;
}

Vec3f Model::normal(int iface, int nthvert)
{
	int idx = m_mesh->faces_[iface][nthvert][2];
	return m_mesh->norms_[idx].normalize();
}
}
