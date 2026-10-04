
#include "model.h"
#include <stdlib.h>  // getenv
#include <string.h>  // memcpy
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include "Bullet3Common/b3Logging.h"
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace TinyRender
{
struct SharedMesh
{
	std::vector<Vec3f> verts_;
	// Three corners per face, one after the other; this Vec3i means vertex/uv/normal.
	std::vector<Vec3i> faces_;
	std::vector<Vec3f> norms_;
	std::vector<Vec2f> uv_;
	unsigned long long hash_;
	int refs_;
	bool registered_;
	bool hasAABB_;
	// -1 until asked: whether every corner's uv index is its vertex index.
	int uvByVertex_;
	Vec3f aabbMin_;
	Vec3f aabbMax_;

	SharedMesh() : hash_(0), refs_(1), registered_(false), hasAABB_(false), uvByVertex_(-1) {}
};

struct SharedTexture
{
	TGAImage img_;
	const unsigned char* source_;  // texels the image was built from; a lookup key, never dereferenced blindly
	// The file the texels came from, empty for a texture handed over as raw bytes; the dedup key when set.
	std::string name_;
	int refs_;
	bool registered_;
	// Halved copies of img_ down to 1x1, built on the first filtered sample and shared like the image.
	// TGAImage's copy constructor does not copy pixels, so the levels live on the heap.
	std::vector<TGAImage*> mips_;
	bool mipsBuilt_;
	// One byte per texel in img_'s row order, empty for an opaque texture.
	std::vector<unsigned char> alpha_;
	// The alpha plane halved like mips_, with each level's width and height, built in the same pass.
	std::vector<std::vector<unsigned char> > alphaMips_;
	std::vector<int> alphaMipW_;
	std::vector<int> alphaMipH_;

	SharedTexture() : source_(0), refs_(1), registered_(false), mipsBuilt_(false) {}
	~SharedTexture()
	{
		for (size_t i = 0; i < mips_.size(); i++)
			delete mips_[i];
	}
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

// The texture registered under this file name, or 0.
static SharedTexture* findTextureByName(const char* name)
{
	if (!name || !name[0])
		return 0;
	for (size_t i = 0; i < gSharedTextures.size(); i++)
	{
		if (gSharedTextures[i]->name_ == name)
			return gSharedTextures[i];
	}
	return 0;
}

bool retainSharedTexture(const char* textureName)
{
	SharedTexture* tex = findTextureByName(textureName);
	if (!tex)
		return false;
	tex->refs_++;
	return true;
}

void releaseSharedTexture(const char* textureName)
{
	releaseTexture(findTextureByName(textureName));
}

// Bitwise comparison of the stored arrays against the raw input, so a hash match alone never selects a block.
static bool meshMatchesArrays(const SharedMesh& mesh, const float* vertices, int numVertices, const int* indices, int numIndices)
{
	if ((int)mesh.verts_.size() != numVertices || (int)mesh.faces_.size() != numIndices)
		return false;
	for (int i = 0; i < numVertices; i++)
	{
		const float* v = vertices + i * 9;
		if (memcmp(&mesh.verts_[i][0], v, 3 * sizeof(float)) != 0 ||
			memcmp(&mesh.norms_[i][0], v + 4, 3 * sizeof(float)) != 0 ||
			memcmp(&mesh.uv_[i][0], v + 7, 2 * sizeof(float)) != 0)
			return false;
	}
	for (int i = 0; i < numIndices; i++)
	{
		if (mesh.faces_[i][0] != indices[i])
			return false;
	}
	return true;
}

Model::Model(const char *filename) : m_mesh(new SharedMesh), m_diffuse(0), normalmap_(), specularmap_(), m_specularColor(0.f, 0.f, 0.f)
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
			// Only the first three corners are kept: every reader of a face has always drawn a triangle.
			Vec3i corners[3];
			int count = 0;
			Vec3i tmp;
			iss >> trash;
			while (iss >> tmp[0] >> trash >> tmp[1] >> trash >> tmp[2])
			{
				for (int i = 0; i < 3; i++) tmp[i]--;  // in wavefront obj all indices start at 1, not zero
				if (count < 3) corners[count] = tmp;
				count++;
			}
			if (count >= 3)
			{
				for (int i = 0; i < 3; i++) m_mesh->faces_.push_back(corners[i]);
			}
		}
	}
	std::cerr << "# v# " << m_mesh->verts_.size() << " f# " << m_mesh->faces_.size() / 3 << " vt# " << m_mesh->uv_.size() << " vn# " << m_mesh->norms_.size() << std::endl;
	m_diffuse = new SharedTexture;
	load_texture(filename, "_diffuse.tga", m_diffuse->img_);
	load_texture(filename, "_nm_tangent.tga", normalmap_);
	load_texture(filename, "_spec.tga", specularmap_);
}

Model::Model() : m_mesh(new SharedMesh), m_diffuse(0), normalmap_(), specularmap_(), m_specularColor(0.f, 0.f, 0.f)
{
}

bool Model::shareDiffuseTextureByName(const char *textureName)
{
	SharedTexture* tex = findTextureByName(textureName);
	if (!tex)
		return false;
	releaseTexture(m_diffuse);
	tex->refs_++;
	m_diffuse = tex;
	return true;
}

void Model::setDiffuseTextureFromData(unsigned char *textureImage, int textureWidth, int textureHeight, const unsigned char *textureAlpha, const char *textureName)
{
	// A file name is the whole identity of a texture, so the texels never have to be compared.
	if (sharingEnabled() && textureName && textureName[0] && shareDiffuseTextureByName(textureName))
		return;
	releaseTexture(m_diffuse);
	m_diffuse = 0;
	if (!textureImage)
		return;

	const int rowBytes = textureWidth * 3;
	if (sharingEnabled() && !(textureName && textureName[0]))
	{
		// Same source texels and size is the candidate; the row compare below makes it certain.
		for (size_t i = 0; i < gSharedTextures.size(); i++)
		{
			SharedTexture* tex = gSharedTextures[i];
			if (tex->source_ != textureImage || tex->img_.get_width() != textureWidth || tex->img_.get_height() != textureHeight ||
				tex->alpha_.empty() != (textureAlpha == 0))
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
	if (textureName)
		m_diffuse->name_ = textureName;
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
	if (textureAlpha)
	{
		// Stored flipped like the image, so one (x, y) addresses the same texel in both.
		m_diffuse->alpha_.resize((size_t)textureWidth * textureHeight);
		for (int y = 0; y < textureHeight; y++)
			memcpy(&m_diffuse->alpha_[(size_t)y * textureWidth], textureAlpha + (size_t)(textureHeight - 1 - y) * textureWidth, (size_t)textureWidth);
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
	m_mesh->uvByVertex_ = -1;
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
	m_mesh->uvByVertex_ = -1;
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
	m_mesh->faces_.push_back(Vec3i(vertexposIndex0, normalIndex0, uvIndex0));
	m_mesh->faces_.push_back(Vec3i(vertexposIndex1, normalIndex1, uvIndex1));
	m_mesh->faces_.push_back(Vec3i(vertexposIndex2, normalIndex2, uvIndex2));
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

void Model::shareFrom(const Model& other)
{
	if (&other == this)
		return;
	other.m_mesh->refs_++;
	releaseMesh(m_mesh);
	m_mesh = other.m_mesh;
	if (other.m_diffuse)
		other.m_diffuse->refs_++;
	releaseTexture(m_diffuse);
	m_diffuse = other.m_diffuse;
	m_colorRGBA = other.m_colorRGBA;
	m_specularColor = other.m_specularColor;
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
	return (int)m_mesh->faces_.size() / 3;
}

bool Model::uvIndexedByVertex() const
{
	if (m_mesh->uvByVertex_ < 0)
	{
		int same = 1;
		for (size_t i = 0; same && i < m_mesh->faces_.size(); i++)
			same = m_mesh->faces_[i][1] == m_mesh->faces_[i][0];
		m_mesh->uvByVertex_ = same;
	}
	return m_mesh->uvByVertex_ != 0;
}

const float* Model::uvArray() const
{
	return m_mesh->uv_.empty() ? 0 : &m_mesh->uv_[0][0];
}

unsigned long long Model::meshHash() const
{
	return m_mesh->hash_;
}

std::vector<int> Model::face(int idx)
{
	std::vector<int> face;
	face.reserve(3);
	for (int i = 0; i < 3; i++)
		face.push_back(m_mesh->faces_[(size_t)idx * 3 + i][0]);
	return face;
}

void Model::faceVertices(int idx, int out[3]) const
{
	for (int i = 0; i < 3; i++)
		out[i] = m_mesh->faces_[(size_t)idx * 3 + i][0];
}

Vec3f Model::vert(int i)
{
	return m_mesh->verts_[i];
}

Vec3f Model::vert(int iface, int nthvert)
{
	return m_mesh->verts_[m_mesh->faces_[(size_t)iface * 3 + nthvert][0]];
}

Vec3f* Model::readWriteVertices()
{
	detachMesh();
	if (m_mesh->verts_.empty())
		return 0;
	return &m_mesh->verts_[0];
}

void Model::recomputeNormals()
{
	detachMesh();
	if (m_mesh->norms_.empty())
		return;
	for (size_t i = 0; i < m_mesh->norms_.size(); i++)
		m_mesh->norms_[i] = Vec3f(0.f, 0.f, 0.f);
	const int numVerts = (int)m_mesh->verts_.size();
	const int numNorms = (int)m_mesh->norms_.size();
	for (size_t f = 0; f + 2 < m_mesh->faces_.size(); f += 3)
	{
		const Vec3i* face = &m_mesh->faces_[f];
		if (face[0][0] < 0 || face[0][0] >= numVerts || face[1][0] < 0 || face[1][0] >= numVerts || face[2][0] < 0 || face[2][0] >= numVerts)
			continue;
		// The cross product carries twice the triangle's area, so a big face pulls the corner normal harder.
		const Vec3f weighted = cross(m_mesh->verts_[face[1][0]] - m_mesh->verts_[face[0][0]], m_mesh->verts_[face[2][0]] - m_mesh->verts_[face[0][0]]);
		for (int k = 0; k < 3; k++)
		{
			const int ni = face[k][2];
			if (ni >= 0 && ni < numNorms)
				m_mesh->norms_[ni] = m_mesh->norms_[ni] + weighted;
		}
	}
	for (size_t i = 0; i < m_mesh->norms_.size(); i++)
	{
		const float length = m_mesh->norms_[i].norm();
		if (length > 0.f)
			m_mesh->norms_[i] = m_mesh->norms_[i] * (1.f / length);
	}
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

// Wraps a texture coordinate into [0, 1).
static float wrapUnit(float value)
{
	// A float's fraction is exact in float: modf's value without the double round trip, and none for an infinity.
	float f = value - std::trunc(value);
	if (f != f)
		f = value != value ? value : 0.f;
	return f < 0.f ? f + 1.f : f;
}

TGAColor Model::diffuse(Vec2f uvf)
{
	if (m_diffuse && m_diffuse->img_.get_width() && m_diffuse->img_.get_height())
	{
		TGAImage& diffusemap_ = m_diffuse->img_;
		uvf[0] = wrapUnit(uvf[0]);
		uvf[1] = wrapUnit(uvf[1]);
        	Vec2i uv(uvf[0] * diffusemap_.get_width(), uvf[1] * diffusemap_.get_height());
		return diffusemap_.get(uv[0], uv[1]);
	}
	return TGAColor(255, 255, 255, 255);
}

bool Model::hasAlpha() const
{
	return m_diffuse && !m_diffuse->alpha_.empty();
}

// The same wrap and nearest-texel pick as diffuse(), read from the alpha plane.
unsigned char Model::alpha(Vec2f uvf) const
{
	if (!hasAlpha())
		return 255;
	const int w = m_diffuse->img_.get_width(), h = m_diffuse->img_.get_height();
	uvf[0] = wrapUnit(uvf[0]);
	uvf[1] = wrapUnit(uvf[1]);
	int x = (int)(uvf[0] * w), y = (int)(uvf[1] * h);
	x = x < 0 ? 0 : (x >= w ? w - 1 : x);
	y = y < 0 ? 0 : (y >= h ? h - 1 : y);
	return m_diffuse->alpha_[(size_t)y * w + x];
}

// Each level halves the one before with an integer 2x2 box average, down to 1x1; the alpha plane is halved alongside.
// Odd sizes drop their last row or column, as a GPU would.
static void buildMips(SharedTexture& tex)
{
	tex.mipsBuilt_ = true;
	if (!tex.alpha_.empty())
	{
		int sw = tex.img_.get_width(), sh = tex.img_.get_height();
		while (sw > 1 || sh > 1)
		{
			const int dw = sw > 1 ? sw >> 1 : 1;
			const int dh = sh > 1 ? sh >> 1 : 1;
			const std::vector<unsigned char>& src = tex.alphaMips_.empty() ? tex.alpha_ : tex.alphaMips_.back();
			std::vector<unsigned char> dst((size_t)dw * dh);
			for (int y = 0; y < dh; y++)
			{
				const int y0 = 2 * y;
				const int y1 = (y0 + 1 < sh) ? y0 + 1 : y0;
				for (int x = 0; x < dw; x++)
				{
					const int x0 = 2 * x;
					const int x1 = (x0 + 1 < sw) ? x0 + 1 : x0;
					dst[(size_t)y * dw + x] = (unsigned char)((src[(size_t)y0 * sw + x0] + src[(size_t)y0 * sw + x1] +
															   src[(size_t)y1 * sw + x0] + src[(size_t)y1 * sw + x1] + 2) >> 2);
				}
			}
			tex.alphaMips_.push_back(dst);
			tex.alphaMipW_.push_back(dw);
			tex.alphaMipH_.push_back(dh);
			sw = dw;
			sh = dh;
		}
	}
	for (;;)
	{
		TGAImage& src = tex.mips_.empty() ? tex.img_ : *tex.mips_[tex.mips_.size() - 1];
		const int sw = src.get_width(), sh = src.get_height(), bpp = src.get_bytespp();
		if (sw <= 1 && sh <= 1)
			return;
		const int dw = sw > 1 ? sw >> 1 : 1;
		const int dh = sh > 1 ? sh >> 1 : 1;
		TGAImage* dst = new TGAImage(dw, dh, bpp);
		const unsigned char* s = src.buffer();
		unsigned char* d = dst->buffer();
		for (int y = 0; y < dh; y++)
		{
			const int y0 = 2 * y;
			const int y1 = (y0 + 1 < sh) ? y0 + 1 : y0;
			// A row wider than one always has both source columns: the same sums, with the texel width fixed and no edge test.
			if (sw > 1 && (bpp == 3 || bpp == 4))
			{
				const unsigned char* top = s + (size_t)y0 * sw * bpp;
				const unsigned char* bottom = s + (size_t)y1 * sw * bpp;
				unsigned char* o = d + (size_t)y * dw * bpp;
				if (bpp == 3)
					for (int x = 0; x < dw; x++)
					{
						const unsigned char* a = top + 6 * x;
						const unsigned char* b = bottom + 6 * x;
						o[3 * x] = (unsigned char)((a[0] + a[3] + b[0] + b[3] + 2) >> 2);
						o[3 * x + 1] = (unsigned char)((a[1] + a[4] + b[1] + b[4] + 2) >> 2);
						o[3 * x + 2] = (unsigned char)((a[2] + a[5] + b[2] + b[5] + 2) >> 2);
					}
				else
					for (int x = 0; x < dw; x++)
					{
						const unsigned char* a = top + 8 * x;
						const unsigned char* b = bottom + 8 * x;
						o[4 * x] = (unsigned char)((a[0] + a[4] + b[0] + b[4] + 2) >> 2);
						o[4 * x + 1] = (unsigned char)((a[1] + a[5] + b[1] + b[5] + 2) >> 2);
						o[4 * x + 2] = (unsigned char)((a[2] + a[6] + b[2] + b[6] + 2) >> 2);
						o[4 * x + 3] = (unsigned char)((a[3] + a[7] + b[3] + b[7] + 2) >> 2);
					}
				continue;
			}
			for (int x = 0; x < dw; x++)
			{
				const int x0 = 2 * x;
				const int x1 = (x0 + 1 < sw) ? x0 + 1 : x0;
				const unsigned char* p00 = s + (x0 + y0 * sw) * bpp;
				const unsigned char* p10 = s + (x1 + y0 * sw) * bpp;
				const unsigned char* p01 = s + (x0 + y1 * sw) * bpp;
				const unsigned char* p11 = s + (x1 + y1 * sw) * bpp;
				unsigned char* o = d + (x + y * dw) * bpp;
				for (int c = 0; c < bpp; c++)
					o[c] = (unsigned char)((p00[c] + p10[c] + p01[c] + p11[c] + 2) >> 2);
			}
		}
		tex.mips_.push_back(dst);
	}
}

// (i % n + n) % n; a wrapped coordinate floors to -1 .. n - 1, which needs no division.
static inline int wrapTexel(int i, int n)
{
	if (i >= 0 && i < n)
		return i;
	if (i == -1)
		return n - 1;
	return (i % n + n) % n;
}

// Four-texel blend inside one level with 8-bit fixed-point weights, repeat wrap.
// u and v are already in [0, 1).
static TGAColor sampleBilinear(TGAImage& img, float u, float v)
{
	const int w = img.get_width(), h = img.get_height(), bpp = img.get_bytespp();
	const float x = u * w - 0.5f;
	const float y = v * h - 0.5f;
	const float fx = std::floor(x);
	const float fy = std::floor(y);
	const int wx = (int)((x - fx) * 256.f);
	const int wy = (int)((y - fy) * 256.f);
	int x0 = wrapTexel((int)fx, w);
	int y0 = wrapTexel((int)fy, h);
	const int x1 = (x0 + 1 == w) ? 0 : x0 + 1;
	const int y1 = (y0 + 1 == h) ? 0 : y0 + 1;
	const unsigned char* s = img.buffer();
	const unsigned char* p00 = s + (x0 + y0 * w) * bpp;
	const unsigned char* p10 = s + (x1 + y0 * w) * bpp;
	const unsigned char* p01 = s + (x0 + y1 * w) * bpp;
	const unsigned char* p11 = s + (x1 + y1 * w) * bpp;
	const int w00 = (256 - wx) * (256 - wy), w10 = wx * (256 - wy);
	const int w01 = (256 - wx) * wy, w11 = wx * wy;
	TGAColor c;
	c.bytespp = (unsigned char)bpp;
	for (int i = 0; i < bpp; i++)
		c.bgra[i] = (unsigned char)((p00[i] * w00 + p10[i] * w10 + p01[i] * w01 + p11[i] * w11 + 32768) >> 16);
	return c;
}

// Two mip levels and the 8-bit weight of the coarser one, which is read only when that weight is not zero.
struct MipPick
{
	TGAImage* m_a;
	TGAImage* m_b;
	int m_weight;
};

// The levels a footprint radius squared of rho2 texels asks for, log2 from the float's own bits.
static MipPick pickLevels(SharedTexture& tex, float rho2)
{
	float lambda = 0.f;
	if (rho2 > 1.f)
	{
		unsigned int bits;
		memcpy(&bits, &rho2, sizeof(bits));
		const int e = (int)((bits >> 23) & 255) - 127;
		bits = (bits & 0x007fffffu) | 0x3f800000u;
		float mant;
		memcpy(&mant, &bits, sizeof(mant));
		lambda = 0.5f * ((float)e + (mant - 1.f));
	}

	std::vector<TGAImage*>& mips = tex.mips_;
	const int last = (int)mips.size();
	int level = (int)lambda;
	float frac = lambda - (float)level;
	if (level >= last)
	{
		level = last;
		frac = 0.f;
	}
	MipPick pick;
	pick.m_a = level == 0 ? &tex.img_ : mips[level - 1];
	pick.m_weight = (int)(frac * 256.f);
	pick.m_b = pick.m_weight == 0 ? 0 : mips[level];
	return pick;
}

// One trilinear read from the levels pickLevels chose.
static TGAColor sampleLevels(const MipPick& pick, float u, float v)
{
	TGAColor a = sampleBilinear(*pick.m_a, u, v);
	const int wl = pick.m_weight;
	if (wl == 0)
		return a;
	TGAColor b = sampleBilinear(*pick.m_b, u, v);
	for (int i = 0; i < (int)a.bytespp; i++)
		a.bgra[i] = (unsigned char)((a.bgra[i] * (256 - wl) + b.bgra[i] * wl + 128) >> 8);
	return a;
}

// Trilinear sample at the pixel footprint; with more taps the footprint is walked along its long side and the reads averaged.
TGAColor Model::diffuseFiltered(Vec2f uvf, Vec2f duvdx, Vec2f duvdy, int maxTaps)
{
	if (!m_diffuse)
		return TGAColor(255, 255, 255, 255);
	const int w = m_diffuse->img_.get_width(), h = m_diffuse->img_.get_height();
	if (!w || !h)
		return TGAColor(255, 255, 255, 255);
	if (!m_diffuse->mipsBuilt_)
		buildMips(*m_diffuse);

	uvf[0] = wrapUnit(uvf[0]);
	uvf[1] = wrapUnit(uvf[1]);

	const float sx = duvdx[0] * w, tx = duvdx[1] * h;
	const float sy = duvdy[0] * w, ty = duvdy[1] * h;
	const float rx2 = sx * sx + tx * tx;
	const float ry2 = sy * sy + ty * ty;
	const float rho2 = rx2 > ry2 ? rx2 : ry2;
	if (maxTaps <= 1)
		return sampleLevels(pickLevels(*m_diffuse, rho2), uvf[0], uvf[1]);

	const bool xMajor = rx2 >= ry2;
	const float major2 = xMajor ? rx2 : ry2, minor2 = xMajor ? ry2 : rx2;
	int taps = 1;
	if (minor2 > 0.f && major2 > minor2)
	{
		// Enough reads that each covers about the shorter side's length along the longer side.
		const float ratio = sqrtf(major2 / minor2);
		taps = (int)ratio;
		if ((float)taps < ratio)
			taps++;
		taps = taps > maxTaps ? maxTaps : (taps < 1 ? 1 : taps);
	}
	if (taps <= 1)
		return sampleLevels(pickLevels(*m_diffuse, rho2), uvf[0], uvf[1]);
	const float perTap2 = major2 / ((float)taps * (float)taps);
	const float tapRho2 = perTap2 > minor2 ? perTap2 : minor2;
	const Vec2f along = xMajor ? duvdx : duvdy;
	const MipPick pick = pickLevels(*m_diffuse, tapRho2);
	int sum[4] = {0, 0, 0, 0};
	unsigned char bytespp = 3;
	for (int k = 0; k < taps; k++)
	{
		const float f = ((float)k + 0.5f) / (float)taps - 0.5f;
		const TGAColor c = sampleLevels(pick, wrapUnit(uvf[0] + along[0] * f), wrapUnit(uvf[1] + along[1] * f));
		bytespp = c.bytespp;
		for (int i = 0; i < (int)c.bytespp; i++)
			sum[i] += c.bgra[i];
	}
	TGAColor out;
	out.bytespp = bytespp;
	for (int i = 0; i < (int)bytespp; i++)
		out.bgra[i] = (unsigned char)((sum[i] + taps / 2) / taps);
	return out;
}

#if defined(__AVX2__)
typedef float FilterLanes __attribute__((vector_size(32)));
typedef int FilterInts __attribute__((vector_size(32)));
typedef long long FilterLongs __attribute__((vector_size(32)));
typedef int FilterInts4 __attribute__((vector_size(16)));
typedef unsigned FilterUints __attribute__((vector_size(32)));

// Per lane: a where the mask is set, b elsewhere.
static inline FilterLanes filterSelect(FilterInts mask, FilterLanes a, FilterLanes b)
{
	return (FilterLanes)((mask & (FilterInts)a) | (~mask & (FilterInts)b));
}

static inline FilterInts filterSelect(FilterInts mask, FilterInts a, FilterInts b)
{
	return (mask & a) | (~mask & b);
}

// wrapUnit on each lane.
static inline FilterLanes wrapUnit8(FilterLanes value)
{
	const FilterLanes zero = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
	FilterLanes f = value - (FilterLanes)_mm256_round_ps((__m256)value, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
	f = filterSelect(f != f, filterSelect(value != value, value, zero), f);
	return filterSelect(f < zero, f + 1.f, f);
}

// The image each lane reads at its level: where its bytes start and its size. A one-texel three-byte image is read from
// a copy with a spare byte in front, which gives the same bytes; the copy lives in `lone`, which must outlive the reads.
struct FilterLevel8
{
	FilterLongs m_baseLo, m_baseHi;
	FilterInts m_w, m_h;
};

static inline void filterLevel8(SharedTexture* const tex[8], FilterInts level, FilterInts bpp, unsigned char lone[8][4], FilterLevel8& out)
{
	for (int l = 0; l < 8; l++)
	{
		TGAImage& img = level[l] == 0 ? tex[l]->img_ : *tex[l]->mips_[level[l] - 1];
		long long base = (long long)(size_t)img.buffer();
		if (bpp[l] == 3 && img.get_width() * img.get_height() == 1)
		{
			lone[l][0] = 0;
			memcpy(&lone[l][1], img.buffer(), 3);
			base = (long long)(size_t)&lone[l][1];
		}
		if (l < 4)
			out.m_baseLo[l] = base;
		else
			out.m_baseHi[l - 4] = base;
		out.m_w[l] = img.get_width();
		out.m_h[l] = img.get_height();
	}
}

// sampleBilinear on each lane, from the level of its own texture `images` names. A texel's bytes come from one
// four-byte read at the texel, or one byte earlier for the last texel of a three-byte image, so no read leaves it.
static inline void bilinear8(const FilterLevel8& images, FilterLanes u, FilterLanes v, FilterInts bpp, FilterInts out[4])
{
	const FilterLongs baseLo = images.m_baseLo, baseHi = images.m_baseHi;
	const FilterInts w = images.m_w, h = images.m_h;
	const FilterLanes x = u * __builtin_convertvector(w, FilterLanes) - 0.5f;
	const FilterLanes y = v * __builtin_convertvector(h, FilterLanes) - 0.5f;
	const FilterLanes fx = (FilterLanes)_mm256_round_ps((__m256)x, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
	const FilterLanes fy = (FilterLanes)_mm256_round_ps((__m256)y, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
	const FilterInts wx = __builtin_convertvector((x - fx) * 256.f, FilterInts);
	const FilterInts wy = __builtin_convertvector((y - fy) * 256.f, FilterInts);
	const FilterInts x0r = __builtin_convertvector(fx, FilterInts), y0r = __builtin_convertvector(fy, FilterInts);
	const FilterInts zero = {0, 0, 0, 0, 0, 0, 0, 0};
	const FilterInts xin = (x0r >= 0) & (x0r < w), yin = (y0r >= 0) & (y0r < h);
	FilterInts x0 = filterSelect(xin, x0r, filterSelect(x0r == -1, w - 1, x0r));
	FilterInts y0 = filterSelect(yin, y0r, filterSelect(y0r == -1, h - 1, y0r));
	if (_mm256_movemask_ps((__m256)((~xin & (x0r != -1)) | (~yin & (y0r != -1)))))
		for (int l = 0; l < 8; l++)
		{
			x0[l] = wrapTexel(x0r[l], w[l]);
			y0[l] = wrapTexel(y0r[l], h[l]);
		}
	const FilterInts x1 = filterSelect(x0 + 1 == w, zero, x0 + 1), y1 = filterSelect(y0 + 1 == h, zero, y0 + 1);
	const FilterInts lastTexel = (w * h - 1) * bpp;
	const FilterInts at[4] = {(x0 + y0 * w) * bpp, (x1 + y0 * w) * bpp, (x0 + y1 * w) * bpp, (x1 + y1 * w) * bpp};
	FilterInts texel[4];
	for (int q = 0; q < 4; q++)
	{
		const FilterInts early = (at[q] == lastTexel) & (bpp == 3);
		const FilterInts from = at[q] - (early & 1);
		const FilterInts4 lo = {from[0], from[1], from[2], from[3]}, hi = {from[4], from[5], from[6], from[7]};
		const __m128i a = _mm256_i64gather_epi32((const int*)0, (__m256i)(baseLo + __builtin_convertvector(lo, FilterLongs)), 1);
		const __m128i b = _mm256_i64gather_epi32((const int*)0, (__m256i)(baseHi + __builtin_convertvector(hi, FilterLongs)), 1);
		const FilterInts bytes = (FilterInts)_mm256_set_m128i(b, a);
		texel[q] = filterSelect(early, (FilterInts)((FilterUints)bytes >> 8), bytes);
	}
	// The four-weight sum, as two blends along x and one along y: t00 (256 - wx) + t10 wx is 256 t00 + (t10 - t00) wx in
	// whole numbers, and so for the rows, so every sum is the same integer with fewer products.
	for (int c = 0; c < 4; c++)
	{
		const int s = 8 * c;
		const FilterInts t00 = (texel[0] >> s) & 255, t10 = (texel[1] >> s) & 255, t01 = (texel[2] >> s) & 255, t11 = (texel[3] >> s) & 255;
		const FilterInts top = (t00 << 8) + (t10 - t00) * wx, bottom = (t01 << 8) + (t11 - t01) * wx;
		out[c] = ((top << 8) + (bottom - top) * wy + 32768) >> 16;
	}
}

// diffuseFiltered on eight reads with maxTaps above one, every lane its read's steps in their order.
static void filtered8(SharedTexture* const tex[8], const Vec2f* uvf, const Vec2f* duvdx, const Vec2f* duvdy, int maxTaps, TGAColor* out)
{
	FilterLanes u, v, dx0, dx1, dy0, dy1, wf, hf;
	FilterInts last, bpp;
	for (int l = 0; l < 8; l++)
	{
		u[l] = uvf[l][0], v[l] = uvf[l][1];
		dx0[l] = duvdx[l][0], dx1[l] = duvdx[l][1], dy0[l] = duvdy[l][0], dy1[l] = duvdy[l][1];
		wf[l] = (float)tex[l]->img_.get_width(), hf[l] = (float)tex[l]->img_.get_height();
		last[l] = (int)tex[l]->mips_.size();
		bpp[l] = tex[l]->img_.get_bytespp();
	}
	const FilterLanes zf = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
	const FilterInts one = {1, 1, 1, 1, 1, 1, 1, 1}, zero = one - 1;
	u = wrapUnit8(u);
	v = wrapUnit8(v);
	const FilterLanes sx = dx0 * wf, tx = dx1 * hf, sy = dy0 * wf, ty = dy1 * hf;
	const FilterLanes rx2 = sx * sx + tx * tx, ry2 = sy * sy + ty * ty;
	const FilterLanes rho2 = filterSelect(rx2 > ry2, rx2, ry2);
	const FilterInts xMajor = rx2 >= ry2;
	const FilterLanes major2 = filterSelect(xMajor, rx2, ry2), minor2 = filterSelect(xMajor, ry2, rx2);
	const FilterLanes ratio = (FilterLanes)_mm256_sqrt_ps((__m256)(major2 / minor2));
	FilterInts taps = __builtin_convertvector(ratio, FilterInts);
	taps = filterSelect(__builtin_convertvector(taps, FilterLanes) < ratio, taps + 1, taps);
	const FilterInts tapLimit = one * maxTaps;
	taps = filterSelect(taps > tapLimit, tapLimit, filterSelect(taps < one, one, taps));
	taps = filterSelect((minor2 > zf) & (major2 > minor2), taps, one);
	const FilterLanes tapsf = __builtin_convertvector(taps, FilterLanes);
	const FilterLanes perTap2 = major2 / (tapsf * tapsf);
	const FilterLanes rho = filterSelect(taps > one, filterSelect(perTap2 > minor2, perTap2, minor2), rho2);
	const FilterLanes along0 = filterSelect(xMajor, dx0, dy0), along1 = filterSelect(xMajor, dx1, dy1);
	// pickLevels on each lane.
	const FilterInts bits = (FilterInts)rho;
	const FilterLanes mant = (FilterLanes)((bits & 0x007fffff) | 0x3f800000);
	const FilterLanes lambda = filterSelect(rho > 1.f, 0.5f * (__builtin_convertvector(((bits >> 23) & 255) - 127, FilterLanes) + (mant - 1.f)), zf);
	FilterInts level = __builtin_convertvector(lambda, FilterInts);
	FilterLanes frac = lambda - __builtin_convertvector(level, FilterLanes);
	const FilterInts over = level >= last;
	level = filterSelect(over, last, level);
	frac = filterSelect(over, zf, frac);
	const FilterInts weight = __builtin_convertvector(frac * 256.f, FilterInts);
	const FilterInts coarser = filterSelect(level + 1 > last, last, level + 1);
	const bool blend = _mm256_movemask_ps((__m256)(weight != 0)) != 0;
	int most = 1;
	for (int l = 0; l < 8; l++)
		most = taps[l] > most ? taps[l] : most;
	// A lane reads the same two levels at every tap, so their images are looked up once.
	unsigned char lone[2][8][4];
	FilterLevel8 fine, coarse;
	filterLevel8(tex, level, bpp, lone[0], fine);
	if (blend)
		filterLevel8(tex, coarser, bpp, lone[1], coarse);
	FilterInts sum[4] = {zero, zero, zero, zero};
	for (int k = 0; k < most; k++)
	{
		const FilterLanes f = ((float)k + 0.5f) / tapsf - 0.5f;
		const FilterInts many = taps > one;
		const FilterLanes uk = filterSelect(many, wrapUnit8(u + along0 * f), u), vk = filterSelect(many, wrapUnit8(v + along1 * f), v);
		FilterInts a[4], b[4];
		bilinear8(fine, uk, vk, bpp, a);
		if (blend)
		{
			bilinear8(coarse, uk, vk, bpp, b);
			for (int c = 0; c < 4; c++)
				// a (256 - w) + b w as 256 a + (b - a) w, the same integer; at w = 0 it is a itself.
				a[c] = ((a[c] << 8) + (b[c] - a[c]) * weight + 128) >> 8;
		}
		const FilterInts active = (zero + k) < taps;
		for (int c = 0; c < 4; c++)
			sum[c] += filterSelect(active, a[c], zero);
	}
	for (int c = 0; c < 4; c++)
	{
		// Whole numbers under 2^24 divide exactly in float, so the truncated quotient is the integer division's.
		const FilterInts many = __builtin_convertvector(__builtin_convertvector(sum[c] + (taps >> 1), FilterLanes) / tapsf, FilterInts);
		const FilterInts byte = filterSelect(taps > one, many, sum[c]);
		for (int l = 0; l < 8; l++)
			out[l].bgra[c] = c < bpp[l] ? (unsigned char)byte[l] : 0;
	}
	for (int l = 0; l < 8; l++)
		out[l].bytespp = (unsigned char)bpp[l];
}
#endif

void Model::diffuseFilteredMany(Model* const* models, const Vec2f* uv, const Vec2f* duvdx, const Vec2f* duvdy, int count, int maxTaps, TGAColor* out)
{
	int k = 0;
#if defined(__AVX2__)
	for (; k + 8 <= count; k += 8)
	{
		SharedTexture* tex[8];
		bool lanes = true;
		for (int l = 0; l < 8 && lanes; l++)
		{
			tex[l] = models[k + l]->m_diffuse;
			lanes = tex[l] && tex[l]->mipsBuilt_ && tex[l]->img_.get_width() && tex[l]->img_.get_height() &&
					(tex[l]->img_.get_bytespp() == 3 || tex[l]->img_.get_bytespp() == 4);
		}
		if (lanes)
			filtered8(tex, uv + k, duvdx + k, duvdy + k, maxTaps, out + k);
		else
			for (int l = 0; l < 8; l++)
				out[k + l] = models[k + l]->diffuseFiltered(uv[k + l], duvdx[k + l], duvdy[k + l], maxTaps);
	}
#endif
	for (; k < count; k++)
		out[k] = models[k]->diffuseFiltered(uv[k], duvdx[k], duvdy[k], maxTaps);
}

TGAColor Model::diffuseMean(Vec2f uvf)
{
	if (!m_diffuse || !m_diffuse->img_.get_width() || !m_diffuse->img_.get_height())
		return TGAColor(255, 255, 255, 255);
	if (!m_diffuse->mipsBuilt_)
		buildMips(*m_diffuse);
	return sampleBilinear(m_diffuse->mips_.empty() ? m_diffuse->img_ : *m_diffuse->mips_.back(), wrapUnit(uvf[0]), wrapUnit(uvf[1]));
}

unsigned char Model::alphaFiltered(Vec2f uvf, float footprintUv2, bool* averaged) const
{
	if (averaged)
		*averaged = false;
	if (!hasAlpha() || !m_diffuse->mipsBuilt_ || m_diffuse->alphaMips_.empty())
		return alpha(uvf);
	const float rho2 = footprintUv2 * (float)m_diffuse->img_.get_width() * (float)m_diffuse->img_.get_height();
	if (!(rho2 > 4.0f))
		return alpha(uvf);
	if (averaged)
		*averaged = true;
	int level = (int)(0.5f * log2f(rho2));
	const int last = (int)m_diffuse->alphaMips_.size();
	level = level > last ? last : level;
	const std::vector<unsigned char>& plane = m_diffuse->alphaMips_[(size_t)level - 1];
	const int w = m_diffuse->alphaMipW_[(size_t)level - 1], h = m_diffuse->alphaMipH_[(size_t)level - 1];
	const float x = wrapUnit(uvf[0]) * w - 0.5f, y = wrapUnit(uvf[1]) * h - 0.5f;
	const float fx = std::floor(x), fy = std::floor(y);
	const float ax = x - fx, ay = y - fy;
	const int x0 = wrapTexel((int)fx, w), y0 = wrapTexel((int)fy, h);
	const int x1 = (x0 + 1 == w) ? 0 : x0 + 1, y1 = (y0 + 1 == h) ? 0 : y0 + 1;
	const float top = plane[(size_t)y0 * w + x0] * (1.f - ax) + plane[(size_t)y0 * w + x1] * ax;
	const float bottom = plane[(size_t)y1 * w + x0] * (1.f - ax) + plane[(size_t)y1 * w + x1] * ax;
	return (unsigned char)(top * (1.f - ay) + bottom * ay + 0.5f);
}

void Model::buildMipmaps()
{
	if (m_diffuse && !m_diffuse->mipsBuilt_ && m_diffuse->img_.get_width() && m_diffuse->img_.get_height())
		buildMips(*m_diffuse);
}

Vec3f Model::storedNormal(int iface, int nthvert) const
{
	return m_mesh->norms_[m_mesh->faces_[(size_t)iface * 3 + nthvert][2]];
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
	return m_mesh->uv_[m_mesh->faces_[(size_t)iface * 3 + nthvert][1]];
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
	int idx = m_mesh->faces_[(size_t)iface * 3 + nthvert][2];
	return m_mesh->norms_[idx].normalize();
}
}
