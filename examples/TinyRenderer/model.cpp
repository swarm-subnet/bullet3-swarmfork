
#include "model.h"
#include <string.h>  // memcpy
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include "Bullet3Common/b3Logging.h"

namespace TinyRender
{
Model::Model(const char *filename) : verts_(), faces_(), norms_(), uv_(), diffusemap_(), normalmap_(), specularmap_(), mips_(), mipsBuilt_(false)
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
			verts_.push_back(v);
		}
		else if (!line.compare(0, 3, "vn "))
		{
			iss >> trash >> trash;
			Vec3f n;
			for (int i = 0; i < 3; i++) iss >> n[i];
			norms_.push_back(n);
		}
		else if (!line.compare(0, 3, "vt "))
		{
			iss >> trash >> trash;
			Vec2f uv;
			for (int i = 0; i < 2; i++) iss >> uv[i];
			uv_.push_back(uv);
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
			faces_.push_back(f);
		}
	}
	std::cerr << "# v# " << verts_.size() << " f# " << faces_.size() << " vt# " << uv_.size() << " vn# " << norms_.size() << std::endl;
	load_texture(filename, "_diffuse.tga", diffusemap_);
	load_texture(filename, "_nm_tangent.tga", normalmap_);
	load_texture(filename, "_spec.tga", specularmap_);
}

Model::Model() : verts_(), faces_(), norms_(), uv_(), diffusemap_(), normalmap_(), specularmap_(), mips_(), mipsBuilt_(false)
{
}

void Model::setDiffuseTextureFromData(unsigned char *textureImage, int textureWidth, int textureHeight)
{
	clearMips();
	{
		B3_PROFILE("new TGAImage");
		diffusemap_ = TGAImage(textureWidth, textureHeight, TGAImage::RGB);
	}
	TGAColor color;
	color.bgra[3] = 255;

	color.bytespp = 3;
	{
		B3_PROFILE("copy texels");
		memcpy(diffusemap_.buffer(), textureImage, textureHeight * textureWidth * 3);
	}
	{
		B3_PROFILE("flip_vertically");
		diffusemap_.flip_vertically();
	}
}

void Model::loadDiffuseTexture(const char *relativeFileName)
{
	clearMips();
	diffusemap_.read_tga_file(relativeFileName);
}

void Model::reserveMemory(int numVertices, int numIndices)
{
	verts_.reserve(numVertices);
	norms_.reserve(numVertices);
	uv_.reserve(numVertices);
	faces_.reserve(numIndices);
}

void Model::addVertex(float x, float y, float z, float normalX, float normalY, float normalZ, float u, float v)
{
	verts_.push_back(Vec3f(x, y, z));
	norms_.push_back(Vec3f(normalX, normalY, normalZ));
	uv_.push_back(Vec2f(u, v));
}
void Model::addTriangle(int vertexposIndex0, int normalIndex0, int uvIndex0,
						int vertexposIndex1, int normalIndex1, int uvIndex1,
						int vertexposIndex2, int normalIndex2, int uvIndex2)
{
	std::vector<Vec3i> f;
	f.push_back(Vec3i(vertexposIndex0, normalIndex0, uvIndex0));
	f.push_back(Vec3i(vertexposIndex1, normalIndex1, uvIndex1));
	f.push_back(Vec3i(vertexposIndex2, normalIndex2, uvIndex2));
	faces_.push_back(f);
}

Model::~Model()
{
	clearMips();
}

int Model::nverts()
{
	return (int)verts_.size();
}

int Model::nfaces()
{
	return (int)faces_.size();
}

std::vector<int> Model::face(int idx)
{
	std::vector<int> face;
        face.reserve((int)faces_[idx].size());
        for (int i = 0; i < (int)faces_[idx].size(); i++)
          face.push_back(faces_[idx][i][0]);
        return face;
}


Vec3f Model::vert(int i)
{
	return verts_[i];
}

Vec3f Model::vert(int iface, int nthvert)
{
	return verts_[faces_[iface][nthvert][0]];
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
	if (diffusemap_.get_width() && diffusemap_.get_height())
	{
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

// TGAImage's copy constructor does not copy pixels, so the levels live on the heap.
void Model::clearMips()
{
	for (int i = 0; i < (int)mips_.size(); i++)
		delete mips_[i];
	mips_.clear();
	mipsBuilt_ = false;
}

// Each level halves the one before with an integer 2x2 box average, down to 1x1.
// Odd sizes drop their last row or column, as a GPU would.
void Model::buildMips()
{
	clearMips();
	mipsBuilt_ = true;
	for (;;)
	{
		TGAImage &src = mips_.empty() ? diffusemap_ : *mips_[mips_.size() - 1];
		const int sw = src.get_width(), sh = src.get_height(), bpp = src.get_bytespp();
		if (sw <= 1 && sh <= 1)
			return;
		const int dw = sw > 1 ? sw >> 1 : 1;
		const int dh = sh > 1 ? sh >> 1 : 1;
		TGAImage *dst = new TGAImage(dw, dh, bpp);
		const unsigned char *s = src.buffer();
		unsigned char *d = dst->buffer();
		for (int y = 0; y < dh; y++)
		{
			const int y0 = 2 * y;
			const int y1 = (y0 + 1 < sh) ? y0 + 1 : y0;
			for (int x = 0; x < dw; x++)
			{
				const int x0 = 2 * x;
				const int x1 = (x0 + 1 < sw) ? x0 + 1 : x0;
				const unsigned char *p00 = s + (x0 + y0 * sw) * bpp;
				const unsigned char *p10 = s + (x1 + y0 * sw) * bpp;
				const unsigned char *p01 = s + (x0 + y1 * sw) * bpp;
				const unsigned char *p11 = s + (x1 + y1 * sw) * bpp;
				unsigned char *o = d + (x + y * dw) * bpp;
				for (int c = 0; c < bpp; c++)
					o[c] = (unsigned char)((p00[c] + p10[c] + p01[c] + p11[c] + 2) >> 2);
			}
		}
		mips_.push_back(dst);
	}
}

// Four-texel blend inside one level with 8-bit fixed-point weights, repeat wrap.
// u and v are already in [0, 1).
static TGAColor sampleBilinear(TGAImage &img, float u, float v)
{
	const int w = img.get_width(), h = img.get_height(), bpp = img.get_bytespp();
	const float x = u * w - 0.5f;
	const float y = v * h - 0.5f;
	const float fx = std::floor(x);
	const float fy = std::floor(y);
	const int wx = (int)((x - fx) * 256.f);
	const int wy = (int)((y - fy) * 256.f);
	int x0 = ((int)fx % w + w) % w;
	int y0 = ((int)fy % h + h) % h;
	const int x1 = (x0 + 1 == w) ? 0 : x0 + 1;
	const int y1 = (y0 + 1 == h) ? 0 : y0 + 1;
	const unsigned char *s = img.buffer();
	const unsigned char *p00 = s + (x0 + y0 * w) * bpp;
	const unsigned char *p10 = s + (x1 + y0 * w) * bpp;
	const unsigned char *p01 = s + (x0 + y1 * w) * bpp;
	const unsigned char *p11 = s + (x1 + y1 * w) * bpp;
	const int w00 = (256 - wx) * (256 - wy), w10 = wx * (256 - wy);
	const int w01 = (256 - wx) * wy, w11 = wx * wy;
	TGAColor c;
	c.bytespp = (unsigned char)bpp;
	for (int i = 0; i < bpp; i++)
		c.bgra[i] = (unsigned char)((p00[i] * w00 + p10[i] * w10 + p01[i] * w01 + p11[i] * w11 + 32768) >> 16);
	return c;
}

// Trilinear sample: the level comes from how many level-0 texels one pixel
// spans, using the float's own exponent and mantissa for log2 so nothing
// depends on the maths library. duvdx and duvdy are the uv steps to the pixel
// to the right and the pixel below.
TGAColor Model::diffuseFiltered(Vec2f uvf, Vec2f duvdx, Vec2f duvdy)
{
	const int w = diffusemap_.get_width(), h = diffusemap_.get_height();
	if (!w || !h)
		return TGAColor(255, 255, 255, 255);
	if (!mipsBuilt_)
		buildMips();

	double val;
	uvf[0] = std::modf(uvf[0], &val);
	if (uvf[0] < 0)
		uvf[0] = uvf[0] + 1;
	uvf[1] = std::modf(uvf[1], &val);
	if (uvf[1] < 0)
		uvf[1] = uvf[1] + 1;

	const float sx = duvdx[0] * w, tx = duvdx[1] * h;
	const float sy = duvdy[0] * w, ty = duvdy[1] * h;
	const float rx2 = sx * sx + tx * tx;
	const float ry2 = sy * sy + ty * ty;
	const float rho2 = rx2 > ry2 ? rx2 : ry2;

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

	const int last = (int)mips_.size();
	int level = (int)lambda;
	float frac = lambda - (float)level;
	if (level >= last)
	{
		level = last;
		frac = 0.f;
	}
	TGAImage &imgA = level == 0 ? diffusemap_ : *mips_[level - 1];
	TGAColor a = sampleBilinear(imgA, uvf[0], uvf[1]);
	const int wl = (int)(frac * 256.f);
	if (wl == 0)
		return a;
	TGAColor b = sampleBilinear(*mips_[level], uvf[0], uvf[1]);
	for (int i = 0; i < (int)a.bytespp; i++)
		a.bgra[i] = (unsigned char)((a.bgra[i] * (256 - wl) + b.bgra[i] * wl + 128) >> 8);
	return a;
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
	return uv_[faces_[iface][nthvert][1]];
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
	int idx = faces_[iface][nthvert][2];
	return norms_[idx].normalize();
}
}
