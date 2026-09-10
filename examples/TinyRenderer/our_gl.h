#ifndef __OUR_GL_H__
#define __OUR_GL_H__
#include "tgaimage.h"
#include "geometry.h"

namespace TinyRender
{
Matrix viewport(int x, int y, int w, int h);
Matrix projection(float coeff = 0.f);  // coeff = -1/c
Matrix lookat(Vec3f eye, Vec3f center, Vec3f up);

struct IShader
{
	float m_nearPlane;
	float m_farPlane;
	// When set, the rasteriser also hands the fragment the perspective-correct
	// barycentrics of the pixel to the right and the pixel below, so the shader
	// can measure how much texture one pixel covers.
	bool m_needUvDerivatives;
	Vec3f m_barDx;
	Vec3f m_barDy;
	IShader() : m_needUvDerivatives(false) {}
	virtual ~IShader();
	virtual Vec4f vertex(int iface, int nthvert) = 0;
	virtual bool fragment(Vec3f bar, TGAColor &color) = 0;
};

void triangle(mat<4, 3, float> &pts, IShader &shader, TGAImage &image, float *zbuffer, const Matrix &viewPortMatrix);
void triangle(mat<4, 3, float> &pts, IShader &shader, TGAImage &image, float *zbuffer, int *segmentationMaskBuffer, const Matrix &viewPortMatrix, int objectIndex);
void triangleClipped(mat<4, 3, float> &clippedPts, mat<4, 3, float> &pts, IShader &shader, TGAImage &image, float *zbuffer, const Matrix &viewPortMatrix);
void triangleClipped(mat<4, 3, float> &clippedPts, mat<4, 3, float> &pts, IShader &shader, TGAImage &image, float *zbuffer, int *segmentationMaskBuffer, const Matrix &viewPortMatrix, int objectIndex);

// bandTid/bandCount split the row loop across threads: each row is filled only by its
// owning thread while the incremental edge accumulators still advance on every row,
// so output stays bit-identical to the serial pass for any thread count.
void triangleDepthOnly(mat<4, 3, float> &clipc, float *zbuffer, int *segmentationMaskBuffer, const Matrix &viewPortMatrix, int objectAndLinkIndex, int width, int height, float nearPlane, float farPlane, int bandTid = 0, int bandCount = 1);
void triangleClippedDepthOnly(mat<4, 3, float> &clipc, float *zbuffer, int *segmentationMaskBuffer, const Matrix &viewPortMatrix, int objectAndLinkIndex, int width, int height, float nearPlane, float farPlane, int bandTid = 0, int bandCount = 1);
}

#endif  //__OUR_GL_H__
