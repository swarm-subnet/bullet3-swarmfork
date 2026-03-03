#include <cmath>
#include <limits>
#include <cstdlib>
#include "our_gl.h"
#include "Bullet3Common/b3MinMax.h"

namespace TinyRender
{
IShader::~IShader() {}

Matrix viewport(int x, int y, int w, int h)
{
	Matrix Viewport;
	Viewport = Matrix::identity();
	Viewport[0][3] = x + w / 2.f;
	Viewport[1][3] = y + h / 2.f;
	Viewport[2][3] = .5f;
	Viewport[0][0] = w / 2.f;
	Viewport[1][1] = h / 2.f;
	Viewport[2][2] = .5f;
	return Viewport;
}

Matrix projection(float coeff)
{
	Matrix Projection;
	Projection = Matrix::identity();
	Projection[3][2] = coeff;
	return Projection;
}

Matrix lookat(Vec3f eye, Vec3f center, Vec3f up)
{
	Vec3f f = (center - eye).normalize();
	Vec3f u = up.normalize();
	Vec3f s = cross(f, u).normalize();
	u = cross(s, f);

	Matrix ModelView;
	ModelView[0][0] = s.x;
	ModelView[0][1] = s.y;
	ModelView[0][2] = s.z;

	ModelView[1][0] = u.x;
	ModelView[1][1] = u.y;
	ModelView[1][2] = u.z;

	ModelView[2][0] = -f.x;
	ModelView[2][1] = -f.y;
	ModelView[2][2] = -f.z;

	ModelView[3][0] = 0.f;
	ModelView[3][1] = 0.f;
	ModelView[3][2] = 0.f;

	ModelView[0][3] = -(s[0] * eye[0] + s[1] * eye[1] + s[2] * eye[2]);
	ModelView[1][3] = -(u[0] * eye[0] + u[1] * eye[1] + u[2] * eye[2]);
	ModelView[2][3] = f[0] * eye[0] + f[1] * eye[1] + f[2] * eye[2];
	ModelView[3][3] = 1.f;

	return ModelView;
}

Vec3d barycentric(Vec2f A1, Vec2f B1, Vec2f C1, Vec2f P1)
{

	Vec2d A(A1.x, A1.y);
	Vec2d B(B1.x, B1.y);
	Vec2d C(C1.x, C1.y);
	Vec2d P(P1.x, P1.y);;

	Vec3d s[2];
	for (int i = 2; i--;)
	{
		s[i][0] = C[i] - A[i];
		s[i][1] = B[i] - A[i];
		s[i][2] = A[i] - P[i];
	}
	Vec3d u = cross(s[0], s[1]);
	if (std::abs(u[2]) > 1e-2)  // dont forget that u[2] is integer. If it is zero then triangle ABC is degenerate
		return Vec3d(1. - (u.x + u.y) / u.z, u.y / u.z, u.x / u.z);
	return Vec3d(-1., 1., 1.);  // in this case generate negative coordinates, it will be thrown away by the rasterizator
}

void triangleClipped(mat<4, 3, float> &clipc, mat<4, 3, float> &orgClipc, IShader &shader, TGAImage &image, float *zbuffer, const Matrix &viewPortMatrix)
{
	triangleClipped(clipc, orgClipc, shader, image, zbuffer, 0, viewPortMatrix, 0);
}

void triangleClipped(mat<4, 3, float> &clipc, mat<4, 3, float> &orgClipc, IShader &shader, TGAImage &image, float *zbuffer, int *segmentationMaskBuffer, const Matrix &viewPortMatrix, int objectAndLinkIndex)
{
	mat<3, 4, float> screenSpacePts = (viewPortMatrix * clipc).transpose();  // transposed to ease access to each of the points

	mat<3, 2, float> pts2;
	for (int i = 0; i < 3; i++)
	{
		pts2[i] = proj<2>(screenSpacePts[i] / screenSpacePts[i][3]);
	}

	Vec2f bboxmin(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
	Vec2f bboxmax(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
	Vec2f clamp(image.get_width() - 1, image.get_height() - 1);

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			bboxmin[j] = b3Max(0.f, b3Min(bboxmin[j], pts2[i][j]));
			bboxmax[j] = b3Min(clamp[j], b3Max(bboxmax[j], pts2[i][j]));
		}
	}

	Vec2i P;
	TGAColor color;

	mat<3, 4, float> orgScreenSpacePts = (viewPortMatrix * orgClipc).transpose();  // transposed to ease access to each of the points

	mat<3, 2, float> orgPts2;
	for (int i = 0; i < 3; i++)
	{
		orgPts2[i] = proj<2>(orgScreenSpacePts[i] / orgScreenSpacePts[i][3]);
	}

	for (P.x = bboxmin.x; P.x <= bboxmax.x; P.x++)
	{
		for (P.y = bboxmin.y; P.y <= bboxmax.y; P.y++)
		{
			double frag_depth = 0;
			{
				Vec3d bc_screen = barycentric(pts2[0], pts2[1], pts2[2], P);
				Vec3d bc_clip = Vec3d(bc_screen.x / screenSpacePts[0][3], bc_screen.y / screenSpacePts[1][3], bc_screen.z / screenSpacePts[2][3]);
				bc_clip = bc_clip / (bc_clip.x + bc_clip.y + bc_clip.z);
				Vec3d clipd(clipc[2].x, clipc[2].y, clipc[2].z);
				frag_depth = -1. * (clipd * bc_clip);

				if (bc_screen.x < 0 || bc_screen.y < 0 || bc_screen.z < 0 ||
					zbuffer[P.x + P.y * image.get_width()] > frag_depth)
					continue;
			}

			Vec3d bc_screen2 = barycentric(orgPts2[0], orgPts2[1], orgPts2[2], P);
			Vec3d bc_clip2 = Vec3d(bc_screen2.x / orgScreenSpacePts[0][3], bc_screen2.y / orgScreenSpacePts[1][3], bc_screen2.z / orgScreenSpacePts[2][3]);
			bc_clip2 = bc_clip2 / (bc_clip2.x + bc_clip2.y + bc_clip2.z);
			Vec3d orgClipd(orgClipc[2].x, orgClipc[2].y, orgClipc[2].z);
			double frag_depth2 = -1. * (orgClipd * bc_clip2);

			Vec3f bc_clip2f(bc_clip2.x, bc_clip2.y, bc_clip2.z);
			bool discard = shader.fragment(bc_clip2f, color);
			
			if (!discard)
			{
				zbuffer[P.x + P.y * image.get_width()] = frag_depth;
				if (segmentationMaskBuffer)
				{
					segmentationMaskBuffer[P.x + P.y * image.get_width()] = objectAndLinkIndex;
				}
				image.set(P.x, P.y, color);
			}
		}
	}
}

void triangle(mat<4, 3, float> &clipc, IShader &shader, TGAImage &image, float *zbuffer, const Matrix &viewPortMatrix)
{
	triangle(clipc, shader, image, zbuffer, 0, viewPortMatrix, 0);
}

void triangle(mat<4, 3, float> &clipc, IShader &shader, TGAImage &image, float *zbuffer, int *segmentationMaskBuffer, const Matrix &viewPortMatrix, int objectAndLinkIndex)
{
	mat<3, 4, float> pts = (viewPortMatrix * clipc).transpose();  // transposed to ease access to each of the points

	mat<3, 2, float> pts2;
	for (int i = 0; i < 3; i++) pts2[i] = proj<2>(pts[i] / pts[i][3]);

	Vec2f bboxmin(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
	Vec2f bboxmax(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
	Vec2f clamp(image.get_width() - 1, image.get_height() - 1);

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			bboxmin[j] = b3Max(0.f, b3Min(bboxmin[j], pts2[i][j]));
			bboxmax[j] = b3Min(clamp[j], b3Max(bboxmax[j], pts2[i][j]));
		}
	}

	Vec2i P;
	TGAColor color;
	for (P.x = bboxmin.x; P.x <= bboxmax.x; P.x++)
	{
		for (P.y = bboxmin.y; P.y <= bboxmax.y; P.y++)
		{
			Vec3d bc_screen = barycentric(pts2[0], pts2[1], pts2[2], P);
			Vec3d bc_clip = Vec3d(bc_screen.x / pts[0][3], bc_screen.y / pts[1][3], bc_screen.z / pts[2][3]);
			bc_clip = bc_clip / (bc_clip.x + bc_clip.y + bc_clip.z);
			Vec3d clipd(clipc[2].x, clipc[2].y, clipc[2].z);
			double frag_depth = -1. * (clipd * bc_clip);
			if (bc_screen.x < 0 || bc_screen.y < 0 || bc_screen.z < 0 ||
				zbuffer[P.x + P.y * image.get_width()] > frag_depth)
				continue;
			Vec3f bc_clipf(bc_clip.x, bc_clip.y, bc_clip.z);
			bool discard = shader.fragment(bc_clipf, color);
			if (frag_depth < -shader.m_farPlane)
				discard = true;
			if (frag_depth > shader.m_nearPlane)
				discard = true;

			if (!discard)
			{
				zbuffer[P.x + P.y * image.get_width()] = frag_depth;
				if (segmentationMaskBuffer)
				{
					segmentationMaskBuffer[P.x + P.y * image.get_width()] = objectAndLinkIndex;
				}
				image.set(P.x, P.y, color);
			}
		}
	}
}

void triangleDepthOnly(mat<4, 3, float> &clipc, float *zbuffer, int *segmentationMaskBuffer, const Matrix &viewPortMatrix, int objectAndLinkIndex, int width, int height, float nearPlane, float farPlane)
{
	mat<3, 4, float> pts = (viewPortMatrix * clipc).transpose();

	mat<3, 2, float> pts2;
	for (int i = 0; i < 3; i++) pts2[i] = proj<2>(pts[i] / pts[i][3]);

	Vec2f bboxmin(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
	Vec2f bboxmax(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
	Vec2f clamp(width - 1, height - 1);

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			bboxmin[j] = b3Max(0.f, b3Min(bboxmin[j], pts2[i][j]));
			bboxmax[j] = b3Min(clamp[j], b3Max(bboxmax[j], pts2[i][j]));
		}
	}

	double Ax = pts2[0].x, Ay = pts2[0].y;
	double Bx = pts2[1].x, By = pts2[1].y;
	double Cx = pts2[2].x, Cy = pts2[2].y;

	double uz = (Cx - Ax) * (By - Ay) - (Bx - Ax) * (Cy - Ay);
	if (std::abs(uz) <= 1e-2)
		return;

	double inv_uz = 1.0 / uz;
	double dux_dx = By - Ay, dux_dy = -(Bx - Ax);
	double duy_dx = -(Cy - Ay), duy_dy = Cx - Ax;

	double inv_w0 = 1.0 / pts[0][3];
	double inv_w1 = 1.0 / pts[1][3];
	double inv_w2 = 1.0 / pts[2][3];
	double cz0 = clipc[2].x, cz1 = clipc[2].y, cz2 = clipc[2].z;

	int x0 = (int)bboxmin.x, y0 = (int)bboxmin.y;
	int x1 = (int)bboxmax.x, y1 = (int)bboxmax.y;

	double ux_row = (Bx - Ax) * (Ay - y0) - (Ax - x0) * (By - Ay);
	double uy_row = (Ax - x0) * (Cy - Ay) - (Cx - Ax) * (Ay - y0);

	for (int py = y0; py <= y1; py++, ux_row += dux_dy, uy_row += duy_dy)
	{
		double ux = ux_row;
		double uy = uy_row;
		int row_off = py * width;

		for (int px = x0; px <= x1; px++, ux += dux_dx, uy += duy_dx)
		{
			double bc0 = (uz - ux - uy) * inv_uz;
			double bc1 = uy * inv_uz;
			double bc2 = ux * inv_uz;

			if (bc0 < 0 || bc1 < 0 || bc2 < 0)
				continue;

			double c0 = bc0 * inv_w0, c1 = bc1 * inv_w1, c2 = bc2 * inv_w2;
			double cs = c0 + c1 + c2;
			c0 /= cs; c1 /= cs; c2 /= cs;

			double frag_depth = -(cz0 * c0 + cz1 * c1 + cz2 * c2);

			int idx = px + row_off;
			if (zbuffer[idx] > frag_depth)
				continue;

			if (frag_depth < -farPlane || frag_depth > nearPlane)
				continue;

			zbuffer[idx] = frag_depth;
			if (segmentationMaskBuffer)
				segmentationMaskBuffer[idx] = objectAndLinkIndex;
		}
	}
}

void triangleClippedDepthOnly(mat<4, 3, float> &clipc, float *zbuffer, int *segmentationMaskBuffer, const Matrix &viewPortMatrix, int objectAndLinkIndex, int width, int height, float nearPlane, float farPlane)
{
	mat<3, 4, float> pts = (viewPortMatrix * clipc).transpose();

	mat<3, 2, float> pts2;
	for (int i = 0; i < 3; i++)
	{
		pts2[i] = proj<2>(pts[i] / pts[i][3]);
	}

	Vec2f bboxmin(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
	Vec2f bboxmax(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
	Vec2f clamp(width - 1, height - 1);

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			bboxmin[j] = b3Max(0.f, b3Min(bboxmin[j], pts2[i][j]));
			bboxmax[j] = b3Min(clamp[j], b3Max(bboxmax[j], pts2[i][j]));
		}
	}

	double Ax = pts2[0].x, Ay = pts2[0].y;
	double Bx = pts2[1].x, By = pts2[1].y;
	double Cx = pts2[2].x, Cy = pts2[2].y;

	double uz = (Cx - Ax) * (By - Ay) - (Bx - Ax) * (Cy - Ay);
	if (std::abs(uz) <= 1e-2)
		return;

	double inv_uz = 1.0 / uz;
	double dux_dx = By - Ay, dux_dy = -(Bx - Ax);
	double duy_dx = -(Cy - Ay), duy_dy = Cx - Ax;

	double inv_w0 = 1.0 / pts[0][3];
	double inv_w1 = 1.0 / pts[1][3];
	double inv_w2 = 1.0 / pts[2][3];
	double cz0 = clipc[2].x, cz1 = clipc[2].y, cz2 = clipc[2].z;

	int x0 = (int)bboxmin.x, y0 = (int)bboxmin.y;
	int x1 = (int)bboxmax.x, y1 = (int)bboxmax.y;

	double ux_row = (Bx - Ax) * (Ay - y0) - (Ax - x0) * (By - Ay);
	double uy_row = (Ax - x0) * (Cy - Ay) - (Cx - Ax) * (Ay - y0);

	for (int py = y0; py <= y1; py++, ux_row += dux_dy, uy_row += duy_dy)
	{
		double ux = ux_row;
		double uy = uy_row;
		int row_off = py * width;

		for (int px = x0; px <= x1; px++, ux += dux_dx, uy += duy_dx)
		{
			double bc0 = (uz - ux - uy) * inv_uz;
			double bc1 = uy * inv_uz;
			double bc2 = ux * inv_uz;

			if (bc0 < 0 || bc1 < 0 || bc2 < 0)
				continue;

			double c0 = bc0 * inv_w0, c1 = bc1 * inv_w1, c2 = bc2 * inv_w2;
			double cs = c0 + c1 + c2;
			c0 /= cs; c1 /= cs; c2 /= cs;

			double frag_depth = -(cz0 * c0 + cz1 * c1 + cz2 * c2);

			int idx = px + row_off;
			if (zbuffer[idx] > frag_depth)
				continue;

			if (frag_depth < -farPlane || frag_depth > nearPlane)
				continue;

			zbuffer[idx] = frag_depth;
			if (segmentationMaskBuffer)
				segmentationMaskBuffer[idx] = objectAndLinkIndex;
		}
	}
}
}
