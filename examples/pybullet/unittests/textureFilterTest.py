"""Proves ER_TEXTURE_FILTER blends texels and picks mip levels, and leaves depth and segmentation untouched."""
import os
import struct
import tempfile
import unittest
import numpy as np
import pybullet as p

# A vertical quad facing -Y, textured with a 64x64 checkerboard of 8-texel squares.
QUAD_VERTICES = [[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]]
QUAD_INDICES = [0, 1, 2, 0, 2, 3]
QUAD_UVS = [[0, 0], [1, 0], [1, 1], [0, 1]]
QUAD_NORMALS = [[0, -1, 0]] * 4  # flat shading, so nearest sampling yields exactly two grey levels
SIZE = 64
TEX = 64
SQUARE = 8


def write_checker_tga(path):
  """Writes an uncompressed 24-bit TGA checkerboard; the texture loader reads it without any extra library."""
  yy, xx = np.mgrid[0:TEX, 0:TEX]
  white = ((xx // SQUARE + yy // SQUARE) % 2 == 0)
  img = np.where(white[..., None], 255, 0).astype(np.uint8).repeat(3, axis=2)
  header = struct.pack('<BBBHHBHHHHBB', 0, 0, 2, 0, 0, 0, 0, 0, TEX, TEX, 24, 0)
  with open(path, 'wb') as f:
    f.write(header)
    f.write(img.tobytes())


class TestTextureFilter(unittest.TestCase):
  """Renders one checkerboard quad close up and far away, with and without the flag."""

  def setUp(self):
    """Connects, writes the checker texture and spawns the textured quad."""
    p.connect(p.DIRECT)
    self.tmp = tempfile.mkdtemp()
    tex_path = os.path.join(self.tmp, 'checker.tga')
    write_checker_tga(tex_path)
    vis = p.createVisualShape(p.GEOM_MESH, vertices=QUAD_VERTICES, indices=QUAD_INDICES,
                              uvs=QUAD_UVS, normals=QUAD_NORMALS, rgbaColor=[1, 1, 1, 1])
    body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis)
    p.changeVisualShape(body, -1, textureUniqueId=p.loadTexture(tex_path))
    self.proj = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 100.0)

  def tearDown(self):
    """Disconnects and removes the temporary texture."""
    p.disconnect()
    for name in os.listdir(self.tmp):
      os.remove(os.path.join(self.tmp, name))
    os.rmdir(self.tmp)

  def render(self, eye_y, flags):
    """Returns (grey image, depth, segmentation) for a camera at (0, eye_y, 0) looking at the quad."""
    view = p.computeViewMatrix([0, eye_y, 0], [0, 0, 0], [0, 0, 1])
    _, _, rgb, depth, seg = p.getCameraImage(SIZE, SIZE, view, self.proj, shadow=0,
                                             renderer=p.ER_TINY_RENDERER, flags=flags)
    grey = np.asarray(rgb).reshape(SIZE, SIZE, 4)[:, :, 0].astype(int)
    return grey, np.asarray(depth).reshape(SIZE, SIZE), np.asarray(seg).reshape(SIZE, SIZE)

  def test_nearest_keeps_two_texel_values_close_up(self):
    """Without the flag a flat-lit checker renders as exactly two grey levels."""
    grey, _, _ = self.render(-1.0, 0)
    self.assertEqual(len(np.unique(grey)), 2)

  def test_filter_blends_the_square_edges_close_up(self):
    """With the flag the square edges become gradients between the same two extremes."""
    grey, _, _ = self.render(-1.0, p.ER_TEXTURE_FILTER)
    nearest, _, _ = self.render(-1.0, 0)
    lo, hi = int(nearest.min()), int(nearest.max())
    self.assertGreater(len(np.unique(grey)), 8)
    self.assertEqual((int(grey.min()), int(grey.max())), (lo, hi))

  def test_filter_averages_a_far_quad(self):
    """A quad covering a few pixels reads a small mip level, so every pixel lands mid-grey."""
    nearest, _, seg = self.render(-40.0, 0)
    grey, _, _ = self.render(-40.0, p.ER_TEXTURE_FILTER)
    quad = seg >= 0
    self.assertGreater(int(quad.sum()), 0)
    lo, hi = int(nearest[quad].min()), int(nearest[quad].max())
    self.assertEqual(lo, 0)
    mid = grey[quad]
    self.assertTrue(((mid > 0.3 * hi) & (mid < 0.7 * hi)).all())

  def test_flag_leaves_depth_and_segmentation_alone(self):
    """The flag changes colour only: depth and segmentation buffers are byte-identical."""
    for eye_y in (-1.0, -40.0):
      _, depth0, seg0 = self.render(eye_y, 0)
      _, depth1, seg1 = self.render(eye_y, p.ER_TEXTURE_FILTER)
      self.assertTrue(np.array_equal(depth0, depth1))
      self.assertTrue(np.array_equal(seg0, seg1))


if __name__ == '__main__':
  unittest.main()
