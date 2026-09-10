import unittest
import numpy as np
import pybullet as p

# One vertical quad facing -Y. A camera at +Y looks at its back.
QUAD_VERTICES = [[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]]
QUAD_INDICES = [0, 1, 2, 0, 2, 3]
SIZE = 64


class TestDoubleSidedMultibody(unittest.TestCase):

  def setUp(self):
    p.connect(p.DIRECT)
    self.proj = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 10.0)

  def tearDown(self):
    p.disconnect()

  def spawn_quad(self, flags=0):
    vis = p.createVisualShape(p.GEOM_MESH, vertices=QUAD_VERTICES, indices=QUAD_INDICES,
                              rgbaColor=[1, 0, 0, 1], flags=flags)
    return p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis)

  def pixels(self, eye_y):
    """Returns (colour, depth, segmentation) hit counts for a camera at (0, eye_y, 0)."""
    view = p.computeViewMatrix([0, eye_y, 0], [0, 0, 0], [0, 0, 1])
    _, _, rgb, depth, seg = p.getCameraImage(SIZE, SIZE, view, self.proj, shadow=0,
                                             renderer=p.ER_TINY_RENDERER)
    rgb = np.asarray(rgb).reshape(-1, 4)[:, :3]
    colour = int((rgb != 255).any(axis=1).sum())
    return colour, int((np.asarray(depth) < 1.0).sum()), int((np.asarray(seg) >= 0).sum())

  def test_front_always_visible(self):
    self.spawn_quad()
    colour, depth, seg = self.pixels(-3)
    self.assertGreater(colour, 0)
    self.assertEqual((colour, colour), (depth, seg))

  def test_unflagged_quad_vanishes_from_behind(self):
    self.spawn_quad()
    self.assertEqual(self.pixels(3), (0, 0, 0))

  def test_soft_body_flag_keeps_ignoring_multibodies(self):
    body = self.spawn_quad(flags=p.VISUAL_SHAPE_DOUBLE_SIDED)
    self.assertEqual(self.pixels(3), (0, 0, 0))
    p.changeVisualShape(body, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED)
    self.assertEqual(self.pixels(3), (0, 0, 0))

  def test_create_flag_shows_back(self):
    self.spawn_quad(flags=p.VISUAL_SHAPE_DOUBLE_SIDED_MULTIBODY)
    back = self.pixels(3)
    self.assertGreater(back[0], 0)
    self.assertEqual(back, self.pixels(-3))

  def test_change_flag_shows_back(self):
    body = self.spawn_quad()
    self.assertEqual(self.pixels(3), (0, 0, 0))
    p.changeVisualShape(body, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED_MULTIBODY)
    self.assertEqual(self.pixels(3), self.pixels(-3))
    p.changeVisualShape(body, -1, flags=0)
    self.assertEqual(self.pixels(3), (0, 0, 0))


if __name__ == '__main__':
  unittest.main()
