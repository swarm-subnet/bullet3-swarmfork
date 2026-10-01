"""Ray-cast frames between changes: whatever changes a body between two frames shows in the second one."""
import os
import struct
import tempfile
import unittest
import numpy as np
import pybullet as p

SIZE = 64
NEEDS_BACKEND = unittest.skipUnless(hasattr(p, "ER_SWARM_RAYCAST"), "wheel built without the ray-cast backend")
# One vertical quad facing -Y; the camera at +Y sees its back.
QUAD_VERTICES = [[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]]
QUAD_INDICES = [0, 1, 2, 0, 2, 3]


def write_blue_tga(path):
  """Writes an uncompressed 24-bit TGA of one blue colour."""
  header = struct.pack('<BBBHHBHHHHBB', 0, 0, 2, 0, 0, 0, 0, 0, 4, 4, 24, 0)
  with open(path, 'wb') as f:
    f.write(header)
    f.write(bytes([255, 0, 0]) * 16)


@NEEDS_BACKEND
class TestRaycastSync(unittest.TestCase):
  """A white quad and a two-link arm, drawn, changed and drawn again on the ray-cast path."""

  def setUp(self):
    """Connects a DIRECT client with the quad at the origin and the arm beside it, and draws a first frame."""
    p.connect(p.DIRECT)
    vis = p.createVisualShape(p.GEOM_MESH, vertices=QUAD_VERTICES, indices=QUAD_INDICES, uvs=[[0, 0], [1, 0], [1, 1], [0, 1]],
                              rgbaColor=[1, 1, 1, 1])
    self.quad = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis)
    link = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.5], rgbaColor=[0, 1, 0, 1])
    shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.5])
    self.arm = p.createMultiBody(baseMass=0, basePosition=[2, 0, -1], linkMasses=[1], linkCollisionShapeIndices=[shape],
                                 linkVisualShapeIndices=[link], linkPositions=[[0, 0, 0]], linkOrientations=[[0, 0, 0, 1]],
                                 linkInertialFramePositions=[[0, 0, 0.5]], linkInertialFrameOrientations=[[0, 0, 0, 1]],
                                 linkParentIndices=[0], linkJointTypes=[p.JOINT_REVOLUTE], linkJointAxis=[[0, 1, 0]])
    self.render()

  def tearDown(self):
    """Drops the client."""
    p.disconnect()

  def render(self, eye_y=-6.0):
    """Returns (rgb HxWx3, seg HxW with link bits) from a camera on the Y axis looking at the origin."""
    view = p.computeViewMatrix([0, eye_y, 0], [0, 0, 0], [0, 0, 1])
    proj = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 20.0)
    _, _, rgb, _, seg = p.getCameraImage(SIZE, SIZE, view, proj, shadow=0, renderer=p.ER_TINY_RENDERER,
                                         flags=p.ER_SWARM_RAYCAST | p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
    return np.asarray(rgb).reshape(SIZE, SIZE, 4)[:, :, :3].astype(int), np.asarray(seg).reshape(SIZE, SIZE)

  def quad_pixels(self, seg):
    """How many pixels show the quad."""
    return int((seg == self.quad).sum())

  def test_moved_body_follows(self):
    """A body moved away between frames leaves the picture, and moved back it returns."""
    before = self.quad_pixels(self.render()[1])
    self.assertGreater(before, 0)
    p.resetBasePositionAndOrientation(self.quad, [0, 0, 50], [0, 0, 0, 1])
    self.assertEqual(self.quad_pixels(self.render()[1]), 0)
    p.resetBasePositionAndOrientation(self.quad, [0, 0, 0], [0, 0, 0, 1])
    self.assertGreater(self.quad_pixels(self.render()[1]), 0)

  def test_moved_link_follows(self):
    """A link turned by its joint between frames is drawn where it went."""
    link = self.arm + (1 << 24)
    before = self.render()[1] == link
    p.resetJointState(self.arm, 0, 1.2)
    after = self.render()[1] == link
    self.assertGreater(int(before.sum()), 0)
    self.assertFalse(np.array_equal(before, after))

  def test_transparent_body_hides(self):
    """A body made fully transparent between frames is gone, and made opaque again it is back."""
    before = self.quad_pixels(self.render()[1])
    p.changeVisualShape(self.quad, -1, rgbaColor=[1, 1, 1, 0])
    self.assertEqual(self.quad_pixels(self.render()[1]), 0)
    p.changeVisualShape(self.quad, -1, rgbaColor=[1, 1, 1, 1])
    self.assertEqual(self.quad_pixels(self.render()[1]), before)

  def test_new_texture_shows(self):
    """A texture given between frames colours the body in the next frame."""
    rgb, seg = self.render()
    white = rgb[seg == self.quad].mean(axis=0)
    with tempfile.TemporaryDirectory() as folder:
      path = os.path.join(folder, "blue.tga")
      write_blue_tga(path)
      p.changeVisualShape(self.quad, -1, textureUniqueId=p.loadTexture(path))
    rgb, seg = self.render()
    shaded = rgb[seg == self.quad].mean(axis=0)
    self.assertEqual(white[0], white[2])
    self.assertGreater(shaded[2], shaded[0] + 50)

  def test_double_sided_flag_shows_back(self):
    """The double-sided flag given between frames shows the back of a face, and taken away hides it again."""
    self.assertEqual(self.quad_pixels(self.render(6.0)[1]), 0)
    p.changeVisualShape(self.quad, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED_MULTIBODY)
    self.assertGreater(self.quad_pixels(self.render(6.0)[1]), 0)
    p.changeVisualShape(self.quad, -1, flags=0)
    self.assertEqual(self.quad_pixels(self.render(6.0)[1]), 0)

  def test_new_body_appears(self):
    """A body created after the first frame is drawn in the next one."""
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.3], rgbaColor=[0, 0, 1, 1])
    body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=[-2, -1, 0])
    self.assertGreater(int((self.render()[1] == body).sum()), 0)


if __name__ == '__main__':
  unittest.main()
