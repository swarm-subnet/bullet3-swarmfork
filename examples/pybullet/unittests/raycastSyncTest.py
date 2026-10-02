"""Ray-cast frames between changes: whatever changes a body between two frames shows in the second one."""
import os
import shutil
import struct
import tempfile
import unittest
import numpy as np
import pybullet as p

from daylightTest import write_obj
from forestBatchTest import forest_world
from instancedStaticTest import INSTANCED, frame
from visualMeshTest import grid

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


@NEEDS_BACKEND
class TestRaycastSyncPaths(unittest.TestCase):
  """The other ways a world changes between two ray-cast frames, seen from straight above."""

  def setUp(self):
    """Connects a DIRECT client and makes a folder for mesh files."""
    p.connect(p.DIRECT)
    self.folder = tempfile.mkdtemp(prefix="raycastsync_")

  def tearDown(self):
    """Drops the client and the folder."""
    p.disconnect()
    shutil.rmtree(self.folder)

  def colour(self):
    """(rgb, depth, seg) of a ray-cast colour frame."""
    return frame(flags=p.ER_SWARM_RAYCAST, shadow=0)

  def depth_only(self):
    """The depth of a ray-cast depth-only frame, which brings the scene up to date as well."""
    return frame(flags=p.ER_SWARM_RAYCAST | p.ER_DEPTH_ONLY, shadow=0)[1]

  def grid_body(self, position=(0, 0, 0)):
    """A flat grid body built from arrays; every body built from the same arrays shares one mesh."""
    vertices, normals, uvs, triangles = grid(9, 1.0)
    shape = p.createVisualShape(p.GEOM_MESH, vertices=vertices, indices=[i for t in triangles for i in t], normals=normals,
                                uvs=uvs, rgbaColor=[1, 1, 1, 1], specularColor=[0, 0, 0])
    return p.createMultiBody(0, -1, shape, basePosition=list(position)), vertices

  def test_rewritten_mesh_follows(self):
    """A mesh rewritten between frames is drawn in its new shape, also after a depth-only frame took the change first."""
    body, vertices = self.grid_body()
    before = self.colour()
    raised = [list(v) for v in vertices]
    raised[len(raised) // 2][2] = 0.8
    p.resetMeshData(body, raised)
    depth = self.depth_only()
    after = self.colour()
    self.assertNotEqual(before[1].tobytes(), after[1].tobytes())
    self.assertEqual(depth.tobytes(), after[1].tobytes())

  def test_change_survives_a_depth_only_frame(self):
    """A body moved before a depth-only frame is still moved in the colour frame after it."""
    box = p.createMultiBody(0, -1, p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5] * 3))
    self.assertGreater(int((self.colour()[2] == box).sum()), 0)
    p.resetBasePositionAndOrientation(box, [0, 0, 50], [0, 0, 0, 1])
    self.depth_only()
    self.assertEqual(int((self.colour()[2] == box).sum()), 0)

  def test_removed_body_and_its_successor(self):
    """A removed body is gone from the next frame, and a body created after it is drawn where it was put."""
    first = p.createMultiBody(0, -1, p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5] * 3), basePosition=[-1.5, 0, 0])
    self.colour()
    p.removeBody(first)
    second = p.createMultiBody(0, -1, p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5] * 3), basePosition=[1.5, 0, 0])
    _, _, seg = self.colour()
    columns = np.nonzero((seg == second).any(axis=0))[0]
    self.assertGreater(len(columns), 0)
    self.assertGreater(columns.min(), seg.shape[1] // 2)
    self.assertEqual(int((seg >= 0).sum()), int((seg == second).sum()))

  def test_reset_simulation_starts_a_new_scene(self):
    """After resetSimulation only the bodies of the new world are drawn."""
    p.createMultiBody(0, -1, p.createVisualShape(p.GEOM_BOX, halfExtents=[2, 2, 0.1]))
    self.colour()
    p.resetSimulation()
    box = p.createMultiBody(0, -1, p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3] * 3))
    _, _, seg = self.colour()
    self.assertGreater(int((seg == box).sum()), 0)
    self.assertEqual(int((seg >= 0).sum()), int((seg == box).sum()))

  def test_instanced_body_moves_and_hides(self):
    """An instanced body moved between frames is drawn where it went, and made transparent it is gone."""
    path = os.path.join(self.folder, "card.obj")
    write_obj(path, 0.5)
    body = p.createMultiBody(0, -1, p.createVisualShape(p.GEOM_MESH, fileName=path, flags=INSTANCED), basePosition=[-1.5, 0, 0])
    before = self.colour()[2] == body
    p.resetBasePositionAndOrientation(body, [1.5, 0, 0], [0, 0, 0, 1])
    after = self.colour()[2] == body
    self.assertGreater(int(after.sum()), 0)
    self.assertFalse((before & after).any())
    p.changeVisualShape(body, -1, rgbaColor=[1, 1, 1, 0])
    self.assertEqual(int((self.colour()[2] == body).sum()), 0)

  def test_forest_moves_with_its_body(self):
    """Every placement of a forest follows its body when the body moves between frames."""
    body = forest_world(self.folder)
    before = self.colour()[2] == body
    p.resetBasePositionAndOrientation(body, [0.5, 0, 0], [0, 0, 0, 1])
    after = self.colour()[2] == body
    self.assertGreater(int(before.sum()), 0)
    self.assertFalse(np.array_equal(before, after))

  def shared_mesh_frame(self, touch_untouched):
    """Two movers sharing one mesh, one of them rewritten: the frame after, with the other body's colour set again or not."""
    p.resetSimulation()
    p.createMultiBody(0, -1, p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2] * 3), basePosition=[0, 2.5, 0])
    self.colour()
    rewritten, vertices = self.grid_body((-1.2, 0, 0))
    untouched, _ = self.grid_body((1.2, 0, 0))
    before = self.colour()
    raised = [list(v) for v in vertices]
    raised[len(raised) // 2][2] = 0.8
    p.resetMeshData(rewritten, raised)
    if touch_untouched:
      p.changeVisualShape(untouched, -1, rgbaColor=[1, 1, 1, 1])
    return before, self.colour()

  def test_shared_mesh_rewrite_draws_as_a_full_sync(self):
    """Rewriting one of two movers that share a mesh draws the same bytes as when the other one is synced too."""
    before, plain = self.shared_mesh_frame(False)
    _, synced = self.shared_mesh_frame(True)
    self.assertNotEqual(before[1].tobytes(), plain[1].tobytes())
    for a, b in zip(plain, synced):
      self.assertEqual(a.tobytes(), b.tobytes())


if __name__ == '__main__':
  unittest.main()
