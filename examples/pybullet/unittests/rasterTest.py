"""ER_SWARM_RASTER: a painted frame shows what the searched frame shows, dot for dot but for samples on a seam two bodies
share, through clipping, back faces, movers, cut-outs and forest trees, searched tree by tree or whole, with the same
bytes at any thread count."""
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pybullet as p

from alphaCutoutTest import build_world as cutout_world
from alphaCutoutTest import write_leaf_tga
from forestBatchTest import forest_world, ground, write_forest, write_meshes

SIZE = 96
RASTER = getattr(p, "ER_SWARM_RASTER", 0)
INSTANCED = getattr(p, "VISUAL_SHAPE_RENDER_INSTANCED", 0)
NEEDS_FLAG = unittest.skipUnless(RASTER, "wheel built without the painted frame")
PICTURE = (getattr(p, "ER_SWARM_RAYCAST", 0) | getattr(p, "ER_SWARM_SHADOW_MAP", 0) | getattr(p, "ER_SWARM_MOVER_SHADOW", 0) |
           getattr(p, "ER_EDGE_ANTIALIAS", 0) | getattr(p, "ER_ALPHA_CUTOUT", 0) | getattr(p, "ER_TEXTURE_FILTER", 0) |
           p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
LIGHT = [0.6, 0.2, 0.8]
# Low and tilted, so the floor runs from behind the eye to far ahead and its triangles cross the near plane.
EYE, TARGET = (0.0, -6.0, 1.5), (0.0, 2.0, 0.0)
IN_VIEW = [1.2, 0.5, 0.3]


def build_world():
  """A floor wider than the view, a red box, a quad wound away from the eye and a twin wound towards it, and a blue box
  that becomes a mover once it moves; returns (blue box, away quad, towards quad)."""
  floor = p.createVisualShape(p.GEOM_MESH, vertices=[[-30, -30, 0], [30, -30, 0], [30, 30, 0], [-30, 30, 0]],
                              indices=[0, 1, 2, 0, 2, 3], normals=[[0, 0, 1]] * 4, rgbaColor=[0.7, 0.7, 0.7, 1])
  p.createMultiBody(baseMass=0, baseVisualShapeIndex=floor)
  box = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.4], rgbaColor=[1, 0, 0, 1])
  p.createMultiBody(baseMass=0, baseVisualShapeIndex=box, basePosition=[-0.8, 0.0, 0.4])
  quad = [[-0.5, 0.0, 0.2], [0.5, 0.0, 0.2], [0.5, 0.0, 1.2], [-0.5, 0.0, 1.2]]
  away = p.createVisualShape(p.GEOM_MESH, vertices=quad, indices=[0, 2, 1, 0, 3, 2], normals=[[0, 1, 0]] * 4,
                             rgbaColor=[0, 1, 0, 1])
  towards = p.createVisualShape(p.GEOM_MESH, vertices=quad, indices=[0, 1, 2, 0, 2, 3], normals=[[0, -1, 0]] * 4,
                                rgbaColor=[1, 1, 0, 1])
  away_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=away, basePosition=[1.5, -2.0, 0.0])
  towards_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=towards, basePosition=[-1.5, -2.0, 0.0])
  small = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.3], rgbaColor=[0, 0, 1, 1])
  blue = p.createMultiBody(baseMass=0, baseVisualShapeIndex=small, basePosition=[9.0, -9.0, 0.3])
  return blue, away_id, towards_id


def render(flags, size=SIZE, eye=EYE, target=TARGET):
  """Colour HxWx3, depth HxW and object map HxW of one frame from eye towards target."""
  view = p.computeViewMatrix(list(eye), list(target), [0, 0, 1])
  proj = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 60.0)
  _, _, rgb, depth, seg = p.getCameraImage(size, size, view, proj, lightDirection=LIGHT, renderer=p.ER_TINY_RENDERER,
                                           flags=flags, shadow=1)
  return (np.asarray(rgb, dtype=np.uint8).reshape(size, size, 4)[:, :, :3], np.asarray(depth, dtype=np.float32).reshape(size, size),
          np.asarray(seg, dtype=np.int32).reshape(size, size))


def assert_same_frame(case, searched, painted):
  """Object maps agree but for seam samples, depth agrees where they do, and colour barely moves."""
  agree = searched[2] == painted[2]
  case.assertGreaterEqual(agree.mean(), 0.998)
  case.assertLess(np.abs(searched[1] - painted[1])[agree].max(), 1e-4)
  steps = np.abs(searched[0].astype(int) - painted[0].astype(int)).max(axis=2)
  case.assertLessEqual((steps > 16).mean(), 0.005)


def bodies(seg):
  """The body ids an object map shows."""
  return set(int(v) & 0xFFFFFF for v in np.unique(seg) if v >= 0)


@NEEDS_FLAG
class TestRaster(unittest.TestCase):
  """Each frame painted against the same frame searched."""

  def setUp(self):
    """Connects and builds the scene; a first frame puts the still bodies in the static tree."""
    p.connect(p.DIRECT)
    self.blue, self.away, self.towards = build_world()
    render(PICTURE)
    p.resetBasePositionAndOrientation(self.blue, [9.5, -9.0, 0.3], [0, 0, 0, 1])

  def tearDown(self):
    """Disconnects."""
    p.disconnect()

  def test_painted_frame_shows_what_the_search_shows(self):
    """Floor through the near plane, boxes and quads: the painted frame is the searched frame."""
    assert_same_frame(self, render(PICTURE), render(PICTURE | RASTER))

  def test_back_face_stays_hidden(self):
    """The quad wound away from the eye shows in neither frame; its twin wound towards it shows in both."""
    searched, painted = render(PICTURE)[2], render(PICTURE | RASTER)[2]
    for seg in (searched, painted):
      self.assertNotIn(self.away, bodies(seg))
      self.assertIn(self.towards, bodies(seg))

  def test_moved_body_is_painted_where_it_stands(self):
    """A mover brought into view is painted where the search finds it."""
    p.resetBasePositionAndOrientation(self.blue, IN_VIEW, [0, 0, 0, 1])
    searched, painted = render(PICTURE), render(PICTURE | RASTER)
    self.assertIn(self.blue, bodies(painted[2]))
    assert_same_frame(self, searched, painted)

  def test_small_frame_is_searched(self):
    """A frame too small to remember its lens is searched as before, to the byte."""
    for a, b in zip(render(PICTURE, size=32), render(PICTURE | RASTER, size=32)):
      self.assertTrue(np.array_equal(a, b))


@NEEDS_FLAG
class TestRasterScenes(unittest.TestCase):
  """Cut-outs and forest trees, the parts the painting tests texel by texel or leaves to rays."""

  def setUp(self):
    """Connects and makes a folder for the textures and forest files."""
    p.connect(p.DIRECT)
    self.folder = tempfile.mkdtemp()

  def tearDown(self):
    """Disconnects and removes the folder."""
    p.disconnect()
    for name in os.listdir(self.folder):
      os.remove(os.path.join(self.folder, name))
    os.rmdir(self.folder)

  def test_cut_out_holes_match(self):
    """A leaf card's see-through centre shows the wall behind in the painted frame as in the searched one."""
    path = os.path.join(self.folder, "leaf.tga")
    write_leaf_tga(path, True)
    _card, wall = cutout_world(path)
    eye, target = (0.0, -4.5, 2.2), (0.0, 0.0, 1.3)
    searched, painted = render(PICTURE, eye=eye, target=target), render(PICTURE | RASTER, eye=eye, target=target)
    self.assertIn(wall, bodies(painted[2]))
    assert_same_frame(self, searched, painted)

  def test_forest_trees_are_found(self):
    """Forest placements, which only rays draw, show in the painted frame where they show in the searched one, against
    the ground and against the open sky alike."""
    forest_world(self.folder)
    for eye, target in (((0.0, -6.0, 6.0), (0.0, 0.0, 0.0)), ((0.0, -7.0, 0.6), (0.0, 0.0, 1.0))):
      assert_same_frame(self, render(PICTURE, eye=eye, target=target), render(PICTURE | RASTER, eye=eye, target=target))

  def test_a_row_of_trees_deeper_than_the_list_is_searched_whole(self):
    """Looking down a row of forty trees, rays enter more boxes than a sample keeps, and the full search draws them."""
    p.resetSimulation()
    ground()
    row = [(k % 2, (0.0, -4.0 + 0.25 * k, 0.4), 0.3 * k, (1.0, 1.0, 1.0)) for k in range(40)]
    path = write_forest(self.folder, write_meshes(self.folder), row)
    shape = p.createVisualShape(p.GEOM_MESH, fileName=path, flags=INSTANCED | p.VISUAL_SHAPE_DOUBLE_SIDED_MULTIBODY,
                                specularColor=[0, 0, 0])
    p.createMultiBody(0, -1, shape)
    eye, target = (0.0, -9.0, 0.7), (0.0, 0.0, 0.5)
    assert_same_frame(self, render(PICTURE, eye=eye, target=target), render(PICTURE | RASTER, eye=eye, target=target))

  def test_same_bytes_for_one_two_and_four_threads(self):
    """Every painted frame of the scripted run hashes the same at every thread count."""
    digests = set()
    for threads in ("1", "2", "4"):
      env = dict(os.environ, SWARM_RENDER_THREADS=threads)
      digests.add(subprocess.check_output([sys.executable, __file__, "--hash"], env=env, text=True).strip())
    self.assertEqual(len(digests), 1)


def run_hash():
  """Prints the sha256 of the painted frames of the main scene, with a mover in view, and of a forest."""
  p.connect(p.DIRECT)
  blue, _away, _towards = build_world()
  digest = hashlib.sha256()
  for position in ([9.5, -9.0, 0.3], IN_VIEW):
    p.resetBasePositionAndOrientation(blue, position, [0, 0, 0, 1])
    for part in render(PICTURE | RASTER):
      digest.update(part.tobytes())
  folder = tempfile.mkdtemp()
  forest_world(folder)
  for part in render(PICTURE | RASTER, eye=(0.0, -6.0, 6.0), target=(0.0, 0.0, 0.0)):
    digest.update(part.tobytes())
  for name in os.listdir(folder):
    os.remove(os.path.join(folder, name))
  os.rmdir(folder)
  print(digest.hexdigest())
  p.disconnect()


if __name__ == '__main__':
  if "--hash" in sys.argv:
    run_hash()
  else:
    unittest.main()
