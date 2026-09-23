"""A forest file draws every placement of its meshes as one body, exactly as the same placements as separate bodies."""
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pybullet as p

from daylightTest import write_obj
from instancedStaticTest import CUTOUT, DAY, INSTANCED, MAP, RAY, frame

# (mesh index, position, yaw in radians, scale) of each placement, with turns and uneven scales on purpose.
PLACEMENTS = [(0, (-2.0, -1.0, 0.4), 0.0, (1.0, 1.0, 1.0)), (0, (1.5, -1.5, 0.2), 0.7, (0.6, 1.3, 1.0)),
              (1, (0.0, 1.5, 0.5), 2.1, (1.0, 1.0, 1.8)), (1, (2.0, 1.0, 0.1), -1.2, (0.8, 0.8, 0.8)),
              (0, (-1.5, 2.0, 0.3), 3.0, (1.2, 0.7, 1.0))]


def write_meshes(folder):
  """Two small meshes: a flat card and a standing card, as the forest's two kinds of tree."""
  write_obj(os.path.join(folder, "flat.obj"), 0.5)
  write_obj(os.path.join(folder, "stand.obj"), 0.5, double_sided_quad=True)
  return ["flat.obj", "stand.obj"]


def write_forest(folder, meshes, placements):
  """Write a forest file naming the meshes and listing each placement as index, xyz, quaternion and scale."""
  lines = ["# a comment line is skipped"] + ["mesh %s" % name for name in meshes]
  for index, position, yaw, scale in placements:
    lines.append("%d %f %f %f 0 0 %f %f %f %f %f" % ((index,) + tuple(position) + (math.sin(yaw / 2), math.cos(yaw / 2))
                                                     + tuple(scale)))
  path = os.path.join(folder, "forest.fst")
  with open(path, "w") as out:
    out.write("\n".join(lines) + "\n")
  return path


def ground():
  """A white receiver under the placements, so shadows land somewhere."""
  shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[5, 5, 0.05], rgbaColor=[1, 1, 1, 1], specularColor=[0, 0, 0])
  p.createMultiBody(0, -1, shape, basePosition=[0, 0, -0.05])


def forest_world(folder, base=(0, 0, 0), flags=INSTANCED):
  """The placements as one forest body; returns its body id."""
  p.resetSimulation()
  ground()
  path = write_forest(folder, write_meshes(folder), PLACEMENTS)
  shape = p.createVisualShape(p.GEOM_MESH, fileName=path, flags=flags | p.VISUAL_SHAPE_DOUBLE_SIDED_MULTIBODY,
                              specularColor=[0, 0, 0])
  return p.createMultiBody(0, -1, shape, basePosition=list(base))


def separate_world(folder, base=(0, 0, 0)):
  """The same placements as one instanced body each, the path the forest must match."""
  p.resetSimulation()
  ground()
  meshes = write_meshes(folder)
  bodies = []
  for index, position, yaw, scale in PLACEMENTS:
    shape = p.createVisualShape(p.GEOM_MESH, fileName=os.path.join(folder, meshes[index]), meshScale=list(scale),
                                flags=INSTANCED | p.VISUAL_SHAPE_DOUBLE_SIDED_MULTIBODY, specularColor=[0, 0, 0])
    bodies.append(p.createMultiBody(0, -1, shape, basePosition=[a + b for a, b in zip(base, position)],
                                    baseOrientation=[0, 0, math.sin(yaw / 2), math.cos(yaw / 2)]))
  return bodies


def child(threads):
  """Hash of a forest frame rendered in a fresh process at a thread count."""
  env = dict(os.environ, SWARM_RENDER_THREADS=threads)
  output = subprocess.check_output([sys.executable, __file__, "identity"], env=env, text=True)
  return json.loads(output.strip().splitlines()[-1])


@unittest.skipUnless(INSTANCED and RAY, "wheel without static instancing")
class TestForestBatch(unittest.TestCase):
  """Compare a forest body with separate bodies on every path that draws or shadows it."""

  def setUp(self):
    """Create an isolated client and a temporary folder for the meshes and the forest file."""
    p.connect(p.DIRECT)
    self.temp = tempfile.TemporaryDirectory()
    self.folder = self.temp.name

  def tearDown(self):
    """Release the client and its files."""
    p.disconnect()
    self.temp.cleanup()

  def assertPixelsEqual(self, left, right):
    """Colour as bytes and depth to its last bit, which two levels of instancing may round differently; not the mask."""
    differing = np.count_nonzero(left[0] != right[0])
    self.assertTrue(left[0].tobytes() == right[0].tobytes(), "colour: %d differing values" % differing)
    self.assertLessEqual(float(np.abs(np.asarray(left[1], dtype=np.float64) - right[1]).max()), 2.5e-7)

  def test_forest_matches_separate_bodies(self):
    """Turned and unevenly scaled placements draw and shadow the same with the ray caster, the map and daylight."""
    for flags in (RAY, RAY | MAP, RAY | MAP | DAY | CUTOUT):
      forest_world(self.folder)
      together = frame(flags=flags, eye=(0, -6, 6))
      separate_world(self.folder)
      apart = frame(flags=flags, eye=(0, -6, 6))
      self.assertPixelsEqual(together, apart)

  def test_body_pose_carries_the_placements(self):
    """A forest body placed off the origin moves every placement with it."""
    forest_world(self.folder, base=(1.0, -0.5, 0.25))
    together = frame(flags=RAY | MAP, eye=(0, -6, 6))
    separate_world(self.folder, base=(1.0, -0.5, 0.25))
    self.assertPixelsEqual(together, frame(flags=RAY | MAP, eye=(0, -6, 6)))

  def test_raster_path_draws_every_placement(self):
    """The rasteriser draws the forest the same as separate bodies too."""
    forest_world(self.folder)
    together = frame(flags=0, shadow=0, eye=(0, -6, 6))
    separate_world(self.folder)
    self.assertPixelsEqual(together, frame(flags=0, shadow=0, eye=(0, -6, 6)))

  def test_mask_names_the_forest_body(self):
    """Every placement's pixels carry the forest body's id in the segmentation mask."""
    body = forest_world(self.folder)
    _, _, mask = frame(flags=RAY, shadow=0, eye=(0, -6, 6))
    separate_world(self.folder)
    _, _, apart = frame(flags=RAY, shadow=0, eye=(0, -6, 6))
    self.assertGreater(int((mask == body).sum()), 0)
    self.assertEqual(int((mask == body).sum()), int((apart > 0).sum()))

  def test_mover_leaves_the_forest_pixels_alone(self):
    """A body moving between frames changes only its own pixels, not the forest's."""
    forest_world(self.folder)
    box = p.createMultiBody(1, -1, p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2] * 3), basePosition=[0, 0, 3])
    before = frame(flags=RAY, shadow=0, eye=(0, -6, 6))
    p.resetBasePositionAndOrientation(box, [0.5, 0, 3], [0, 0, 0, 1])
    after = frame(flags=RAY, shadow=0, eye=(0, -6, 6))
    untouched = (before[2] != box) & (after[2] != box)
    self.assertTrue((before[0][untouched] == after[0][untouched]).all())

  def test_without_the_flag_nothing_is_drawn(self):
    """A forest file needs the instanced flag; without it the body draws nothing and nothing fails."""
    forest_world(self.folder, flags=0)
    _, _, mask = frame(flags=RAY, shadow=0, eye=(0, -6, 6))
    self.assertEqual(int((mask > 0).sum()), 0)

  def test_threads_give_the_same_bytes(self):
    """Fresh processes render identical colour, depth and mask at one, two and four threads."""
    hashes = [child(t) for t in ("1", "2", "4")]
    self.assertEqual(hashes[0], hashes[1])
    self.assertEqual(hashes[0], hashes[2])


if __name__ == "__main__":
  if len(sys.argv) > 1 and sys.argv[1] == "identity":
    p.connect(p.DIRECT)
    with tempfile.TemporaryDirectory() as folder:
      forest_world(folder)
      buffers = frame(RAY | MAP | DAY | CUTOUT | p.ER_EDGE_ANTIALIAS | p.ER_TEXTURE_FILTER, eye=(0, -6, 6))
      print(json.dumps(hashlib.sha256(b"".join(a.tobytes() for a in buffers)).hexdigest()))
    p.disconnect()
  else:
    unittest.main()
