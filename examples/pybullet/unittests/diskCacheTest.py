"""Model, texture and ray-cast tree files under SWARM_BVH_CACHE_DIR: a process with the folder draws the same bytes as
one without it, whether it writes the files, reads them back, or finds them broken."""
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pybullet as p
import pybullet_data

RAY = getattr(p, "ER_SWARM_RAYCAST", 0)


def frame_hash():
  """Hash of a ray-cast and a TinyRenderer frame of the textured duck, loaded from its .obj and .mtl."""
  p.connect(p.DIRECT)
  shape = p.createVisualShape(p.GEOM_MESH, fileName=os.path.join(pybullet_data.getDataPath(), "duck.obj"), meshScale=[2, 2, 2])
  p.createMultiBody(baseMass=0, baseVisualShapeIndex=shape)
  view = p.computeViewMatrix([0.4, -0.5, 0.4], [0, 0, 0], [0, 0, 1])
  proj = p.computeProjectionMatrixFOV(60, 1.0, 0.01, 10.0)
  digest = hashlib.sha256()
  for flags in (RAY, 0):
    _, _, rgb, depth, seg = p.getCameraImage(96, 96, view, proj, renderer=p.ER_TINY_RENDERER, flags=flags)
    for part in (rgb, depth, seg):
      digest.update(np.ascontiguousarray(part).tobytes())
  p.disconnect()
  return digest.hexdigest()


def child(cache_dir):
  """The frame hash from a fresh process, with the cache folder set when given."""
  env = dict(os.environ)
  env.pop("SWARM_BVH_CACHE_DIR", None)
  if cache_dir:
    env["SWARM_BVH_CACHE_DIR"] = cache_dir
  return subprocess.check_output([sys.executable, __file__, "--hash"], env=env, text=True).strip().splitlines()[-1]


class TestDiskCache(unittest.TestCase):
  """The duck drawn without the folder, writing it, reading it, and with every file in it cut short."""

  def setUp(self):
    """Makes an empty cache folder."""
    self.folder = tempfile.mkdtemp()

  def tearDown(self):
    """Removes the folder and whatever was written to it."""
    for name in os.listdir(self.folder):
      os.remove(os.path.join(self.folder, name))
    os.rmdir(self.folder)

  def kinds(self):
    """The file suffixes in the cache folder."""
    return sorted({name.rsplit(".", 1)[-1] for name in os.listdir(self.folder)})

  def test_written_and_read_files_draw_the_same_bytes(self):
    """The first process writes the model and texture files, the next reads them, and both draw what none draws."""
    plain = child(None)
    self.assertEqual(child(self.folder), plain)
    self.assertIn("objc", self.kinds())
    self.assertIn("texc", self.kinds())
    self.assertEqual(child(self.folder), plain)

  def test_broken_files_are_parsed_afresh(self):
    """Every cache file cut to half its length is ignored, and the frame is the one drawn without the folder."""
    plain = child(None)
    child(self.folder)
    for name in os.listdir(self.folder):
      path = os.path.join(self.folder, name)
      with open(path, "rb") as handle:
        data = handle.read()
      with open(path, "wb") as handle:
        handle.write(data[:len(data) // 2])
    self.assertEqual(child(self.folder), plain)


if __name__ == '__main__':
  if "--hash" in sys.argv:
    print(frame_hash())
  else:
    unittest.main()
