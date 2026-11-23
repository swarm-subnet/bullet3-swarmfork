"""The per-client file cache frees a file's bytes once its last handle closes."""
import shutil
import tempfile
import unittest
from pathlib import Path

import pybullet_data

import pybullet as p

DATA = Path(pybullet_data.getDataPath())


class FileCacheTest(unittest.TestCase):
  """Loads meshes from a scratch copy so files can be deleted while the client is connected."""

  def setUp(self) -> None:
    """Connects a direct client and copies cube.obj into a scratch folder."""
    self.folder = Path(tempfile.mkdtemp())
    self.mesh = str(self.folder / "cube.obj")
    shutil.copy(DATA / "cube.obj", self.mesh)
    self.client = p.connect(p.DIRECT)

  def tearDown(self) -> None:
    """Disconnects and removes the scratch folder."""
    p.disconnect(self.client)
    shutil.rmtree(self.folder, ignore_errors=True)

  def test_deleted_file_is_not_served_from_memory(self) -> None:
    """A mesh file removed after its shape was made is gone for the client too."""
    first = p.createCollisionShape(p.GEOM_MESH, fileName=self.mesh, physicsClientId=self.client)
    self.assertGreaterEqual(first, 0)
    Path(self.mesh).unlink()
    with self.assertRaises(p.error):
      p.createCollisionShape(p.GEOM_MESH,
                             fileName=self.mesh,
                             meshScale=[2, 2, 2],
                             physicsClientId=self.client)

  def test_reopened_file_gives_the_same_shape(self) -> None:
    """A convex mesh read again from disk after its bytes were freed builds the same hull."""
    boxes = []
    for _ in range(2):
      shape = p.createCollisionShape(p.GEOM_MESH,
                                     fileName=self.mesh,
                                     meshScale=[0.5, 1.0, 2.0],
                                     physicsClientId=self.client)
      body = p.createMultiBody(baseMass=0,
                               baseCollisionShapeIndex=shape,
                               physicsClientId=self.client)
      boxes.append(p.getAABB(body, physicsClientId=self.client))
    self.assertEqual(boxes[0], boxes[1])

  def test_files_load_again_after_release(self) -> None:
    """A texture loaded twice and a mesh opened by its probe and parse each read the file again."""
    texture = str(self.folder / "tex.png")
    shutil.copy(DATA / "tex256.png", texture)
    self.assertGreaterEqual(p.loadTexture(texture, physicsClientId=self.client), 0)
    self.assertGreaterEqual(p.loadTexture(texture, physicsClientId=self.client), 0)
    visual = p.createVisualShape(p.GEOM_MESH, fileName=self.mesh, physicsClientId=self.client)
    body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual, physicsClientId=self.client)
    self.assertGreaterEqual(body, 0)


if __name__ == "__main__":
  unittest.main()
