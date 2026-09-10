"""Concave mesh collision trees cached on disk behind GEOM_CONCAVE_BVH_CACHE and SWARM_BVH_CACHE_DIR."""
import glob
import os
import shutil
import tempfile
import unittest
import pybullet as p

# A bumpy 12 x 12 grid, so the concave mesh has a real tree with many leaves.
N = 12
VERTICES = [[x * 0.5, y * 0.5, 0.2 * ((x * 7 + y * 3) % 5)] for y in range(N) for x in range(N)]
INDICES = []
for y in range(N - 1):
  for x in range(N - 1):
    a = y * N + x
    INDICES += [a, a + 1, a + N, a + 1, a + N + 1, a + N]
FLAGS = p.GEOM_FORCE_CONCAVE_TRIMESH | p.GEOM_CONCAVE_BVH_CACHE
RAYS_FROM = [[x * 0.37, y * 0.41, 5.0] for x in range(16) for y in range(16)]
RAYS_TO = [[x, y, -1.0] for x, y, _ in RAYS_FROM]


class TestConcaveBvhCache(unittest.TestCase):
  """Builds the same concave mesh in fresh worlds and checks what the cache folder does to it."""

  def setUp(self):
    """Hand the engine an empty cache folder."""
    self.cache_dir = tempfile.mkdtemp(prefix="bvhcache")
    os.environ["SWARM_BVH_CACHE_DIR"] = self.cache_dir

  def tearDown(self):
    """Drop the folder and the variable."""
    os.environ.pop("SWARM_BVH_CACHE_DIR", None)
    shutil.rmtree(self.cache_dir, ignore_errors=True)

  def cache_files(self):
    """Every file the engine left in the cache folder."""
    return sorted(glob.glob(os.path.join(self.cache_dir, "*")))

  def hits(self, flags=FLAGS, scale=(1, 1, 1)):
    """Ray hits and the body AABB from a fresh world holding one concave mesh."""
    cli = p.connect(p.DIRECT)
    col = p.createCollisionShape(p.GEOM_MESH, vertices=VERTICES, indices=INDICES, flags=flags,
                                 meshScale=list(scale), physicsClientId=cli)
    body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, physicsClientId=cli)
    rays = p.rayTestBatch(RAYS_FROM, RAYS_TO, physicsClientId=cli)
    result = ([(r[0], r[2], r[3], r[4]) for r in rays], p.getAABB(body, physicsClientId=cli))
    p.disconnect(physicsClientId=cli)
    return result

  def test_no_file_without_flag(self):
    """A concave mesh created without the flag writes nothing."""
    self.hits(flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
    self.assertEqual(self.cache_files(), [])

  def test_no_file_without_cache_dir(self):
    """The flag alone, with no folder set, writes nothing."""
    del os.environ["SWARM_BVH_CACHE_DIR"]
    self.hits()
    self.assertEqual(self.cache_files(), [])

  def test_built_then_loaded_are_identical(self):
    """The first world writes one tree file, the second loads it untouched, and both hit the same points."""
    built = self.hits()
    files = self.cache_files()
    self.assertEqual(len(files), 1)
    self.assertTrue(files[0].endswith(".bvh"))
    stamp = (os.path.getsize(files[0]), os.path.getmtime(files[0]))
    loaded = self.hits()
    self.assertEqual(built, loaded)
    self.assertEqual((os.path.getsize(files[0]), os.path.getmtime(files[0])), stamp)
    self.assertTrue(any(fraction < 1.0 for _, fraction, _, _ in built[0]))

  def test_scale_gets_its_own_file(self):
    """A different mesh scale is a different tree, so it gets a second file."""
    self.hits()
    self.hits(scale=(2, 2, 2))
    self.assertEqual(len(self.cache_files()), 2)

  def test_corrupt_file_is_rebuilt(self):
    """Garbage in the cache file is ignored, rebuilt over, and never changes the hits."""
    built = self.hits()
    path = self.cache_files()[0]
    with open(path, "wb") as f:
      f.write(b"not a tree")
    self.assertEqual(self.hits(), built)
    self.assertGreater(os.path.getsize(path), 64)
    self.assertEqual(self.hits(), built)


if __name__ == '__main__':
  unittest.main()
