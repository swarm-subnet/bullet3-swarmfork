"""resetMeshData on a visual mesh: a body's drawn surface takes new vertex positions each frame, keeping its faces,
uvs and texture, with normals rebuilt from the new shape. One body stays one surface, so an animal never has to be
cut into rigid pieces. Both colour paths, and the ray caster's tree follows the change."""
import math
import os
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pybullet as p

SIZE = 96
SUN = [0.2, -0.4, 0.9]
RAYCAST = getattr(p, "ER_SWARM_RAYCAST", 0)
PICTURE = (RAYCAST | getattr(p, "ER_SWARM_SHADOW_MAP", 0) | getattr(p, "ER_TEXTURE_FILTER", 0)
           | getattr(p, "ER_SWARM_LINEAR_LIGHT", 0))


def write_png(path, rgba):
  """Writes an RGBA uint8 array as a PNG with the standard library only."""
  import struct
  import zlib
  height, width = rgba.shape[:2]
  raw = b"".join(b"\x00" + rgba[y].tobytes() for y in range(height))
  def chunk(kind, data):
    """One PNG chunk with its CRC."""
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xffffffff)
  with open(path, "wb") as handle:
    handle.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
                 + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def grid(side, half):
  """A flat square of side by side vertices in the xy plane at z = 0, with uvs across it and its triangles."""
  xs = np.linspace(-half, half, side)
  vertices, uvs = [], []
  for j in range(side):
    for i in range(side):
      vertices.append([xs[i], xs[j], 0.0])
      uvs.append([i / (side - 1.0), j / (side - 1.0)])
  triangles = []
  for j in range(side - 1):
    for i in range(side - 1):
      a = j * side + i
      triangles += [[a, a + 1, a + side + 1], [a, a + side + 1, a + side]]
  normals = [[0.0, 0.0, 1.0]] * len(vertices)
  return vertices, normals, uvs, triangles


class TestVisualMesh(unittest.TestCase):
  """Renders a body whose mesh is rewritten between frames and checks what changed and what did not."""

  def setUp(self):
    """A DIRECT client, a camera looking down the scene, and a flat grid body."""
    p.connect(p.DIRECT)
    self.folder = tempfile.mkdtemp(prefix="visualmesh_")
    self.proj = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 100.0)
    self.vertices, self.normals, self.uvs, self.triangles = grid(9, 1.0)

  def tearDown(self):
    """Drops the client."""
    p.disconnect()

  def body(self, rgba=(1, 1, 1, 1)):
    """A visual-only body built from the grid arrays."""
    shape = p.createVisualShape(p.GEOM_MESH, vertices=self.vertices, indices=[i for t in self.triangles for i in t],
                                normals=self.normals, uvs=self.uvs, rgbaColor=list(rgba), specularColor=[0, 0, 0])
    return p.createMultiBody(0, -1, shape)

  def render(self, flags=0, eye=(0, -2.6, 1.6), target=(0, 0, 0)):
    """Returns (rgb, depth) of one frame."""
    view = p.computeViewMatrix(list(eye), list(target), [0, 0, 1])
    _, _, rgb, depth, _ = p.getCameraImage(SIZE, SIZE, view, self.proj, shadow=0, lightDirection=SUN,
                                           renderer=p.ER_TINY_RENDERER, flags=flags)
    return np.asarray(rgb).reshape(SIZE, SIZE, 4)[:, :, :3].astype(int), np.asarray(depth).reshape(SIZE, SIZE)

  def raised(self, height):
    """The grid with its middle vertex lifted."""
    moved = [list(v) for v in self.vertices]
    moved[len(moved) // 2][2] = height
    return moved

  def test_new_positions_change_the_picture(self):
    """Rewriting the vertices moves the drawn surface, on the ray-cast path and on the rasteriser."""
    for flags in (0, PICTURE):
      with self.subTest(flags=flags):
        p.resetSimulation()
        uid = self.body()
        flat, flat_depth = self.render(flags)
        p.resetMeshData(uid, self.raised(0.8))
        bumped, bumped_depth = self.render(flags)
        self.assertNotEqual(flat.tobytes(), bumped.tobytes())
        # Depth follows the surface: the buffer changes, and nothing behind the old one becomes the nearest thing seen.
        self.assertNotEqual(flat_depth.tobytes(), bumped_depth.tobytes())
        self.assertLessEqual(float(bumped_depth.min()), float(flat_depth.min()) + 1e-6)

  def test_a_wrong_vertex_count_is_refused(self):
    """A mesh of another size is left alone and the call raises, instead of half writing."""
    uid = self.body()
    before, _ = self.render(PICTURE)
    with self.assertRaises(p.error):
      p.resetMeshData(uid, self.vertices[:-3])
    after, _ = self.render(PICTURE)
    self.assertEqual(before.tobytes(), after.tobytes())

  def test_a_body_without_a_visual_mesh_is_refused(self):
    """Uploading to a body that has no mesh of that size fails rather than reporting success."""
    box = p.createMultiBody(0, -1, p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3] * 3))
    with self.assertRaises(p.error):
      p.resetMeshData(box, self.vertices)

  def test_two_bodies_from_one_mesh_move_apart(self):
    """Two bodies built from the same arrays share their mesh until one is rewritten; then only that one changes."""
    first = self.body()
    second = self.body()
    p.resetBasePositionAndOrientation(second, [2.2, 0, 0], [0, 0, 0, 1])
    eye, target = (1.1, -3.4, 1.7), (1.1, 0, 0)
    before, _ = self.render(PICTURE, eye, target)
    p.resetMeshData(first, self.raised(0.9))
    after, _ = self.render(PICTURE, eye, target)
    left_before, right_before = before[:, :SIZE // 2], before[:, SIZE // 2:]
    left_after, right_after = after[:, :SIZE // 2], after[:, SIZE // 2:]
    self.assertNotEqual(left_before.tobytes(), left_after.tobytes())
    self.assertEqual(right_before.tobytes(), right_after.tobytes())

  def test_texture_and_uvs_survive_the_rewrite(self):
    """The body keeps its texture and its uvs, so only the shape moves: the two halves stay their own colours."""
    texels = np.zeros((16, 16, 4), dtype=np.uint8)
    texels[:, :8] = [220, 30, 30, 255]
    texels[:, 8:] = [30, 30, 220, 255]
    path = os.path.join(self.folder, "halves.png")
    write_png(path, texels)
    uid = self.body()
    p.changeVisualShape(uid, -1, textureUniqueId=p.loadTexture(path))
    before, _ = self.render(PICTURE, (0, -0.1, 3.2), (0, 0, 0))
    p.resetMeshData(uid, self.raised(0.35))
    after, _ = self.render(PICTURE, (0, -0.1, 3.2), (0, 0, 0))
    for frame in (before, after):
      left = frame[:, 10:40].reshape(-1, 3).mean(axis=0)
      right = frame[:, SIZE - 40:SIZE - 10].reshape(-1, 3).mean(axis=0)
      self.assertGreater(left[0], left[2] + 30)
      self.assertGreater(right[2], right[0] + 30)

  def test_normals_are_rebuilt_from_the_new_shape(self):
    """A face that tilts is shaded differently afterwards, so the normals followed the positions."""
    uid = self.body()
    flat, _ = self.render(PICTURE)
    tilted = [[v[0], v[1], 0.35 * v[0]] for v in self.vertices]
    p.resetMeshData(uid, tilted)
    after, _ = self.render(PICTURE)
    self.assertGreater(abs(float(after.mean()) - float(flat.mean())), 1.0)

  def test_a_mover_keeps_up_with_its_mesh(self):
    """A body that has already moved still shows a rewritten mesh, so the ray caster refits its tree."""
    uid = self.body()
    self.render(PICTURE)
    p.resetBasePositionAndOrientation(uid, [0.2, 0.1, 0], [0, 0, 0, 1])
    moved, _ = self.render(PICTURE)
    p.resetMeshData(uid, self.raised(0.9))
    after, _ = self.render(PICTURE)
    self.assertNotEqual(moved.tobytes(), after.tobytes())

  def test_the_same_upload_twice_gives_the_same_bytes(self):
    """Rewriting with the same positions twice renders the same frame, so nothing accumulates."""
    uid = self.body()
    p.resetMeshData(uid, self.raised(0.6))
    once, _ = self.render(PICTURE)
    p.resetMeshData(uid, self.raised(0.6))
    twice, _ = self.render(PICTURE)
    self.assertEqual(once.tobytes(), twice.tobytes())

  def test_going_back_restores_the_first_frame(self):
    """The flat mesh renders the same bytes after a rewrite and a rewrite back."""
    uid = self.body()
    first, _ = self.render(PICTURE)
    p.resetMeshData(uid, self.raised(0.7))
    self.render(PICTURE)
    p.resetMeshData(uid, self.vertices)
    back, _ = self.render(PICTURE)
    self.assertEqual(first.tobytes(), back.tobytes())

  def test_threads_give_the_same_bytes(self):
    """The frame is byte-identical at every render thread count the build allows."""
    frames = []
    for threads in ("1", "2", "4"):
      os.environ["SWARM_RENDER_THREADS"] = threads
      p.disconnect()
      p.connect(p.DIRECT)
      uid = self.body()
      p.resetMeshData(uid, self.raised(0.8))
      rgb, _ = self.render(PICTURE | getattr(p, "ER_EDGE_ANTIALIAS", 0))
      frames.append(rgb.tobytes())
    self.assertEqual(frames[0], frames[1])
    self.assertEqual(frames[0], frames[2])

  def test_the_upload_is_released_after_the_call(self):
    """resetMeshData and createVisualShape let go of the lists they read, so a fresh upload every frame is freed."""
    uid = self.body()
    upload = self.raised(0.5)
    indices = [i for t in self.triangles for i in t]
    lists = (upload, upload[0], self.vertices, self.normals, self.uvs, self.uvs[0], indices)
    held = [sys.getrefcount(x) for x in lists]
    p.resetMeshData(uid, upload)
    p.createVisualShape(p.GEOM_MESH, vertices=self.vertices, indices=indices, normals=self.normals, uvs=self.uvs)
    self.assertEqual([sys.getrefcount(x) for x in lists], held)

  def test_an_array_upload_draws_the_same_bytes_as_a_list(self):
    """An N x 3 float64 or float32 array, contiguous or not, draws the colour and depth of the same numbers as a list."""
    raised = np.array(self.raised(0.8))
    single = raised.astype(np.float32)
    strided = np.hstack([raised, raised])[:, :3]
    groups = ((raised.tolist(), raised, strided), (single.tolist(), single))
    for flags in (0, PICTURE, RAYCAST | getattr(p, "ER_SWARM_THERMAL", 0)):
      for group in groups:
        frames = []
        for upload in group:
          p.resetSimulation()
          uid = self.body()
          self.render(flags)
          p.resetMeshData(uid, upload)
          rgb, depth = self.render(flags)
          frames.append(rgb.tobytes() + depth.tobytes())
        with self.subTest(flags=flags, dtype=str(group[1].dtype)):
          self.assertEqual(len(set(frames)), 1)

  def test_a_wrong_array_is_refused(self):
    """An array with the wrong vertex count, or two numbers per vertex, raises as the list does."""
    uid = self.body()
    with self.assertRaises(p.error):
      p.resetMeshData(uid, np.array(self.vertices)[:-3])
    with self.assertRaises(p.error):
      p.resetMeshData(uid, np.array(self.vertices)[:, :2])

  def test_array_uploads_give_the_same_bytes_at_every_thread_count(self):
    """Array uploads render byte-identical frames at 1, 2 and 4 render threads, each count in its own process."""
    frames = []
    for threads in ("1", "2", "4"):
      out = os.path.join(self.folder, "frame_%s.npy" % threads)
      env = dict(os.environ, SWARM_RENDER_THREADS=threads)
      subprocess.check_call([sys.executable, os.path.abspath(__file__), "--frame", out], env=env,
                            cwd=os.path.dirname(os.path.abspath(__file__)))
      frames.append(np.load(out))
    self.assertTrue(np.array_equal(frames[0], frames[1]))
    self.assertTrue(np.array_equal(frames[0], frames[2]))


def save_frame(path):
  """Rewrites a grid from a float32 and then a float64 array, rendering each, and saves the frames for the thread test."""
  p.connect(p.DIRECT)
  vertices, normals, uvs, triangles = grid(9, 1.0)
  shape = p.createVisualShape(p.GEOM_MESH, vertices=vertices, indices=[i for t in triangles for i in t],
                              normals=normals, uvs=uvs, rgbaColor=[1, 1, 1, 1], specularColor=[0, 0, 0])
  uid = p.createMultiBody(0, -1, shape)
  view = p.computeViewMatrix([0, -2.6, 1.6], [0, 0, 0], [0, 0, 1])
  proj = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 100.0)
  frames = []
  for height, dtype in ((0.8, np.float32), (0.4, np.float64)):
    raised = np.array(vertices, dtype=dtype)
    raised[len(raised) // 2, 2] = height
    p.resetMeshData(uid, raised)
    _, _, rgb, depth, _ = p.getCameraImage(SIZE, SIZE, view, proj, shadow=0, lightDirection=SUN,
                                           renderer=p.ER_TINY_RENDERER,
                                           flags=PICTURE | getattr(p, "ER_EDGE_ANTIALIAS", 0))
    frames.append(np.concatenate([np.asarray(rgb, dtype=np.float64).ravel(), np.asarray(depth, dtype=np.float64).ravel()]))
  np.save(path, np.stack(frames))
  p.disconnect()


if __name__ == '__main__':
  if len(sys.argv) == 3 and sys.argv[1] == "--frame":
    save_frame(sys.argv[2])
  else:
    unittest.main()
