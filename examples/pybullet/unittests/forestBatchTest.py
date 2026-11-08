"""A forest file draws every placement of its meshes as one body, exactly as the same placements as separate bodies."""
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import unittest

import numpy as np
import pybullet as p

from daylightTest import write_obj
from instancedStaticTest import CUTOUT, DAY, INSTANCED, MAP, RAY, frame

BINARY_ROWS = unittest.skipUnless(getattr(p, "FOREST_FILE_BINARY", 0), "wheel that reads text forest rows only")

# (mesh index, position, yaw in radians, scale) of each placement, with turns and uneven scales on purpose.
PLACEMENTS = [(0, (-2.0, -1.0, 0.4), 0.0, (1.0, 1.0, 1.0)), (0, (1.5, -1.5, 0.2), 0.7, (0.6, 1.3, 1.0)),
              (1, (0.0, 1.5, 0.5), 2.1, (1.0, 1.0, 1.8)), (1, (2.0, 1.0, 0.1), -1.2, (0.8, 0.8, 0.8)),
              (0, (-1.5, 2.0, 0.3), 3.0, (1.2, 0.7, 1.0))]


def write_meshes(folder):
  """Two small meshes: a flat card and a standing card, as the forest's two kinds of tree."""
  write_obj(os.path.join(folder, "flat.obj"), 0.5)
  write_obj(os.path.join(folder, "stand.obj"), 0.5, double_sided_quad=True)
  return ["flat.obj", "stand.obj"]


def write_forest(folder, meshes, placements, binary=False):
  """Write a forest file naming the meshes and listing each placement as index, xyz, quaternion and scale.

  binary writes the rows as doubles after a "binary" line, each the number its text form would parse to.
  """
  lines = ["# a comment line is skipped"] + ["mesh %s" % name for name in meshes]
  rows = []
  for index, position, yaw, scale in placements:
    rows.append("%d %f %f %f 0 0 %f %f %f %f %f" % ((index,) + tuple(position) + (math.sin(yaw / 2), math.cos(yaw / 2))
                                                    + tuple(scale)))
  path = os.path.join(folder, "forest.fst")
  with open(path, "wb") as out:
    if binary:
      out.write(("\n".join(lines + ["binary %d" % len(rows)]) + "\n").encode())
      out.write(np.array([[float(v) for v in row.split()] for row in rows], dtype="<f8").tobytes())
    else:
      out.write(("\n".join(lines + rows) + "\n").encode())
  return path


def ground():
  """A white receiver under the placements, so shadows land somewhere."""
  shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[5, 5, 0.05], rgbaColor=[1, 1, 1, 1], specularColor=[0, 0, 0])
  p.createMultiBody(0, -1, shape, basePosition=[0, 0, -0.05])


def forest_world(folder, base=(0, 0, 0), flags=INSTANCED, binary=False):
  """The placements as one forest body; returns its body id."""
  p.resetSimulation()
  ground()
  path = write_forest(folder, write_meshes(folder), PLACEMENTS, binary)
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


def identity_hash(folder):
  """Hash of the forest frame drawn with every picture flag, the frame the thread and process checks compare."""
  forest_world(folder)
  buffers = frame(RAY | MAP | DAY | CUTOUT | p.ER_EDGE_ANTIALIAS | p.ER_TEXTURE_FILTER, eye=(0, -6, 6))
  return hashlib.sha256(b"".join(a.tobytes() for a in buffers)).hexdigest()


def raster_hash():
  """Hash of the forest's colour, depth and mask on the raster path."""
  return hashlib.sha256(b"".join(a.tobytes() for a in frame(flags=0, shadow=0, eye=(0, -6, 6)))).hexdigest()


def files_gone_hashes(folder, how):
  """Raster hashes of a forest never ray-cast and of one rasterised after its first ray-cast frame lost its mesh files.

  how is "deleted" (files removed), "changed" (caching switched off, which empties the parse cache, then files
  rewritten) or "uncached" (caching off from the start, then files removed).
  """
  if how == "uncached":
    p.setPhysicsEngineParameter(enableFileCaching=0)
  forest_world(folder)
  fresh = raster_hash()
  forest_world(folder)
  frame(flags=RAY, eye=(0, -6, 6))
  if how == "changed":
    p.setPhysicsEngineParameter(enableFileCaching=0)
  for name in ("flat.obj", "stand.obj"):
    os.remove(os.path.join(folder, name))
  if how == "changed":
    write_obj(os.path.join(folder, "flat.obj"), 0.8)
    write_obj(os.path.join(folder, "stand.obj"), 0.8, double_sided_quad=True)
  return fresh, raster_hash()


def child(threads, mode="identity"):
  """Last JSON line printed by this file run in a fresh process at a thread count, in one of its modes."""
  env = dict(os.environ, SWARM_RENDER_THREADS=threads)
  output = subprocess.check_output([sys.executable, __file__, mode], env=env, text=True)
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

  @BINARY_ROWS
  def test_binary_rows_draw_the_text_rows_bytes(self):
    """Rows given as doubles draw the same colour, depth and mask as the same rows given as text."""
    forest_world(self.folder)
    text = frame(flags=RAY | MAP | DAY | CUTOUT, eye=(0, -6, 6))
    forest_world(self.folder, binary=True)
    binary = frame(flags=RAY | MAP | DAY | CUTOUT, eye=(0, -6, 6))
    for left, right in zip(text, binary):
      self.assertEqual(np.asarray(left).tobytes(), np.asarray(right).tobytes())

  @BINARY_ROWS
  def test_a_damaged_binary_file_draws_nothing_and_fails_nothing(self):
    """A row count past the end of the file refuses it, and rows whose mesh index is NaN or out of range are skipped."""
    meshes = write_meshes(self.folder)
    rows = np.array([[0, 0, 0, 0.5, 0, 0, 0, 1, 1, 1, 1]] * 3, dtype="<f8")
    rows[1, 0], rows[2, 0] = np.nan, 7.0
    for count, body in ((10 ** 15, rows), (2, rows[1:])):
      p.resetSimulation()
      path = os.path.join(self.folder, "damaged.fst")
      with open(path, "wb") as out:
        out.write(("".join("mesh %s\n" % name for name in meshes) + "binary %d\n" % count).encode() + body.tobytes())
      shape = p.createVisualShape(p.GEOM_MESH, fileName=path, flags=INSTANCED | p.VISUAL_SHAPE_DOUBLE_SIDED_MULTIBODY)
      p.createMultiBody(0, -1, shape)
      _, _, mask = frame(flags=RAY, shadow=0, eye=(0, -6, 6))
      self.assertEqual(int((mask >= 0).sum()), 0)

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

  def test_raster_after_the_ray_caster_draws_the_same(self):
    """A forest whose mesh the ray caster's tree took over still rasterises to the bytes of one never ray-cast."""
    forest_world(self.folder)
    frame(flags=RAY, eye=(0, -6, 6))
    after = frame(flags=0, shadow=0, eye=(0, -6, 6))
    forest_world(self.folder)
    fresh = frame(flags=0, shadow=0, eye=(0, -6, 6))
    for left, right in zip(after, fresh):
      self.assertEqual(np.asarray(left).tobytes(), np.asarray(right).tobytes())

  def test_raster_after_the_mesh_files_are_gone_draws_the_same(self):
    """Mesh files deleted, rewritten with the parse cache emptied, or never cached: the raster frame after a ray-cast
    frame still equals a never ray-cast forest's, drawn in a fresh process so a bad read fails this test alone."""
    for how in ("deleted", "changed", "uncached"):
      fresh, after = child("2", how)
      self.assertEqual(fresh, after, how)

  def test_rendering_while_another_client_changes_meshes_keeps_every_byte(self):
    """Forest release and refill during another client's mesh churn complete with the serial frame bytes."""
    forest_world(self.folder)
    expected = [tuple(a.tobytes() for a in frame(flags=flags, shadow=0, eye=(0, -6, 6), size=96))
                for flags in (RAY, 0)]
    other = p.connect(p.DIRECT)
    errors = []
    ready = threading.Barrier(2, timeout=10)
    go = threading.Event()

    def change_meshes():
      """Create shared and distinct meshes and remove their owners during each forest render pair."""
      try:
        for _ in range(40):
          ready.wait()
          if not go.wait(timeout=10):
            raise TimeoutError("forest renderer did not start")
          bodies = []
          for i in range(16):
            shape = p.createVisualShape(p.GEOM_MESH, fileName=os.path.join(self.folder, "flat.obj"),
                                        meshScale=[1 + (i % 4) * 0.125, 1, 1], physicsClientId=other)
            bodies.append(p.createMultiBody(0, -1, shape, physicsClientId=other))
          for body in bodies:
            p.removeBody(body, physicsClientId=other)
          p.resetSimulation(physicsClientId=other)
          ready.wait()
      except Exception as error:
        errors.append(error)
        ready.abort()

    worker = threading.Thread(target=change_meshes, daemon=True)
    worker.start()
    try:
      for _ in range(40):
        forest_world(self.folder)
        go.clear()
        ready.wait()
        interval = sys.getswitchinterval()
        sys.setswitchinterval(1000.0)
        try:
          go.set()
          for flags, alone in zip((RAY, 0), expected):
            actual = frame(flags=flags, shadow=0, eye=(0, -6, 6), size=96)
            self.assertEqual(tuple(a.tobytes() for a in actual), alone)
        finally:
          sys.setswitchinterval(interval)
        ready.wait()
    except BaseException:
      ready.abort()
      raise
    finally:
      worker.join(timeout=10)
      if not worker.is_alive():
        p.disconnect(physicsClientId=other)
    self.assertFalse(worker.is_alive(), "mesh worker did not finish")
    self.assertEqual(errors, [])

  def test_forest_added_again_draws_the_same(self):
    """A forest removed after its first frame and created again from the same file draws the first frame's bytes."""
    body = forest_world(self.folder)
    first = frame(flags=RAY | MAP, eye=(0, -6, 6))
    p.removeBody(body)
    frame(flags=RAY | MAP, eye=(0, -6, 6))
    shape = p.createVisualShape(p.GEOM_MESH, fileName=os.path.join(self.folder, "forest.fst"),
                                flags=INSTANCED | p.VISUAL_SHAPE_DOUBLE_SIDED_MULTIBODY, specularColor=[0, 0, 0])
    p.createMultiBody(0, -1, shape)
    again = frame(flags=RAY | MAP, eye=(0, -6, 6))
    for left, right in zip(first[:2], again[:2]):
      self.assertEqual(np.asarray(left).tobytes(), np.asarray(right).tobytes())

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

  def test_later_world_draws_what_a_fresh_process_draws(self):
    """A forest drawn after other worlds in the process, its mesh trees kept from them, gives a fresh process's bytes."""
    for _ in range(2):
      forest_world(self.folder)
      frame(flags=RAY, eye=(0, -6, 6))
      p.disconnect()
      p.connect(p.DIRECT)
    self.assertEqual(identity_hash(self.folder), child("2"))

  def test_threads_give_the_same_bytes(self):
    """Fresh processes render identical colour, depth and mask at one, two and four threads."""
    hashes = [child(t) for t in ("1", "2", "4")]
    self.assertEqual(hashes[0], hashes[1])
    self.assertEqual(hashes[0], hashes[2])


if __name__ == "__main__":
  if len(sys.argv) > 1 and sys.argv[1] == "identity":
    p.connect(p.DIRECT)
    with tempfile.TemporaryDirectory() as folder:
      print(json.dumps(identity_hash(folder)))
    p.disconnect()
  elif len(sys.argv) > 1:
    p.connect(p.DIRECT)
    with tempfile.TemporaryDirectory() as folder:
      print(json.dumps(files_gone_hashes(folder, sys.argv[1])))
    p.disconnect()
  else:
    unittest.main()
