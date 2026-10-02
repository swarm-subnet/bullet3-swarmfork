"""ER_SWARM_FRAME_REUSE: a still camera gets its last frame back, and every frame it gets is the one it would have drawn:
after a mover goes by out of sight and shade, after one comes into view or throws its shadow in, after a colour change,
with new grain at night and in thermal, and with the same bytes at any thread count."""
import hashlib
import math
import os
import subprocess
import sys
import time
import unittest

import numpy as np
import pybullet as p

SIZE = 96
REUSE = getattr(p, "ER_SWARM_FRAME_REUSE", 0)
NEEDS_FLAG = unittest.skipUnless(REUSE, "wheel built without frame reuse")
PICTURE = (getattr(p, "ER_SWARM_RAYCAST", 0) | getattr(p, "ER_SWARM_SHADOW_MAP", 0) | getattr(p, "ER_SWARM_MOVER_SHADOW", 0) |
           getattr(p, "ER_EDGE_ANTIALIAS", 0) | getattr(p, "ER_ALPHA_CUTOUT", 0) | getattr(p, "ER_TEXTURE_FILTER", 0))
LOW_LIGHT = getattr(p, "ER_SWARM_RAYCAST", 0) | getattr(p, "ER_SWARM_LINEAR_LIGHT", 0) | getattr(p, "ER_SWARM_LOW_LIGHT", 0)
THERMAL = getattr(p, "ER_SWARM_RAYCAST", 0) | getattr(p, "ER_SWARM_THERMAL", 0) | getattr(p, "ER_ALPHA_CUTOUT", 0)
LIGHT = [0.6, 0.0, 0.8]
EYE = [0.0, 0.0, 4.0]
FAR_AWAY = [[8.0, -8.0, 0.3], [8.5, -9.0, 0.3], [7.0, -10.0, 0.6]]  # out of view, and off the light's way from it
IN_VIEW = [0.8, 0.8, 0.3]
SHADING = [2.5, 0.5, 4.0]  # above the camera's eye, out of view, on the light's way from the floor at (-0.5, 0.5)


def build_world():
  """A grey floor, a red box on it, and a small blue box that becomes a mover once it moves; returns the blue box."""
  floor = p.createVisualShape(p.GEOM_MESH, vertices=[[-6, -6, 0], [6, -6, 0], [6, 6, 0], [-6, 6, 0]],
                              indices=[0, 1, 2, 0, 2, 3], normals=[[0, 0, 1]] * 4, rgbaColor=[0.7, 0.7, 0.7, 1])
  p.createMultiBody(baseMass=0, baseVisualShapeIndex=floor)
  box = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.4], rgbaColor=[1, 0, 0, 1])
  p.createMultiBody(baseMass=0, baseVisualShapeIndex=box, basePosition=[0.6, -0.6, 0.4])
  small = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.3], rgbaColor=[0, 0, 1, 1])
  return p.createMultiBody(baseMass=0, baseVisualShapeIndex=small, basePosition=FAR_AWAY[0])


def render(flags, size=SIZE, eye=EYE, **extra):
  """Colour, depth and mask bytes of one frame straight down from eye, as one bytes string."""
  view = p.computeViewMatrix(list(eye), [eye[0], eye[1], 0.0], [0, 1, 0])
  proj = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 30.0)
  _, _, rgb, depth, seg = p.getCameraImage(size, size, view, proj, lightDirection=LIGHT, renderer=p.ER_TINY_RENDERER,
                                           flags=flags, **extra)
  return (np.asarray(rgb, dtype=np.uint8).tobytes() + np.asarray(depth, dtype=np.float32).tobytes() +
          np.asarray(seg, dtype=np.int32).tobytes())


def both(flags, **extra):
  """The frame asked with frame reuse, and the same frame drawn without it."""
  return render(flags | REUSE, **extra), render(flags, **extra)


def move(body, position):
  """Puts a body at a position, unturned."""
  p.resetBasePositionAndOrientation(body, position, [0, 0, 0, 1])


def sequence(body):
  """Every frame of the scripted run with frame reuse on, in order: still, movers away and near, a new colour, grain."""
  frames = [render(PICTURE | REUSE, shadow=1) for _ in range(3)]
  for position in FAR_AWAY + [IN_VIEW, IN_VIEW, SHADING, SHADING]:
    move(body, position)
    frames.append(render(PICTURE | REUSE, shadow=1))
    frames.append(render(PICTURE | REUSE, shadow=1))
  p.changeVisualShape(body, -1, rgbaColor=[0, 1, 0, 1])
  frames.append(render(PICTURE | REUSE, shadow=1))
  for seed in (1, 2, 3):
    frames.append(render(LOW_LIGHT | REUSE, sensorPhotons=40.0, sensorReadNoise=2.0, sensorGainCap=400.0, sensorSeed=seed))
  for seed in (1, 2, 3):
    frames.append(render(THERMAL | REUSE, airTemperature=15.0, skyTemperature=-20.0, thermalSeed=seed))
  return frames


@NEEDS_FLAG
class TestFrameReuse(unittest.TestCase):
  """Each frame asked with the flag against the same frame drawn without it, through the changes reuse must notice."""

  def setUp(self):
    """Connects, builds the scene, draws it once so the boxes join the static tree, and moves the blue box once."""
    p.connect(p.DIRECT)
    self.body = build_world()
    render(PICTURE, shadow=1)
    move(self.body, FAR_AWAY[1])

  def tearDown(self):
    """Disconnects."""
    p.disconnect()

  def test_still_camera_gets_its_frame(self):
    """The same request again and again gives the bytes the frame is drawn with."""
    first, plain = both(PICTURE, shadow=1)
    for _ in range(3):
      self.assertEqual(render(PICTURE | REUSE, shadow=1), first)
    self.assertEqual(first, plain)

  def test_mover_out_of_sight_and_shade_changes_nothing(self):
    """A mover that moves where no ray of the frame goes, nor the light's way back from what they met, leaves it as drawn."""
    kept, _ = both(PICTURE, shadow=1)
    kept, _ = both(PICTURE, shadow=1)
    for position in FAR_AWAY:
      move(self.body, position)
      reused, plain = both(PICTURE, shadow=1)
      self.assertEqual(reused, plain)
      self.assertEqual(reused, kept)

  def test_mover_coming_into_view_is_drawn(self):
    """A mover moving into view shows at once, and the frame after it too."""
    before, _ = both(PICTURE, shadow=1)
    before, _ = both(PICTURE, shadow=1)
    move(self.body, IN_VIEW)
    for _ in range(2):
      reused, plain = both(PICTURE, shadow=1)
      self.assertEqual(reused, plain)
      self.assertNotEqual(reused, before)

  def test_mover_shadow_from_out_of_view_is_drawn(self):
    """A mover out of view whose shadow falls into it changes the frame as the shadow does."""
    before, _ = both(PICTURE, shadow=1)
    before, _ = both(PICTURE, shadow=1)
    move(self.body, SHADING)
    reused, plain = both(PICTURE, shadow=1)
    self.assertEqual(reused, plain)
    self.assertNotEqual(reused, before)
    move(self.body, FAR_AWAY[0])
    reused, plain = both(PICTURE, shadow=1)
    self.assertEqual(reused, plain)
    self.assertEqual(reused, before)

  def test_new_colour_is_drawn(self):
    """A colour change of a body in view gives the new colour."""
    move(self.body, IN_VIEW)
    before, _ = both(PICTURE, shadow=1)
    before, _ = both(PICTURE, shadow=1)
    p.changeVisualShape(self.body, -1, rgbaColor=[0, 1, 0, 1])
    reused, plain = both(PICTURE, shadow=1)
    self.assertEqual(reused, plain)
    self.assertNotEqual(reused, before)

  def test_moved_camera_draws_anew(self):
    """A camera moved by a millimetre draws its own frame."""
    both(PICTURE, shadow=1)
    both(PICTURE, shadow=1)
    reused, plain = both(PICTURE, shadow=1, eye=[0.001, 0.0, 4.0])
    self.assertEqual(reused, plain)

  @unittest.skipUnless(getattr(p, "ER_SWARM_LOW_LIGHT", 0), "wheel built without the low-light camera")
  def test_night_grain_is_new_every_frame(self):
    """At night a still camera still gets the grain of each request's seed."""
    frames = []
    for seed in (1, 1, 2, 3):
      reused, plain = both(LOW_LIGHT, sensorPhotons=40.0, sensorReadNoise=2.0, sensorGainCap=400.0, sensorSeed=seed)
      self.assertEqual(reused, plain)
      frames.append(reused)
    self.assertEqual(len(set(frames)), 3)

  @unittest.skipUnless(getattr(p, "ER_SWARM_THERMAL", 0), "wheel built without thermal")
  def test_thermal_grain_is_new_every_frame(self):
    """A still thermal camera gets the grain of each request's seed."""
    frames = []
    for seed in (1, 1, 2, 3):
      reused, plain = both(THERMAL, airTemperature=15.0, skyTemperature=-20.0, thermalSeed=seed)
      self.assertEqual(reused, plain)
      frames.append(reused)
    self.assertEqual(len(set(frames)), 3)

  def test_kept_frame_skips_the_tracing(self):
    """A large frame asked again costs a fraction of drawing it: the frame really is handed back."""
    def spent(eye):
      """Seconds one request from eye takes."""
      start = time.perf_counter()
      render(PICTURE | REUSE, size=320, eye=eye, shadow=1)
      return time.perf_counter() - start
    fresh = min(spent([0.001 * k, 0.0, 4.0]) for k in range(1, 6))
    spent(EYE)
    spent(EYE)
    still = min(spent(EYE) for _ in range(5))
    self.assertLess(still * 3.0, fresh)


TAN = math.tan(math.radians(30.0))
ROW = 48
ROW_NDC = 1.0 - (2.0 * ROW + 2.0) / SIZE
# Where the edge pass probes the uncovered fifth of the left column's square at ROW, half a pixel past the frame's side.
STRIP = [25.0, 25.0 * TAN * (1.0 + 0.8 / SIZE), 1.0 + 25.0 * TAN * ROW_NDC]


def border_world():
  """A plate 2 m ahead of an eye at (0, 0, 1) looking along +x, whose corner covers the right four fifths of the left
  column's square at ROW and nothing above it, so that pixel's rest is probed past the frame's left side; and a 2 cm
  cube far away that becomes a mover once it moves. Returns the cube."""
  edge = 2.0 * TAN * (1.0 + 0.6 / SIZE)
  top = 1.0 + 2.0 * TAN * (ROW_NDC + 1.5 / SIZE)
  plate = p.createVisualShape(p.GEOM_MESH, vertices=[[2, edge, -1], [2, -3, -1], [2, -3, top], [2, edge, top]],
                              indices=[0, 1, 2, 0, 2, 3], normals=[[-1, 0, 0]] * 4, rgbaColor=[0.2, 0.6, 0.2, 1])
  p.createMultiBody(baseMass=0, baseVisualShapeIndex=plate)
  cube = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.01, 0.01, 0.01], rgbaColor=[1, 0, 1, 1])
  return p.createMultiBody(baseMass=0, baseVisualShapeIndex=cube, basePosition=[60, -60, 0])


def along_x(flags):
  """The frame seen from (0, 0, 1) along +x, colour, depth and mask as one bytes string."""
  view = p.computeViewMatrix([0, 0, 1], [1, 0, 1], [0, 0, 1])
  proj = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 30.0)
  _, _, rgb, depth, seg = p.getCameraImage(SIZE, SIZE, view, proj, lightDirection=LIGHT, renderer=p.ER_TINY_RENDERER,
                                           flags=flags, shadow=1)
  return (np.asarray(rgb, dtype=np.uint8).tobytes() + np.asarray(depth, dtype=np.float32).tobytes() +
          np.asarray(seg, dtype=np.int32).tobytes())


@NEEDS_FLAG
class TestFrameReuseBorder(unittest.TestCase):
  """Edge-pass probes on the left column and bottom row reach half a pixel past the frame's side."""

  def setUp(self):
    """Connects, builds the plate and the cube, draws once so both join the static tree, and moves the cube once."""
    p.connect(p.DIRECT)
    self.cube = border_world()
    along_x(PICTURE)
    move(self.cube, [60, -61, 0])

  def tearDown(self):
    """Disconnects."""
    p.disconnect()

  def test_mover_met_only_by_a_border_probe(self):
    """A cube only the left column's probe meets shows in the frame, and leaving takes it out of the frame."""
    empty = along_x(PICTURE)
    move(self.cube, STRIP)
    for _ in range(3):
      self.assertEqual(along_x(PICTURE | REUSE), along_x(PICTURE))
    seen = along_x(PICTURE)
    self.assertNotEqual(seen, empty)
    move(self.cube, [60, -60, 0])
    self.assertEqual(along_x(PICTURE | REUSE), along_x(PICTURE))

  def test_mover_crossing_the_side_of_the_frame(self):
    """A cube stepping a centimetre at a time out through the frame's left side gives the frame drawn without reuse at
    every step."""
    along_x(PICTURE | REUSE)
    for step in range(40):
      move(self.cube, [STRIP[0], STRIP[1] - 0.2 + 0.01 * step, STRIP[2]])
      for _ in range(2):
        self.assertEqual(along_x(PICTURE | REUSE), along_x(PICTURE), step)


@NEEDS_FLAG
class TestFrameReuseThreads(unittest.TestCase):
  """The scripted run in its own process at 1, 2 and 4 render threads."""

  def test_same_bytes_for_one_two_and_four_threads(self):
    """Every frame of the run hashes the same at every thread count."""
    digests = set()
    for threads in ("1", "2", "4"):
      env = dict(os.environ, SWARM_RENDER_THREADS=threads)
      out = subprocess.check_output([sys.executable, __file__, "--hash"], env=env, text=True)
      digests.add(out.strip())
    self.assertEqual(len(digests), 1)


def run_hash():
  """Prints the sha256 of every frame of the scripted run."""
  p.connect(p.DIRECT)
  body = build_world()
  render(PICTURE, shadow=1)
  move(body, FAR_AWAY[1])
  digest = hashlib.sha256()
  for frame in sequence(body):
    digest.update(frame)
  print(digest.hexdigest())
  p.disconnect()


if __name__ == '__main__':
  if "--hash" in sys.argv:
    run_hash()
  else:
    unittest.main()
