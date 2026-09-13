"""Shadows of moving bodies on the ray-cast path: a mover casts the shadow the ray would, it follows the body, a hidden mover casts none, the flag alone changes nothing, identical bytes at any thread count."""
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pybullet as p

from raycastColourTest import LIGHT, SIZE, build_world, write_checker_tga
from shadowMapTest import BOX, MAP, RAY, darker, render

NEEDS_MOVER = unittest.skipUnless(hasattr(p, "ER_SWARM_MOVER_SHADOW"), "wheel built without mover shadows")
MOVER = MAP | getattr(p, "ER_SWARM_MOVER_SHADOW", 0)
MOVED = [-1.0, 1.5, 0.5]  # where the box goes once it moves: its shadow falls on open floor in view
MOVED_AGAIN = [-1.5, -1.0, 0.5]  # a second place whose shadow is in view even with a box still at the origin


def make_mover(position):
  """Renders one frame so the box enters the static tree at its start pose, then moves it, which makes it a mover."""
  render(MAP, 1)
  p.resetBasePositionAndOrientation(BOX, position, [0, 0, 0, 1])


@NEEDS_MOVER
class TestMoverShadow(unittest.TestCase):
  """Renders the colour test scene with the shadow map alone and with mover shadows on top, against the shadow ray."""

  def setUp(self):
    """Connects and builds the scene with its checker texture."""
    p.connect(p.DIRECT)
    self.tmp = tempfile.mkdtemp()
    self.tex_path = os.path.join(self.tmp, 'checker.tga')
    write_checker_tga(self.tex_path)
    build_world(self.tex_path)

  def tearDown(self):
    """Disconnects and removes the temporary texture."""
    p.disconnect()
    os.remove(self.tex_path)
    os.rmdir(self.tmp)

  def assert_mover_shadow(self, lit, seg):
    """The mover-shadowed frame darkens floor the map leaves lit, never brightens a pixel, and agrees with the ray away from the edge band."""
    ray, _ = render(RAY, 1)
    mapped, _ = render(MAP, 1)
    mover, _ = render(MOVER, 1)
    self.assertFalse((mover > lit).any())
    added = darker(mover, lit) & ~darker(mapped, lit) & (seg == 0)
    self.assertGreater(int(added.sum()), 20)
    self.assertTrue(darker(ray, lit)[added].all())
    self.assertLess(int((ray != mover).any(axis=2).sum()), SIZE * SIZE // 100)
    return mover

  def test_moved_body_casts_a_shadow_that_follows_it(self):
    """Once the box moves it is a mover: with the flag it throws the ray's shadow at its new place, and again after a second move."""
    make_mover(MOVED)
    lit_a, seg_a = render(RAY, 0)
    mover_a = self.assert_mover_shadow(lit_a, seg_a)
    p.resetBasePositionAndOrientation(BOX, MOVED_AGAIN, [0, 0, 0, 1])
    lit_b, seg_b = render(RAY, 0)
    mover_b = self.assert_mover_shadow(lit_b, seg_b)
    shadow_a = darker(mover_a, lit_a) & (seg_a == 0)
    shadow_b = darker(mover_b, lit_b) & (seg_b == 0)
    self.assertGreater(int((shadow_a ^ shadow_b).sum()), 20)

  def test_body_added_after_the_world_was_built_casts_a_shadow(self):
    """A body created after the first frame never enters the static tree, so only the mover flag gives it a shadow."""
    render(MAP, 1)
    shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], rgbaColor=[0, 1, 0, 1])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=shape, basePosition=MOVED_AGAIN)
    lit, seg = render(RAY, 0)
    self.assert_mover_shadow(lit, seg)

  def test_hidden_mover_casts_none_and_returns_when_shown(self):
    """Alpha zero takes the mover out of the shadow tree, so its frame equals the map's; alpha one brings the very same bytes back."""
    make_mover(MOVED)
    shown, _ = render(MOVER, 1)
    self.assertNotEqual(shown.tobytes(), render(MAP, 1)[0].tobytes())
    p.changeVisualShape(BOX, -1, rgbaColor=[1, 0, 0, 0])
    hidden, _ = render(MOVER, 1)
    mapped, _ = render(MAP, 1)
    self.assertEqual(hidden.tobytes(), mapped.tobytes())
    self.assertNotEqual(hidden.tobytes(), shown.tobytes())
    p.changeVisualShape(BOX, -1, rgbaColor=[1, 0, 0, 1])
    self.assertEqual(render(MOVER, 1)[0].tobytes(), shown.tobytes())

  def test_flag_alone_changes_nothing(self):
    """With no mover in the world the flag gives the map's bytes; without the map, without shadow=1 or on the rasterised path it is inert."""
    self.assertEqual(render(MOVER, 1)[0].tobytes(), render(MAP, 1)[0].tobytes())
    self.assertEqual(render(MOVER, 0)[0].tobytes(), render(RAY, 0)[0].tobytes())
    make_mover(MOVED)
    no_map = RAY | p.ER_SWARM_MOVER_SHADOW
    self.assertEqual(render(no_map, 1)[0].tobytes(), render(RAY, 1)[0].tobytes())
    tiny = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
    self.assertEqual(render(tiny | p.ER_SWARM_MOVER_SHADOW, 1)[0].tobytes(), render(tiny, 1)[0].tobytes())


@NEEDS_MOVER
class TestMoverShadowThreads(unittest.TestCase):
  """The mover shadow does not depend on how many render threads are used."""

  def test_same_bytes_for_one_two_and_four_threads(self):
    """Renders the scene in a fresh process per thread count and compares the colour hashes."""
    digests = set()
    for threads in ("1", "2", "4"):
      env = dict(os.environ, SWARM_RENDER_THREADS=threads)
      out = subprocess.check_output([sys.executable, __file__, "--hash"], env=env, text=True)
      digests.add(out.strip())
    self.assertEqual(len(digests), 1)


def mover_hash():
  """Prints the sha256 of two mover-shadowed frames: after the box moved, and after it moved again."""
  p.connect(p.DIRECT)
  tmp = tempfile.mkdtemp()
  tex_path = os.path.join(tmp, 'checker.tga')
  write_checker_tga(tex_path)
  build_world(tex_path)
  digest = hashlib.sha256()
  make_mover(MOVED)
  digest.update(render(MOVER, 1)[0].astype(np.uint8).tobytes())
  p.resetBasePositionAndOrientation(BOX, MOVED_AGAIN, [0, 0, 0, 1])
  digest.update(render(MOVER, 1)[0].astype(np.uint8).tobytes())
  print(digest.hexdigest())
  p.disconnect()
  os.remove(tex_path)
  os.rmdir(tmp)


if __name__ == '__main__':
  if "--hash" in sys.argv:
    mover_hash()
  else:
    unittest.main()
