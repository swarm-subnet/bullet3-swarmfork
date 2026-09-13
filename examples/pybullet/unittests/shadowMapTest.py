"""Sun shadows from the map on the ray-cast path: the same shadow as the ray, follows the light, no stale shadow, identical bytes at any thread count."""
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pybullet as p

from raycastColourTest import LIGHT, SIZE, build_world, write_checker_tga

NEEDS_MAP = unittest.skipUnless(hasattr(p, "ER_SWARM_SHADOW_MAP"), "wheel built without the shadow map")
RAY = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX | getattr(p, "ER_SWARM_RAYCAST", 0)
MAP = RAY | getattr(p, "ER_SWARM_SHADOW_MAP", 0)
BOX = 1  # body index of the red box in build_world
MIRRORED = [-LIGHT[0], LIGHT[1], LIGHT[2]]  # the sun on the other side of the box


def render(flags, shadow, light=LIGHT, eye=(3.0, -3.0, 2.5), target=(0, 0, 0.4)):
  """Returns (rgb HxWx3 as int, seg HxW) from a camera at eye looking at target under the given light."""
  view = p.computeViewMatrix(list(eye), list(target), [0, 0, 1])
  proj = p.computeProjectionMatrixFOV(70, 1.0, 0.1, 30.0)
  _, _, rgb, _, seg = p.getCameraImage(SIZE, SIZE, view, proj, shadow=shadow, lightDirection=list(light),
                                       renderer=p.ER_TINY_RENDERER, flags=flags)
  return np.asarray(rgb).reshape(SIZE, SIZE, 4)[:, :, :3].astype(int), np.asarray(seg).reshape(SIZE, SIZE)


def darker(a, b):
  """Pixels where frame a is darker than frame b in any channel."""
  return (a < b).any(axis=2)


@NEEDS_MAP
class TestShadowMap(unittest.TestCase):
  """Renders the colour test scene with the shadow ray and with the shadow map and compares them."""

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

  def test_map_agrees_with_the_shadow_ray_away_from_shadow_edges(self):
    """The map darkens the same floor as the ray and never brightens a pixel; the pixels that differ are a thin edge band."""
    lit, seg = render(RAY, 0)
    ray, _ = render(RAY, 1)
    mapped, _ = render(MAP, 1)
    self.assertFalse((mapped > lit).any())
    self.assertGreater(int((darker(mapped, lit) & (seg == 0)).sum()), 20)
    differ = (ray != mapped).any(axis=2)
    self.assertLess(int(differ.sum()), SIZE * SIZE // 100)

  def test_map_follows_the_light(self):
    """A second sun direction moves the shadow, and the map agrees with the ray under it as well."""
    lit_a, seg = render(RAY, 0)
    map_a, _ = render(MAP, 1)
    lit_b, _ = render(RAY, 0, light=MIRRORED)
    ray_b, _ = render(RAY, 1, light=MIRRORED)
    map_b, _ = render(MAP, 1, light=MIRRORED)
    floor = seg == 0
    shadow_a = darker(map_a, lit_a) & floor
    shadow_b = darker(map_b, lit_b) & floor
    self.assertGreater(int(shadow_b.sum()), 20)
    self.assertGreater(int((shadow_a ^ shadow_b).sum()), 20)
    self.assertLess(int((ray_b != map_b).any(axis=2).sum()), SIZE * SIZE // 100)

  def test_moved_body_drops_its_old_shadow_and_casts_none(self):
    """Once the box moves, its cells are recast so its old shadow goes, and as a mover it throws no new one."""
    before, seg0 = render(MAP, 1)
    lit0, _ = render(RAY, 0)
    old_shadow = darker(before, lit0) & (seg0 == 0)
    p.resetBasePositionAndOrientation(BOX, [-1.0, 1.5, 0.5], [0, 0, 0, 1])
    lit, seg = render(RAY, 0)
    ray, _ = render(RAY, 1)
    mapped, _ = render(MAP, 1)
    released = old_shadow & (seg == 0) & ~darker(mapped, lit)
    self.assertGreater(int(released.sum()), 20)
    self.assertTrue((mapped[released] == lit[released]).all())
    kept = old_shadow & darker(mapped, lit)
    self.assertTrue(darker(ray, lit)[kept].all())
    mover_shadow = darker(ray, lit) & ~darker(mapped, lit) & (seg == 0)
    self.assertGreater(int(mover_shadow.sum()), 20)

  def test_hidden_body_leaves_the_map_and_returns_when_shown(self):
    """Alpha zero recasts the box's cells so its shadow goes; alpha one brings the very same bytes back."""
    lit, seg = render(RAY, 0)
    shown, _ = render(MAP, 1)
    p.changeVisualShape(BOX, -1, rgbaColor=[1, 0, 0, 0])
    hidden, seg_hidden = render(MAP, 1)
    released = darker(shown, lit) & (seg == 0) & (seg_hidden == 0) & ~darker(hidden, lit)
    self.assertGreater(int(released.sum()), 20)
    p.changeVisualShape(BOX, -1, rgbaColor=[1, 0, 0, 1])
    again, _ = render(MAP, 1)
    self.assertEqual(again.tobytes(), shown.tobytes())

  def test_flag_alone_changes_nothing(self):
    """Without shadow=1 the flag is inert, and the rasterised path ignores it."""
    self.assertEqual(render(MAP, 0)[0].tobytes(), render(RAY, 0)[0].tobytes())
    tiny = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
    self.assertEqual(render(tiny | p.ER_SWARM_SHADOW_MAP, 1)[0].tobytes(), render(tiny, 1)[0].tobytes())


@NEEDS_MAP
class TestShadowMapThreads(unittest.TestCase):
  """The map and the frames read from it do not depend on how many render threads are used."""

  def test_same_bytes_for_one_two_and_four_threads(self):
    """Renders the scene in a fresh process per thread count and compares the colour hashes."""
    digests = set()
    for threads in ("1", "2", "4"):
      env = dict(os.environ, SWARM_RENDER_THREADS=threads)
      out = subprocess.check_output([sys.executable, __file__, "--hash"], env=env, text=True)
      digests.add(out.strip())
    self.assertEqual(len(digests), 1)


def shadow_hash():
  """Prints the sha256 of two map-shadowed frames: one from the full map, one after the box moved and its cells were recast."""
  p.connect(p.DIRECT)
  tmp = tempfile.mkdtemp()
  tex_path = os.path.join(tmp, 'checker.tga')
  write_checker_tga(tex_path)
  build_world(tex_path)
  digest = hashlib.sha256()
  digest.update(render(MAP, 1)[0].astype(np.uint8).tobytes())
  p.resetBasePositionAndOrientation(BOX, [-1.0, 1.5, 0.5], [0, 0, 0, 1])
  digest.update(render(MAP, 1)[0].astype(np.uint8).tobytes())
  print(digest.hexdigest())
  p.disconnect()
  os.remove(tex_path)
  os.rmdir(tmp)


if __name__ == '__main__':
  if "--hash" in sys.argv:
    shadow_hash()
  else:
    unittest.main()
