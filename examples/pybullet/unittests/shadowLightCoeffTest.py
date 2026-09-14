"""Shadow strength on the ray-cast path: the default keeps the rasteriser's floor, zero leaves only the ambient light in shadow from every shadow source, the rasterised path ignores it, identical bytes at any thread count."""
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pybullet as p

from raycastColourTest import SIZE, build_world, render, write_checker_tga
from shadowMapTest import MAP, RAY, darker
from moverShadowTest import MOVED, MOVER, make_mover

NEEDS_MOVER = unittest.skipUnless(hasattr(p, "ER_SWARM_MOVER_SHADOW"), "wheel built without mover shadows")
FLOOR = 0.8  # the share of the direct light TinyRenderer's shader keeps where its shadow buffer says blocked
TINY = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX


def shaded(flags, coeff):
  """Returns (rgb, seg) with shadow=1 and the given share of direct light in shadow; a negative coeff leaves the server's value alone."""
  rgb, _, seg = render(flags, 1, shadowLightCoeff=coeff, lightDiffuseCoeff=0.35, lightSpecularCoeff=0.05)
  return rgb, seg


def ambient_only():
  """Returns the scene lit by the ambient term alone, which is what a fully shadowed surface must show."""
  return render(RAY, 0, lightDiffuseCoeff=0.0, lightSpecularCoeff=0.0)[0]


@NEEDS_MOVER
class TestShadowLightCoeff(unittest.TestCase):
  """Renders the colour test scene with the shadow ray, the shadow map and the mover pass at the default and at zero."""

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

  def assert_full_shadow(self, flags):
    """At zero the pixels that change against the default all show the ambient-only colour, nothing brightens, and the shadow is there; returns the changed mask."""
    ambient = ambient_only()
    default, _ = shaded(flags, FLOOR)
    full, _ = shaded(flags, 0.0)
    changed = (full != default).any(axis=2)
    self.assertGreater(int(changed.sum()), 20)
    self.assertTrue((full[changed] == ambient[changed]).all())
    self.assertFalse((full > default).any())
    return changed

  def test_default_is_the_rasteriser_floor(self):
    """A render that never set the coefficient gives the bytes of an explicit 0.8, and that frame does carry a shadow."""
    untouched, seg = shaded(RAY, -1.0)
    self.assertEqual(untouched.tobytes(), shaded(RAY, FLOOR)[0].tobytes())
    lit = render(RAY, 0)[0]
    self.assertGreater(int((darker(untouched, lit) & (seg == 0)).sum()), 20)

  def test_zero_leaves_only_ambient_light_from_the_ray_and_the_map(self):
    """The shadow ray and the shadow map both hand the coefficient the same blocked pixels: at zero they are ambient only."""
    self.assert_full_shadow(RAY)
    self.assert_full_shadow(MAP)

  def test_zero_leaves_only_ambient_light_under_a_mover(self):
    """A mover's shadow, which the map cannot hold, takes the coefficient too: at zero it adds ambient-only pixels the map alone leaves lit."""
    make_mover(MOVED)
    from_map = self.assert_full_shadow(MAP)
    from_mover = self.assert_full_shadow(MOVER)
    self.assertGreater(int((from_mover & ~from_map).sum()), 20)

  def test_inert_without_shadow_and_on_the_rasterised_path(self):
    """With shadow=0 the coefficient changes nothing, and TinyRenderer keeps its own floor whatever is passed."""
    self.assertEqual(render(RAY, 0, shadowLightCoeff=0.0)[0].tobytes(), render(RAY, 0)[0].tobytes())
    self.assertEqual(shaded(TINY, 0.0)[0].tobytes(), shaded(TINY, FLOOR)[0].tobytes())


@NEEDS_MOVER
class TestShadowLightCoeffThreads(unittest.TestCase):
  """A full shadow does not depend on how many render threads are used."""

  def test_same_bytes_for_one_two_and_four_threads(self):
    """Renders the scene in a fresh process per thread count and compares the colour hashes."""
    digests = set()
    for threads in ("1", "2", "4"):
      env = dict(os.environ, SWARM_RENDER_THREADS=threads)
      out = subprocess.check_output([sys.executable, __file__, "--hash"], env=env, text=True)
      digests.add(out.strip())
    self.assertEqual(len(digests), 1)


def full_shadow_hash():
  """Prints the sha256 of two fully shadowed frames: from the map, and from the map with a mover on top."""
  p.connect(p.DIRECT)
  tmp = tempfile.mkdtemp()
  tex_path = os.path.join(tmp, 'checker.tga')
  write_checker_tga(tex_path)
  build_world(tex_path)
  digest = hashlib.sha256()
  digest.update(shaded(MAP, 0.0)[0].astype(np.uint8).tobytes())
  make_mover(MOVED)
  digest.update(shaded(MOVER, 0.0)[0].astype(np.uint8).tobytes())
  print(digest.hexdigest())
  p.disconnect()
  os.remove(tex_path)
  os.rmdir(tmp)


if __name__ == '__main__':
  if "--hash" in sys.argv:
    full_shadow_hash()
  else:
    unittest.main()
