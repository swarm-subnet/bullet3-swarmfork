"""Leaf cards and the shadow map (ER_SWARM_LEAF_NO_SHADOW): a double-sided cut-out card casts no shadow into
the map or along a shadow ray and takes its own light from the sunward side, an opaque body still casts, and
without the flag nothing moves. Ray-cast colour path only, off by default."""
import os
import tempfile
import unittest
import numpy as np
import pybullet as p

from daylightTest import DAYLIGHT, LOW_SUN, PICTURE, RAYCAST, SKY, SUN_COLOR, write_obj, write_png

SIZE = 96
LEAF = getattr(p, "ER_SWARM_LEAF_NO_SHADOW", 0)
# Straight down from 8 m with a 60 degree lens: 4.62 m of ground a side, 10.4 pixels a metre, the origin mid frame.
# The sun stands 45 degrees up on the +x side, so the leaf's shadow lies 1.5 m to -x of it and the box's at +1.5 m.
EYE = ([0, 0, 8], [0, 0, 0])
UNDER_LEAF = (slice(40, 56), slice(24, 37))
UNDER_BOX = (slice(43, 53), slice(60, 68))
OPEN = (slice(4, 20), slice(4, 20))


@unittest.skipUnless(RAYCAST and DAYLIGHT and SKY and LEAF, "wheel without the leaf shadow flag")
class TestLeafNoShadow(unittest.TestCase):
  """A leaf card and a box above a white ground under a low sun, rendered with the flag off and on."""

  def setUp(self):
    """A DIRECT client, a temporary folder, the ground, a horizontal leaf card 1.5 m up and a box beside it."""
    p.connect(p.DIRECT)
    self.folder = tempfile.mkdtemp(prefix="leaf_shadow_")
    self.proj = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 100.0)
    ground = os.path.join(self.folder, "ground.obj")
    write_obj(ground, 20.0)
    vis = p.createVisualShape(p.GEOM_MESH, fileName=ground, rgbaColor=[1, 1, 1, 1], specularColor=[0, 0, 0])
    self.ground = p.createMultiBody(0, -1, vis)
    photo = np.zeros((32, 32, 4), dtype=np.uint8)
    photo[:] = (60, 160, 60, 255)
    photo[12:20, 12:20, 3] = 0
    path = os.path.join(self.folder, "leaf.png")
    write_png(path, photo)
    leaf = os.path.join(self.folder, "leaf.obj")
    write_obj(leaf, 1.0)
    vis = p.createVisualShape(p.GEOM_MESH, fileName=leaf, rgbaColor=[1, 1, 1, 1], specularColor=[0, 0, 0],
                              flags=p.VISUAL_SHAPE_DOUBLE_SIDED_MULTIBODY)
    self.leaf = p.createMultiBody(0, -1, vis, basePosition=[0, 0, 1.5])
    p.changeVisualShape(self.leaf, -1, textureUniqueId=p.loadTexture(path))
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.05], rgbaColor=[1, 1, 1, 1], specularColor=[0, 0, 0])
    self.box = p.createMultiBody(0, -1, vis, basePosition=[3, 0, 1.5])

  def tearDown(self):
    """Drops the client."""
    p.disconnect()

  def render(self, flags, shadow=1):
    """Returns (rgb, seg) of the frame straight down."""
    eye, target = EYE
    view = p.computeViewMatrix(list(eye), list(target), [0, 1, 0])
    _, _, rgb, _, seg = p.getCameraImage(SIZE, SIZE, view, self.proj, shadow=shadow, lightDirection=LOW_SUN,
                                         lightColor=SUN_COLOR, lightAmbientCoeff=1.0, lightDiffuseCoeff=3.0,
                                         renderer=p.ER_TINY_RENDERER, flags=flags, shadowLightCoeff=0.0)
    return np.asarray(rgb).reshape(SIZE, SIZE, 4)[:, :, :3].astype(int), np.asarray(seg).reshape(SIZE, SIZE)

  def ground_under(self, rgb, seg, window):
    """Mean green byte of the ground pixels inside a window, which must hold some."""
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[window] = True
    mask &= seg == self.ground
    self.assertGreater(int(mask.sum()), 20)
    return float(rgb[:, :, 1][mask].mean())

  def test_without_the_flag_the_leaf_shadows_the_ground(self):
    """With the shadow map alone the ground under the leaf is as dark as the ground under the box."""
    rgb, seg = self.render(PICTURE | DAYLIGHT)
    open_ground = self.ground_under(rgb, seg, OPEN)
    self.assertLess(self.ground_under(rgb, seg, UNDER_LEAF), open_ground - 20)
    self.assertLess(self.ground_under(rgb, seg, UNDER_BOX), open_ground - 20)

  def test_with_the_flag_the_leaf_casts_nothing_and_the_box_still_does(self):
    """The ground under the leaf is lit like open ground; the box's shadow stays."""
    rgb, seg = self.render(PICTURE | DAYLIGHT | LEAF)
    open_ground = self.ground_under(rgb, seg, OPEN)
    self.assertAlmostEqual(self.ground_under(rgb, seg, UNDER_LEAF), open_ground, delta=2.0)
    self.assertLess(self.ground_under(rgb, seg, UNDER_BOX), open_ground - 20)

  def test_the_flag_leaves_the_lit_ground_alone(self):
    """Pixels away from both shadows are byte-identical with and without the flag."""
    off, seg = self.render(PICTURE | DAYLIGHT)
    on, _ = self.render(PICTURE | DAYLIGHT | LEAF)
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[OPEN] = True
    mask &= seg == self.ground
    self.assertTrue(np.array_equal(off[mask], on[mask]))

  def test_the_flag_needs_shadows(self):
    """With shadow=0 the flag changes no byte."""
    off, _ = self.render(PICTURE | DAYLIGHT, shadow=0)
    on, _ = self.render(PICTURE | DAYLIGHT | LEAF, shadow=0)
    self.assertTrue(np.array_equal(off, on))

  def test_threads_give_the_same_bytes(self):
    """The frame with the flag is byte-identical at every render thread count the build allows."""
    frames = []
    for threads in ("1", "2", "4"):
      os.environ["SWARM_RENDER_THREADS"] = threads
      p.disconnect()
      self.setUp()
      frames.append(self.render(PICTURE | DAYLIGHT | LEAF)[0])
    os.environ.pop("SWARM_RENDER_THREADS", None)
    self.assertTrue(np.array_equal(frames[0], frames[1]))
    self.assertTrue(np.array_equal(frames[0], frames[2]))


if __name__ == "__main__":
  unittest.main()
