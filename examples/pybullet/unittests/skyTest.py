"""Sky background on getCameraImage: colour buffer only, per call, off by default."""
import unittest
import numpy as np
import pybullet as p

SIZE = 64
# Chosen so that value * 255 is never within float error of a .5 rounding boundary.
HORIZON = [0.85, 0.55, 0.25]
ZENITH = [0.15, 0.35, 0.95]
# Camera at the origin, Z up: level along +X, straight up, straight down.
EYES = {"level": ([1, 0, 0], [0, 0, 1]), "up": ([0, 0, 1], [1, 0, 0]), "down": ([0, 0, -1], [1, 0, 0])}


def as_bytes(color):
  """The 8-bit triple the renderer writes for a 0..1 colour."""
  return np.array([int(c * 255 + 0.5) for c in color], dtype=np.uint8)


class TestSkyBackground(unittest.TestCase):
  """Renders an empty world and one box through the TinyRenderer with and without a sky."""

  def setUp(self):
    """Connects a DIRECT client with a 90 degree square camera."""
    p.connect(p.DIRECT)
    self.proj = p.computeProjectionMatrixFOV(90, 1.0, 0.1, 10.0)

  def tearDown(self):
    """Drops the client."""
    p.disconnect()

  def render(self, look="level", flags=0, **sky):
    """Returns (rgb HxWx3 or None, depth, segmentation) for a camera at the origin looking `look`."""
    target, up = EYES[look]
    view = p.computeViewMatrix([0, 0, 0], target, up)
    _, _, rgb, depth, seg = p.getCameraImage(SIZE, SIZE, view, self.proj, shadow=0,
                                             renderer=p.ER_TINY_RENDERER, flags=flags, **sky)
    rgb = None if rgb is None else np.asarray(rgb).reshape(SIZE, SIZE, 4)[:, :, :3]
    return rgb, np.asarray(depth), seg

  def test_default_stays_white(self):
    """Without the sky arguments every empty pixel is still white."""
    rgb, _, _ = self.render()
    self.assertTrue((rgb == 255).all())

  def test_one_colour_is_a_flat_sky(self):
    """Either colour alone fills the whole background with that colour."""
    rgb, _, _ = self.render(skyHorizonColor=HORIZON)
    self.assertTrue((rgb == as_bytes(HORIZON)).all())
    rgb, _, _ = self.render(skyZenithColor=ZENITH)
    self.assertTrue((rgb == as_bytes(ZENITH)).all())

  def test_gradient_follows_the_view(self):
    """Horizon colour at and below eye level, zenith colour straight up, blended in between."""
    rgb, _, _ = self.render("level", skyHorizonColor=HORIZON, skyZenithColor=ZENITH)
    # Bottom rows look below the horizon: exactly the horizon colour. Top rows look 45 degrees up.
    self.assertTrue((rgb[-1] == as_bytes(HORIZON)).all())
    self.assertTrue((rgb[0, :, 2] > rgb[-1, :, 2]).all())
    self.assertTrue((rgb[0, :, 0] < rgb[-1, :, 0]).all())
    up_rgb, _, _ = self.render("up", skyHorizonColor=HORIZON, skyZenithColor=ZENITH)
    self.assertTrue((up_rgb[SIZE // 2, SIZE // 2] == as_bytes(ZENITH)).all())
    self.assertTrue((up_rgb[:, :, 2] > as_bytes(HORIZON)[2]).all())
    self.assertGreater(up_rgb[:, :, 2].mean(), rgb[:, :, 2].mean())
    down_rgb, _, _ = self.render("down", skyHorizonColor=HORIZON, skyZenithColor=ZENITH)
    self.assertTrue((down_rgb == as_bytes(HORIZON)).all())

  def test_sky_is_per_call(self):
    """A sky on one call does not leak into the next call without one."""
    self.render(skyHorizonColor=HORIZON)
    rgb, _, _ = self.render()
    self.assertTrue((rgb == 255).all())

  def test_objects_depth_and_mask_unchanged(self):
    """The sky changes only pixels where nothing was drawn; depth and mask are byte-identical."""
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], rgbaColor=[1, 0, 0, 1])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=[3, 0, 0.5])
    plain_rgb, plain_depth, plain_seg = self.render("level")
    sky_rgb, sky_depth, sky_seg = self.render("level", skyHorizonColor=HORIZON, skyZenithColor=ZENITH)
    self.assertEqual(plain_depth.tobytes(), sky_depth.tobytes())
    self.assertEqual(np.asarray(plain_seg).tobytes(), np.asarray(sky_seg).tobytes())
    hit = np.asarray(plain_seg).reshape(SIZE, SIZE) >= 0
    self.assertGreater(int(hit.sum()), 0)
    self.assertTrue((plain_rgb[hit] == sky_rgb[hit]).all())
    self.assertTrue((plain_rgb[~hit] == 255).all())
    self.assertFalse((sky_rgb[~hit] == 255).all())

  def test_depth_only_ignores_the_sky(self):
    """Depth-only renders return no colour and the same depth whether or not a sky is given."""
    flags = p.ER_DEPTH_ONLY | p.ER_NO_SEGMENTATION_MASK
    _, plain_depth, _ = self.render("level", flags=flags)
    rgb, sky_depth, _ = self.render("level", flags=flags, skyHorizonColor=HORIZON)
    self.assertIsNone(rgb)
    self.assertEqual(plain_depth.tobytes(), sky_depth.tobytes())


if __name__ == '__main__':
  unittest.main()
