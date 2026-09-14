"""The sky computed from the sun (ER_SWARM_SKY_SUN): colour buffer only, per call, off by default."""
import time
import unittest
import numpy as np
import pybullet as p

SIZE = 64
SUN_COLOR = [1.0, 0.72, 0.48]
NOON = [0.0, 0.0, 1.0]
LOW_SUN = [0.9986, 0.0, 0.0523]  # 3 degrees above the horizon along +X
RAYCAST = getattr(p, "ER_SWARM_RAYCAST", 0)


def as_bytes(color):
  """The 8-bit triple the renderer writes for a 0..1 colour."""
  return np.array([int(c * 255 + 0.5) for c in color], dtype=np.uint8)


class TestSkyFromSun(unittest.TestCase):
  """Renders an empty world and one box with the sun sky on and off."""

  def setUp(self):
    """Connects a DIRECT client with a 90 degree square camera."""
    p.connect(p.DIRECT)
    self.proj = p.computeProjectionMatrixFOV(90, 1.0, 0.1, 10.0)

  def tearDown(self):
    """Drops the client."""
    p.disconnect()

  def render(self, target=(1, 0, 0), up=(0, 0, 1), flags=0, sun=NOON, fov=None, **kwargs):
    """Returns (rgb HxWx3, depth, segmentation) for a camera at the origin looking at target."""
    view = p.computeViewMatrix([0, 0, 0], target, up)
    proj = self.proj if fov is None else p.computeProjectionMatrixFOV(fov, 1.0, 0.1, 10.0)
    _, _, rgb, depth, seg = p.getCameraImage(SIZE, SIZE, view, proj, shadow=0, lightDirection=sun,
                                             lightColor=SUN_COLOR, renderer=p.ER_TINY_RENDERER, flags=flags, **kwargs)
    rgb = None if rgb is None else np.asarray(rgb).reshape(SIZE, SIZE, 4)[:, :, :3]
    return rgb, np.asarray(depth), seg

  def test_default_stays_white(self):
    """Without the flag every empty pixel is still white."""
    rgb, _, _ = self.render()
    self.assertTrue((rgb == 255).all())

  def test_noon_sky_is_blue_above_and_pale_at_the_horizon(self):
    """Overhead the sky is blue; towards the horizon it is brighter and less saturated."""
    rgb, _, _ = self.render(flags=p.ER_SWARM_SKY_SUN)
    top, horizon = rgb[0].astype(int), rgb[SIZE // 2].astype(int)
    self.assertTrue((top[:, 2] > top[:, 0]).all())
    self.assertGreater(horizon.sum(), top.sum())
    self.assertLess((horizon[:, 2] - horizon[:, 0]).mean(), (top[:, 2] - top[:, 0]).mean())
    self.assertFalse((rgb == 255).all())

  def test_low_sun_paints_a_warm_glow_and_its_disc(self):
    """Looking straight at a low sun through a narrow lens, the centre is the sun colour and the glow is warm."""
    rgb, _, _ = self.render(target=LOW_SUN, flags=p.ER_SWARM_SKY_SUN, sun=LOW_SUN, fov=30)
    centre = rgb[SIZE // 2, SIZE // 2]
    self.assertTrue((centre == as_bytes(SUN_COLOR)).all())
    ring = rgb[SIZE // 2, SIZE // 2 + 8].astype(int)
    self.assertFalse((ring == as_bytes(SUN_COLOR)).all())
    self.assertGreater(ring[0], ring[2])
    away, _, _ = self.render(target=[-LOW_SUN[0], 0, LOW_SUN[2]], flags=p.ER_SWARM_SKY_SUN, sun=LOW_SUN, fov=30)
    self.assertGreater(rgb.astype(int).sum(), away.astype(int).sum())

  def test_sky_moves_with_the_sun(self):
    """Two sun directions give two different skies from the same camera."""
    east, _, _ = self.render(flags=p.ER_SWARM_SKY_SUN, sun=[0.7, 0.0, 0.7])
    west, _, _ = self.render(flags=p.ER_SWARM_SKY_SUN, sun=[-0.7, 0.0, 0.7])
    self.assertFalse((east == west).all())

  def test_same_sun_gives_the_same_bytes_and_reuses_the_map(self):
    """A second frame under the same sun is byte-identical and does not rebuild the map."""
    first, _, _ = self.render(flags=p.ER_SWARM_SKY_SUN)
    start = time.perf_counter()
    second, _, _ = self.render(flags=p.ER_SWARM_SKY_SUN)
    reused = time.perf_counter() - start
    self.assertEqual(first.tobytes(), second.tobytes())
    start = time.perf_counter()
    self.render(flags=p.ER_SWARM_SKY_SUN, sun=[0.5, 0.5, 0.7])
    rebuilt = time.perf_counter() - start
    self.assertLess(reused, rebuilt)

  def test_clouds_are_seeded(self):
    """Clouds change the sky, the same seed repeats them and another seed moves them."""
    clear, _, _ = self.render(flags=p.ER_SWARM_SKY_SUN)
    seed_a, _, _ = self.render(flags=p.ER_SWARM_SKY_SUN, skyCloudSeed=11)
    seed_a_again, _, _ = self.render(flags=p.ER_SWARM_SKY_SUN, skyCloudSeed=11)
    seed_b, _, _ = self.render(flags=p.ER_SWARM_SKY_SUN, skyCloudSeed=12)
    self.assertFalse((clear == seed_a).all())
    self.assertEqual(seed_a.tobytes(), seed_a_again.tobytes())
    self.assertFalse((seed_a == seed_b).all())
    after, _, _ = self.render(flags=p.ER_SWARM_SKY_SUN)
    self.assertEqual(clear.tobytes(), after.tobytes())

  def test_objects_depth_and_mask_unchanged(self):
    """The sky fills only empty pixels; depth and mask are byte-identical, hit pixels change only by the ambient tint."""
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], rgbaColor=[1, 1, 1, 1])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=[3, 0, 0.5])
    plain_rgb, plain_depth, plain_seg = self.render()
    sky_rgb, sky_depth, sky_seg = self.render(flags=p.ER_SWARM_SKY_SUN)
    self.assertEqual(plain_depth.tobytes(), sky_depth.tobytes())
    self.assertEqual(np.asarray(plain_seg).tobytes(), np.asarray(sky_seg).tobytes())
    hit = np.asarray(plain_seg).reshape(SIZE, SIZE) >= 0
    self.assertGreater(int(hit.sum()), 0)
    self.assertTrue((plain_rgb[~hit] == 255).all())
    self.assertFalse((sky_rgb[~hit] == 255).all())
    # A noon sky is blue, so its ambient tint takes red away from the white box and leaves blue alone.
    self.assertTrue((sky_rgb[hit][:, 0] <= plain_rgb[hit][:, 0]).all())
    self.assertTrue((sky_rgb[hit][:, 2] == plain_rgb[hit][:, 2]).all())
    self.assertLess(int(sky_rgb[hit][:, 0].sum()), int(plain_rgb[hit][:, 0].sum()))

  def test_depth_only_ignores_the_sky(self):
    """Depth-only renders return no colour and the same depth with or without the flag."""
    flags = p.ER_DEPTH_ONLY | p.ER_NO_SEGMENTATION_MASK
    _, plain_depth, _ = self.render(flags=flags)
    rgb, sky_depth, _ = self.render(flags=flags | p.ER_SWARM_SKY_SUN)
    self.assertIsNone(rgb)
    self.assertEqual(plain_depth.tobytes(), sky_depth.tobytes())

  @unittest.skipUnless(RAYCAST, "wheel without the ray-cast backend")
  def test_raycast_path_paints_the_same_sky(self):
    """The ray-cast path and the rasterised path write the same sky bytes where nothing is drawn."""
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], rgbaColor=[1, 1, 1, 1])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=[3, 0, 0.5])
    tiny_rgb, _, tiny_seg = self.render(flags=p.ER_SWARM_SKY_SUN, skyCloudSeed=5)
    ray_rgb, _, ray_seg = self.render(flags=p.ER_SWARM_SKY_SUN | RAYCAST, skyCloudSeed=5)
    miss = (np.asarray(tiny_seg).reshape(SIZE, SIZE) < 0) & (np.asarray(ray_seg).reshape(SIZE, SIZE) < 0)
    self.assertGreater(int(miss.sum()), SIZE * SIZE // 2)
    self.assertTrue((tiny_rgb[miss] == ray_rgb[miss]).all())


if __name__ == '__main__':
  unittest.main()
