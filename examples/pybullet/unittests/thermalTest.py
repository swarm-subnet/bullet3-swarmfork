"""The thermal camera (ER_SWARM_THERMAL): White Hot grey with each frame's own range, temperatures set per shape or
painted by a heat map, passive surfaces cooled by the open night sky and warmed by the sun, glass mirroring the cold
sky, depth and mask unchanged, the same bytes at every thread count. Ray-cast path only, off by default."""
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pybullet as p

from daylightTest import RAYCAST, write_obj, write_png

SIZE = 96
THERMAL = getattr(p, "ER_SWARM_THERMAL", 0)
FRAME = RAYCAST | THERMAL | getattr(p, "ER_ALPHA_CUTOUT", 0)
COLOUR = RAYCAST | getattr(p, "ER_TEXTURE_FILTER", 0) | getattr(p, "ER_ALPHA_CUTOUT", 0)
NIGHT = [0.3, 0.2, -0.9]
SUN_45 = [0.7071, 0.0, 0.7071]
AIR, SKY = 20.0, -20.0
DOWN = ((0, 0, 10), (0, 0, 0))
SIDE = ((0, -6, 1), (0, 0, 0))


class Scene(object):
  """A grey ground 40 m across; the pieces each test adds sit on it."""

  def __init__(self):
    """A DIRECT client, a temporary folder and the ground."""
    p.connect(p.DIRECT)
    self.folder = tempfile.mkdtemp(prefix="thermal_")
    self.proj = p.computeProjectionMatrixFOV(90, 1.0, 0.1, 100.0)
    path = os.path.join(self.folder, "ground.obj")
    write_obj(path, 20.0)
    vis = p.createVisualShape(p.GEOM_MESH, fileName=path, rgbaColor=[0.3, 0.3, 0.3, 1], specularColor=[0, 0, 0])
    self.ground = p.createMultiBody(0, -1, vis)

  def box(self, centre, half, flags=0, colour=(0.5, 0.5, 0.5, 1)):
    """A box body with no specular colour, or with the given visual shape flags."""
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=list(half), rgbaColor=list(colour), specularColor=[0, 0, 0], flags=flags)
    return p.createMultiBody(0, -1, vis, basePosition=list(centre))

  def heat_quad(self, centre, low, high):
    """A 2 m square whose heat map is cold on its -x half and hot on its +x half."""
    path = os.path.join(self.folder, "quad.obj")
    write_obj(path, 1.0)
    vis = p.createVisualShape(p.GEOM_MESH, fileName=path, rgbaColor=[0.5, 0.5, 0.5, 1], specularColor=[0, 0, 0])
    body = p.createMultiBody(0, -1, vis, basePosition=list(centre))
    heat = np.zeros((16, 16, 4), dtype=np.uint8)
    heat[:, :, 3] = 255
    heat[:, 8:, :3] = 255
    png = os.path.join(self.folder, "heat.png")
    write_png(png, heat)
    p.changeVisualShape(body, -1, thermalTextureUniqueId=p.loadTexture(png), temperatureRange=[low, high])
    return body

  def render(self, flags=FRAME, eye=DOWN, sun=NIGHT, seed=1, renderer=p.ER_TINY_RENDERER, air=AIR, sky=SKY):
    """Returns (grey or rgb, depth, seg, view) of one frame; thermal arguments are passed whatever the flags."""
    view = p.computeViewMatrix(list(eye[0]), list(eye[1]), [0, 1, 0] if eye == DOWN else [0, 0, 1])
    _, _, rgb, depth, seg = p.getCameraImage(SIZE, SIZE, view, self.proj, lightDirection=sun, renderer=renderer,
                                             flags=flags, airTemperature=air, skyTemperature=sky, thermalSeed=seed)
    rgb = np.asarray(rgb, dtype=np.uint8).reshape(SIZE, SIZE, 4)[:, :, :3]
    return rgb, np.asarray(depth, dtype=np.float32).reshape(SIZE, SIZE), np.asarray(seg).reshape(SIZE, SIZE), view

  def world(self, depth, view):
    """World position of every pixel, from its depth and the camera."""
    inverse = np.linalg.inv(np.array(self.proj).reshape(4, 4).T @ np.array(view).reshape(4, 4).T)
    xs = (np.arange(SIZE) + 0.5) / SIZE * 2 - 1
    ys = 1 - (np.arange(SIZE) + 0.5) / SIZE * 2
    gx, gy = np.meshgrid(xs, ys)
    clip = np.stack([gx, gy, depth * 2 - 1, np.ones_like(gx)], -1) @ inverse.T
    return clip[..., :3] / clip[..., 3:4]


def full_scene(scene=None):
  """Every thermal term at once: a sheltering roof, a warm box, a glass slab and a heat map, added to the given scene or
  a new one; returns the scene."""
  scene = scene or Scene()
  scene.box((0, 0, 3), (0.5, 0.5, 0.02))
  p.changeVisualShape(scene.box((3, 3, 0.5), (0.3, 0.3, 0.5)), -1, temperature=33.0)
  scene.box((-3, 3, 1), (0.8, 0.8, 0.02), flags=p.VISUAL_SHAPE_GLASS)
  scene.heat_quad((3, -3, 0.5), 20.0, 40.0)
  return scene


@unittest.skipUnless(RAYCAST and THERMAL, "wheel without the thermal camera")
class TestThermal(unittest.TestCase):
  """Small worlds rendered through the thermal camera, checked term by term."""

  def setUp(self):
    """A fresh ground for every test."""
    self.scene = Scene()

  def tearDown(self):
    """Drops the client."""
    p.disconnect()

  def region(self, grey, depth, seg, view, body, x=None, y=None):
    """Mean grey of a body's pixels, kept to world x and y ranges when given; the region must hold some."""
    mask = seg == body
    points = self.scene.world(depth, view)
    if x is not None:
      mask &= (points[..., 0] > x[0]) & (points[..., 0] < x[1])
    if y is not None:
      mask &= (points[..., 1] > y[0]) & (points[..., 1] < y[1])
    self.assertGreater(int(mask.sum()), 5)
    return float(grey[mask].mean())

  def test_frame_is_grey_from_black_to_white(self):
    """Every pixel has R = G = B, and the frame's coldest point is 0 and its hottest 255."""
    p.changeVisualShape(self.scene.box((0, 0, 0.5), (0.5, 0.5, 0.5)), -1, temperature=33.0)
    rgb, _, _, _ = self.scene.render()
    self.assertTrue(np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 0], rgb[..., 2]))
    self.assertEqual(int(rgb.min()), 0)
    self.assertEqual(int(rgb.max()), 255)

  def test_range_follows_each_frame(self):
    """A hotter box stays the white end and pushes a 24 C box darker; a box colder than everything turns black."""
    box = self.scene.box((2, 0, 0.5), (0.5, 0.5, 0.5))
    middle = self.scene.box((-2, 0, 0.5), (0.5, 0.5, 0.5))
    p.changeVisualShape(middle, -1, temperature=24.0)
    means = []
    for temperature in (33.0, 60.0):
      p.changeVisualShape(box, -1, temperature=temperature)
      rgb, depth, seg, view = self.scene.render()
      means.append(self.region(rgb[..., 0], depth, seg, view, middle))
      self.assertGreaterEqual(int(np.percentile(rgb[..., 0][seg == box], 75)), 250)
    self.assertLess(means[1], means[0] - 5)
    p.changeVisualShape(box, -1, temperature=-40.0)
    rgb, _, seg, _ = self.scene.render()
    self.assertLessEqual(int(np.percentile(rgb[..., 0][seg == box], 25)), 5)

  def test_hotter_surface_is_brighter(self):
    """Of two boxes set to 25 C and 35 C the warmer one is brighter."""
    cool = self.scene.box((-2, 0, 0.5), (0.5, 0.5, 0.5))
    warm = self.scene.box((2, 0, 0.5), (0.5, 0.5, 0.5))
    p.changeVisualShape(cool, -1, temperature=25.0)
    p.changeVisualShape(warm, -1, temperature=35.0)
    rgb, depth, seg, view = self.scene.render()
    self.assertGreater(self.region(rgb[..., 0], depth, seg, view, warm), self.region(rgb[..., 0], depth, seg, view, cool) + 20)

  def test_sheltered_ground_is_warmer_at_night(self):
    """Ground under a roof, seen from the side, is lighter than open ground in the same frame."""
    self.scene.box((0, 0, 3), (1.5, 1.5, 0.02))
    rgb, depth, seg, view = self.scene.render(eye=SIDE)
    under = self.region(rgb[..., 0], depth, seg, view, self.scene.ground, x=(-1.0, 1.0), y=(-1.0, 1.0))
    open_ground = self.region(rgb[..., 0], depth, seg, view, self.scene.ground, x=(-1.0, 1.0), y=(-4.0, -2.5))
    self.assertGreater(under, open_ground + 20)

  def test_glass_mirrors_the_cold_sky(self):
    """A glass slab under the open night sky is darker than a plain slab beside it at the same temperature."""
    glass = self.scene.box((-1.5, 0, 1), (1, 1, 0.02), flags=p.VISUAL_SHAPE_GLASS)
    plain = self.scene.box((1.5, 0, 1), (1, 1, 0.02))
    rgb, depth, seg, view = self.scene.render()
    self.assertLess(self.region(rgb[..., 0], depth, seg, view, glass), self.region(rgb[..., 0], depth, seg, view, plain) - 10)

  def test_heat_map_paints_temperatures(self):
    """The hot half of a heat map is brighter than its cold half; removing the map makes the square passive again."""
    quad = self.scene.heat_quad((0, 0, 0.5), 20.0, 40.0)
    rgb, depth, seg, view = self.scene.render()
    cold = self.region(rgb[..., 0], depth, seg, view, quad, x=(-0.9, -0.1))
    hot = self.region(rgb[..., 0], depth, seg, view, quad, x=(0.1, 0.9))
    self.assertGreater(hot, cold + 40)
    p.changeVisualShape(quad, -1, thermalTextureUniqueId=-1)
    rgb, depth, seg, view = self.scene.render()
    cold = self.region(rgb[..., 0], depth, seg, view, quad, x=(-0.9, -0.1))
    hot = self.region(rgb[..., 0], depth, seg, view, quad, x=(0.1, 0.9))
    self.assertAlmostEqual(hot, cold, delta=3.0)

  def test_nan_makes_a_surface_passive_again(self):
    """A box set hot and then set to NaN renders the same bytes as a box that was never set."""
    box = self.scene.box((0, 0, 0.5), (0.5, 0.5, 0.5))
    never, _, _, _ = self.scene.render()
    p.changeVisualShape(box, -1, temperature=60.0)
    p.changeVisualShape(box, -1, temperature=float("nan"))
    again, _, _, _ = self.scene.render()
    self.assertTrue(np.array_equal(never, again))

  def test_lower_emissivity_reflects_more_cold_sky(self):
    """Of two boxes at 33 C under the night sky, the one with emissivity 0.3 is darker than the default one."""
    plain = self.scene.box((-2, 0, 0.5), (0.5, 0.5, 0.5))
    shiny = self.scene.box((2, 0, 0.5), (0.5, 0.5, 0.5))
    p.changeVisualShape(plain, -1, temperature=33.0)
    p.changeVisualShape(shiny, -1, temperature=33.0, emissivity=0.3)
    rgb, depth, seg, view = self.scene.render()
    self.assertLess(self.region(rgb[..., 0], depth, seg, view, shiny), self.region(rgb[..., 0], depth, seg, view, plain) - 20)

  def test_emissivity_above_one_is_one(self):
    """An emissivity of 5 draws the bytes of an emissivity of 1."""
    box = self.scene.box((0, 0, 0.5), (0.5, 0.5, 0.5))
    p.changeVisualShape(box, -1, temperature=33.0, emissivity=1.0)
    one, _, _, _ = self.scene.render()
    p.changeVisualShape(box, -1, emissivity=5.0)
    five, _, _, _ = self.scene.render()
    self.assertTrue(np.array_equal(one, five))

  def test_air_temperature_warms_passive_ground(self):
    """Passive ground is darker than a 25 C box in 10 C air and lighter than it in 40 C air."""
    box = self.scene.box((2, 0, 0.5), (0.5, 0.5, 0.5))
    p.changeVisualShape(box, -1, temperature=25.0)
    for air, warmer in ((10.0, False), (40.0, True)):
      rgb, depth, seg, view = self.scene.render(air=air)
      ground = self.region(rgb[..., 0], depth, seg, view, self.scene.ground, x=(-4.0, -1.0))
      fixed = self.region(rgb[..., 0], depth, seg, view, box)
      if warmer:
        self.assertGreater(ground, fixed + 10)
      else:
        self.assertLess(ground, fixed - 10)

  def test_sky_temperature_cools_open_ground(self):
    """In 20 C air, open ground is darker than a 17 C box under a -40 C sky and lighter than it under a 10 C sky."""
    box = self.scene.box((2, 0, 0.5), (0.5, 0.5, 0.5))
    p.changeVisualShape(box, -1, temperature=17.0)
    for sky, warmer in ((-40.0, False), (10.0, True)):
      rgb, depth, seg, view = self.scene.render(air=20.0, sky=sky)
      ground = self.region(rgb[..., 0], depth, seg, view, self.scene.ground, x=(-4.0, -1.0))
      fixed = self.region(rgb[..., 0], depth, seg, view, box)
      if warmer:
        self.assertGreater(ground, fixed + 10)
      else:
        self.assertLess(ground, fixed - 10)

  def test_nan_temperatures_draw_a_frame(self):
    """NaN air, sky and heat map range give a grey frame instead of reading outside the camera's table."""
    p.changeVisualShape(self.scene.box((0, 0, 0.5), (0.5, 0.5, 0.5)), -1, temperature=33.0)
    quad = self.scene.heat_quad((3, -3, 0.5), 20.0, 40.0)
    rgb, _, _, _ = self.scene.render(air=float("nan"), sky=float("nan"))
    self.assertEqual(rgb.shape, (SIZE, SIZE, 3))
    self.assertTrue(np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 0], rgb[..., 2]))
    texture = p.loadTexture(os.path.join(self.scene.folder, "heat.png"))
    p.changeVisualShape(quad, -1, thermalTextureUniqueId=texture, temperatureRange=[float("nan"), float("nan")])
    rgb, _, _, _ = self.scene.render()
    self.assertEqual(rgb.shape, (SIZE, SIZE, 3))

  def test_unknown_heat_map_fails_and_keeps_the_map(self):
    """A heat map id no texture has raises, and the shape keeps the map it had."""
    quad = self.scene.heat_quad((0, 0, 0.5), 20.0, 40.0)
    before, _, _, _ = self.scene.render()
    with self.assertRaises(p.error):
      p.changeVisualShape(quad, -1, thermalTextureUniqueId=987654, temperatureRange=[0.0, 1.0])
    after, _, _, _ = self.scene.render()
    self.assertTrue(np.array_equal(before, after))

  def test_seed_keeps_its_low_32_bits(self):
    """A seed past 64 bits draws the grain of its low 32 bits."""
    p.changeVisualShape(self.scene.box((0, 0, 0.5), (0.5, 0.5, 0.5)), -1, temperature=33.0)
    small, _, _, _ = self.scene.render(seed=7)
    large, _, _, _ = self.scene.render(seed=(1 << 70) + 7)
    self.assertTrue(np.array_equal(small, large))

  def test_sun_heats_and_shade_cools_by_day(self):
    """With the sun up, open ground is lighter than the ground in a roof's shadow."""
    self.scene.box((0, 0, 3), (0.5, 0.5, 0.02))
    rgb, depth, seg, view = self.scene.render(sun=SUN_45)
    shade = self.region(rgb[..., 0], depth, seg, view, self.scene.ground, x=(-3.3, -2.7), y=(-0.3, 0.3))
    sunlit = self.region(rgb[..., 0], depth, seg, view, self.scene.ground, x=(2.7, 3.3), y=(-0.3, 0.3))
    self.assertGreater(sunlit, shade + 20)

  def test_depth_and_mask_match_the_colour_frame(self):
    """The thermal frame's depth and segmentation are byte-identical to the ray-cast colour frame's."""
    full_scene(self.scene)
    _, depth, seg, _ = self.scene.render()
    _, colour_depth, colour_seg, _ = self.scene.render(flags=COLOUR)
    self.assertTrue(np.array_equal(depth, colour_depth))
    self.assertTrue(np.array_equal(seg, colour_seg))

  def test_colour_frames_ignore_every_thermal_setting(self):
    """A colour frame is byte-identical before and after temperatures, a heat map and thermal arguments are set."""
    box = self.scene.box((0, 0, 0.5), (0.5, 0.5, 0.5))
    before, _, _, _ = self.scene.render(flags=COLOUR, sun=SUN_45)
    heat = np.full((4, 4, 4), 255, dtype=np.uint8)
    png = os.path.join(self.scene.folder, "flat_heat.png")
    write_png(png, heat)
    p.changeVisualShape(box, -1, temperature=50.0, emissivity=0.3)
    p.changeVisualShape(self.scene.ground, -1, thermalTextureUniqueId=p.loadTexture(png), temperatureRange=[10.0, 30.0])
    after, _, _, _ = self.scene.render(flags=COLOUR, sun=SUN_45, seed=99)
    self.assertTrue(np.array_equal(before, after))

  def test_without_the_ray_caster_the_flag_does_nothing(self):
    """On the rasteriser the thermal bit changes no byte."""
    self.scene.box((0, 0, 0.5), (0.5, 0.5, 0.5))
    plain, _, _, _ = self.scene.render(flags=0)
    flagged, _, _, _ = self.scene.render(flags=THERMAL)
    self.assertTrue(np.array_equal(plain, flagged))

  def test_seed_draws_the_grain(self):
    """The same seed gives the same bytes; another seed moves some pixels, by little."""
    p.changeVisualShape(self.scene.box((0, 0, 0.5), (0.5, 0.5, 0.5)), -1, temperature=33.0)
    first, _, _, _ = self.scene.render(seed=1)
    same, _, _, _ = self.scene.render(seed=1)
    other, _, _, _ = self.scene.render(seed=2)
    self.assertTrue(np.array_equal(first, same))
    self.assertFalse(np.array_equal(first, other))
    self.assertLess(float(np.abs(first.astype(int) - other.astype(int)).mean()), 5.0)

  def test_threads_give_the_same_bytes(self):
    """The full scene is byte-identical at 1, 2 and 4 render threads, each count in its own process."""
    frames = []
    for threads in ("1", "2", "4"):
      out = os.path.join(self.scene.folder, "frame_%s.npy" % threads)
      env = dict(os.environ, SWARM_RENDER_THREADS=threads)
      subprocess.check_call([sys.executable, os.path.abspath(__file__), "--frame", out], env=env,
                            cwd=os.path.dirname(os.path.abspath(__file__)))
      frames.append(np.load(out))
    self.assertTrue(np.array_equal(frames[0], frames[1]))
    self.assertTrue(np.array_equal(frames[0], frames[2]))


def save_frame(path):
  """Renders the full scene from the side and saves its grey bytes, for the thread test's child processes."""
  scene = full_scene()
  rgb, _, _, _ = scene.render(eye=SIDE)
  np.save(path, rgb)
  p.disconnect()


if __name__ == "__main__":
  if len(sys.argv) == 3 and sys.argv[1] == "--frame":
    save_frame(sys.argv[2])
  else:
    unittest.main()
