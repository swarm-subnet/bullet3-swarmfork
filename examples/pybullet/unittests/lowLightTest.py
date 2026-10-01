"""The camera at night: ER_SWARM_LOW_LIGHT (white balance, auto exposure up to a gain cap, grain from the light each
pixel collected, colour fading with it), ER_SWARM_NEAR_INFRARED (grey, surfaces by their near-infrared albedo) and the
per-request spot light. Ray-cast colour path only; off by default and byte-identical when off."""
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pybullet as p

from daylightTest import RAYCAST, write_obj

SIZE = 96
LOW_LIGHT = getattr(p, "ER_SWARM_LOW_LIGHT", 0)
NEAR_INFRARED = getattr(p, "ER_SWARM_NEAR_INFRARED", 0)
LINEAR = RAYCAST | getattr(p, "ER_SWARM_LINEAR_LIGHT", 0) | getattr(p, "ER_ALPHA_CUTOUT", 0)
THERMAL = getattr(p, "ER_SWARM_THERMAL", 0)
MOON = [0.3, 0.2, 0.93]
MOON_COLOUR = [0.62, 0.70, 0.90]
HEIGHT = 10.0
FLAT = (slice(24, 72), slice(24, 72))


def to_linear(rgb):
  """Linear light of sRGB bytes."""
  c = np.asarray(rgb, dtype=np.float64) / 255.0
  return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def luma(linear):
  """Rec. 709 luminance of linear light."""
  return linear[..., 0] * 0.2126 + linear[..., 1] * 0.7152 + linear[..., 2] * 0.0722


class Scene(object):
  """A grey ground 40 m across under a weak moon, seen straight down from 10 m; tests add boxes to it."""

  def __init__(self):
    """A DIRECT client, a temporary folder and the ground."""
    p.connect(p.DIRECT)
    self.folder = tempfile.mkdtemp(prefix="lowlight_")
    self.proj = p.computeProjectionMatrixFOV(90, 1.0, 0.1, 100.0)
    path = os.path.join(self.folder, "ground.obj")
    write_obj(path, 20.0)
    vis = p.createVisualShape(p.GEOM_MESH, fileName=path, rgbaColor=[0.5, 0.5, 0.5, 1], specularColor=[0, 0, 0])
    self.ground = p.createMultiBody(0, -1, vis)

  def box(self, centre, half, colour):
    """A box with no specular colour."""
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=list(half), rgbaColor=list(colour), specularColor=[0, 0, 0])
    return p.createMultiBody(0, -1, vis, basePosition=list(centre))

  def render(self, flags=LINEAR | LOW_LIGHT, ambient=0.2, diffuse=0.1, colour=(1, 1, 1), height=HEIGHT, **camera):
    """Returns (rgb, seg) of one frame looking straight down; `camera` carries the sensor and spot arguments."""
    view = p.computeViewMatrix([0, 0, height], [0, 0, 0], [0, 1, 0])
    _, _, rgb, _, seg = p.getCameraImage(SIZE, SIZE, view, self.proj, lightDirection=MOON, lightColor=list(colour),
                                         lightAmbientCoeff=ambient, lightDiffuseCoeff=diffuse, lightSpecularCoeff=0.0, shadow=0,
                                         renderer=p.ER_TINY_RENDERER, flags=flags, **camera)
    return np.asarray(rgb, dtype=np.uint8).reshape(SIZE, SIZE, 4)[:, :, :3], np.asarray(seg).reshape(SIZE, SIZE)


def spot_down(height=HEIGHT, angle=20.0, reach=50.0, intensity=100.0):
  """A spot light at the camera pointing straight down."""
  return [0.0, 0.0, height, 0.0, 0.0, -1.0, angle, reach, intensity]


@unittest.skipUnless(RAYCAST and LOW_LIGHT and NEAR_INFRARED, "wheel without the low-light camera")
class TestLowLight(unittest.TestCase):
  """Small worlds rendered through the low-light camera, the near infrared and the spot light, term by term."""

  def setUp(self):
    """A fresh ground for every test."""
    self.scene = Scene()

  def tearDown(self):
    """Drops the client."""
    p.disconnect()

  def grain(self, rgb):
    """Standard deviation of the ground's luminance over its mean, in linear light."""
    y = luma(to_linear(rgb))[FLAT]
    return float(y.std() / y.mean())

  def test_arguments_without_the_flags_change_nothing(self):
    """Sensor arguments on a frame with neither flag and no spot light leave every byte as it was."""
    plain, _ = self.scene.render(flags=LINEAR)
    argued, _ = self.scene.render(flags=LINEAR, sensorPhotons=50.0, sensorReadNoise=3.0, sensorGainCap=100.0, sensorSeed=7)
    self.assertTrue(np.array_equal(plain, argued))

  def test_exposure_lifts_a_dark_frame_to_mid_grey(self):
    """A frame lit 50 times too weakly comes out at mid grey, while the plain frame stays dark."""
    plain, _ = self.scene.render(flags=LINEAR, ambient=0.02, diffuse=0.0)
    exposed, _ = self.scene.render(ambient=0.02, diffuse=0.0, sensorPhotons=0.0, sensorGainCap=1000.0)
    self.assertLess(float(luma(to_linear(plain)).mean()), 0.01)
    self.assertAlmostEqual(float(luma(to_linear(exposed)).mean()), 0.18, delta=0.01)

  def test_gain_cap_keeps_a_dark_frame_dark(self):
    """With the gain capped at 2 the frame is twice the plain frame's light, not mid grey."""
    plain, _ = self.scene.render(flags=LINEAR, ambient=0.02, diffuse=0.0)
    capped, _ = self.scene.render(ambient=0.02, diffuse=0.0, sensorPhotons=0.0, sensorGainCap=2.0)
    ratio = float(luma(to_linear(capped)).mean() / luma(to_linear(plain)).mean())
    self.assertAlmostEqual(ratio, 2.0, delta=0.1)

  def test_grain_grows_as_the_light_falls(self):
    """Ten times less light, exposed to the same grey, carries clearly more grain even after the noise reduction."""
    bright, _ = self.scene.render(ambient=0.2, diffuse=0.0, sensorPhotons=2000.0, sensorGainCap=1000.0, sensorSeed=3)
    dim, _ = self.scene.render(ambient=0.02, diffuse=0.0, sensorPhotons=2000.0, sensorGainCap=1000.0, sensorSeed=3)
    self.assertAlmostEqual(float(luma(to_linear(bright)).mean()), float(luma(to_linear(dim)).mean()), delta=0.02)
    self.assertGreater(self.grain(dim), 1.5 * self.grain(bright))

  def test_noise_reduction_smooths_a_dim_frame(self):
    """In deep dark the grain left is far below what the photons alone would give; where colour survives its noise is
    blotchy, not per pixel: neighbouring pixels share most of their colour."""
    dark, _ = self.scene.render(ambient=0.002, diffuse=0.0, sensorPhotons=2000.0, sensorReadNoise=5.0,
                                sensorGainCap=10000.0, sensorSeed=4)
    electrons = 0.002 * 0.212 * 2000.0
    raw = (electrons + 25.0) ** 0.5 / electrons
    self.assertLess(self.grain(dark), 0.5 * raw)
    dim, _ = self.scene.render(ambient=0.02, diffuse=0.0, sensorPhotons=2000.0, sensorGainCap=1000.0, sensorSeed=4)
    linear = to_linear(dim)[FLAT]
    chroma = linear[..., 0] - linear[..., 2]
    self.assertGreater(float(chroma.std()), 0.0)
    self.assertGreater(float(np.corrcoef(chroma[:, :-1].ravel(), chroma[:, 1:].ravel())[0, 1]), 0.5)

  def test_read_noise_adds_grain_in_the_dark(self):
    """Read noise raises the grain of a dim frame."""
    clean, _ = self.scene.render(ambient=0.02, diffuse=0.0, sensorPhotons=2000.0, sensorReadNoise=0.0, sensorGainCap=1000.0)
    noisy, _ = self.scene.render(ambient=0.02, diffuse=0.0, sensorPhotons=2000.0, sensorReadNoise=10.0, sensorGainCap=1000.0)
    self.assertGreater(self.grain(noisy), self.grain(clean) * 1.5)

  def test_no_photons_is_a_noiseless_sensor(self):
    """With sensorPhotons 0 the flat ground is one value."""
    rgb, _ = self.scene.render(ambient=0.05, diffuse=0.0, sensorPhotons=0.0, sensorGainCap=1000.0)
    self.assertEqual(len(np.unique(rgb[FLAT].reshape(-1, 3), axis=0)), 1)

  def test_seed_draws_the_grain(self):
    """The same seed gives the same bytes; another seed moves the grain but not the exposure."""
    first, _ = self.scene.render(sensorPhotons=500.0, sensorGainCap=1000.0, sensorSeed=1)
    same, _ = self.scene.render(sensorPhotons=500.0, sensorGainCap=1000.0, sensorSeed=1)
    other, _ = self.scene.render(sensorPhotons=500.0, sensorGainCap=1000.0, sensorSeed=2)
    self.assertTrue(np.array_equal(first, same))
    self.assertFalse(np.array_equal(first, other))
    self.assertAlmostEqual(float(luma(to_linear(first)).mean()), float(luma(to_linear(other)).mean()), delta=0.01)

  def test_white_balance_turns_the_moonlight_white(self):
    """Grey ground under the blue moon is blue on the plain frame and grey through the camera."""
    plain, _ = self.scene.render(flags=LINEAR, colour=MOON_COLOUR, ambient=0.0, diffuse=0.3)
    balanced, _ = self.scene.render(colour=MOON_COLOUR, ambient=0.0, diffuse=0.3, sensorPhotons=0.0, sensorGainCap=1000.0)
    plain_mean = plain[FLAT].reshape(-1, 3).mean(axis=0)
    balanced_mean = balanced[FLAT].reshape(-1, 3).mean(axis=0)
    self.assertGreater(plain_mean[2], plain_mean[0] + 10)
    self.assertLess(float(balanced_mean.max() - balanced_mean.min()), 2.0)

  def test_colour_fades_as_the_noise_grows(self):
    """A red box keeps less of its colour, averaged over the box, when it collects fewer photons."""
    box = self.scene.box((0, 0, 0.5), (3, 3, 0.5), (0.8, 0.1, 0.1, 1))

    def red_share(photons):
      """Mean red over mean green of the box's pixels."""
      rgb, seg = self.scene.render(sensorPhotons=photons, sensorGainCap=1000.0, sensorSeed=5)
      mean = to_linear(rgb)[seg == box].mean(axis=0)
      return float(mean[0] / mean[1])

    self.assertGreater(red_share(100000.0), 2.0 * red_share(20.0))

  def test_near_infrared_is_grey_and_foliage_bright(self):
    """Every pixel is grey; a green box outshines a blue one twice its visible luminance."""
    green = self.scene.box((-4, 0, 0.5), (2, 2, 0.5), (0.1, 0.4, 0.1, 1))
    blue = self.scene.box((4, 0, 0.5), (2, 2, 0.5), (0.1, 0.1, 0.9, 1))
    colour, seg = self.scene.render(flags=LINEAR)
    grey, _ = self.scene.render(flags=LINEAR | NEAR_INFRARED)
    self.assertTrue(np.array_equal(grey[..., 0], grey[..., 1]) and np.array_equal(grey[..., 0], grey[..., 2]))
    visible = luma(to_linear(colour))
    self.assertGreater(float(visible[seg == blue].mean()), 0.5 * float(visible[seg == green].mean()))
    infrared = to_linear(grey)[..., 0]
    self.assertGreater(float(infrared[seg == green].mean()), 2.0 * float(infrared[seg == blue].mean()))

  def test_spot_light_lights_its_cone_only(self):
    """In the dark the spot's centre is lit, its edge carries half as much as the axis would, and past it is black."""
    rgb, _ = self.scene.render(flags=LINEAR, ambient=0.0, diffuse=0.0, spotLight=spot_down(angle=40.0))
    light = luma(to_linear(rgb))
    self.assertGreater(float(light[SIZE // 2, SIZE // 2]), 0.05)
    self.assertEqual(int(rgb[2, 2].max()), 0)
    # At 20 degrees off the axis the ground is cos^3 dimmer for its distance and slant, and the cone halves it.
    radius = int(round(np.tan(np.radians(20.0)) * SIZE / 2))
    edge = float(np.mean([light[SIZE // 2, SIZE // 2 + radius], light[SIZE // 2, SIZE // 2 - radius]]))
    expected = float(light[SIZE // 2, SIZE // 2]) * 0.5 * np.cos(np.radians(20.0)) ** 3
    self.assertAlmostEqual(edge / expected, 1.0, delta=0.15)

  def test_spot_light_ends_at_its_range(self):
    """Ground 10 m below is dark for a 9 m spot and lit for a 20 m one."""
    short, _ = self.scene.render(flags=LINEAR, ambient=0.0, diffuse=0.0, spotLight=spot_down(reach=9.0))
    long_reach, _ = self.scene.render(flags=LINEAR, ambient=0.0, diffuse=0.0, spotLight=spot_down(reach=20.0))
    self.assertEqual(int(short.max()), 0)
    self.assertGreater(int(long_reach.max()), 0)

  def test_spot_light_falls_with_the_square_of_distance(self):
    """From twice as high the centre of the spot gets a quarter of the light."""
    near, _ = self.scene.render(flags=LINEAR, ambient=0.0, diffuse=0.0, height=5.0, spotLight=spot_down(height=5.0, reach=1000.0))
    far, _ = self.scene.render(flags=LINEAR, ambient=0.0, diffuse=0.0, height=10.0, spotLight=spot_down(height=10.0, reach=1000.0))
    centre = (slice(SIZE // 2 - 2, SIZE // 2 + 2), slice(SIZE // 2 - 2, SIZE // 2 + 2))
    ratio = float(luma(to_linear(near))[centre].mean() / luma(to_linear(far))[centre].mean())
    self.assertAlmostEqual(ratio, 4.0, delta=0.2)

  def test_spot_light_lasts_one_request(self):
    """A frame after a lit one, asked without the spot, is the frame that was never lit."""
    never, _ = self.scene.render(flags=LINEAR)
    self.scene.render(flags=LINEAR, spotLight=spot_down())
    after, _ = self.scene.render(flags=LINEAR)
    self.assertTrue(np.array_equal(never, after))

  def test_other_paths_ignore_the_flags(self):
    """On the rasteriser and on a thermal frame the two bits and a spot light change no byte."""
    self.scene.box((0, 0, 0.5), (1, 1, 0.5), (0.8, 0.1, 0.1, 1))
    for base in (0, RAYCAST | THERMAL):
      plain, _ = self.scene.render(flags=base)
      flagged, _ = self.scene.render(flags=base | LOW_LIGHT | NEAR_INFRARED, sensorPhotons=100.0, spotLight=spot_down())
      self.assertTrue(np.array_equal(plain, flagged), base)

  def test_threads_give_the_same_bytes(self):
    """Grainy colour and near infrared with the spot are byte-identical at 1, 2 and 4 threads, each in its own process."""
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
  """Renders a grainy colour frame and a lit near-infrared one and saves both, for the thread test's children."""
  scene = Scene()
  scene.box((0, 0, 0.5), (2, 2, 0.5), (0.8, 0.1, 0.1, 1))
  scene.box((-5, 3, 0.5), (2, 2, 0.5), (0.1, 0.4, 0.1, 1))
  colour, _ = scene.render(ambient=0.03, sensorPhotons=300.0, sensorReadNoise=4.0, sensorGainCap=500.0, sensorSeed=11)
  infrared, _ = scene.render(flags=LINEAR | LOW_LIGHT | NEAR_INFRARED, ambient=0.01, sensorPhotons=300.0,
                             sensorReadNoise=4.0, sensorGainCap=500.0, sensorSeed=12, spotLight=spot_down())
  np.save(path, np.stack([colour, infrared]))
  p.disconnect()


if __name__ == "__main__":
  if len(sys.argv) == 3 and sys.argv[1] == "--frame":
    save_frame(sys.argv[2])
  else:
    unittest.main()
