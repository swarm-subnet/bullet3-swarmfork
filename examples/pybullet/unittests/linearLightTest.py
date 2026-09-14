"""Linear-light shading on getCameraImage: behind ER_SWARM_LINEAR_LIGHT the ray-cast path lights, blends and glints in linear light and writes sRGB bytes; without it, and on the rasteriser, nothing moves."""
import unittest
import numpy as np
import pybullet as p

SIZE = 96
# Straight down onto the slab with the light straight above it, so the diffuse term is exactly one.
EYE = ([0, 0, 6], [0, 0, 0], [0, 1, 0])
LIGHT = [0.0, 0.0, 1.0]
SKY = {"skyHorizonColor": [0.9, 0.3, 0.3], "skyZenithColor": [0.3, 0.3, 0.9]}
HAS_RAYCAST = hasattr(p, "ER_SWARM_RAYCAST")


def build_world(slab_rgba, box_rgba=(1, 1, 1, 1)):
  """A large flat slab and a box standing on it, each in one plain colour; returns their body ids."""
  slab = p.createVisualShape(p.GEOM_BOX, halfExtents=[20, 20, 0.05], rgbaColor=list(slab_rgba))
  slab_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=slab, basePosition=[0, 0, -0.05])
  box = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.7, 0.7, 0.7], rgbaColor=list(box_rgba))
  box_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=box, basePosition=[0.3, 0.2, 0.7])
  return slab_id, box_id


def render(flags=0, ambient=0.6, diffuse=0.35, specular=0.05, **extra):
  """Returns (rgb HxWx3 int, depth, seg) for the fixed camera; flags are added to the segmentation request."""
  eye, target, up = EYE
  view = p.computeViewMatrix(eye, target, up)
  proj = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 60.0)
  _, _, rgb, depth, seg = p.getCameraImage(SIZE, SIZE, view, proj, shadow=0, lightDirection=LIGHT,
                                           lightAmbientCoeff=ambient, lightDiffuseCoeff=diffuse,
                                           lightSpecularCoeff=specular, renderer=p.ER_TINY_RENDERER,
                                           flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX | flags, **extra)
  rgb = np.asarray(rgb).reshape(SIZE, SIZE, 4)[:, :, :3].astype(int)
  return rgb, np.asarray(depth), np.asarray(seg).reshape(SIZE, SIZE)


def interior(seg, body):
  """Pixels of `body` whose eight neighbours are the same body: away from every silhouette."""
  same = np.ones((SIZE - 2, SIZE - 2), dtype=bool)
  centre = seg[1:-1, 1:-1]
  for dy in (-1, 0, 1):
    for dx in (-1, 0, 1):
      same &= seg[1 + dy:SIZE - 1 + dy, 1 + dx:SIZE - 1 + dx] == centre
  mask = np.zeros((SIZE, SIZE), dtype=bool)
  mask[1:-1, 1:-1] = same & (centre == body)
  return mask


@unittest.skipUnless(HAS_RAYCAST, "wheel built without the ray-cast backend")
class TestLinearLight(unittest.TestCase):
  """Renders a white slab and a white box on the ray-cast path with the linear-light flag off and on."""

  RAY = p.ER_SWARM_RAYCAST if HAS_RAYCAST else 0
  LINEAR = getattr(p, "ER_SWARM_LINEAR_LIGHT", 0)

  def setUp(self):
    """Connects a DIRECT client and builds the white scene."""
    p.connect(p.DIRECT)
    self.slab, self.box = build_world((1, 1, 1, 1))

  def tearDown(self):
    """Drops the client."""
    p.disconnect()

  def test_flag_is_exported(self):
    """The module carries the flag at its own bit."""
    self.assertEqual(p.ER_SWARM_LINEAR_LIGHT, 4096)

  def test_rasteriser_ignores_the_flag(self):
    """On the rasteriser the flag changes no byte."""
    plain, _, _ = render()
    flagged, _, _ = render(self.LINEAR)
    self.assertEqual(plain.tobytes(), flagged.tobytes())

  def test_full_ambient_round_trips_exactly(self):
    """With only a full ambient term the decode and the encode cancel: the bytes match the flag-off frame."""
    plain, _, _ = render(self.RAY, ambient=1.0, diffuse=0.0, specular=0.0)
    linear, _, _ = render(self.RAY | self.LINEAR, ambient=1.0, diffuse=0.0, specular=0.0)
    self.assertEqual(plain.tobytes(), linear.tobytes())

  def test_half_ambient_is_half_the_light_not_half_the_byte(self):
    """Half ambient on white gives byte 127 on encoded values and byte 188, the sRGB of 0.5, in linear light."""
    plain, _, seg = render(self.RAY, ambient=0.5, diffuse=0.0, specular=0.0)
    linear, _, _ = render(self.RAY | self.LINEAR, ambient=0.5, diffuse=0.0, specular=0.0)
    mask = interior(seg, self.slab)
    self.assertGreater(int(mask.sum()), SIZE * SIZE // 4)
    self.assertTrue((plain[mask] == 127).all())
    self.assertTrue((linear[mask] == 188).all())

  def test_grey_body_colour_is_decoded_too(self):
    """A mid-grey body under half ambient goes darker than half its byte in linear light, as light does."""
    p.changeVisualShape(self.slab, -1, rgbaColor=[0.5, 0.5, 0.5, 1])
    plain, _, seg = render(self.RAY, ambient=0.5, diffuse=0.0, specular=0.0)
    linear, _, _ = render(self.RAY | self.LINEAR, ambient=0.5, diffuse=0.0, specular=0.0)
    mask = interior(seg, self.slab)
    self.assertTrue((plain[mask] == 63).all())
    self.assertTrue((linear[mask] == 92).all())

  def test_edge_blend_averages_light(self):
    """Where anti-aliasing mixes the white box into a black slab, the mix lands above the byte average in linear light."""
    p.changeVisualShape(self.slab, -1, rgbaColor=[0, 0, 0, 1])
    aa = self.RAY | p.ER_EDGE_ANTIALIAS
    plain, _, seg = render(aa, ambient=1.0, diffuse=0.0, specular=0.0)
    linear, _, _ = render(aa | self.LINEAR, ambient=1.0, diffuse=0.0, specular=0.0)
    edge = ~(interior(seg, self.slab) | interior(seg, self.box))
    mixed = edge & (plain[:, :, 0] > 0) & (plain[:, :, 0] < 255)
    self.assertGreater(int(mixed.sum()), 20)
    self.assertTrue((linear[mixed] >= plain[mixed]).all())
    self.assertGreater(float((linear[mixed] - plain[mixed]).mean()), 10.0)
    self.assertEqual(plain[~edge].tobytes(), linear[~edge].tobytes())

  def test_glint_blends_in_linear(self):
    """With the glint on, the flag still moves a glossy slab towards the sky and leaves a matte box alone."""
    p.changeVisualShape(self.slab, -1, specularColor=[1, 1, 1])
    p.changeVisualShape(self.box, -1, specularColor=[0, 0, 0])
    linear = self.RAY | self.LINEAR
    plain, _, seg = render(linear, ambient=0.5, diffuse=0.0, specular=0.0, **SKY)
    glinted, _, _ = render(linear | p.ER_SPECULAR_GLINT, ambient=0.5, diffuse=0.0, specular=0.0, **SKY)
    slab = interior(seg, self.slab)
    box = interior(seg, self.box)
    self.assertFalse((plain[slab] == glinted[slab]).all())
    self.assertTrue((plain[box] == glinted[box]).all())

  def test_sun_sky_ambient_tint_is_lit_in_linear(self):
    """The sun sky's ambient tint reaches the linear path: its slab differs from the untinted one and from the encoded tint."""
    sun = self.RAY | getattr(p, "ER_SWARM_SKY_SUN", 0)
    untinted, _, seg = render(self.RAY | self.LINEAR, ambient=0.5, diffuse=0.0, specular=0.0)
    tinted, _, _ = render(sun | self.LINEAR, ambient=0.5, diffuse=0.0, specular=0.0)
    encoded, _, _ = render(sun, ambient=0.5, diffuse=0.0, specular=0.0)
    mask = interior(seg, self.slab)
    self.assertFalse((tinted[mask] == untinted[mask]).all())
    self.assertTrue((tinted[mask] > encoded[mask]).all())

  def test_depth_and_mask_untouched(self):
    """The flag changes colour only; depth and segmentation are byte-identical."""
    _, plain_depth, plain_seg = render(self.RAY)
    _, linear_depth, linear_seg = render(self.RAY | self.LINEAR)
    self.assertEqual(plain_depth.tobytes(), linear_depth.tobytes())
    self.assertEqual(plain_seg.tobytes(), linear_seg.tobytes())


if __name__ == "__main__":
  unittest.main()
