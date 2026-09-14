"""Glass glint on getCameraImage: the per-object specular colour reflects the sky, more at grazing angles, only behind ER_SPECULAR_GLINT."""
import os
import tempfile
import unittest
import numpy as np
import pybullet as p

SIZE = 96
LIGHT = [0.3, -0.4, 0.86]
RED_SKY = {"skyHorizonColor": [1.0, 0.2, 0.2], "skyZenithColor": [0.2, 0.2, 1.0]}
BLUE_SKY = {"skyHorizonColor": [0.2, 0.2, 1.0], "skyZenithColor": [1.0, 0.2, 0.2]}
# Straight down onto the floor, and along it from one metre up so the far floor is seen at a grazing angle.
EYES = {"down": ([0, 0, 8], [0, 0, 0], [0, 1, 0]), "grazing": ([0, -12, 1.0], [0, 0, 0], [0, 0, 1])}
URDF = """<robot name="glass">
  <material name="pane"><color rgba="0.5 0.5 0.5 1"/><specular rgb="1 1 1"/></material>
  <link name="base">
    <visual><geometry><box size="40 40 0.1"/></geometry><material name="pane"/></visual>
  </link>
</robot>
"""
HAS_RAYCAST = hasattr(p, "ER_SWARM_RAYCAST")


def build_world(floor_specular, box_specular=(0, 0, 0)):
  """A large grey floor slab and a red box, each with its own specular colour; returns their body ids."""
  floor = p.createVisualShape(p.GEOM_BOX, halfExtents=[20, 20, 0.05], rgbaColor=[0.5, 0.5, 0.5, 1],
                              specularColor=list(floor_specular))
  floor_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=floor, basePosition=[0, 0, -0.05])
  box = p.createVisualShape(p.GEOM_BOX, halfExtents=[1, 1, 1], rgbaColor=[1, 0, 0, 1],
                            specularColor=list(box_specular))
  box_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=box, basePosition=[0, -1, 1])
  return floor_id, box_id


def render(look="grazing", flags=0, **sky):
  """Returns (rgb HxWx3 int, depth, seg) for the named camera; flags are added to the segmentation request."""
  eye, target, up = EYES[look]
  view = p.computeViewMatrix(eye, target, up)
  proj = p.computeProjectionMatrixFOV(70, 1.0, 0.1, 60.0)
  _, _, rgb, depth, seg = p.getCameraImage(SIZE, SIZE, view, proj, shadow=0, lightDirection=LIGHT,
                                           renderer=p.ER_TINY_RENDERER,
                                           flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX | flags, **sky)
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


class TestGlint(unittest.TestCase):
  """Renders a glossy floor and a matte box with and without the glint flag."""

  def setUp(self):
    """Connects a DIRECT client and builds the scene with a mirror-white floor."""
    p.connect(p.DIRECT)
    self.floor, self.box = build_world((1, 1, 1))

  def tearDown(self):
    """Drops the client."""
    p.disconnect()

  def floor_shift(self, look, flags=0, **sky):
    """Mean per-channel change on the floor interior when the glint is switched on."""
    plain, _, seg = render(look, flags, **sky)
    glint, _, _ = render(look, flags | p.ER_SPECULAR_GLINT, **sky)
    mask = interior(seg, self.floor)
    self.assertGreater(int(mask.sum()), SIZE * SIZE // 8)
    return (glint[mask] - plain[mask]).mean(axis=0)

  def test_flag_off_ignores_the_specular_colour(self):
    """Without the flag, a mirror floor and a matte floor give the same bytes."""
    mirror, _, _ = render("grazing", **RED_SKY)
    p.changeVisualShape(self.floor, -1, specularColor=[0, 0, 0])
    matte, _, _ = render("grazing", **RED_SKY)
    self.assertEqual(mirror.tobytes(), matte.tobytes())

  def test_grazing_view_reflects_more_sky_than_straight_down(self):
    """The floor brightens towards the sky at a grazing angle and barely changes seen head on."""
    grazing = float(np.abs(self.floor_shift("grazing", **RED_SKY)).max())
    down = float(np.abs(self.floor_shift("down", **RED_SKY)).max())
    self.assertGreater(grazing, 15.0)
    self.assertLess(down, 8.0)
    self.assertGreater(grazing, 3.0 * down)

  def test_glint_carries_the_sky_colour(self):
    """A red horizon pushes the grazing floor towards red, a blue one towards blue."""
    red = self.floor_shift("grazing", **RED_SKY)
    blue = self.floor_shift("grazing", **BLUE_SKY)
    self.assertGreater(red[0], red[2] + 10)
    self.assertGreater(blue[2], blue[0] + 10)

  def test_no_sky_reflects_white(self):
    """With no sky given, the glint blends towards the white background, every channel alike."""
    shift = self.floor_shift("grazing")
    self.assertGreater(float(shift.min()), 10.0)
    self.assertLess(float(shift.max() - shift.min()), 2.0)

  def test_specular_zero_keeps_the_colour(self):
    """The matte box keeps its bytes while the floor around it changes."""
    plain, _, seg = render("grazing", **RED_SKY)
    glint, _, _ = render("grazing", p.ER_SPECULAR_GLINT, **RED_SKY)
    box = interior(seg, self.box)
    self.assertGreater(int(box.sum()), 20)
    self.assertTrue((plain[box] == glint[box]).all())
    self.assertFalse((plain[interior(seg, self.floor)] == glint[interior(seg, self.floor)]).all())

  def test_change_visual_shape_updates_the_glint(self):
    """A specular colour set after creation reaches the renderer."""
    before = self.floor_shift("grazing", **RED_SKY)
    p.changeVisualShape(self.floor, -1, specularColor=[0, 0, 0])
    after = self.floor_shift("grazing", **RED_SKY)
    self.assertGreater(float(np.abs(before).max()), 15.0)
    self.assertEqual(float(np.abs(after).max()), 0.0)

  def test_depth_and_mask_untouched(self):
    """The glint changes colour only; depth and segmentation are byte-identical."""
    _, plain_depth, plain_seg = render("grazing", **RED_SKY)
    _, glint_depth, glint_seg = render("grazing", p.ER_SPECULAR_GLINT, **RED_SKY)
    self.assertEqual(plain_depth.tobytes(), glint_depth.tobytes())
    self.assertEqual(plain_seg.tobytes(), glint_seg.tobytes())

  @unittest.skipUnless(HAS_RAYCAST, "wheel built without the ray-cast backend")
  def test_ray_cast_path_glints_the_same_way(self):
    """Both colour paths agree on the glinted floor away from silhouettes."""
    tiny, _, seg = render("grazing", p.ER_SPECULAR_GLINT, **RED_SKY)
    ray, _, _ = render("grazing", p.ER_SPECULAR_GLINT | p.ER_SWARM_RAYCAST, **RED_SKY)
    mask = interior(seg, self.floor)
    diff = np.abs(tiny[mask] - ray[mask])
    self.assertLess(float(diff.mean()), 2.0)
    self.assertLessEqual(int(np.percentile(diff, 99)), 8)


class TestGlintFromUrdf(unittest.TestCase):
  """The specular colour of a URDF material reaches the renderer."""

  def test_urdf_material_specular_glints(self):
    """A URDF slab whose material carries a white specular colour brightens towards the sky at a grazing angle."""
    p.connect(p.DIRECT)
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "glass.urdf")
    with open(path, "w") as handle:
      handle.write(URDF)
    try:
      slab = p.loadURDF(path, basePosition=[0, 0, -0.05], useFixedBase=True)
      plain, _, seg = render("grazing", **RED_SKY)
      glint, _, _ = render("grazing", p.ER_SPECULAR_GLINT, **RED_SKY)
      mask = interior(seg, slab)
      self.assertGreater(int(mask.sum()), SIZE * SIZE // 8)
      self.assertGreater(float((glint[mask] - plain[mask]).mean(axis=0)[0]), 20.0)
    finally:
      p.disconnect()
      os.remove(path)
      os.rmdir(tmp)


if __name__ == '__main__':
  unittest.main()
