"""Alpha cut-outs on the ray-cast path: a see-through texel is a miss in colour, depth and shadow alike."""
import hashlib
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pybullet as p

SIZE = 96
TEX = 32
FRAME = 6  # opaque texels around the edge of the leaf card; everything inside is a hole
LIGHT = [0.0, 0.25, 0.968]  # sun high behind the card: its shadow lands on the floor in front, clear of the wall's
NEEDS_BACKEND = unittest.skipUnless(hasattr(p, "ER_ALPHA_CUTOUT") and hasattr(p, "ER_SWARM_RAYCAST"),
                                    "wheel built without the ray-cast backend")
RAY = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX | getattr(p, "ER_SWARM_RAYCAST", 0)
CUT = RAY | getattr(p, "ER_ALPHA_CUTOUT", 0)


def write_leaf_tga(path, with_alpha):
  """Writes a green TGA whose centre is transparent; a 24-bit copy without the alpha channel when asked."""
  yy, xx = np.mgrid[0:TEX, 0:TEX]
  edge = (xx < FRAME) | (xx >= TEX - FRAME) | (yy < FRAME) | (yy >= TEX - FRAME)
  bgr = np.zeros((TEX, TEX, 3), dtype=np.uint8)
  bgr[..., 1] = 160
  bgr[..., 0] = 40
  bgr[..., 2] = 40
  if with_alpha:
    alpha = np.where(edge, 255, 0).astype(np.uint8)[..., None]
    pixels = np.concatenate([bgr, alpha], axis=2)
  else:
    pixels = bgr
  header = struct.pack('<BBBHHBHHHHBB', 0, 0, 2, 0, 0, 0, 0, 0, TEX, TEX, 8 * pixels.shape[2], 8 if with_alpha else 0)
  with open(path, 'wb') as f:
    f.write(header)
    f.write(pixels.tobytes())


def build_world(tex_path):
  """A grey floor, a leaf card standing on it and a red wall behind the card; returns (card id, wall id)."""
  floor = p.createVisualShape(p.GEOM_BOX, halfExtents=[6, 6, 0.05], rgbaColor=[0.7, 0.7, 0.7, 1])
  p.createMultiBody(baseMass=0, baseVisualShapeIndex=floor, basePosition=[0, 0, -0.05])
  card = p.createVisualShape(p.GEOM_MESH, vertices=[[-1, 0, 0.5], [1, 0, 0.5], [1, 0, 2.5], [-1, 0, 2.5]],
                             indices=[0, 1, 2, 0, 2, 3], uvs=[[0, 0], [1, 0], [1, 1], [0, 1]],
                             normals=[[0, -1, 0]] * 4, rgbaColor=[1, 1, 1, 1])
  card_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=card)
  p.changeVisualShape(card_id, -1, textureUniqueId=p.loadTexture(tex_path), flags=p.VISUAL_SHAPE_DOUBLE_SIDED_MULTIBODY)
  wall = p.createVisualShape(p.GEOM_BOX, halfExtents=[3, 0.1, 3], rgbaColor=[1, 0, 0, 1])
  wall_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=wall, basePosition=[0, 2.5, 3])
  return card_id, wall_id


def render(flags, shadow=0, eye=(0.0, -4.5, 2.2), target=(0.0, 0.0, 1.3)):
  """Returns (rgb HxWx3 or None, depth HxW, seg HxW or None) from a camera at eye looking at target."""
  view = p.computeViewMatrix(list(eye), list(target), [0, 0, 1])
  proj = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 30.0)
  _, _, rgb, depth, seg = p.getCameraImage(SIZE, SIZE, view, proj, shadow=shadow, lightDirection=LIGHT,
                                           renderer=p.ER_TINY_RENDERER, flags=flags)
  rgb = None if rgb is None else np.asarray(rgb).reshape(SIZE, SIZE, 4)[:, :, :3].astype(int)
  seg = None if seg is None else np.asarray(seg).reshape(SIZE, SIZE)
  return rgb, np.asarray(depth).reshape(SIZE, SIZE), seg


@NEEDS_BACKEND
class TestAlphaCutout(unittest.TestCase):
  """One leaf card with a transparent centre, rendered with and without the cut-out flag."""

  def setUp(self):
    """Connects and builds the scene around the see-through leaf texture."""
    p.connect(p.DIRECT)
    self.tmp = tempfile.mkdtemp()
    self.tex_path = os.path.join(self.tmp, 'leaf.tga')
    write_leaf_tga(self.tex_path, with_alpha=True)
    self.card, self.wall = build_world(self.tex_path)

  def tearDown(self):
    """Disconnects and removes the temporary textures."""
    p.disconnect()
    for name in os.listdir(self.tmp):
      os.remove(os.path.join(self.tmp, name))
    os.rmdir(self.tmp)

  def hole(self):
    """Pixels the card covers without the flag and gives up with it: the transparent centre."""
    _, _, solid = render(RAY)
    _, _, cut = render(CUT)
    hole = (solid == self.card) & (cut != self.card)
    self.assertGreater(int(hole.sum()), 100)
    self.assertGreater(int((cut == self.card).sum()), 50)
    return hole

  def test_hole_shows_the_wall_behind_in_colour_and_segmentation(self):
    """Through the hole every ray reaches the red wall: its id in the mask and only red in the colour."""
    hole = self.hole()
    rgb, _, seg = render(CUT)
    self.assertTrue((seg[hole] == self.wall).all())
    self.assertTrue((rgb[hole][:, 1:] == 0).all())
    self.assertTrue((rgb[hole][:, 0] > 0).all())

  def test_depth_sees_the_same_hole_as_colour(self):
    """The hole is farther than the card frame in the depth buffer, and a depth-only render gives the same bytes."""
    hole = self.hole()
    _, depth, seg = render(CUT)
    _, depth_only, _ = render(p.ER_NO_SEGMENTATION_MASK | p.ER_SWARM_RAYCAST | p.ER_DEPTH_ONLY | p.ER_ALPHA_CUTOUT)
    self.assertEqual(depth.tobytes(), depth_only.tobytes())
    self.assertGreater(float(depth[hole].min()), float(depth[seg == self.card].max()))

  def test_shadow_falls_through_the_hole(self):
    """The card's shadow on the floor keeps the hole: fewer shaded floor pixels than with a solid card."""
    shaded_solid = self.shaded_floor(RAY)
    shaded_cut = self.shaded_floor(CUT)
    self.assertGreater(shaded_cut, 20)
    self.assertLess(shaded_cut, shaded_solid - 20)

  def test_shadow_map_shadow_falls_through_the_hole(self):
    """Shadows read from the light's depth map keep the hole too, and the map is recast when the flag flips."""
    smap = p.ER_SWARM_SHADOW_MAP
    shaded_solid = self.shaded_floor(RAY | smap)
    shaded_cut = self.shaded_floor(CUT | smap)
    self.assertGreater(shaded_cut, 20)
    self.assertLess(shaded_cut, shaded_solid - 20)
    # Back to the solid setting on the same client: a stale map would hand back the cut-out counts.
    self.assertEqual(self.shaded_floor(RAY | smap), shaded_solid)

  def test_mover_shadow_falls_through_the_hole(self):
    """A card that moved casts its mover shadow through the hole as well."""
    flags = p.ER_SWARM_SHADOW_MAP | p.ER_SWARM_MOVER_SHADOW
    render(RAY)
    p.resetBasePositionAndOrientation(self.card, [0.3, 0.2, 0.0], [0, 0, 0, 1])
    self.assertLess(self.shaded_floor(CUT | flags), self.shaded_floor(RAY | flags) - 20)

  def test_moved_card_keeps_its_holes_and_lets_light_through(self):
    """A card moved after the first frame, which the scene then tracks as an instance, cuts out the same way."""
    render(RAY)
    p.resetBasePositionAndOrientation(self.card, [0.3, 0.2, 0.0], [0, 0, 0, 1])
    hole = self.hole()
    _, _, seg = render(CUT)
    self.assertTrue((seg[hole] == self.wall).all())
    self.assertLess(self.shaded_floor(CUT), self.shaded_floor(RAY) - 20)

  def test_rasteriser_ignores_the_flag(self):
    """TinyRenderer's own path draws the card solid and gives the same bytes with the flag and without."""
    tiny = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
    rgb_a, depth_a, seg_a = render(tiny, shadow=1)
    rgb_b, depth_b, seg_b = render(tiny | p.ER_ALPHA_CUTOUT, shadow=1)
    self.assertEqual(rgb_a.tobytes(), rgb_b.tobytes())
    self.assertEqual(depth_a.tobytes(), depth_b.tobytes())
    self.assertEqual(seg_a.tobytes(), seg_b.tobytes())
    _, _, ray_solid = render(RAY)
    self.assertEqual(int((seg_a == self.card).sum()), int((ray_solid == self.card).sum()))

  def test_opaque_texture_is_untouched_by_the_flag(self):
    """The same picture without an alpha channel renders byte for byte the same whether the flag is set or not."""
    opaque = os.path.join(self.tmp, 'leaf_opaque.tga')
    write_leaf_tga(opaque, with_alpha=False)
    p.changeVisualShape(self.card, -1, textureUniqueId=p.loadTexture(opaque))
    rgb_a, depth_a, seg_a = render(RAY, shadow=1)
    rgb_b, depth_b, seg_b = render(CUT, shadow=1)
    self.assertEqual(rgb_a.tobytes(), rgb_b.tobytes())
    self.assertEqual(depth_a.tobytes(), depth_b.tobytes())
    self.assertGreater(int((seg_a == self.card).sum()), 150)

  def shaded_floor(self, flags):
    """Number of floor pixels that get darker when shadows are switched on."""
    lit, _, seg = render(flags, shadow=0)
    shaded, _, _ = render(flags, shadow=1)
    return int(((shaded < lit).any(axis=2) & (seg == 0)).sum())


@NEEDS_BACKEND
class TestAlphaCutoutThreads(unittest.TestCase):
  """The cut-out frame does not depend on how many render threads are used."""

  def test_same_bytes_for_one_two_and_four_threads(self):
    """Renders the scene in a fresh process per thread count and compares the frame hashes."""
    digests = set()
    for threads in ("1", "2", "4"):
      env = dict(os.environ, SWARM_RENDER_THREADS=threads)
      out = subprocess.check_output([sys.executable, __file__, "--hash"], env=env, text=True)
      digests.add(out.strip())
    self.assertEqual(len(digests), 1)


def frame_hash():
  """Prints the sha256 of one shadowed, filtered, cut-out colour and depth frame on the ray-cast path."""
  p.connect(p.DIRECT)
  tmp = tempfile.mkdtemp()
  tex_path = os.path.join(tmp, 'leaf.tga')
  write_leaf_tga(tex_path, with_alpha=True)
  build_world(tex_path)
  rgb, depth, _ = render(CUT | p.ER_TEXTURE_FILTER, shadow=1)
  print(hashlib.sha256(rgb.astype(np.uint8).tobytes() + depth.tobytes()).hexdigest())
  p.disconnect()
  os.remove(tex_path)
  os.rmdir(tmp)


if __name__ == '__main__':
  if "--hash" in sys.argv:
    frame_hash()
  else:
    unittest.main()
