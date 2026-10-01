"""Edge-only anti-aliasing on the ray-cast path: flat pixels, depth and the mask keep their bytes, edges blend."""
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pybullet as p

SIZE = 96
LIGHT = [-0.3, 0.6, 0.74]
NEEDS_FLAG = unittest.skipUnless(hasattr(p, "ER_EDGE_ANTIALIAS"), "wheel built without edge anti-aliasing")


def build_world():
  """A white floor, a red box and a grey sphere: one plane, one creased body, one curved body."""
  floor = p.createVisualShape(p.GEOM_MESH, vertices=[[-4, -4, 0], [4, -4, 0], [4, 4, 0], [-4, 4, 0]],
                              indices=[0, 1, 2, 0, 2, 3], uvs=[[0, 0], [4, 0], [4, 4], [0, 4]],
                              normals=[[0, 0, 1]] * 4, rgbaColor=[1, 1, 1, 1])
  p.createMultiBody(baseMass=0, baseVisualShapeIndex=floor)
  box = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], rgbaColor=[1, 0, 0, 1])
  p.createMultiBody(baseMass=0, baseVisualShapeIndex=box, basePosition=[0, 0, 0.5])
  sphere = p.createVisualShape(p.GEOM_SPHERE, radius=0.6, rgbaColor=[0.6, 0.6, 0.6, 1])
  p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphere, basePosition=[1.6, 1.0, 0.6])


def render(flags, eye=(3.0, -3.0, 2.5), target=(0, 0, 0.4), **extra):
  """Returns (rgb HxWx3 int, depth HxW, seg HxW) from a camera at eye looking at target."""
  view = p.computeViewMatrix(list(eye), list(target), [0, 0, 1])
  proj = p.computeProjectionMatrixFOV(70, 1.0, 0.1, 30.0)
  _, _, rgb, depth, seg = p.getCameraImage(SIZE, SIZE, view, proj, lightDirection=LIGHT,
                                           renderer=p.ER_TINY_RENDERER, flags=flags, **extra)
  rgb = None if rgb is None else np.asarray(rgb).reshape(SIZE, SIZE, 4)[:, :, :3].astype(int)
  seg = None if seg is None else np.asarray(seg).reshape(SIZE, SIZE)
  return rgb, np.asarray(depth).reshape(SIZE, SIZE), seg


def square(x, y, z, r):
  """Corners of a level square of half side r centred on (x, y, z), anticlockwise from above."""
  return [[x - r, y - r, z], [x + r, y - r, z], [x + r, y + r, z], [x - r, y + r, z]]


def near_a_mask_change(seg, radius):
  """Pixels within radius of a neighbour that belongs to another object."""
  change = np.zeros((SIZE, SIZE), dtype=bool)
  change[:, 1:] |= seg[:, 1:] != seg[:, :-1]
  change[:, :-1] |= seg[:, 1:] != seg[:, :-1]
  change[1:, :] |= seg[1:, :] != seg[:-1, :]
  change[:-1, :] |= seg[1:, :] != seg[:-1, :]
  out = change.copy()
  for _ in range(radius):
    grown = out.copy()
    grown[:, 1:] |= out[:, :-1]
    grown[:, :-1] |= out[:, 1:]
    grown[1:, :] |= out[:-1, :]
    grown[:-1, :] |= out[1:, :]
    out = grown
  return out


@NEEDS_FLAG
class TestEdgeAntialias(unittest.TestCase):
  """Renders one scene with and without the flag and compares the buffers."""

  BASE = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX | p.ER_SWARM_RAYCAST
  AA = BASE | p.ER_EDGE_ANTIALIAS

  def setUp(self):
    """Connects and builds the scene."""
    p.connect(p.DIRECT)
    build_world()

  def tearDown(self):
    """Disconnects."""
    p.disconnect()

  def test_depth_and_mask_keep_their_bytes(self):
    """The flag changes no depth and no segmentation byte, with or without the mask."""
    _, depth, seg = render(self.BASE)
    _, depth_aa, seg_aa = render(self.AA)
    self.assertEqual(depth.tobytes(), depth_aa.tobytes())
    self.assertEqual(seg.tobytes(), seg_aa.tobytes())
    _, depth, _ = render(p.ER_NO_SEGMENTATION_MASK | p.ER_SWARM_RAYCAST)
    _, depth_aa, _ = render(p.ER_NO_SEGMENTATION_MASK | p.ER_SWARM_RAYCAST | p.ER_EDGE_ANTIALIAS)
    self.assertEqual(depth.tobytes(), depth_aa.tobytes())

  def test_flat_pixels_keep_their_bytes_and_edges_change(self):
    """Floor pixels away from every silhouette are untouched; pixels next to a silhouette blend."""
    rgb, _, seg = render(self.BASE)
    rgb_aa, _, _ = render(self.AA)
    changed = (rgb != rgb_aa).any(axis=2)
    flat_floor = (seg == 0) & ~near_a_mask_change(seg, 2)
    self.assertGreater(int(flat_floor.sum()), SIZE * SIZE // 8)
    self.assertFalse(changed[flat_floor].any())
    silhouette = near_a_mask_change(seg, 1)
    self.assertGreater(int((changed & silhouette).sum()), SIZE // 2)
    # Everything that moved is a silhouette pixel or lies on the creased box or the curved sphere.
    self.assertFalse((changed & ~silhouette & (seg == 0)).any())
    self.assertFalse((changed & ~silhouette & (seg < 0)).any())

  def test_blend_lies_between_the_two_sides(self):
    """On the red box against the white floor, a blended pixel stays between red and the floor colour."""
    rgb, _, seg = render(self.BASE)
    rgb_aa, _, _ = render(self.AA)
    changed = (rgb != rgb_aa).any(axis=2)
    box_edge = changed & (seg == 1)
    self.assertGreater(int(box_edge.sum()), 10)
    # The box has no green or blue; a blended box pixel gains some from the floor and keeps its red.
    self.assertTrue((rgb[box_edge][:, 1:] == 0).all())
    self.assertGreater(int((rgb_aa[box_edge][:, 1] > 0).sum()), 10)
    self.assertTrue((rgb_aa[box_edge][:, 0] > 0).all())

  def test_sky_side_of_a_silhouette_blends_with_the_sky(self):
    """With a sky, a missed pixel next to a floating box takes the box colour into the sky, not white."""
    box = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.8, 0.8, 0.8], rgbaColor=[1, 0, 0, 1])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=box, basePosition=[0, 0, 3.0],
                      baseOrientation=p.getQuaternionFromEuler([0.3, 0.2, 0.4]))
    sky = {"skyHorizonColor": [0.2, 0.4, 0.8], "skyZenithColor": [0.1, 0.2, 0.6], "eye": (0.0, -6.0, 3.0), "target": (0, 0, 3.0)}
    rgb, _, seg = render(self.BASE, **sky)
    rgb_aa, _, _ = render(self.AA, **sky)
    changed = (rgb != rgb_aa).any(axis=2)
    sky_edge = changed & (seg < 0)
    self.assertGreater(int(sky_edge.sum()), 10)
    self.assertFalse((rgb_aa[sky_edge] == 255).all(axis=1).any())
    self.assertTrue((rgb_aa[sky_edge][:, 0] >= rgb[sky_edge][:, 0]).all())
    self.assertGreater(int((rgb_aa[sky_edge][:, 0] > rgb[sky_edge][:, 0]).sum()), 10)

  def test_tiny_renderer_ignores_the_flag(self):
    """Without the ray-cast flag the bytes are exactly the rasterised path's."""
    rgb, depth, seg = render(p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
    rgb_aa, depth_aa, seg_aa = render(p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX | p.ER_EDGE_ANTIALIAS)
    self.assertEqual(rgb.tobytes(), rgb_aa.tobytes())
    self.assertEqual(depth.tobytes(), depth_aa.tobytes())
    self.assertEqual(seg.tobytes(), seg_aa.tobytes())


@unittest.skipUnless(hasattr(p, "ER_SWARM_EDGE_OUTLINE"), "wheel built without outline-only edges")
class TestEdgeOutline(unittest.TestCase):
  """ER_SWARM_EDGE_OUTLINE: silhouettes and depth jumps blend as before, creases inside one body keep the first ray."""

  BASE = TestEdgeAntialias.BASE
  AA = TestEdgeAntialias.AA
  OUTLINE = AA | getattr(p, "ER_SWARM_EDGE_OUTLINE", 0)

  def setUp(self):
    """Connects and builds the scene."""
    p.connect(p.DIRECT)
    build_world()

  def tearDown(self):
    """Disconnects."""
    p.disconnect()

  def test_without_edge_antialias_nothing_changes(self):
    """The flag alone leaves every colour, depth and mask byte as it was."""
    rgb, depth, seg = render(self.BASE)
    rgb_o, depth_o, seg_o = render(self.BASE | p.ER_SWARM_EDGE_OUTLINE)
    self.assertEqual(rgb.tobytes(), rgb_o.tobytes())
    self.assertEqual(depth.tobytes(), depth_o.tobytes())
    self.assertEqual(seg.tobytes(), seg_o.tobytes())

  def test_each_pixel_is_the_first_ray_or_the_blend(self):
    """Depth and mask keep their bytes; every pixel is either the unblended one or the fully blended one, the
    silhouettes all blended and some creases on the box and the sphere left as the first ray drew them."""
    rgb, depth, seg = render(self.BASE)
    rgb_aa, _, _ = render(self.AA)
    rgb_o, depth_o, seg_o = render(self.OUTLINE)
    self.assertEqual(depth.tobytes(), depth_o.tobytes())
    self.assertEqual(seg.tobytes(), seg_o.tobytes())
    plain = (rgb_o == rgb).all(axis=2)
    blended = (rgb_o == rgb_aa).all(axis=2)
    self.assertTrue((plain | blended).all())
    silhouette = near_a_mask_change(seg, 0)
    self.assertTrue(blended[silhouette].all())
    crease = (rgb_aa != rgb).any(axis=2) & ~silhouette & (seg >= 1)
    self.assertGreater(int(crease.sum()), 10)
    self.assertGreater(int((crease & plain & ~blended).sum()), 10)

  def test_a_jump_in_depth_inside_one_body_still_blends(self):
    """A small tilted-lit square floating high over a large one, both in one body, blends its outline as without the
    flag."""
    shape = p.createVisualShape(p.GEOM_MESH, vertices=square(0, 0, 0, 6) + square(0.3, 0.2, 2.5, 0.7),
                                indices=[0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7],
                                normals=[[0, 0, 1]] * 4 + [[0.6, 0, 0.8]] * 4, rgbaColor=[0.2, 0.8, 0.3, 1])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=shape, basePosition=[10, 0, 0])
    above = {"eye": (10.2, -0.1, 6.0), "target": (10.0, 0.0, 0.0)}
    rgb, _, seg = render(self.BASE, **above)
    rgb_aa, _, _ = render(self.AA, **above)
    rgb_o, _, _ = render(self.OUTLINE, **above)
    self.assertEqual(len(np.unique(seg)), 1)
    outline = (rgb_aa != rgb).any(axis=2)
    self.assertGreater(int(outline.sum()), SIZE // 2)
    self.assertTrue((rgb_o[outline] == rgb_aa[outline]).all())


@NEEDS_FLAG
class TestEdgeAntialiasThreads(unittest.TestCase):
  """The blended bytes do not depend on how many render threads are used."""

  def test_same_bytes_for_one_two_and_four_threads(self):
    """Renders the scene, also with outline-only edges and the leaves with crease fill, in a fresh process per thread
    count and compares the hashes."""
    digests = set()
    for threads in ("1", "2", "4"):
      env = dict(os.environ, SWARM_RENDER_THREADS=threads)
      out = subprocess.check_output([sys.executable, __file__, "--hash"], env=env, text=True)
      digests.add(out.strip())
    self.assertEqual(len(digests), 1)


def build_foliage():
  """A floor and one body of 300 small leaves at many depths, so most of its pixels are creases inside one body."""
  build_world()
  rng = np.random.default_rng(7)
  vertices, indices = [], []
  for k in range(300):
    centre = rng.uniform([-0.8, -0.8, 1.2], [0.8, 0.8, 2.2])
    a, b = rng.normal(size=3), rng.normal(size=3)
    a = a / np.linalg.norm(a) * 0.09
    b = np.cross(a, b)
    b = b / np.linalg.norm(b) * 0.09
    vertices += [list(centre - a - b), list(centre + a - b), list(centre + a + b), list(centre - a + b)]
    indices += [4 * k, 4 * k + 1, 4 * k + 2, 4 * k, 4 * k + 2, 4 * k + 3]
  leaves = p.createVisualShape(p.GEOM_MESH, vertices=vertices, indices=indices, rgbaColor=[0.2, 0.6, 0.2, 1],
                               flags=p.VISUAL_SHAPE_DOUBLE_SIDED)
  p.createMultiBody(baseMass=0, baseVisualShapeIndex=leaves)


@unittest.skipUnless(hasattr(p, "ER_SWARM_CREASE_FILL"), "wheel built without crease fill")
class TestCreaseFill(unittest.TestCase):
  """ER_SWARM_CREASE_FILL: a crease inside one body fills its uncovered share from its neighbours, every other pixel
  keeps the full blend."""

  BASE = TestEdgeAntialias.BASE
  AA = TestEdgeAntialias.AA
  FILL = AA | getattr(p, "ER_SWARM_CREASE_FILL", 0)
  EYE = dict(eye=(2.5, -2.5, 3.0), target=(0, 0, 1.4))

  def setUp(self):
    """Connects and builds the foliage scene."""
    p.connect(p.DIRECT)
    build_foliage()

  def tearDown(self):
    """Disconnects."""
    p.disconnect()

  def test_without_edge_antialias_nothing_changes(self):
    """The flag alone leaves every colour, depth and mask byte as it was."""
    rgb, depth, seg = render(self.BASE, **self.EYE)
    rgb_f, depth_f, seg_f = render(self.BASE | p.ER_SWARM_CREASE_FILL, **self.EYE)
    self.assertEqual(rgb.tobytes(), rgb_f.tobytes())
    self.assertEqual(depth.tobytes(), depth_f.tobytes())
    self.assertEqual(seg.tobytes(), seg_f.tobytes())

  def test_only_creases_inside_one_body_change(self):
    """Depth and mask keep their bytes; a pixel next to another body or the frame's border is the full blend, and some
    creases among the leaves come out differently."""
    rgb_aa, depth_aa, seg_aa = render(self.AA, **self.EYE)
    rgb_f, depth_f, seg_f = render(self.FILL, **self.EYE)
    self.assertEqual(depth_aa.tobytes(), depth_f.tobytes())
    self.assertEqual(seg_aa.tobytes(), seg_f.tobytes())
    outline = near_a_mask_change(seg_aa, 0)
    outline[0, :] = outline[-1, :] = outline[:, 0] = outline[:, -1] = True
    self.assertTrue((rgb_f[outline] == rgb_aa[outline]).all())
    self.assertGreater(int((rgb_f != rgb_aa).any(axis=2).sum()), 10)


def colour_hash():
  """Prints the sha256 of one shadowed, filtered, anti-aliased colour frame on the ray-cast path."""
  p.connect(p.DIRECT)
  build_world()
  rgb, _, _ = render(p.ER_NO_SEGMENTATION_MASK | p.ER_SWARM_RAYCAST | p.ER_TEXTURE_FILTER | p.ER_EDGE_ANTIALIAS, shadow=1)
  print(hashlib.sha256(rgb.astype(np.uint8).tobytes()).hexdigest())
  if hasattr(p, "ER_SWARM_EDGE_OUTLINE"):
    rgb, _, _ = render(p.ER_NO_SEGMENTATION_MASK | p.ER_SWARM_RAYCAST | p.ER_TEXTURE_FILTER | p.ER_EDGE_ANTIALIAS |
                       p.ER_SWARM_EDGE_OUTLINE, shadow=1)
    print(hashlib.sha256(rgb.astype(np.uint8).tobytes()).hexdigest())
  p.disconnect()
  if hasattr(p, "ER_SWARM_CREASE_FILL"):
    p.connect(p.DIRECT)
    build_foliage()
    rgb, _, _ = render(p.ER_NO_SEGMENTATION_MASK | p.ER_SWARM_RAYCAST | p.ER_TEXTURE_FILTER | p.ER_EDGE_ANTIALIAS |
                       p.ER_SWARM_CREASE_FILL, shadow=1, **TestCreaseFill.EYE)
    print(hashlib.sha256(rgb.astype(np.uint8).tobytes()).hexdigest())
    p.disconnect()


if __name__ == '__main__':
  if "--hash" in sys.argv:
    colour_hash()
  else:
    unittest.main()
