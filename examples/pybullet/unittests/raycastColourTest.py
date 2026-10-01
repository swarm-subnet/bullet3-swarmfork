"""Colour on the ray-cast path: the same light model as TinyRenderer, a real shadow, identical bytes at any thread count."""
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
TEX = 64
SQUARE = 8
LIGHT = [-0.3, 0.6, 0.74]  # sun behind the box as seen from the camera, so its shadow falls into view
NEEDS_BACKEND = unittest.skipUnless(hasattr(p, "ER_SWARM_RAYCAST"), "wheel built without the ray-cast backend")


def write_checker_tga(path):
  """Writes an uncompressed 24-bit TGA checkerboard the texture loader reads without any extra library."""
  yy, xx = np.mgrid[0:TEX, 0:TEX]
  white = ((xx // SQUARE + yy // SQUARE) % 2 == 0)
  img = np.where(white[..., None], 255, 0).astype(np.uint8).repeat(3, axis=2)
  header = struct.pack('<BBBHHBHHHHBB', 0, 0, 2, 0, 0, 0, 0, 0, TEX, TEX, 24, 0)
  with open(path, 'wb') as f:
    f.write(header)
    f.write(img.tobytes())


def build_world(tex_path):
  """A checker-textured floor, a red box, a grey sphere and a fully transparent box."""
  floor = p.createVisualShape(p.GEOM_MESH, vertices=[[-4, -4, 0], [4, -4, 0], [4, 4, 0], [-4, 4, 0]],
                              indices=[0, 1, 2, 0, 2, 3], uvs=[[0, 0], [4, 0], [4, 4], [0, 4]],
                              normals=[[0, 0, 1]] * 4, rgbaColor=[1, 1, 1, 1])
  body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=floor)
  p.changeVisualShape(body, -1, textureUniqueId=p.loadTexture(tex_path))
  box = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], rgbaColor=[1, 0, 0, 1])
  p.createMultiBody(baseMass=0, baseVisualShapeIndex=box, basePosition=[0, 0, 0.5])
  sphere = p.createVisualShape(p.GEOM_SPHERE, radius=0.6, rgbaColor=[0.6, 0.6, 0.6, 1])
  p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphere, basePosition=[1.6, 1.0, 0.6])
  ghost = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.4], rgbaColor=[0, 0, 1, 0])
  p.createMultiBody(baseMass=0, baseVisualShapeIndex=ghost, basePosition=[-1.5, 0.5, 0.4])


def render(flags, shadow=0, eye=(3.0, -3.0, 2.5), target=(0, 0, 0.4), **sky):
  """Returns (rgb HxWx3, depth HxW, seg HxW) from a camera at eye looking at target."""
  view = p.computeViewMatrix(list(eye), list(target), [0, 0, 1])
  proj = p.computeProjectionMatrixFOV(70, 1.0, 0.1, 30.0)
  _, _, rgb, depth, seg = p.getCameraImage(SIZE, SIZE, view, proj, shadow=shadow, lightDirection=LIGHT,
                                           renderer=p.ER_TINY_RENDERER, flags=flags, **sky)
  rgb = None if rgb is None else np.asarray(rgb).reshape(SIZE, SIZE, 4)[:, :, :3].astype(int)
  seg = None if seg is None else np.asarray(seg).reshape(SIZE, SIZE)
  return rgb, np.asarray(depth).reshape(SIZE, SIZE), seg


def interior(seg):
  """Pixels whose eight neighbours belong to the same object: away from every silhouette."""
  same = np.ones((SIZE - 2, SIZE - 2), dtype=bool)
  centre = seg[1:-1, 1:-1]
  for dy in (-1, 0, 1):
    for dx in (-1, 0, 1):
      same &= seg[1 + dy:SIZE - 1 + dy, 1 + dx:SIZE - 1 + dx] == centre
  mask = np.zeros((SIZE, SIZE), dtype=bool)
  mask[1:-1, 1:-1] = same
  return mask


@NEEDS_BACKEND
class TestRaycastColour(unittest.TestCase):
  """Renders one lit scene through both backends and compares them."""

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

  def test_colour_agrees_with_tiny_renderer_away_from_edges(self):
    """Same texture, light and object colour on both paths: interior pixels differ by a few levels at most."""
    tiny_rgb, _, tiny_seg = render(p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
    ray_rgb, _, ray_seg = render(p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX | p.ER_SWARM_RAYCAST)
    mask = interior(tiny_seg) & interior(ray_seg) & (tiny_seg == ray_seg) & (tiny_seg >= 0)
    self.assertGreater(int(mask.sum()), SIZE * SIZE // 4)
    diff = np.abs(tiny_rgb - ray_rgb)[mask]
    self.assertLess(float(diff.mean()), 2.0)
    self.assertLessEqual(int(np.percentile(diff, 99)), 8)

  def test_misses_keep_the_clear_colour_and_depth_is_untouched(self):
    """A ray that hits nothing leaves the white clear colour; asking for colour changes no depth byte."""
    rgb, depth, seg = render(p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX | p.ER_SWARM_RAYCAST)
    _, depth_only, _ = render(p.ER_NO_SEGMENTATION_MASK | p.ER_SWARM_RAYCAST | p.ER_DEPTH_ONLY)
    self.assertTrue((rgb[seg < 0] == 255).all())
    self.assertGreater(int((seg < 0).sum()), 0)
    self.assertEqual(depth.tobytes(), depth_only.tobytes())

  def test_object_colour_and_alpha_apply(self):
    """The red box keeps only its red channel; the alpha-zero box is not drawn at all."""
    rgb, _, seg = render(p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX | p.ER_SWARM_RAYCAST)
    box = seg == 1
    self.assertGreater(int(box.sum()), 0)
    self.assertTrue((rgb[box][:, 1:] == 0).all())
    self.assertTrue((rgb[box][:, 0] > 0).all())
    self.assertFalse((seg == 3).any())

  def test_shadow_ray_darkens_the_floor_behind_the_box(self):
    """With shadow=1 the floor pixels the box hides from the light drop to 0.8 of their lit value; nothing else moves."""
    lit, _, seg = render(p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX | p.ER_SWARM_RAYCAST, shadow=0)
    shaded, _, _ = render(p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX | p.ER_SWARM_RAYCAST, shadow=1)
    floor = seg == 0
    darker = (shaded < lit).any(axis=2)
    self.assertGreater(int((darker & floor).sum()), 20)
    self.assertFalse((shaded > lit).any())
    self.assertTrue((lit[~darker] == shaded[~darker]).all())

  def test_single_sided_roof_still_blocks_the_sun(self):
    """A one-sided surface the camera sees through from below still hides the light from the ground."""
    roof = p.createVisualShape(p.GEOM_MESH, vertices=[[-1.2, -1.2, 0], [1.2, -1.2, 0], [1.2, 1.2, 0], [-1.2, 1.2, 0]],
                               indices=[0, 1, 2, 0, 2, 3], uvs=[[0, 0], [1, 0], [1, 1], [0, 1]],
                               normals=[[0, 0, 1]] * 4, rgbaColor=[0.2, 0.3, 0.9, 1])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=roof, basePosition=[0, 0, 2.2])
    overhead = {"eye": (5.0, -5.0, 3.5), "target": (0, 0, 0.3)}
    lit, _, seg = render(p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX | p.ER_SWARM_RAYCAST, shadow=0, **overhead)
    shaded, _, _ = render(p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX | p.ER_SWARM_RAYCAST, shadow=1, **overhead)
    floor_in_shade = ((shaded < lit).any(axis=2)) & (seg == 0)
    self.assertGreater(int(floor_in_shade.sum()), 20)

  def test_texture_footprint_is_right_on_a_face_seen_from_behind(self):
    """A double-sided texture keeps its detail from either side; the footprint must not flip with the winding."""
    quad = p.createVisualShape(p.GEOM_MESH, vertices=[[-1.5, 0, -1.5], [1.5, 0, -1.5], [1.5, 0, 1.5], [-1.5, 0, 1.5]],
                               indices=[0, 1, 2, 0, 2, 3], uvs=[[0, 0], [3, 0], [3, 3], [0, 3]],
                               normals=[[0, -1, 0]] * 4, rgbaColor=[1, 1, 1, 1])
    body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=quad, basePosition=[0, 4.0, 1.5])
    p.changeVisualShape(body, -1, textureUniqueId=p.loadTexture(self.tex_path),
                        flags=p.VISUAL_SHAPE_DOUBLE_SIDED_MULTIBODY)
    flags = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX | p.ER_SWARM_RAYCAST | p.ER_TEXTURE_FILTER
    contrast = []
    for eye in ((0.0, 1.0, 1.5), (0.0, 7.0, 1.5)):
      rgb, _, seg = render(flags, eye=eye, target=(0, 4.0, 1.5))
      on = seg == 4
      self.assertGreater(int(on.sum()), 100)
      contrast.append(int(rgb[on][:, 0].max() - rgb[on][:, 0].min()))
    self.assertGreater(min(contrast), max(contrast) // 2)

  def test_sky_shows_through_the_misses_the_same_way_up_as_tiny_renderer(self):
    """A sky behind the scene fills the missed pixels and its gradient rises to the zenith on both paths."""
    sky = {"skyHorizonColor": [0.9, 0.5, 0.2], "skyZenithColor": [0.1, 0.3, 0.9]}
    level = {"eye": (0.0, -6.0, 1.0), "target": (0.0, 0.0, 1.0)}
    tiny, _, seg = render(p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX, **level, **sky)
    ray, _, ray_seg = render(p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX | p.ER_SWARM_RAYCAST, **level, **sky)
    missed = (seg < 0) & (ray_seg < 0)
    self.assertGreater(int(missed.sum()), SIZE)
    # Blue rises towards the top of the frame, which looks up, on both paths.
    for name, rgb in (("tiny", tiny), ("raycast", ray)):
        top = rgb[0][missed[0]][:, 2]
        bottom = rgb[SIZE // 2 - 2][missed[SIZE // 2 - 2]][:, 2]
        self.assertGreater(int(top.mean()), int(bottom.mean()) + 20, name)
    self.assertFalse((ray[missed] == 255).all())

  def test_texture_filter_averages_the_far_floor(self):
    """With the filter flag the far checker floor reads a coarse mip level and loses its pure black and white."""
    nearest, _, seg = render(p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX | p.ER_SWARM_RAYCAST, eye=(0.0, -14.0, 1.2))
    filtered, _, _ = render(p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX | p.ER_SWARM_RAYCAST | p.ER_TEXTURE_FILTER,
                            eye=(0.0, -14.0, 1.2))
    far_floor = (seg == 0) & (np.arange(SIZE)[:, None] < SIZE * 3 // 5)
    self.assertGreater(int(far_floor.sum()), 50)
    self.assertLess(int(filtered[far_floor][:, 0].max() - filtered[far_floor][:, 0].min()),
                    int(nearest[far_floor][:, 0].max() - nearest[far_floor][:, 0].min()))


@NEEDS_BACKEND
class TestRaycastDepthHint(unittest.TestCase):
  """The last frame of a lens only tells each ray how far to search; it never changes a byte."""

  def test_frame_after_a_nearby_one_equals_a_first_frame(self):
    """A frame drawn after a nearby view, with a body moved in between, has the bytes of the same frame drawn first."""
    first = subprocess.check_output([sys.executable, __file__, "--hint", "first"], text=True)
    warm = subprocess.check_output([sys.executable, __file__, "--hint", "warm"], text=True)
    self.assertEqual(first, warm)

  def test_coincident_faces_resolve_as_in_a_first_frame(self):
    """Two boxes sharing a top face among other bodies, seen from 50 places, resolve every tie as in a first frame."""
    first = subprocess.check_output([sys.executable, __file__, "--ties", "first"], text=True)
    warm = subprocess.check_output([sys.executable, __file__, "--ties", "warm"], text=True)
    self.assertEqual(len(first.split()), 50)
    self.assertEqual(first, warm)


@NEEDS_BACKEND
class TestRaycastColourThreads(unittest.TestCase):
  """The colour bytes do not depend on how many render threads are used."""

  def test_same_bytes_for_one_two_and_four_threads(self):
    """Renders the scene in a fresh process per thread count and compares the colour hashes."""
    digests = set()
    for threads in ("1", "2", "4"):
      env = dict(os.environ, SWARM_RENDER_THREADS=threads)
      out = subprocess.check_output([sys.executable, __file__, "--hash"], env=env, text=True)
      digests.add(out.strip())
    self.assertEqual(len(digests), 1)


def colour_hash():
  """Prints the sha256 of one shadowed, filtered colour frame on the ray-cast path."""
  p.connect(p.DIRECT)
  tmp = tempfile.mkdtemp()
  tex_path = os.path.join(tmp, 'checker.tga')
  write_checker_tga(tex_path)
  build_world(tex_path)
  rgb, _, _ = render(p.ER_NO_SEGMENTATION_MASK | p.ER_SWARM_RAYCAST | p.ER_TEXTURE_FILTER, shadow=1)
  print(hashlib.sha256(rgb.astype(np.uint8).tobytes()).hexdigest())
  p.disconnect()
  os.remove(tex_path)
  os.rmdir(tmp)


def hint_hash(warm):
  """Prints the sha256 of the colour, depth and mask of one frame, drawn first or right after a nearby view."""
  p.connect(p.DIRECT)
  tmp = tempfile.mkdtemp()
  tex_path = os.path.join(tmp, 'checker.tga')
  write_checker_tga(tex_path)
  build_world(tex_path)
  flags = (p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX | p.ER_SWARM_RAYCAST | p.ER_TEXTURE_FILTER |
           getattr(p, "ER_EDGE_ANTIALIAS", 0) | getattr(p, "ER_SWARM_SHADOW_MAP", 0) | getattr(p, "ER_SWARM_MOVER_SHADOW", 0))
  if warm:
    render(flags, shadow=1)
  else:
    # The same opening frame at another size is another lens, so the frame compared below has no last frame to use.
    view = p.computeViewMatrix([3.0, -3.0, 2.5], [0, 0, 0.4], [0, 0, 1])
    proj = p.computeProjectionMatrixFOV(70, 1.0, 0.1, 30.0)
    p.getCameraImage(SIZE // 2, SIZE // 2, view, proj, shadow=1, lightDirection=LIGHT, renderer=p.ER_TINY_RENDERER, flags=flags)
  p.resetBasePositionAndOrientation(2, [1.2, 1.4, 0.6], [0, 0, 0, 1])
  rgb, depth, seg = render(flags, shadow=1, eye=(3.3, -2.7, 2.3))
  print(hashlib.sha256(rgb.astype(np.uint8).tobytes() + depth.tobytes() + seg.tobytes()).hexdigest())
  p.disconnect()
  os.remove(tex_path)
  os.rmdir(tmp)


def build_coincident(flags, proj):
  """Two 4 x 4 m boxes of different colours moved together to one pose, two small bodies nearer the cameras and four
  further along the view, all moved once so each is its own instance box in the top tree; the opening frame is too
  small to leave a last frame."""
  bodies = []
  for colour in ([1, 0, 0, 1], [0, 0, 1, 1]):
    shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[2, 2, 0.25], rgbaColor=colour)
    bodies.append((p.createMultiBody(baseMass=0, baseVisualShapeIndex=shape), [0.1, 0.2, 0.25]))
  for k in range(6):
    shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.3], rgbaColor=[0.15 * k, 0.8, 0.3, 1])
    place = [-2.6 + 5.2 * (k % 2), -2.5, 0.8] if k < 2 else [-3.0 + 2.0 * (k - 2), 6.0 + 1.5 * (k - 2), 0.6]
    bodies.append((p.createMultiBody(baseMass=0, baseVisualShapeIndex=shape), place))
  p.getCameraImage(16, 16, p.computeViewMatrix([0, -6, 4], [0, 0, 0], [0, 0, 1]), proj, renderer=p.ER_TINY_RENDERER, flags=flags)
  for body, place in bodies:
    p.resetBasePositionAndOrientation(body, place, [0, 0, 0, 1])


def ties_hashes(warm):
  """Prints one hash per camera place over the coincident boxes, each frame drawn first or right after a nearby one."""
  p.connect(p.DIRECT)
  rng = np.random.default_rng(7)
  flags = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX | p.ER_SWARM_RAYCAST | getattr(p, "ER_EDGE_ANTIALIAS", 0)
  proj = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 60.0)
  build_coincident(flags, proj)
  digests = []
  for _ in range(50):
    height = rng.uniform(2.0, 8.0)
    pitch = np.radians(rng.uniform(30.0, 60.0))
    eye = [rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5) - height / np.tan(pitch), height]
    target = [eye[0], eye[1] + height / np.tan(pitch), 0.5]
    if warm:
      p.getCameraImage(SIZE, SIZE, p.computeViewMatrix([eye[0] + 0.3, eye[1] - 0.2, eye[2] + 0.2], target, [0, 0, 1]), proj,
                       lightDirection=LIGHT, renderer=p.ER_TINY_RENDERER, flags=flags)
    else:
      p.resetSimulation()
      build_coincident(flags, proj)
    _, _, rgb, depth, seg = p.getCameraImage(SIZE, SIZE, p.computeViewMatrix(eye, target, [0, 0, 1]), proj,
                                             lightDirection=LIGHT, renderer=p.ER_TINY_RENDERER, flags=flags)
    blob = np.asarray(rgb, dtype=np.uint8).tobytes() + np.asarray(depth, dtype=np.float32).tobytes() + np.asarray(seg, dtype=np.int32).tobytes()
    digests.append(hashlib.sha256(blob).hexdigest()[:16])
  print(" ".join(digests))
  p.disconnect()


if __name__ == '__main__':
  if "--hash" in sys.argv:
    colour_hash()
  elif "--ties" in sys.argv:
    ties_hashes(sys.argv[-1] == "warm")
  elif "--hint" in sys.argv:
    hint_hash(sys.argv[-1] == "warm")
  else:
    unittest.main()
