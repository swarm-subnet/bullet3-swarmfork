"""The daylight model (ER_SWARM_DAYLIGHT): sky light by direction, a sun several times the sky, the
film curve and exposure, a photographed sky, soft shadows from two grids, haze, sun through leaves,
texture taps along the footprint. Ray-cast colour path only, off by default."""
import os
import tempfile
import unittest
import numpy as np
import pybullet as p

SIZE = 96
SUN_COLOR = [1.0, 0.9, 0.8]
NOON = [0.0, 0.0, 1.0]
LOW_SUN = [0.7071, 0.0, 0.7071]
RAYCAST = getattr(p, "ER_SWARM_RAYCAST", 0)
DAYLIGHT = getattr(p, "ER_SWARM_DAYLIGHT", 0)
SKY = getattr(p, "ER_SWARM_SKY_SUN", 0)
PICTURE = (RAYCAST | SKY | getattr(p, "ER_SWARM_SHADOW_MAP", 0) | getattr(p, "ER_SWARM_MOVER_SHADOW", 0)
           | getattr(p, "ER_TEXTURE_FILTER", 0) | getattr(p, "ER_SPECULAR_GLINT", 0) | getattr(p, "ER_ALPHA_CUTOUT", 0)
           | getattr(p, "ER_SWARM_LINEAR_LIGHT", 0))


def write_png(path, rgba):
  """Writes an RGBA uint8 array as a PNG with the standard library only."""
  import struct
  import zlib
  height, width = rgba.shape[:2]
  raw = b"".join(b"\x00" + rgba[y].tobytes() for y in range(height))
  def chunk(kind, data):
    """One PNG chunk with its CRC."""
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xffffffff)
  with open(path, "wb") as handle:
    handle.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
                 + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


AGX_INSET = np.array([[0.544814746488245, 0.373787398372697, 0.0813978551390581],
                      [0.140416948464053, 0.754137554567394, 0.105445496968552],
                      [0.0888104196149096, 0.178871756420858, 0.732317823964232]])
AGX_OUTSET = np.array([[1.96488741169489, -0.855988495690215, -0.108898916004672],
                       [-0.299313364904742, 1.32639796461980, -0.0270845997150571],
                       [-0.164352742528393, -0.238183969428088, 1.40253671195648]])


def agx(lin):
  """The engine's film curve on one linear RGB triple, as display values 0..1."""
  v = AGX_INSET @ np.maximum(np.asarray(lin, dtype=float), 1e-10)
  v = np.clip((np.log2(v) + 12.47393) / (4.026069 + 12.47393), 0.0, 1.0)
  x2, x4 = v * v, v ** 4
  e = 15.5 * x4 * x2 - 40.14 * x4 * v + 31.96 * x4 - 6.868 * x2 * v + 0.4298 * x2 + 0.1191 * v - 0.00232
  return np.clip(AGX_OUTSET @ e, 0.0, 1.0)


def linear(bytes_):
  """Linear grey light behind display bytes, by inverting the film curve on the green channel."""
  out = []
  for byte in np.atleast_1d(np.asarray(bytes_, dtype=float)):
    lo, hi = 1e-6, 64.0
    for _ in range(50):
      mid = (lo + hi) / 2.0
      if agx([mid, mid, mid])[1] * 255.0 + 0.5 < byte:
        lo = mid
      else:
        hi = mid
    out.append((lo + hi) / 2.0)
  return np.array(out)


def write_obj(path, size, double_sided_quad=False):
  """A square of half side `size` in the xy plane at z = 0 (or the xz plane when a standing quad), with uvs."""
  if double_sided_quad:
    verts = [(-size, 0, 0), (size, 0, 0), (size, 0, 2 * size), (-size, 0, 2 * size)]
  else:
    verts = [(-size, -size, 0), (size, -size, 0), (size, size, 0), (-size, size, 0)]
  lines = ["mtllib none.mtl"] + ["v %f %f %f" % v for v in verts] + ["vt 0 0", "vt 1 0", "vt 1 1", "vt 0 1"]
  lines += ["vn 0 0 1" if not double_sided_quad else "vn 0 -1 0", "f 1/1/1 2/2/1 3/3/1", "f 1/1/1 3/3/1 4/4/1"]
  with open(path, "w") as handle:
    handle.write("\n".join(lines) + "\n")


@unittest.skipUnless(RAYCAST and DAYLIGHT and SKY, "wheel without the daylight model")
class TestDaylight(unittest.TestCase):
  """Renders small worlds with the daylight flag on and off and checks what each term does."""

  def setUp(self):
    """A DIRECT client, a temporary folder for meshes and textures, a 90 degree camera."""
    p.connect(p.DIRECT)
    self.folder = tempfile.mkdtemp(prefix="daylight_")
    self.proj = p.computeProjectionMatrixFOV(90, 1.0, 0.1, 100.0)

  def tearDown(self):
    """Drops the client."""
    p.disconnect()

  def render(self, eye=(0, -4, 2), target=(0, 0, 0), flags=PICTURE, sun=LOW_SUN, shadow=1, size=SIZE,
             ambient=1.0, diffuse=3.0, **kwargs):
    """Returns (rgb, depth, seg) of one frame at the given eye and target."""
    view = p.computeViewMatrix(list(eye), list(target), [0, 0, 1])
    _, _, rgb, depth, seg = p.getCameraImage(size, size, view, self.proj, shadow=shadow, lightDirection=sun,
                                             lightColor=SUN_COLOR, lightAmbientCoeff=ambient, lightDiffuseCoeff=diffuse,
                                             renderer=p.ER_TINY_RENDERER, flags=flags, shadowLightCoeff=0.0, **kwargs)
    rgb = None if rgb is None else np.asarray(rgb).reshape(size, size, 4)[:, :, :3].astype(int)
    return rgb, np.asarray(depth), np.asarray(seg).reshape(size, size)

  def ground(self, size=20.0, texture=None):
    """A white ground square, optionally textured, as a static body."""
    path = os.path.join(self.folder, "ground%.0f.obj" % size)
    write_obj(path, size)
    vis = p.createVisualShape(p.GEOM_MESH, fileName=path, rgbaColor=[1, 1, 1, 1], specularColor=[0, 0, 0])
    uid = p.createMultiBody(0, -1, vis)
    if texture is not None:
      p.changeVisualShape(uid, -1, textureUniqueId=texture)
    return uid

  def box(self, position, half=0.5, specular=(0, 0, 0), mass=0):
    """A white box at a position."""
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[half] * 3, rgbaColor=[1, 1, 1, 1], specularColor=list(specular))
    return p.createMultiBody(mass, -1, vis, basePosition=list(position))

  def test_flag_off_leaves_every_byte(self):
    """Without the flag the picture path renders exactly as before, even after a daylight frame."""
    self.ground()
    self.box((0, 0, 0.5))
    before, depth_before, seg_before = self.render(ambient=0.45, diffuse=0.35)
    with_flag, depth_flag, seg_flag = self.render(flags=PICTURE | DAYLIGHT, exposure=1.0, hazeDistance=50.0, shadowCoreRadius=5.0)
    after, depth_after, seg_after = self.render(ambient=0.45, diffuse=0.35)
    self.assertEqual(before.tobytes(), after.tobytes())
    self.assertFalse((before == with_flag).all())
    self.assertEqual(depth_before.tobytes(), depth_flag.tobytes())
    self.assertEqual(depth_before.tobytes(), depth_after.tobytes())
    self.assertEqual(seg_before.tobytes(), seg_flag.tobytes())

  def shadow_scene(self):
    """A ground and a box whose shadow, thrown along -x by a 45 degree sun, fills a good part of the view."""
    self.ground()
    self.box((0, 0, 1.0), half=1.0)
    return dict(eye=(-5, -3, 3), target=(-2, 0, 0), sun=[0.7071, 0.0, 0.7071])

  def test_sun_is_several_times_the_sky(self):
    """Lit ground is far brighter than the ground in the box's shadow, and the ratio follows the sun coefficient."""
    scene = self.shadow_scene()
    rgb, _, seg = self.render(flags=PICTURE | DAYLIGHT, diffuse=3.0, **scene)
    ground = seg == 0
    bytes_ = rgb[:, :, 1][ground]
    lit, shadowed = linear(np.percentile(bytes_, 95)), linear(np.percentile(bytes_, 2))
    # A white ground under a sun of three times the sky, at 45 degrees: lit is about 1 + 3 * 0.7 times the shadow.
    self.assertGreater(lit / shadowed, 2.4)
    self.assertGreater(int((bytes_ < np.percentile(bytes_, 95) - 15).sum()), 50)
    strong, _, _ = self.render(flags=PICTURE | DAYLIGHT, diffuse=6.0, **scene)
    strong_bytes = strong[:, :, 1][ground]
    self.assertGreater(linear(np.percentile(strong_bytes, 95)) / linear(np.percentile(strong_bytes, 2)), lit / shadowed)
    dim, _, _ = self.render(flags=PICTURE | DAYLIGHT, diffuse=0.3, **scene)
    self.assertLess(np.percentile(dim[:, :, 1][ground], 95), np.percentile(bytes_, 95))

  def test_sky_light_follows_the_direction_a_surface_faces(self):
    """With no sun term, a box's top under the sky is brighter than its side, and its underside darker still."""
    uid = self.box((0, 0, 0.5))
    above, _, seg = self.render(eye=(0, -3, 2.5), target=(0, 0, 0.5), flags=PICTURE | DAYLIGHT, sun=NOON, diffuse=0.0, ambient=1.0)
    hit = seg == uid
    rows = np.where(hit.any(axis=1))[0]
    top_rows, side_rows = rows[: len(rows) // 3], rows[-len(rows) // 3:]
    top = above[top_rows][hit[top_rows]].mean()
    side = above[side_rows][hit[side_rows]].mean()
    below, _, seg_below = self.render(eye=(0, -3, -1.5), target=(0, 0, 0.5), flags=PICTURE | DAYLIGHT, sun=NOON, diffuse=0.0, ambient=1.0)
    hit_below = seg_below == uid
    rows = np.where(hit_below.any(axis=1))[0]
    under_rows = rows[-len(rows) // 3:]
    under = below[under_rows][hit_below[under_rows]].mean()
    self.assertGreater(top, side)
    self.assertGreater(side, under)

  def test_exposure_scales_the_picture_through_a_soft_shoulder(self):
    """More exposure never darkens a pixel, and a lit white surface at four times the light is not yet clipped."""
    self.ground()
    one, _, seg = self.render(flags=PICTURE | DAYLIGHT, exposure=1.0)
    four, _, _ = self.render(flags=PICTURE | DAYLIGHT, exposure=4.0)
    ground = seg == 0
    self.assertTrue((four[ground] >= one[ground]).all())
    self.assertLess(four[ground].max(), 255)
    self.assertGreater(four[ground].mean(), one[ground].mean())

  def test_photo_sky_is_painted_and_turned_by_yaw(self):
    """A two-colour photo shows its colours along the right headings, and yaw 180 swaps them."""
    photo = np.zeros((64, 128, 4), dtype=np.uint8)
    photo[:, :64] = (200, 40, 40, 255)
    photo[:, 64:] = (40, 40, 200, 255)
    path = os.path.join(self.folder, "sky.png")
    write_png(path, photo)
    tex = p.loadTexture(path)
    self.assertGreaterEqual(tex, 0)
    # The red half spans headings 0 to 180 degrees, so +y (90 degrees) sits in its middle.
    east, _, seg = self.render(eye=(0, 0, 1), target=(0, 1, 1.2), flags=PICTURE | DAYLIGHT, sun=NOON, skyTextureId=tex, skyYaw=0.0)
    turned, _, _ = self.render(eye=(0, 0, 1), target=(0, 1, 1.2), flags=PICTURE | DAYLIGHT, sun=NOON, skyTextureId=tex, skyYaw=180.0)
    miss = seg < 0
    self.assertTrue(miss.all())
    centre = east[SIZE // 2, SIZE // 2]
    self.assertGreater(centre[0], centre[2])
    centre_turned = turned[SIZE // 2, SIZE // 2]
    self.assertGreater(centre_turned[2], centre_turned[0])
    plain, _, _ = self.render(eye=(0, 0, 1), target=(0, 1, 1.2), flags=PICTURE | DAYLIGHT, sun=NOON)
    self.assertFalse((plain == east).all())

  def test_photo_sky_is_rebuilt_after_a_world_reset(self):
    """A photo loaded after resetSimulation paints its own colours, even if it lands where the old one was."""
    red = np.zeros((32, 64, 4), dtype=np.uint8)
    red[:] = (200, 40, 40, 255)
    blue = red.copy()
    blue[:] = (40, 40, 200, 255)
    red_path, blue_path = os.path.join(self.folder, "red.png"), os.path.join(self.folder, "blue.png")
    write_png(red_path, red)
    write_png(blue_path, blue)
    first, _, _ = self.render(eye=(0, 0, 1), target=(0, 1, 1.2), flags=PICTURE | DAYLIGHT, sun=NOON, skyTextureId=p.loadTexture(red_path))
    p.resetSimulation()
    second, _, _ = self.render(eye=(0, 0, 1), target=(0, 1, 1.2), flags=PICTURE | DAYLIGHT, sun=NOON, skyTextureId=p.loadTexture(blue_path))
    self.assertGreater(first[SIZE // 2, SIZE // 2][0], first[SIZE // 2, SIZE // 2][2])
    self.assertGreater(second[SIZE // 2, SIZE // 2][2], second[SIZE // 2, SIZE // 2][0])

  def test_photo_sky_needs_no_sun_sky_flag(self):
    """A photo with the daylight flag alone is painted; with neither sky flag nor photo the background stays white."""
    photo = np.zeros((32, 64, 4), dtype=np.uint8)
    photo[:] = (40, 40, 200, 255)
    path = os.path.join(self.folder, "blue_only.png")
    write_png(path, photo)
    tex = p.loadTexture(path)
    without_sun_sky = PICTURE & ~SKY
    painted, _, seg = self.render(eye=(0, 0, 1), target=(0, 1, 1.2), flags=without_sun_sky | DAYLIGHT, sun=NOON, skyTextureId=tex)
    self.assertTrue((seg < 0).all())
    self.assertGreater(painted[SIZE // 2, SIZE // 2][2], painted[SIZE // 2, SIZE // 2][0])
    white, _, _ = self.render(eye=(0, 0, 1), target=(0, 1, 1.2), flags=without_sun_sky | DAYLIGHT, sun=NOON)
    self.assertTrue((white == 255).all())

  def test_shadow_map_keeps_up_with_the_scene_when_daylight_turns_off(self):
    """A body hidden after a daylight frame casts no shadow in the next flag-off frame, as if daylight had never run."""
    scene = self.shadow_scene()
    uid = p.getNumBodies() - 1
    reference, _, _ = self.render(flags=PICTURE, ambient=0.45, diffuse=0.35, **scene)
    self.render(flags=PICTURE | DAYLIGHT, shadowCoreRadius=10.0, **scene)
    p.changeVisualShape(uid, -1, rgbaColor=[1, 1, 1, 0])
    hidden, _, _ = self.render(flags=PICTURE, ambient=0.45, diffuse=0.35, **scene)
    p.changeVisualShape(uid, -1, rgbaColor=[1, 1, 1, 1])
    shown, _, _ = self.render(flags=PICTURE, ambient=0.45, diffuse=0.35, **scene)
    self.assertEqual(reference.tobytes(), shown.tobytes())
    self.assertFalse((hidden == reference).all())
    self.assertGreater(hidden[:, :, 1].mean(), reference[:, :, 1].mean())

  def test_haze_fades_far_ground_towards_the_sky(self):
    """With a short haze distance the far ground moves towards the horizon colour; without haze it does not."""
    self.ground(size=80.0)
    clear, _, seg = self.render(eye=(0, -1, 1.5), target=(0, 20, 0.5), flags=PICTURE | DAYLIGHT, sun=NOON, hazeDistance=0.0)
    hazy, _, _ = self.render(eye=(0, -1, 1.5), target=(0, 20, 0.5), flags=PICTURE | DAYLIGHT, sun=NOON, hazeDistance=15.0)
    ground = seg == 0
    rows = np.where(ground.any(axis=1))[0]
    far, near = rows[0], rows[-1]
    far_change = np.abs(hazy[far][ground[far]].astype(int) - clear[far][ground[far]].astype(int)).mean()
    near_change = np.abs(hazy[near][ground[near]].astype(int) - clear[near][ground[near]].astype(int)).mean()
    self.assertGreater(far_change, near_change + 5)

  def test_soft_shadow_has_grey_between_lit_and_dark(self):
    """The shadow edge on the ground carries bytes between the lit and the shadowed value, with and without the core grid."""
    scene = self.shadow_scene()
    for core in (0.0, 10.0):
      rgb, _, seg = self.render(flags=PICTURE | DAYLIGHT, shadowCoreRadius=core, **scene)
      lum = rgb[:, :, 1][seg == 0]
      lit, dark = np.percentile(lum, 95), np.percentile(lum, 2)
      self.assertGreater(lit, dark + 20, core)
      between = int(((lum > dark + 5) & (lum < lit - 5)).sum())
      self.assertGreater(between, 0, core)

  def test_shadow_edge_moves_smoothly_inside_a_cell(self):
    """Ground points a tenth of a coarse grid cell apart across a shadow edge darken by small steps, not one jump per cell."""
    self.ground(size=2000.0)
    self.box((0, 0, 8.0), half=3.0)
    shades = []
    for step in range(10):
      x = -2.0 + 0.138 * step
      rgb, _, seg = self.render(eye=(x, -2, 4), target=(x, 0, 0), flags=PICTURE | DAYLIGHT, size=33,
                                sun=[0.7071, 0.0, 0.7071], ambient=0.3)
      self.assertEqual(seg[16, 16], 0)
      shades.append(int(rgb[16, 16, 1]))
    self.assertGreater(max(shades), min(shades) + 10, shades)
    self.assertGreaterEqual(sum(b != a for a, b in zip(shades, shades[1:])), 7, shades)

  def test_leaf_lets_the_sun_through_from_behind(self):
    """A cut-out double-sided quad seen from its back is brighter with the sun behind it than with the sun ahead."""
    photo = np.zeros((32, 32, 4), dtype=np.uint8)
    photo[:] = (60, 160, 60, 255)
    photo[8:24, 8:24, 3] = 0
    path = os.path.join(self.folder, "leaf.png")
    write_png(path, photo)
    tex = p.loadTexture(path)
    obj = os.path.join(self.folder, "leaf.obj")
    write_obj(obj, 1.0, double_sided_quad=True)
    vis = p.createVisualShape(p.GEOM_MESH, fileName=obj, rgbaColor=[1, 1, 1, 1], specularColor=[0, 0, 0],
                              flags=p.VISUAL_SHAPE_DOUBLE_SIDED_MULTIBODY)
    uid = p.createMultiBody(0, -1, vis)
    p.changeVisualShape(uid, -1, textureUniqueId=tex)
    behind, _, seg = self.render(eye=(0, 3, 1), target=(0, 0, 1), flags=PICTURE | DAYLIGHT, sun=[0.0, -0.7071, 0.7071], diffuse=3.0)
    ahead, _, _ = self.render(eye=(0, 3, 1), target=(0, 0, 1), flags=PICTURE | DAYLIGHT, sun=[0.0, 0.7071, 0.7071], diffuse=3.0)
    hit = seg == uid
    self.assertGreater(int(hit.sum()), 0)
    self.assertGreater(behind[hit].sum(), 0)
    self.assertLess(behind[hit].mean(), ahead[hit].mean())
    self.assertGreater(behind[hit].mean(), ahead[hit].mean() * 0.2)

  def chain_link_card(self):
    """A 60 by 20 m card of thin dark lines every 25 cm in front of the sky, 18 % of it wire; returns its body."""
    tex = np.zeros((256, 256, 4), dtype=np.uint8)
    tex[..., :3] = 60
    lines = np.zeros((256, 256), dtype=bool)
    for k in range(0, 256, 32):
      lines[k:k + 3, :] = True
      lines[:, k:k + 3] = True
    tex[lines, 3] = 255
    png = os.path.join(self.folder, "link.png")
    write_png(png, tex)
    obj = os.path.join(self.folder, "link.obj")
    with open(obj, "w") as handle:
      handle.write("v -30 0 0\nv 30 0 0\nv 30 0 20\nv -30 0 20\nvt 0 0\nvt 30 0\nvt 30 10\nvt 0 10\nvn 0 -1 0\n"
                   "f 1/1/1 2/2/1 3/3/1\nf 1/1/1 3/3/1 4/4/1\n")
    vis = p.createVisualShape(p.GEOM_MESH, fileName=obj, rgbaColor=[1, 1, 1, 1], specularColor=[0, 0, 0],
                              flags=p.VISUAL_SHAPE_DOUBLE_SIDED_MULTIBODY)
    uid = p.createMultiBody(0, -1, vis)
    p.changeVisualShape(uid, -1, textureUniqueId=p.loadTexture(png))
    return uid

  def test_far_chain_link_is_an_even_veil_and_near_stays_sharp(self):
    """Far, thin wire darkens the sky behind it evenly by its coverage; per-texel reads break it up; near, lines stay crisp."""
    eye, target, band = (0, -25, 10), (0, 0, 10), slice(40, 56)
    sky, _, _ = self.render(eye=eye, target=target, flags=PICTURE | DAYLIGHT, sun=NOON)
    card_id = self.chain_link_card()
    veil, _, _ = self.render(eye=eye, target=target, flags=PICTURE | DAYLIGHT, sun=NOON)
    point, _, _ = self.render(eye=eye, target=target, flags=(PICTURE & ~getattr(p, "ER_TEXTURE_FILTER", 0)) | DAYLIGHT, sun=NOON)
    sky, veil, point = (a[band, :, 1].astype(float) for a in (sky, veil, point))
    darkening = (sky - veil).mean()
    self.assertGreater(darkening, 2.0)
    self.assertLess(darkening, 0.5 * (sky.mean() - 60.0))
    self.assertLess((veil - sky).std(), (point - sky).std() / 3.0)
    near, _, seg = self.render(eye=(0, -0.5, 10), target=(0, 0, 10), flags=PICTURE | DAYLIGHT, sun=NOON)
    card = seg == card_id
    self.assertGreater(int(card.sum()), 0)
    self.assertLess(int(card.sum()), card.size // 2)

  def test_threads_give_the_same_bytes(self):
    """The frame is byte-identical at every render thread count the build allows."""
    self.ground()
    self.box((0.5, 0, 0.5), specular=(1, 1, 1))
    frames = []
    for threads in ("1", "2", "4"):
      os.environ["SWARM_RENDER_THREADS"] = threads
      p.disconnect()
      p.connect(p.DIRECT)
      self.ground()
      self.box((0.5, 0, 0.5), specular=(1, 1, 1))
      rgb, _, _ = self.render(flags=PICTURE | DAYLIGHT | getattr(p, "ER_EDGE_ANTIALIAS", 0), hazeDistance=30.0, shadowCoreRadius=5.0)
      frames.append(rgb.tobytes())
    self.assertEqual(frames[0], frames[1])
    self.assertEqual(frames[0], frames[2])


if __name__ == '__main__':
  unittest.main()
