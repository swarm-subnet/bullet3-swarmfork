"""Thin glass (VISUAL_SHAPE_GLASS) on the ray-cast daylight path: the sky by the Fresnel of the pane's two
faces, the surface behind through the tint for the rest, one more ray on glass pixels only. Off by default."""
import os
import tempfile
import unittest
import numpy as np
import pybullet as p

SIZE = 96
SUN_COLOR = [1.0, 0.9, 0.8]
SUN = [0.0, -0.5, 0.866]
RAYCAST = getattr(p, "ER_SWARM_RAYCAST", 0)
DAYLIGHT = getattr(p, "ER_SWARM_DAYLIGHT", 0)
SKY = getattr(p, "ER_SWARM_SKY_SUN", 0)
GLASS = getattr(p, "VISUAL_SHAPE_GLASS", 0)
TWO_SIDED = getattr(p, "VISUAL_SHAPE_DOUBLE_SIDED_MULTIBODY", 0)
PICTURE = (RAYCAST | SKY | getattr(p, "ER_SWARM_SHADOW_MAP", 0) | getattr(p, "ER_SWARM_MOVER_SHADOW", 0)
           | getattr(p, "ER_TEXTURE_FILTER", 0) | getattr(p, "ER_SPECULAR_GLINT", 0) | getattr(p, "ER_ALPHA_CUTOUT", 0)
           | getattr(p, "ER_SWARM_LINEAR_LIGHT", 0))
DAY = PICTURE | DAYLIGHT


def write_pane(path, half):
  """A standing square of half side `half` in the xz plane at y = 0, from z = 0 to 2 half, facing -y, with uvs."""
  verts = [(-half, 0, 0), (half, 0, 0), (half, 0, 2 * half), (-half, 0, 2 * half)]
  lines = ["mtllib none.mtl"] + ["v %f %f %f" % v for v in verts] + ["vt 0 0", "vt 1 0", "vt 1 1", "vt 0 1"]
  lines += ["vn 0 -1 0", "f 1/1/1 2/2/1 3/3/1", "f 1/1/1 3/3/1 4/4/1"]
  with open(path, "w") as handle:
    handle.write("\n".join(lines) + "\n")


@unittest.skipUnless(RAYCAST and DAYLIGHT and SKY and GLASS, "wheel without thin glass")
class TestGlass(unittest.TestCase):
  """Renders a pane in front of a coloured wall and checks what the glass bit does to the pixels on it."""

  def setUp(self):
    """A DIRECT client, a temporary folder for the pane mesh, a 60 degree camera."""
    p.connect(p.DIRECT)
    self.folder = tempfile.mkdtemp(prefix="glass_")
    self.proj = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 100.0)

  def tearDown(self):
    """Drops the client."""
    p.disconnect()

  def render(self, eye=(0, -4, 1), target=(0, 0, 1), flags=DAY, size=SIZE, **kwargs):
    """Returns (rgb, depth, seg) of one frame looking from eye at target."""
    view = p.computeViewMatrix(list(eye), list(target), [0, 0, 1])
    _, _, rgb, depth, seg = p.getCameraImage(size, size, view, self.proj, shadow=1, lightDirection=SUN, lightColor=SUN_COLOR,
                                             lightAmbientCoeff=1.0, lightDiffuseCoeff=3.0, renderer=p.ER_TINY_RENDERER,
                                             flags=flags, shadowLightCoeff=0.0, exposure=1.0, **kwargs)
    rgb = None if rgb is None else np.asarray(rgb).reshape(size, size, 4)[:, :, :3].astype(int)
    return rgb, np.asarray(depth), np.asarray(seg).reshape(size, size)

  def pane(self, flags=GLASS | TWO_SIDED, rgba=(1, 1, 1, 1), half=1.0):
    """A pane at the origin, glass and double-sided unless told otherwise."""
    path = os.path.join(self.folder, "pane%.1f.obj" % half)
    write_pane(path, half)
    vis = p.createVisualShape(p.GEOM_MESH, fileName=path, flags=flags, rgbaColor=list(rgba), specularColor=[0, 0, 0])
    return p.createMultiBody(0, -1, vis)

  def wall(self, rgba=(1, 0, 0, 1), position=(0, 3, 1), half=(3, 0.2, 1)):
    """A matte box behind the pane, red unless told otherwise."""
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=list(half), rgbaColor=list(rgba), specularColor=[0, 0, 0])
    return p.createMultiBody(0, -1, vis, basePosition=list(position))

  def centre(self, rgb):
    """Mean colour of the middle of the frame, which the pane fills."""
    c = SIZE // 2
    return rgb[c - 4:c + 4, c - 4:c + 4].reshape(-1, 3).mean(axis=0)

  def test_glass_shows_the_wall_behind(self):
    """Without the bit the pane is an opaque white board; with it the red wall shows through."""
    self.wall()
    self.pane(flags=TWO_SIDED)
    opaque = self.centre(self.render()[0])
    p.resetSimulation()
    self.wall()
    self.pane()
    glass = self.centre(self.render()[0])
    self.assertLess(abs(opaque[0] - opaque[1]), 12)
    self.assertGreater(glass[0], glass[1] + 40)
    self.assertGreater(glass[0], glass[2] + 40)

  def test_bit_does_nothing_without_daylight(self):
    """On the picture path without the daylight model the glass bit changes no byte."""
    self.wall()
    self.pane(flags=TWO_SIDED)
    plain = self.render(flags=PICTURE)
    p.resetSimulation()
    self.wall()
    self.pane()
    glass = self.render(flags=PICTURE)
    self.assertEqual(plain[0].tobytes(), glass[0].tobytes())

  def test_depth_and_mask_keep_the_pane(self):
    """The pane stays the surface the depth and the mask report; only the colour looks through."""
    self.wall()
    uid = self.pane(flags=TWO_SIDED)
    _, depth_opaque, seg_opaque = self.render()
    p.changeVisualShape(uid, -1, flags=GLASS | TWO_SIDED)
    _, depth_glass, seg_glass = self.render()
    self.assertEqual(depth_opaque.tobytes(), depth_glass.tobytes())
    self.assertEqual(seg_opaque.tobytes(), seg_glass.tobytes())

  def test_tint_colours_the_view_through(self):
    """A green pane over a white wall gives a green pixel."""
    self.wall(rgba=(1, 1, 1, 1))
    self.pane(rgba=(0.2, 1.0, 0.2, 1))
    seen = self.centre(self.render()[0])
    self.assertGreater(seen[1], seen[0] + 40)
    self.assertGreater(seen[1], seen[2] + 40)

  def test_reflection_rises_as_the_view_grazes(self):
    """Face-on the wall dominates the pane; from a grazing angle the sky does, so the red share drops."""
    self.wall(position=(0, 3, 3), half=(40, 0.2, 3))
    self.pane(half=3.0)
    face_on = self.centre(self.render(eye=(0, -4, 3), target=(0, 0, 3))[0])
    grazing = self.centre(self.render(eye=(-9, -1.0, 3), target=(0, 0, 3))[0])
    self.assertGreater(face_on[0] - face_on[2], (grazing[0] - grazing[2]) + 30)

  def test_sky_comes_through_an_empty_pane(self):
    """With nothing behind it a pane shows almost the sky seen beside it."""
    self.pane(half=1.0)
    rgb, _, _ = self.render(eye=(0, -4, 1.5), target=(0, 0, 2.5))
    c = SIZE // 2
    on_pane = rgb[c + 10:c + 18, c - 4:c + 4].reshape(-1, 3).mean(axis=0)
    beside = rgb[c + 10:c + 18, 2:10].reshape(-1, 3).mean(axis=0)
    self.assertLess(np.abs(on_pane - beside).max(), 30)

  def test_two_panes_still_show_the_wall(self):
    """A cab has a window on each side: the ray passes the second pane too and finds the red wall."""
    self.wall()
    self.pane()
    path = os.path.join(self.folder, "second.obj")
    write_pane(path, 1.0)
    vis = p.createVisualShape(p.GEOM_MESH, fileName=path, flags=GLASS | TWO_SIDED, rgbaColor=[1, 1, 1, 1], specularColor=[0, 0, 0])
    p.createMultiBody(0, -1, vis, basePosition=[0, 1.0, 0])
    seen = self.centre(self.render()[0])
    self.assertGreater(seen[0], seen[1] + 40)

  def test_change_visual_shape_turns_glass_on_and_off(self):
    """The bit can be set and cleared on a body after creation."""
    self.wall()
    uid = self.pane(flags=TWO_SIDED)
    before = self.centre(self.render()[0])
    p.changeVisualShape(uid, -1, flags=GLASS | TWO_SIDED)
    on = self.centre(self.render()[0])
    p.changeVisualShape(uid, -1, flags=TWO_SIDED)
    off = self.centre(self.render()[0])
    self.assertGreater(on[0], on[1] + 40)
    self.assertLess(abs(before[0] - before[1]), 12)
    self.assertLess(abs(off[0] - off[1]), 12)

  def test_glass_on_a_mover_too(self):
    """A pane that moved after the first frame keeps looking through."""
    self.wall()
    uid = self.pane()
    self.render()
    p.resetBasePositionAndOrientation(uid, [0.2, 0.3, 0], [0, 0, 0, 1])
    moved = self.centre(self.render()[0])
    self.assertGreater(moved[0], moved[1] + 40)

  def test_threads_give_the_same_bytes(self):
    """The frame is byte-identical at every render thread count the build allows."""
    frames = []
    for threads in ("1", "2", "4"):
      os.environ["SWARM_RENDER_THREADS"] = threads
      p.disconnect()
      p.connect(p.DIRECT)
      self.wall()
      self.pane(rgba=(0.8, 0.9, 0.8, 1))
      rgb, _, _ = self.render(flags=DAY | getattr(p, "ER_EDGE_ANTIALIAS", 0), hazeDistance=30.0)
      frames.append(rgb.tobytes())
    self.assertEqual(frames[0], frames[1])
    self.assertEqual(frames[0], frames[2])


if __name__ == '__main__':
  unittest.main()
