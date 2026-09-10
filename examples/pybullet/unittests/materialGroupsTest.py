"""Tests for VISUAL_SHAPE_MATERIALS_FROM_MTL: one render object per material in a multi-material OBJ."""
import os
import shutil
import struct
import tempfile
import unittest

import numpy as np
import pybullet as p

# Two vertical quads facing -Y in one OBJ object, Blender style: one "o", one "usemtl" per material.
# The left quad is flat red, the right quad carries a green texture.
OBJ = """mtllib pair.mtl
o Pair
v -2 0 -1
v -0.5 0 -1
v -0.5 0 1
v -2 0 1
v 0.5 0 -1
v 2 0 -1
v 2 0 1
v 0.5 0 1
vt 0 0
vt 1 0
vt 1 1
vt 0 1
usemtl red
f 1 2 3
f 1 3 4
usemtl green_tex
f 5/1 6/2 7/3
f 5/1 7/3 8/4
"""
MTL = """newmtl red
Kd 1 0 0
newmtl green_tex
Kd 1 1 1
map_Kd green.tga
"""
URDF = """<robot name="pair"><link name="base"><visual><geometry><mesh filename="{obj}"/></geometry></visual></link></robot>"""
SIZE = 64


def write_tga(path, rgb, width=2, height=2):
  """Writes an uncompressed 24-bit TGA filled with one colour."""
  header = struct.pack("<BBBHHBHHHHBB", 0, 0, 2, 0, 0, 0, 0, 0, width, height, 24, 0)
  with open(path, "wb") as f:
    f.write(header + bytes(rgb[::-1]) * (width * height))


class TestMaterialGroups(unittest.TestCase):
  """Renders the two-material quad pair with and without the flag and checks each half's colour."""

  def setUp(self):
    """Writes the OBJ, MTL, texture and URDF into a fresh temp folder and opens a DIRECT client."""
    self.tmp = tempfile.mkdtemp()
    self.obj = os.path.join(self.tmp, "pair.obj")
    with open(self.obj, "w") as f:
      f.write(OBJ)
    with open(os.path.join(self.tmp, "pair.mtl"), "w") as f:
      f.write(MTL)
    write_tga(os.path.join(self.tmp, "green.tga"), (0, 255, 0))
    self.urdf = os.path.join(self.tmp, "pair.urdf")
    with open(self.urdf, "w") as f:
      f.write(URDF.format(obj=self.obj))
    p.connect(p.DIRECT)
    self.proj = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 10.0)
    self.view = p.computeViewMatrix([0, -4, 0], [0, 0, 0], [0, 0, 1])

  def tearDown(self):
    """Closes the client and removes the temp folder."""
    p.disconnect()
    shutil.rmtree(self.tmp, ignore_errors=True)

  def spawn(self, flags=0):
    """Creates one static body from the OBJ with the given createVisualShape flags."""
    vis = p.createVisualShape(p.GEOM_MESH, fileName=self.obj, flags=flags)
    return p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis)

  def render(self):
    """Returns the colour image as an (H, W, 3) array and the depth buffer."""
    _, _, rgb, depth, _ = p.getCameraImage(SIZE, SIZE, self.view, self.proj, shadow=0,
                                           renderer=p.ER_TINY_RENDERER)
    rgb = np.asarray(rgb).reshape(SIZE, SIZE, 4)[:, :, :3].astype(int)
    return rgb, np.asarray(depth).reshape(SIZE, SIZE)

  def halves(self):
    """Returns (red, green) hit counts for the left and right halves of the image."""
    rgb, _ = self.render()
    red = (rgb[:, :, 0] > rgb[:, :, 1]) & (rgb[:, :, 0] > rgb[:, :, 2])
    green = (rgb[:, :, 1] > rgb[:, :, 0]) & (rgb[:, :, 1] > rgb[:, :, 2])
    half = SIZE // 2
    left = (int(red[:, :half].sum()), int(green[:, :half].sum()))
    right = (int(red[:, half:].sum()), int(green[:, half:].sum()))
    return left, right

  def assert_one_material(self, left, right):
    """Asserts the old behaviour: the last material's texture paints both quads."""
    self.assertEqual(left[0], 0)
    self.assertGreater(left[1], 0)
    self.assertEqual(right[0], 0)
    self.assertGreater(right[1], 0)

  def assert_two_materials(self, left, right):
    """Asserts the new behaviour: red on the left quad, green texture on the right quad."""
    self.assertGreater(left[0], 0)
    self.assertEqual(left[1], 0)
    self.assertEqual(right[0], 0)
    self.assertGreater(right[1], 0)

  def test_without_flag_one_material_paints_everything(self):
    """Without the flag both quads get the last material, as before."""
    self.spawn()
    self.assert_one_material(*self.halves())

  def test_flag_keeps_each_material(self):
    """With the flag each quad keeps its own colour and texture inside one body."""
    self.spawn(flags=p.VISUAL_SHAPE_MATERIALS_FROM_MTL)
    self.assert_two_materials(*self.halves())

  def test_flag_does_not_change_depth_or_shape_count(self):
    """Depth is bit-identical with and without the flag and the body still reports one visual shape."""
    body = self.spawn()
    _, depth_plain = self.render()
    p.removeBody(body)
    body = self.spawn(flags=p.VISUAL_SHAPE_MATERIALS_FROM_MTL)
    _, depth_flag = self.render()
    self.assertTrue(np.array_equal(depth_plain, depth_flag))
    self.assertEqual(len(p.getVisualShapeData(body)), 1)

  def test_flagged_load_does_not_leak_into_plain_load(self):
    """A plain load after a flagged load of the same file still behaves the old way."""
    body = self.spawn(flags=p.VISUAL_SHAPE_MATERIALS_FROM_MTL)
    p.removeBody(body)
    self.spawn()
    self.assert_one_material(*self.halves())

  def test_urdf_flag(self):
    """loadURDF honours URDF_USE_MATERIALS_FROM_MTL on every visual of the model."""
    body = p.loadURDF(self.urdf, useFixedBase=True)
    self.assert_one_material(*self.halves())
    p.removeBody(body)
    p.loadURDF(self.urdf, useFixedBase=True, flags=p.URDF_USE_MATERIALS_FROM_MTL)
    self.assert_two_materials(*self.halves())


if __name__ == '__main__':
  unittest.main()
