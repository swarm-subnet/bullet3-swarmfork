"""Shared mesh and texture blocks: same pixels with sharing on or off, and edits stay per instance."""
import hashlib
import json
import os
import subprocess
import sys
import unittest

import numpy as np
import pybullet as p
import pybullet_data

SIZE = 96
QUAD_VERTICES = [[-0.4, -0.4, 0], [0.4, -0.4, 0], [0.4, 0.4, 0], [-0.4, 0.4, 0]]
QUAD_INDICES = [0, 1, 2, 0, 2, 3]


def render(eye, target):
  """Returns the colour, depth and segmentation buffers seen from eye toward target."""
  view = p.computeViewMatrix(eye, target, [0, 1, 0])
  proj = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 20.0)
  _, _, rgb, depth, seg = p.getCameraImage(SIZE, SIZE, view, proj, shadow=0,
                                           renderer=p.ER_TINY_RENDERER)
  return np.asarray(rgb), np.asarray(depth), np.asarray(seg)


def digest(buffers):
  """SHA-256 over every buffer in order."""
  h = hashlib.sha256()
  for buf in buffers:
    h.update(np.ascontiguousarray(buf).tobytes())
  return h.hexdigest()


def spawn_quad(rgba, position):
  """One quad built from in-memory vertices, the path map builders use for material meshes."""
  vis = p.createVisualShape(p.GEOM_MESH, vertices=QUAD_VERTICES, indices=QUAD_INDICES, rgbaColor=rgba)
  return p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=position)


def scene_hashes():
  """Builds repeated textured and untextured instances, edits some, removes one, and hashes each phase."""
  p.connect(p.DIRECT)
  p.setAdditionalSearchPath(pybullet_data.getDataPath())
  eye, target = [0.75, 0.5, 4.0], [0.75, 0.5, 0.0]
  duck = p.createVisualShape(p.GEOM_MESH, fileName="duck.obj", meshScale=[0.08] * 3)
  ducks = [p.createMultiBody(baseMass=0, baseVisualShapeIndex=duck, basePosition=[0.5 * (i % 4), 0.5 * (i // 4), 0])
           for i in range(12)]
  quads = [spawn_quad([0, 0, 1, 1], [0.5 * (i % 4), 0.5 * (i // 4), -0.5]) for i in range(12)]
  phases = {"built": digest(render(eye, target))}

  p.changeVisualShape(ducks[0], -1, rgbaColor=[1, 0, 0, 1])
  p.changeVisualShape(quads[0], -1, rgbaColor=[0, 1, 0, 1])
  tex = p.loadTexture("tex256.png")
  for body in (ducks[1], ducks[2], quads[1]):
    p.changeVisualShape(body, -1, textureUniqueId=tex)
  phases["edited"] = digest(render(eye, target))

  p.removeBody(ducks[3])
  p.removeBody(quads[3])
  phases["removed"] = digest(render(eye, target))

  p.resetSimulation()
  p.createMultiBody(baseMass=0, baseVisualShapeIndex=p.createVisualShape(p.GEOM_MESH, fileName="duck.obj", meshScale=[0.08] * 3))
  spawn_quad([0, 0, 1, 1], [0.5, 0, -0.5])
  phases["reset"] = digest(render(eye, target))
  p.disconnect()
  return phases


def run_scene(share):
  """Runs scene_hashes in a fresh interpreter with sharing switched on or off."""
  env = dict(os.environ, SWARM_SHARE_MESH="1" if share else "0")
  out = subprocess.run([sys.executable, os.path.abspath(__file__), "--scene"], env=env,
                       check=True, capture_output=True, text=True)
  return json.loads(out.stdout.strip().splitlines()[-1])


class TestSharedMesh(unittest.TestCase):
  """Pixel identity across sharing modes, and per-instance edits on shared blocks."""

  def test_sharing_changes_no_pixel(self):
    """Every phase hashes the same with private copies and with shared blocks."""
    shared, private = run_scene(True), run_scene(False)
    self.assertEqual(shared, private)
    self.assertEqual(len(set(shared.values())), 4)

  def test_edits_stay_per_instance(self):
    """Recolouring one instance leaves its twins untouched, and removing one leaves the rest drawn."""
    p.connect(p.DIRECT)
    quads = [spawn_quad([1, 0, 0, 1], [i * 1.0, 0, 0]) for i in range(3)]
    p.changeVisualShape(quads[1], -1, rgbaColor=[0, 1, 0, 1])
    rgb, _, seg = render([1.0, 0, 4.0], [1.0, 0, 0])
    red, green = rgb[..., 0].astype(int), rgb[..., 1].astype(int)
    self.assertTrue(((red > 0) & (green == 0)).any())
    self.assertTrue(((green > 0) & (red == 0)).any())
    self.assertEqual(sorted(set(seg[seg >= 0].tolist())), quads)
    p.removeBody(quads[0])
    _, _, seg = render([1.0, 0, 4.0], [1.0, 0, 0])
    self.assertEqual(sorted(set(seg[seg >= 0].tolist())), quads[1:])
    p.disconnect()


if __name__ == "__main__":
  if "--scene" in sys.argv:
    print(json.dumps(scene_hashes()))
  else:
    unittest.main()
