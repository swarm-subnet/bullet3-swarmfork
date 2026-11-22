"""A thermal frame's hits shaded together per tile give the bytes they gave one at a time.

At every thread count: textured, heated and passive surfaces under a roof's shelter and the sun's
shadow.
"""

import hashlib
import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
from daylightTest import RAYCAST, write_obj, write_png
from thermalTest import FRAME, SUN_45, THERMAL, Scene, full_scene

import pybullet as p

# sha256 of the frame below drawn one hit at a time, before tiles shaded thermal hits together.
EXPECTED = "d881ec20a758209eefdd0485693d4198b32ca4dd66fb1d6e270b2af311d4dbf8"
# Looking down past the slab at the roof's shadow, the textured ground filling most of the frame.
EYE = ((-1, -5, 3), (-1, 0, 0))


def textured_scene() -> Scene:
  """The full thermal scene with a noise texture on the ground and on a passive slab.

  Footprints, mip levels and several taps all reach the thermal reads.
  """
  scene: Scene = full_scene()  # type: ignore[no-untyped-call]
  folder = Path(scene.folder)
  rng = np.random.default_rng(124)
  noise = rng.integers(0, 256, size=(64, 64, 4), dtype=np.uint8)
  noise[:, :, 3] = 255
  png = folder / "noise.png"
  write_png(str(png), noise)  # type: ignore[no-untyped-call]
  texture = p.loadTexture(str(png))
  p.changeVisualShape(scene.ground, -1, textureUniqueId=texture)
  path = folder / "slab.obj"
  write_obj(str(path), 1.5)  # type: ignore[no-untyped-call]
  vis = p.createVisualShape(p.GEOM_MESH, fileName=str(path), rgbaColor=[0.8, 0.7, 0.6, 1],
                            specularColor=[0, 0, 0])
  tilt = p.getQuaternionFromEuler([0.3, 0.0, 0.4])
  slab = p.createMultiBody(0, -1, vis, basePosition=[-2.0, -2.0, 0.3], baseOrientation=tilt)
  p.changeVisualShape(slab, -1, textureUniqueId=texture)
  return scene


def frame_hash() -> str:
  """sha256 of the textured scene's thermal frame with the sun up."""
  scene = textured_scene()
  rgb, _, _, _ = scene.render(flags=FRAME, eye=EYE, sun=SUN_45)  # type: ignore[no-untyped-call]
  p.disconnect()
  return hashlib.sha256(rgb.tobytes()).hexdigest()


def child_hash(threads: str) -> str:
  """frame_hash from this file run in a fresh process at a thread count."""
  env = dict(os.environ, SWARM_RENDER_THREADS=threads)
  # Runs this same test file with this same interpreter.
  here = Path(__file__).resolve()
  output = subprocess.check_output([sys.executable, str(here), "--hash"], env=env, text=True,  # noqa: S603
                                   cwd=str(here.parent))
  return output.strip().splitlines()[-1]


@unittest.skipUnless(RAYCAST and THERMAL, "wheel without the thermal camera")
class TestThermalTile(unittest.TestCase):
  """The tile-batched thermal shading against the bytes of the one-hit-at-a-time shading."""

  def test_threads_give_the_recorded_bytes(self) -> None:
    """The frame is the same at 1, 2 and 3 render threads and equal to the one-at-a-time frame."""
    digests = [child_hash(threads) for threads in ("1", "2", "3")]
    self.assertEqual(len(set(digests)), 1)
    self.assertEqual(digests[0], EXPECTED)


if __name__ == "__main__":
  if sys.argv[1:] == ["--hash"]:
    print(frame_hash())
  else:
    unittest.main()
