"""One team of render threads serves every parallel pass of a ray-cast frame.

The colour frame with its edge pass and depth hint, the painted frame, the low-light and the
thermal camera chains give the same bytes at 1, 2 and 3 render threads, and frames keep coming
while another thread builds and drops bodies on a second client.
"""

import hashlib
import os
import subprocess
import sys
import threading
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as p

from daylightTest import RAYCAST
from thermalTest import FRAME, NIGHT, SUN_45, THERMAL, full_scene

LOW_LIGHT = getattr(p, "ER_SWARM_LOW_LIGHT", 0)
RASTER = getattr(p, "ER_SWARM_RASTER", 0)
COLOUR = (RAYCAST | getattr(p, "ER_SWARM_SHADOW_MAP", 0) | getattr(p, "ER_EDGE_ANTIALIAS", 0)
          | getattr(p, "ER_TEXTURE_FILTER", 0) | getattr(p, "ER_ALPHA_CUTOUT", 0)
          | getattr(p, "ER_SWARM_LINEAR_LIGHT", 0) | p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
# sha256 of the sequence below drawn by the release wheel, one parallel region per pass.
EXPECTED = "caf913a305fc26ceadb5f015637a89b23564f08b58fcc4b19dd5bbcc92d9731c"
# Wide enough for several pixel chunks, with tiles cut short at the right and bottom edges.
WIDTH, HEIGHT = 200, 120
# The static box moves before this shot, so its cells of the shadow map are cast again.
BOX_MOVES_AT = 2
# Frames, and rounds of the other thread, the busy check waits for at least.
MIN_FRAMES, MIN_ROUNDS = 30, 10
EYE = ((-1, -6, 3), (0, 0, 0.5))
SHOTS: tuple[tuple[int, dict[str, Any]], ...] = (
  (COLOUR, {"lightDirection": SUN_45}),
  (COLOUR, {"lightDirection": SUN_45}),
  (COLOUR | RASTER, {"lightDirection": SUN_45}),
  (COLOUR | LOW_LIGHT, {"lightDirection": NIGHT, "lightAmbientCoeff": 0.03, "sensorPhotons": 300.0,
                        "sensorReadNoise": 4.0, "sensorGainCap": 500.0, "sensorSeed": 11}),
  (FRAME, {"lightDirection": SUN_45, "airTemperature": 20.0, "skyTemperature": -20.0,
           "thermalSeed": 5}),
)


def camera() -> tuple[list[float], list[float]]:
  """The view and projection every shot is drawn with."""
  view = p.computeViewMatrix(list(EYE[0]), list(EYE[1]), [0, 0, 1])
  return view, p.computeProjectionMatrixFOV(70, WIDTH / HEIGHT, 0.1, 100.0)


def shot(flags: int, arguments: dict[str, Any], client: int = 0) -> bytes:
  """Colour, depth and mask of one frame of the client's world."""
  view, proj = camera()
  _, _, rgb, depth, seg = p.getCameraImage(WIDTH, HEIGHT, view, proj, renderer=p.ER_TINY_RENDERER,
                                           flags=flags, physicsClientId=client, **arguments)
  colour = np.asarray(rgb, dtype=np.uint8).tobytes()
  mask = np.asarray(seg, dtype=np.int32).tobytes()
  frame: bytes = colour + np.asarray(depth, dtype=np.float32).tobytes() + mask
  return frame


def sequence_hash() -> str:
  """sha256 of every shot, in order, of the thermal test's full scene.

  A static box moves after the first two shots, so its cells of the shadow map are cast again.
  """
  full_scene()  # type: ignore[no-untyped-call]
  vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.4, 0.8], rgbaColor=[0.7, 0.6, 0.5, 1],
                            specularColor=[0, 0, 0])
  box = p.createMultiBody(0, -1, vis, basePosition=[-2, -1, 0.8])
  digest = hashlib.sha256()
  for i, (flags, arguments) in enumerate(SHOTS):
    if i == BOX_MOVES_AT:
      p.resetBasePositionAndOrientation(box, [-1.5, -0.5, 0.8], [0, 0, 0, 1])
    digest.update(shot(flags, arguments))
  p.disconnect()
  return digest.hexdigest()


def busy() -> str:
  """Colour frames in a row while a second thread builds, draws and drops bodies on its own client.

  At least thirty frames and ten of the other thread's rounds; ok when every frame equals the one
  drawn before the thread started and the thread raised nothing.
  """
  full_scene()  # type: ignore[no-untyped-call]
  alone = shot(*SHOTS[0])
  stop = threading.Event()
  errors: list[Exception] = []
  rounds = [0]

  def churn() -> None:
    """Adds a box to a client of its own, draws it and removes it, until told to stop."""
    other = p.connect(p.DIRECT)
    try:
      while not stop.is_set():
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], physicsClientId=other)
        body = p.createMultiBody(0, -1, vis, basePosition=[0, 0, 0.5], physicsClientId=other)
        shot(COLOUR, {"lightDirection": SUN_45}, other)
        p.removeBody(body, physicsClientId=other)
        rounds[0] += 1
    except Exception as error:
      errors.append(error)
    finally:
      p.disconnect(physicsClientId=other)

  worker = threading.Thread(target=churn)
  worker.start()
  frames: list[bytes] = []
  try:
    while (len(frames) < MIN_FRAMES or rounds[0] < MIN_ROUNDS) and not errors:
      frames.append(shot(*SHOTS[0]))
  finally:
    stop.set()
    worker.join()
  p.disconnect()
  return "ok" if not errors and all(frame == alone for frame in frames) else f"different {errors}"


def child(mode: str, threads: str) -> str:
  """The last line this file prints in a fresh process at a thread count, run in the given mode."""
  env = dict(os.environ, SWARM_RENDER_THREADS=threads)
  here = Path(__file__).resolve()
  # Runs this same test file with this same interpreter; a hang fails the test, not the suite.
  output = subprocess.run([sys.executable, str(here), mode], env=env,  # noqa: S603
                          cwd=str(here.parent), text=True, stdout=subprocess.PIPE, check=True,
                          timeout=600).stdout
  return output.strip().splitlines()[-1]


@unittest.skipUnless(RAYCAST and THERMAL and LOW_LIGHT and RASTER,
                     "wheel without the ray-cast cameras")
class TestFrameTeam(unittest.TestCase):
  """Frames drawn by one team of render threads against one parallel region per pass."""

  def test_threads_give_the_recorded_bytes(self) -> None:
    """The sequence is the same at 1, 2 and 3 render threads and equal to the release wheel's."""
    digests = [child("--hash", threads) for threads in ("1", "2", "3")]
    self.assertEqual(len(set(digests)), 1)
    self.assertEqual(digests[0], EXPECTED)

  def test_frames_render_while_another_client_changes(self) -> None:
    """Frames on one client keep their bytes while a second thread changes another client."""
    self.assertEqual(child("--busy", "3"), "ok")


if __name__ == "__main__":
  if sys.argv[1:] == ["--hash"]:
    print(sequence_hash())
  elif sys.argv[1:] == ["--busy"]:
    print(busy())
  else:
    unittest.main()
