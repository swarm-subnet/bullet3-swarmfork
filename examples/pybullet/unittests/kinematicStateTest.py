"""Check exact states and contacts while a large fixed world changes membership and mass."""

import hashlib
import random
import struct
import unittest

import pybullet as p

RELEASE_FINGERPRINT = "c6205ace00499680df0dbed0a2ff7383edc00825a3a3429f03154d6ba227edd8"


def fingerprint() -> str:
  """Hash all poses, velocities and full ordered contacts over a fixed random schedule."""
  rng = random.Random(124)  # noqa: S311
  digest = hashlib.sha256()
  cli = p.connect(p.DIRECT)
  try:
    p.setGravity(0, 0, -10, physicsClientId=cli)
    box = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.48, 0.48, 0.1], physicsClientId=cli)
    sphere = p.createCollisionShape(p.GEOM_SPHERE, radius=0.2, physicsClientId=cli)
    bodies = [p.createMultiBody(0, box, basePosition=[x, y, 0], useMaximalCoordinates=True,
                                physicsClientId=cli)
              for x in range(-15, 15) for y in range(-10, 10)]
    moving = [p.createMultiBody(1, sphere, basePosition=[k * 0.6, 0, 1],
                                useMaximalCoordinates=True, physicsClientId=cli) for k in range(6)]
    bodies.extend(moving)
    for step in range(320):
      if step % 17 == 0:
        body = rng.choice(bodies[:-6])
        place = [rng.randrange(-10, 10), rng.randrange(-8, 8), rng.choice([0, 0.1])]
        p.resetBasePositionAndOrientation(body, place, [0, 0, 0, 1], physicsClientId=cli)
        p.changeDynamics(body, -1, activationState=rng.choice([
            p.ACTIVATION_STATE_SLEEP, p.ACTIVATION_STATE_WAKE_UP,
            p.ACTIVATION_STATE_DISABLE_SLEEPING]), physicsClientId=cli)
      if step % 31 == 0:
        body = rng.choice(moving)
        p.changeDynamics(body, -1, mass=rng.choice([0, 1]), physicsClientId=cli)
      if step % 43 == 0:
        body = bodies.pop(rng.randrange(len(bodies) - 6))
        p.removeBody(body, physicsClientId=cli)
        bodies.insert(0, p.createMultiBody(0, box, basePosition=[rng.randrange(-10, 10), 2, 0],
                                           useMaximalCoordinates=True, physicsClientId=cli))
      p.stepSimulation(physicsClientId=cli)
      for body in bodies:
        pos, orn = p.getBasePositionAndOrientation(body, physicsClientId=cli)
        lin, ang = p.getBaseVelocity(body, physicsClientId=cli)
        digest.update(struct.pack("i13d", body, *pos, *orn, *lin, *ang))
      for contact in p.getContactPoints(physicsClientId=cli):
        digest.update(struct.pack("5i19d", *contact[:5], *contact[5], *contact[6], *contact[7],
                                  *contact[8:11], *contact[11], contact[12], *contact[13]))
    return digest.hexdigest()
  finally:
    p.disconnect(cli)


class KinematicStateTest(unittest.TestCase):
  """Require exact release results after world membership and rigid body flags change."""

  def test_release_states_and_contacts(self) -> None:
    """Every recorded value matches the release wheel exactly."""
    self.assertEqual(fingerprint(), RELEASE_FINGERPRINT)


if __name__ == "__main__":
  unittest.main()
