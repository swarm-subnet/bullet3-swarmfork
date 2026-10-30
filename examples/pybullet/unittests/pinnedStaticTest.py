"""Fixed bodies pinned out of the broadphase's per-step restaging keep every body state and contact of the release."""
import hashlib
import struct
import unittest

import pybullet as p


def fingerprint():
  """Hash every body state and contact over steps that pin, move, wake, sleep, re-pin and remove fixed bodies."""
  digest = hashlib.sha256()
  cli = p.connect(p.DIRECT)
  try:
    p.setGravity(0, 0, -10, physicsClientId=cli)
    slab = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.1], physicsClientId=cli)
    ball = p.createCollisionShape(p.GEOM_SPHERE, radius=0.2, physicsClientId=cli)
    fixed = [p.createMultiBody(0, slab, basePosition=[x, y, 0], useMaximalCoordinates=True, physicsClientId=cli)
             for x in range(-3, 4) for y in range(-3, 4)]
    balls = [p.createMultiBody(1, ball, basePosition=[0.3 * k - 1, 0.2 * k - 1, 1 + 0.3 * k], physicsClientId=cli)
             for k in range(8)]
    # Several bodies go to sleep in one step, so more than one leaf leaves the dynamic tree at once.
    script = {
        60: lambda: p.resetBasePositionAndOrientation(fixed[24], [0, 0, -0.05], [0, 0, 0, 1], physicsClientId=cli),
        90: lambda: [p.changeDynamics(body, -1, activationState=p.ACTIVATION_STATE_SLEEP, physicsClientId=cli)
                     for body in (fixed[10], fixed[11], fixed[30], balls[7])],
        150: lambda: [p.changeDynamics(body, -1, activationState=p.ACTIVATION_STATE_WAKE_UP, physicsClientId=cli)
                      for body in (fixed[10], balls[7])],
        170: lambda: p.changeDynamics(fixed[10], -1, activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING, physicsClientId=cli),
        200: lambda: p.removeBody(fixed[17], physicsClientId=cli),
        230: lambda: p.resetBasePositionAndOrientation(fixed[24], [0, 0, 0], [0, 0, 0, 1], physicsClientId=cli),
    }
    alive = fixed[:17] + fixed[18:] + balls
    for step in range(300):
      if step in script:
        script[step]()
      p.stepSimulation(physicsClientId=cli)
      for body in (alive if step >= 200 else fixed + balls):
        position, orientation = p.getBasePositionAndOrientation(body, physicsClientId=cli)
        linear, angular = p.getBaseVelocity(body, physicsClientId=cli)
        digest.update(struct.pack("13d", *position, *orientation, *linear, *angular))
      for contact in p.getContactPoints(physicsClientId=cli):
        digest.update(struct.pack("4i", *contact[1:5]) + struct.pack("10d", *contact[5], *contact[6], *contact[7], contact[8]))
    return digest.hexdigest()
  finally:
    p.disconnect(cli)


class PinnedStaticTest(unittest.TestCase):
  """Require the release wheel's exact physics for fixed bodies that stay put, move, sleep and wake."""

  def test_pinned_fixed_bodies_keep_release_states_and_contacts(self):
    """Every body state and every contact, in order, matches the release wheel bit for bit."""
    self.assertEqual(fingerprint(), "687c4cebc11907f63768c57d1b3a2964e189f9d8d65012371f5dd502a9f2acb3")


if __name__ == '__main__':
  unittest.main()
