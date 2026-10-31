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


def tree_order_fingerprint():
  """Hash, every step, the broadphase's leaf order, every contact in order and every ball's state, while pinned and
  ordinary fixed bodies go stale in the same step, wake, re-pin, move and are removed."""
  digest = hashlib.sha256()
  cli = p.connect(p.DIRECT)
  try:
    p.setGravity(0, 0, -10, physicsClientId=cli)
    slab = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.1], physicsClientId=cli)
    ball = p.createCollisionShape(p.GEOM_SPHERE, radius=0.2, physicsClientId=cli)
    # Balls created between the fixed bodies, so ordinary and pinned bodies interleave in the world's order.
    fixed, balls = [], []
    for x in range(-4, 5):
      for y in range(-4, 5):
        fixed.append(p.createMultiBody(0, slab, basePosition=[1.1 * x, 1.1 * y, 0], useMaximalCoordinates=True,
                                       physicsClientId=cli))
        if (x + y) % 3 == 0:
          balls.append(p.createMultiBody(1, ball, basePosition=[1.1 * x + 0.1, 1.1 * y, 1 + 0.1 * len(balls)],
                                         physicsClientId=cli))
    state = lambda mode, *bodies: [p.changeDynamics(b, -1, activationState=mode, physicsClientId=cli) for b in bodies]
    sleep = lambda *bodies: state(p.ACTIVATION_STATE_SLEEP, *bodies)
    wake = lambda *bodies: state(p.ACTIVATION_STATE_WAKE_UP, *bodies)
    pin = lambda *bodies: state(p.ACTIVATION_STATE_DISABLE_SLEEPING, *bodies)
    move = lambda body, z: p.resetBasePositionAndOrientation(
        body, p.getBasePositionAndOrientation(body, physicsClientId=cli)[0][:2] + (z,), [0, 0, 0, 1], physicsClientId=cli)
    # Every third fixed body stays an ordinary awake one.
    pin(*[b for k, b in enumerate(fixed) if k % 3])
    wake(*fixed[::3])
    script = {
        # Pinned and ordinary fixed bodies go stale in one step: the release fills the fixed tree in its list order.
        10: lambda: sleep(*fixed[::5]),
        20: lambda: sleep(*fixed[1::4] + balls[::2]),
        30: lambda: wake(*fixed[::10] + [balls[5]]),
        31: lambda: pin(*fixed[::20]),
        50: lambda: (sleep(fixed[41], fixed[42], fixed[43], balls[0]), p.removeBody(fixed[44], physicsClientId=cli)),
        70: lambda: (move(fixed[52], -0.05), sleep(*fixed[51:62:2] + [balls[3]])),
        90: lambda: (wake(*fixed[1:30:4]), sleep(*fixed[62:70])),
        91: lambda: (pin(*fixed[1:30:8]), move(fixed[52], 0)),
        110: lambda: (sleep(*fixed[2::6] + [balls[2]]), wake(balls[3])),
    }
    for step in range(140):
      if step in script:
        script[step]()
      p.stepSimulation(physicsClientId=cli)
      # The broadphase reports overlaps walking its moving tree and then its fixed tree, each in leaf order.
      hits = p.getOverlappingObjects([-50, -50, -50], [50, 50, 50], physicsClientId=cli) or []
      digest.update(struct.pack("%di" % (2 * len(hits)), *[v for hit in hits for v in hit]))
      # Only the balls: reading a fixed body's state counts as a transform write, which would keep it from pinning.
      for body in balls:
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

  def test_pinned_and_ordinary_bodies_leave_the_tree_in_release_order(self):
    """Leaf order, contact order and every state match the release wheel when pinned and ordinary bodies go stale together."""
    self.assertEqual(tree_order_fingerprint(), "cfcbe31cce182994badf58f66169a801fd9eef5197c1cd29e225358189342db9")


if __name__ == '__main__':
  unittest.main()
