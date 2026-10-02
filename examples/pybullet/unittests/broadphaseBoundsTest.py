"""The broadphase keeps a fixed body's bounds while it stays put and follows it the step after it moves."""

import unittest

import pybullet as p


def _overlapping(cli, low, high):
  """The bodies whose broadphase bounds overlap a box."""
  return {uid for uid, _link in (p.getOverlappingObjects(low, high, physicsClientId=cli) or [])}


class BroadphaseBoundsTest(unittest.TestCase):
  """Broadphase bounds of fixed bodies that stay put, move, or change their collision margin."""

  def test_a_resting_ball_keeps_its_floor_until_the_floor_moves(self):
    """A ball rests on a fixed slab for many steps; once the slab is moved away the broadphase finds it at its new
    place only, and the ball falls through where it was."""
    cli = p.connect(p.DIRECT)
    try:
      p.setGravity(0, 0, -10, physicsClientId=cli)
      slab = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1, 1, 0.1], physicsClientId=cli)
      floor = p.createMultiBody(0, slab, basePosition=[0, 0, 0], useMaximalCoordinates=True, physicsClientId=cli)
      sphere = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1, physicsClientId=cli)
      ball = p.createMultiBody(1, sphere, basePosition=[0, 0, 0.3], physicsClientId=cli)
      for _ in range(240):
        p.stepSimulation(physicsClientId=cli)
      self.assertTrue(p.getContactPoints(ball, floor, physicsClientId=cli))
      self.assertAlmostEqual(p.getBasePositionAndOrientation(ball, physicsClientId=cli)[0][2], 0.2, delta=0.01)
      self.assertIn(floor, _overlapping(cli, [-0.5, -0.5, -0.05], [0.5, 0.5, 0.05]))

      p.resetBasePositionAndOrientation(floor, [10, 0, 0], [0, 0, 0, 1], physicsClientId=cli)
      for _ in range(60):
        p.stepSimulation(physicsClientId=cli)
      self.assertNotIn(floor, _overlapping(cli, [-0.5, -0.5, -0.05], [0.5, 0.5, 0.05]))
      self.assertIn(floor, _overlapping(cli, [9.5, -0.5, -0.05], [10.5, 0.5, 0.05]))
      self.assertFalse(p.getContactPoints(ball, floor, physicsClientId=cli))
      self.assertLess(p.getBasePositionAndOrientation(ball, physicsClientId=cli)[0][2], 0.0)
    finally:
      p.disconnect(cli)

  def test_a_slab_put_back_under_a_ball_catches_it_again(self):
    """Moving a fixed slab away and back to its exact place, a step apart, leaves the ball resting on it as before."""
    cli = p.connect(p.DIRECT)
    try:
      p.setGravity(0, 0, -10, physicsClientId=cli)
      slab = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1, 1, 0.1], physicsClientId=cli)
      floor = p.createMultiBody(0, slab, basePosition=[0, 0, 0], useMaximalCoordinates=True, physicsClientId=cli)
      sphere = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1, physicsClientId=cli)
      ball = p.createMultiBody(1, sphere, basePosition=[0, 0, 0.3], physicsClientId=cli)
      for _ in range(240):
        p.stepSimulation(physicsClientId=cli)
      p.resetBasePositionAndOrientation(floor, [0, 0, 5], [0, 0, 0, 1], physicsClientId=cli)
      p.stepSimulation(physicsClientId=cli)
      p.resetBasePositionAndOrientation(floor, [0, 0, 0], [0, 0, 0, 1], physicsClientId=cli)
      for _ in range(120):
        p.stepSimulation(physicsClientId=cli)
      self.assertTrue(p.getContactPoints(ball, floor, physicsClientId=cli))
      self.assertGreater(p.getBasePositionAndOrientation(ball, physicsClientId=cli)[0][2], 0.15)
    finally:
      p.disconnect(cli)


if __name__ == "__main__":
  unittest.main()
