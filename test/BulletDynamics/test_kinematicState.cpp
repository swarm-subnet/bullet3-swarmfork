#include <btBulletDynamicsCommon.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

static std::vector<int> calls;

// The ids the full walk of the world array visits, as saveKinematicState did before it kept a list.
static std::vector<int> expectedCalls(btDiscreteDynamicsWorld& world);

struct MotionState : btMotionState
{
	int id;
	btTransform transform;
	// A body this motion state makes kinematic the next time it is read, as a scripted motion state might.
	btRigidBody* makeKinematic;
	MotionState(int index) : id(index), makeKinematic(0)
	{
		transform.setIdentity();
	}
	void getWorldTransform(btTransform& worldTrans) const
	{
		calls.push_back(id);
		worldTrans = transform;
		if (makeKinematic)
			makeKinematic->setCollisionFlags(makeKinematic->getCollisionFlags() | btCollisionObject::CF_KINEMATIC_OBJECT);
	}
	void setWorldTransform(const btTransform& worldTrans)
	{
		transform = worldTrans;
	}
};

struct World : btDiscreteDynamicsWorld
{
	World(btDispatcher* dispatcher, btBroadphaseInterface* broadphase, btConstraintSolver* solver,
		btCollisionConfiguration* configuration)
		: btDiscreteDynamicsWorld(dispatcher, broadphase, solver, configuration) {}
	using btDiscreteDynamicsWorld::saveKinematicState;
};

int main()
{
	btDefaultCollisionConfiguration configuration;
	btCollisionDispatcher dispatcher(&configuration);
	btDbvtBroadphase broadphase;
	btSequentialImpulseConstraintSolver solver;
	btBoxShape shape(btVector3(1, 1, 1));
	std::vector<MotionState*> states;
	std::vector<btRigidBody*> bodies;
	for (int i = 0; i < 2408; i++)
	{
		states.push_back(new MotionState(i));
		bodies.push_back(new btRigidBody(0, states.back(), &shape));
	}
	{
		World world(&dispatcher, &broadphase, &solver, &configuration);
		for (int i = 0; i < int(bodies.size()); i++)
			world.addRigidBody(bodies[i]);
		unsigned int random = 124;
		for (int step = 0; step < 400; step++)
		{
			random = random * 1664525u + 1013904223u;
			btRigidBody* body = bodies[random % bodies.size()];
			body->setCollisionFlags(body->getCollisionFlags() ^ btCollisionObject::CF_KINEMATIC_OBJECT);
			body->forceActivationState(step % 3 ? DISABLE_DEACTIVATION : ISLAND_SLEEPING);
			body->setMassProps(step % 2, btVector3(0, 0, 0));
			btTransform transform = body->getWorldTransform();
			transform.setOrigin(btVector3(step, 0, 0));
			body->setWorldTransform(transform);
			if (step % 7 == 0)
			{
				world.removeCollisionObject(body);
				world.addCollisionObject(body);
			}
			const std::vector<int> expected = expectedCalls(world);
			calls.clear();
			world.saveKinematicState(btScalar(1.) / btScalar(60.));
			if (calls != expected) return EXIT_FAILURE;
			calls.clear();
			world.saveKinematicState(btScalar(1.) / btScalar(60.));
			if (calls != expected) return EXIT_FAILURE;
		}
	}
	{
		// The array reordered through its accessor, without an add or a remove: the walk follows the new order.
		World world(&dispatcher, &broadphase, &solver, &configuration);
		for (int i = 0; i < 4; i++)
		{
			bodies[i]->setCollisionFlags(btCollisionObject::CF_KINEMATIC_OBJECT);
			bodies[i]->forceActivationState(DISABLE_DEACTIVATION);
			world.addRigidBody(bodies[i]);
		}
		world.saveKinematicState(btScalar(1.) / btScalar(60.));
		world.getCollisionObjectArray().swap(0, 3);
		world.getCollisionObjectArray()[0]->setWorldArrayIndex(0);
		world.getCollisionObjectArray()[3]->setWorldArrayIndex(3);
		calls.clear();
		world.saveKinematicState(btScalar(1.) / btScalar(60.));
		if (calls != expectedCalls(world)) return EXIT_FAILURE;

		// A motion state that makes a later body kinematic while it is read: that body is visited in the same step.
		bodies[5]->setCollisionFlags(0);
		bodies[5]->forceActivationState(DISABLE_DEACTIVATION);
		world.addRigidBody(bodies[5]);
		world.saveKinematicState(btScalar(1.) / btScalar(60.));
		states[bodies[0]->getWorldArrayIndex() == 0 ? 0 : 3]->makeKinematic = bodies[5];
		const std::vector<int> before = expectedCalls(world);
		calls.clear();
		world.saveKinematicState(btScalar(1.) / btScalar(60.));
		std::vector<int> wanted = before;
		wanted.push_back(5);
		if (calls != wanted) return EXIT_FAILURE;
		states[0]->makeKinematic = 0;
		states[3]->makeKinematic = 0;
		for (int i = 0; i < 4; i++)
			world.removeRigidBody(bodies[i]);
		world.removeRigidBody(bodies[5]);
	}
	// Bodies may outlive their world and change flags before joining another one.
	for (int i = 0; i < int(bodies.size()); i++)
	{
		bodies[i]->setCollisionFlags(btCollisionObject::CF_KINEMATIC_OBJECT);
		delete bodies[i];
		delete states[i];
	}
	std::puts("Kinematic state order: PASS (400 steps, 2408 bodies)");
	return EXIT_SUCCESS;
}

static std::vector<int> expectedCalls(btDiscreteDynamicsWorld& world)
{
	std::vector<int> expected;
	for (int i = 0; i < world.getNumCollisionObjects(); i++)
	{
		btRigidBody* candidate = btRigidBody::upcast(world.getCollisionObjectArray()[i]);
		if (candidate && candidate->isKinematicObject() && candidate->getActivationState() != ISLAND_SLEEPING)
			expected.push_back(static_cast<MotionState*>(candidate->getMotionState())->id);
	}
	return expected;
}
