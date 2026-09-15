IF(BUILD_BULLET3)
	#SUBDIRS( TestBullet3OpenCL )
	SUBDIRS(  InverseDynamics SharedMemory )
ENDIF(BUILD_BULLET3)

SUBDIRS(  gtest-1.7.0 collision BulletDynamics )

# docs/swarm_api.md against the source; reads text only, so it needs no built module.
ADD_TEST(Test_SwarmApiDoc_PASS python3 ${CMAKE_SOURCE_DIR}/examples/pybullet/unittests/swarmApiDocTest.py)

