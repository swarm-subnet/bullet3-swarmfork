
INCLUDE_DIRECTORIES(
	.
	../../src
	../gtest-1.7.0/include
	../../examples
	../../examples/ThirdPartyLibs
)


#ADD_DEFINITIONS(-DGTEST_HAS_PTHREAD=1)
ADD_DEFINITIONS(-DPHYSICS_LOOP_BACK -DPHYSICS_SERVER_DIRECT -DENABLE_GTEST -D_VARIADIC_MAX=10)

IF (USE_SOFT_BODY_MULTI_BODY_DYNAMICS_WORLD)
	LINK_LIBRARIES(BulletSoftBody)
ENDIF()

LINK_LIBRARIES(
 BulletInverseDynamicsUtils BulletInverseDynamics BulletFileLoader BulletWorldImporter Bullet3Common BulletDynamics BulletCollision LinearMath gtest BussIK
)

IF (NOT WIN32)
	FIND_PACKAGE(Threads)
	LINK_LIBRARIES( ${CMAKE_THREAD_LIBS_INIT} )
	IF (NOT APPLE)
		LINK_LIBRARIES(${DL})
	ENDIF()
ENDIF()

	ADD_EXECUTABLE(Test_PhysicsClientServer
		gtestwrap.cpp
		../../src/SharedMemory/PhysicsClient.cpp
		../../src/SharedMemory/plugins/collisionFilterPlugin/collisionFilterPlugin.cpp
		../../src/SharedMemory/plugins/pdControlPlugin/pdControlPlugin.cpp
		../../src/SharedMemory/plugins/pdControlPlugin/pdControlPlugin.h
		../../src/SharedMemory/b3RobotSimulatorClientAPI_NoDirect.cpp
		../../src/SharedMemory/b3RobotSimulatorClientAPI_NoDirect.h
		../../src/SharedMemory/IKTrajectoryHelper.cpp
		../../src/SharedMemory/IKTrajectoryHelper.h
		../../src/SharedMemory/PhysicsClient.h
		../../src/SharedMemory/PhysicsServer.cpp
		../../src/SharedMemory/PhysicsServer.h
		../../src/SharedMemory/PhysicsServerSharedMemory.cpp
		../../src/SharedMemory/PhysicsServerSharedMemory.h
		../../src/SharedMemory/PhysicsDirect.cpp
		../../src/SharedMemory/PhysicsDirect.h
		../../src/SharedMemory/PhysicsDirectC_API.cpp
		../../src/SharedMemory/PhysicsDirectC_API.h
		../../src/SharedMemory/PhysicsServerCommandProcessor.cpp
		../../src/SharedMemory/PhysicsServerCommandProcessor.h
		../../src/SharedMemory/b3PluginManager.cpp
		../../src/SharedMemory/PhysicsClientSharedMemory.cpp
		../../src/SharedMemory/PhysicsClientSharedMemory.h
		../../src/SharedMemory/PhysicsClientSharedMemory_C_API.cpp
		../../src/SharedMemory/PhysicsClientSharedMemory_C_API.h
		../../src/SharedMemory/PhysicsClientC_API.cpp
		../../src/SharedMemory/PhysicsClientC_API.h
		../../src/SharedMemory/PhysicsLoopBack.cpp
		../../src/SharedMemory/PhysicsLoopBack.h
		../../src/SharedMemory/PhysicsLoopBackC_API.cpp
		../../src/SharedMemory/PhysicsLoopBackC_API.h
		../../src/SharedMemory/Win32SharedMemory.cpp
		../../src/SharedMemory/Win32SharedMemory.h
		../../src/SharedMemory/PosixSharedMemory.cpp
		../../src/SharedMemory/PosixSharedMemory.h
		../../examples/Utils/b3ResourcePath.cpp
		../../examples/Utils/b3ResourcePath.h
		../../examples/Utils/b3Clock.cpp
		../../examples/Utils/b3Clock.h
		../../examples/Utils/ChromeTraceUtil.cpp
		../../examples/Utils/ChromeTraceUtil.h
		../../examples/Utils/RobotLoggingUtil.cpp
		../../examples/Utils/RobotLoggingUtil.h
		../../src/SharedMemory/plugins/tinyRendererPlugin/tinyRendererPlugin.cpp
		../../src/SharedMemory/plugins/tinyRendererPlugin/TinyRendererVisualShapeConverter.cpp
		../../examples/OpenGLWindow/SimpleCamera.cpp
		../../examples/OpenGLWindow/SimpleCamera.h
		../../src/TinyRenderer/geometry.cpp
		../../src/TinyRenderer/model.cpp
		../../src/TinyRenderer/tgaimage.cpp
		../../src/TinyRenderer/our_gl.cpp
		../../src/TinyRenderer/TinyRenderer.cpp
		../../examples/ThirdPartyLibs/tinyxml2/tinyxml2.cpp
		../../examples/ThirdPartyLibs/Wavefront/tiny_obj_loader.cpp
		../../examples/ThirdPartyLibs/Wavefront/tiny_obj_loader.h
		../../examples/Importers/ImportColladaDemo/LoadMeshFromCollada.cpp
		../../examples/Importers/ImportObjDemo/LoadMeshFromObj.cpp
		../../examples/Importers/ImportObjDemo/Wavefront2GLInstanceGraphicsShape.cpp
		../../examples/Importers/ImportMJCFDemo/BulletMJCFImporter.cpp
		../../examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp
		../../examples/Importers/ImportURDFDemo/MyMultiBodyCreator.cpp
		../../examples/Importers/ImportURDFDemo/URDF2Bullet.cpp
		../../examples/Importers/ImportURDFDemo/UrdfParser.cpp
		../../examples/Importers/ImportURDFDemo/urdfStringSplit.cpp
		../../examples/Importers/ImportMeshUtility/b3ImportMeshUtility.cpp
		../../examples/ThirdPartyLibs/stb_image/stb_image.cpp
                        
	)

ADD_TEST(Test_PhysicsClientServer_PASS Test_PhysicsClientServer)

IF (INTERNAL_ADD_POSTFIX_EXECUTABLE_NAMES)
			SET_TARGET_PROPERTIES(Test_PhysicsClientServer PROPERTIES  DEBUG_POSTFIX "_Debug")
			SET_TARGET_PROPERTIES(Test_PhysicsClientServer  PROPERTIES  MINSIZEREL_POSTFIX "_MinsizeRel")
			SET_TARGET_PROPERTIES(Test_PhysicsClientServer  PROPERTIES  RELWITHDEBINFO_POSTFIX "_RelWithDebugInfo")
ENDIF(INTERNAL_ADD_POSTFIX_EXECUTABLE_NAMES)
