
INCLUDE_DIRECTORIES( 
		${BULLET_PHYSICS_SOURCE_DIR}/src
		${BULLET_PHYSICS_SOURCE_DIR}/examples
		${BULLET_PHYSICS_SOURCE_DIR}/examples/ThirdPartyLibs
		${BULLET_PHYSICS_SOURCE_DIR}/examples/ThirdPartyLibs/enet/include
		${BULLET_PHYSICS_SOURCE_DIR}/examples/ThirdPartyLibs/clsocket/src
  )


SET(RobotSimulator_SRCS
		TwoJointMain.cpp
		../../src/SharedMemory/plugins/collisionFilterPlugin/collisionFilterPlugin.cpp
		../../src/SharedMemory/plugins/pdControlPlugin/pdControlPlugin.cpp
		../../src/SharedMemory/plugins/pdControlPlugin/pdControlPlugin.h
		../../src/SharedMemory/b3RobotSimulatorClientAPI_NoDirect.cpp
		../../src/SharedMemory/b3RobotSimulatorClientAPI_NoDirect.h
		../../src/SharedMemory/IKTrajectoryHelper.cpp
		../../src/SharedMemory/IKTrajectoryHelper.h
		../../examples/ExampleBrowser/InProcessExampleBrowser.cpp
		../../src/SharedMemory/plugins/tinyRendererPlugin/tinyRendererPlugin.cpp
		../../src/SharedMemory/plugins/tinyRendererPlugin/TinyRendererVisualShapeConverter.cpp
		../../src/TinyRenderer/geometry.cpp
		../../src/TinyRenderer/model.cpp
		../../src/TinyRenderer/tgaimage.cpp
		../../src/TinyRenderer/our_gl.cpp
		../../src/TinyRenderer/TinyRenderer.cpp
		../../src/SharedMemory/InProcessMemory.cpp
		../../src/SharedMemory/PhysicsClient.cpp
		../../src/SharedMemory/PhysicsClient.h
		../../src/SharedMemory/PhysicsServer.cpp
		../../src/SharedMemory/PhysicsServer.h
		../../src/SharedMemory/GraphicsServerExample.cpp
		../../src/SharedMemory/GraphicsClientExample.cpp
		../../src/SharedMemory/RemoteGUIHelper.cpp
		../../src/SharedMemory/RemoteGUIHelperTCP.cpp
		../../src/SharedMemory/GraphicsServerExample.h
		../../src/SharedMemory/GraphicsClientExample.h
		../../src/SharedMemory/RemoteGUIHelper.h
		../../src/SharedMemory/GraphicsSharedMemoryCommands.h
		../../src/SharedMemory/GraphicsSharedMemoryPublic.h
		../../src/SharedMemory/PhysicsServerExample.cpp
		../../src/SharedMemory/PhysicsServerExampleBullet2.cpp
		../../src/SharedMemory/SharedMemoryInProcessPhysicsC_API.cpp
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
		../../src/SharedMemory/Win32SharedMemory.cpp
		../../src/SharedMemory/Win32SharedMemory.h
		../../src/SharedMemory/PosixSharedMemory.cpp
		../../src/SharedMemory/PosixSharedMemory.h
		../../examples/Utils/b3ResourcePath.cpp
		../../examples/Utils/b3ResourcePath.h
		../../examples/Utils/RobotLoggingUtil.cpp
		../../examples/Utils/RobotLoggingUtil.h
		../../examples/ThirdPartyLibs/tinyxml2/tinyxml2.cpp
		../../examples/ThirdPartyLibs/Wavefront/tiny_obj_loader.cpp
		../../examples/ThirdPartyLibs/Wavefront/tiny_obj_loader.h
		../../examples/ThirdPartyLibs/stb_image/stb_image.cpp
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
		../../examples/MultiThreading/b3PosixThreadSupport.cpp
		../../examples/MultiThreading/b3Win32ThreadSupport.cpp
		../../examples/MultiThreading/b3ThreadSupportInterface.cpp
		
)

IF(BUILD_CLSOCKET)
 ADD_DEFINITIONS(-DBT_ENABLE_CLSOCKET)
ENDIF(BUILD_CLSOCKET)

IF(WIN32)
	LINK_LIBRARIES(
		${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY}
	)
	IF(BUILD_ENET)
		ADD_DEFINITIONS(-DWIN32 -DBT_ENABLE_ENET)
	ENDIF(BUILD_ENET)
	IF(BUILD_CLSOCKET)
		ADD_DEFINITIONS(-DWIN32)
	ENDIF(BUILD_CLSOCKET)

ELSE(WIN32)
	IF(BUILD_ENET)
		ADD_DEFINITIONS(-DHAS_SOCKLEN_T -DBT_ENABLE_ENET)
	ENDIF(BUILD_ENET)

	IF(BUILD_CLSOCKET)	
		ADD_DEFINITIONS(${OSDEF})
	ENDIF(BUILD_CLSOCKET)
ENDIF(WIN32)


IF(BUILD_ENET)
	set(RobotSimulator_SRCS ${RobotSimulator_SRCS} 
		../../src/SharedMemory/PhysicsClientUDP.cpp
		../../src/SharedMemory/PhysicsClientUDP_C_API.cpp
		../../src/SharedMemory/PhysicsClientUDP.h		
		../../src/SharedMemory/PhysicsClientUDP_C_API.h
		../../examples/ThirdPartyLibs/enet/win32.c
		../../examples/ThirdPartyLibs/enet/unix.c
		../../examples/ThirdPartyLibs/enet/callbacks.c
		../../examples/ThirdPartyLibs/enet/compress.c
		../../examples/ThirdPartyLibs/enet/host.c
		../../examples/ThirdPartyLibs/enet/list.c
		../../examples/ThirdPartyLibs/enet/packet.c
		../../examples/ThirdPartyLibs/enet/peer.c
		../../examples/ThirdPartyLibs/enet/protocol.c
	)
ENDIF(BUILD_ENET)

IF(BUILD_CLSOCKET)
	set(RobotSimulator_SRCS ${RobotSimulator_SRCS}
		 ../../src/SharedMemory/PhysicsClientTCP.cpp
                 ../../src/SharedMemory/PhysicsClientTCP.h
                 ../../src/SharedMemory/PhysicsClientTCP_C_API.cpp
                 ../../src/SharedMemory/PhysicsClientTCP_C_API.h
		 ../../examples/ThirdPartyLibs/clsocket/src/SimpleSocket.cpp
		 ../../examples/ThirdPartyLibs/clsocket/src/ActiveSocket.cpp
		 ../../examples/ThirdPartyLibs/clsocket/src/PassiveSocket.cpp
	)
ENDIF()


#some code to support OpenGL and Glew cross platform
IF (WIN32)
	INCLUDE_DIRECTORIES(
			${BULLET_PHYSICS_SOURCE_DIR}/btgui/OpenGLWindow/GlewWindows
	)
	ADD_DEFINITIONS(-DGLEW_STATIC)
	LINK_LIBRARIES( ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY} )
ELSE(WIN32)
	IF(APPLE)
		find_library(COCOA NAMES Cocoa)
		MESSAGE(${COCOA})
		link_libraries(${COCOA} ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY})

	ELSE(APPLE)
		INCLUDE_DIRECTORIES(
				${BULLET_PHYSICS_SOURCE_DIR}/btgui/OpenGLWindow/GlewWindows
		)
		ADD_DEFINITIONS("-DGLEW_INIT_OPENGL11_FUNCTIONS=1")
		ADD_DEFINITIONS("-DGLEW_STATIC")
		ADD_DEFINITIONS("-DGLEW_DYNAMIC_LOAD_ALL_GLX_FUNCTIONS=1")

		LINK_LIBRARIES(  pthread ${DL} )
	ENDIF(APPLE)
ENDIF(WIN32)


ADD_EXECUTABLE(App_TwoJoint  ${RobotSimulator_SRCS})

SET_TARGET_PROPERTIES(App_TwoJoint PROPERTIES VERSION ${BULLET_VERSION})
SET_TARGET_PROPERTIES(App_TwoJoint PROPERTIES DEBUG_POSTFIX "_d")


IF(WIN32)
	IF(BUILD_ENET OR BUILD_CLSOCKET)
		TARGET_LINK_LIBRARIES(App_TwoJoint ws2_32 )
	ENDIF(BUILD_ENET OR BUILD_CLSOCKET)
ENDIF(WIN32)



TARGET_LINK_LIBRARIES(App_TwoJoint BulletExampleBrowserLib BulletFileLoader BulletWorldImporter BulletSoftBody BulletDynamics BulletCollision BulletInverseDynamicsUtils BulletInverseDynamics LinearMath OpenGLWindow gwen BussIK Bullet3Common)


