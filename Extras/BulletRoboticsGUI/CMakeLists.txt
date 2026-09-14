
INCLUDE_DIRECTORIES(
		${BULLET_PHYSICS_SOURCE_DIR}/src
		${BULLET_PHYSICS_SOURCE_DIR}/examples
		${BULLET_PHYSICS_SOURCE_DIR}/src/SharedMemory
		${BULLET_PHYSICS_SOURCE_DIR}/examples/ThirdPartyLibs
		${BULLET_PHYSICS_SOURCE_DIR}/examples/ThirdPartyLibs/enet/include
		${BULLET_PHYSICS_SOURCE_DIR}/examples/ThirdPartyLibs/clsocket/src
)

SET(BulletRoboticsGUI_INCLUDES
  ../../examples/CommonInterfaces/Common2dCanvasInterface.h
  ../../examples/CommonInterfaces/CommonCallbacks.h
  ../../examples/CommonInterfaces/CommonCameraInterface.h
  ../../examples/CommonInterfaces/CommonExampleInterface.h
  ../../examples/CommonInterfaces/CommonFileIOInterface.h
  ../../examples/CommonInterfaces/CommonGraphicsAppInterface.h
  ../../examples/CommonInterfaces/CommonGUIHelperInterface.h
  ../../examples/CommonInterfaces/CommonMultiBodyBase.h
  ../../examples/CommonInterfaces/CommonParameterInterface.h
  ../../examples/CommonInterfaces/CommonRenderInterface.h
  ../../examples/CommonInterfaces/CommonRigidBodyBase.h
  ../../examples/CommonInterfaces/CommonWindowInterface.h

  ../../examples/OpenGLWindow/SimpleCamera.h

  ../../src/TinyRenderer/geometry.h
  ../../src/TinyRenderer/model.h
  ../../src/TinyRenderer/tgaimage.h
  ../../src/TinyRenderer/our_gl.h
  ../../src/TinyRenderer/TinyRenderer.h

  ../../src/SharedMemory/plugins/collisionFilterPlugin/collisionFilterPlugin.h
  ../../src/SharedMemory/plugins/pdControlPlugin/pdControlPlugin.h
  ../../src/SharedMemory/b3RobotSimulatorClientAPI_NoGUI.h
  ../../src/SharedMemory/b3RobotSimulatorClientAPI_NoDirect.h
  ../../src/SharedMemory/IKTrajectoryHelper.h
  ../../src/SharedMemory/plugins/tinyRendererPlugin/tinyRendererPlugin.h
  ../../src/SharedMemory/plugins/tinyRendererPlugin/TinyRendererVisualShapeConverter.h
  ../../src/SharedMemory/InProcessMemory.h
  ../../src/SharedMemory/PhysicsServer.h
  ../../src/SharedMemory/PhysicsClient.h
  ../../src/SharedMemory/PhysicsServerSharedMemory.h
  ../../src/SharedMemory/PhysicsDirect.h
  ../../src/SharedMemory/PhysicsDirectC_API.h
  ../../src/SharedMemory/PhysicsServerCommandProcessor.h
  ../../src/SharedMemory/b3PluginManager.h
  ../../src/SharedMemory/PhysicsClientSharedMemory.h
  ../../src/SharedMemory/PhysicsClientSharedMemory_C_API.h
  ../../src/SharedMemory/PhysicsClientC_API.h
  ../../src/SharedMemory/SharedMemoryPublic.h
  ../../src/SharedMemory/Win32SharedMemory.h
  ../../src/SharedMemory/PosixSharedMemory.h

  ../../examples/Utils/b3ResourcePath.h
  ../../examples/Utils/RobotLoggingUtil.h
  ../../examples/Utils/b3Clock.h
  ../../examples/Utils/b3ResourcePath.h
  ../../examples/Utils/ChromeTraceUtil.h
  ../../examples/Utils/b3ERPCFMHelper.hpp
  ../../examples/Utils/b3ReferenceFrameHelper.hpp  

  ../../examples/ThirdPartyLibs/tinyxml2/tinyxml2.h
  ../../examples/ThirdPartyLibs/Wavefront/tiny_obj_loader.h
  ../../examples/ThirdPartyLibs/stb_image/stb_image.h
  ../../examples/ThirdPartyLibs/BussIK/Jacobian.h
  ../../examples/ThirdPartyLibs/BussIK/LinearR2.h
  ../../examples/ThirdPartyLibs/BussIK/LinearR3.h
  ../../examples/ThirdPartyLibs/BussIK/LinearR4.h
  ../../examples/ThirdPartyLibs/BussIK/MatrixRmn.h
  ../../examples/ThirdPartyLibs/BussIK/Node.h
  ../../examples/ThirdPartyLibs/BussIK/Tree.h
  ../../examples/ThirdPartyLibs/BussIK/VectorRn.h

  ../../examples/Importers/ImportColladaDemo/LoadMeshFromCollada.h
  ../../examples/Importers/ImportObjDemo/LoadMeshFromObj.h
  ../../examples/Importers/ImportObjDemo/Wavefront2GLInstanceGraphicsShape.h
  ../../examples/Importers/ImportMJCFDemo/BulletMJCFImporter.h
  ../../examples/Importers/ImportURDFDemo/BulletUrdfImporter.h
  ../../examples/Importers/ImportURDFDemo/MyMultiBodyCreator.h
  ../../examples/Importers/ImportURDFDemo/URDF2Bullet.h
  ../../examples/Importers/ImportURDFDemo/UrdfParser.h
  ../../examples/Importers/ImportURDFDemo/urdfStringSplit.h
  ../../examples/Importers/ImportURDFDemo/URDFImporterInterface.h
  ../../examples/Importers/ImportURDFDemo/URDFJointTypes.h
  ../../examples/Importers/ImportURDFDemo/SDFAudioTypes.h
  ../../examples/Importers/ImportURDFDemo/UrdfRenderingInterface.h
  ../../examples/Importers/ImportURDFDemo/MultiBodyCreationInterface.h
  ../../examples/Importers/ImportMeshUtility/b3ImportMeshUtility.h
  
  ../../examples/MultiThreading/b3PosixThreadSupport.h
  ../../examples/MultiThreading/b3Win32ThreadSupport.h
  ../../examples/MultiThreading/b3ThreadSupportInterface.h
)

SET(BulletRoboticsGUI_SRCS ${BulletRoboticsGUI_INCLUDES}
	../../examples/ExampleBrowser/InProcessExampleBrowser.cpp
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
)

IF(BUILD_CLSOCKET)
 ADD_DEFINITIONS(-DBT_ENABLE_CLSOCKET)
ENDIF(BUILD_CLSOCKET)

IF(WIN32)
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
		
	IF(NOT APPLE)
		FIND_PACKAGE(Threads)
		LINK_LIBRARIES( ${CMAKE_THREAD_LIBS_INIT} ${DL} )
	ENDIF(NOT APPLE)
ENDIF(WIN32)

IF(BUILD_ENET)
  set(BulletRoboticsGUI_ENET_INCLUDES 
    ../../src/SharedMemory/PhysicsClientUDP.h		
    ../../src/SharedMemory/PhysicsClientUDP_C_API.h
    ../../examples/ThirdPartyLibs/enet/include/enet/win32.h
		../../examples/ThirdPartyLibs/enet/include/enet/unix.h
		../../examples/ThirdPartyLibs/enet/include/enet/callbacks.h
		../../examples/ThirdPartyLibs/enet/include/enet/list.h
		../../examples/ThirdPartyLibs/enet/include/enet/protocol.h
  )

  set(BulletRoboticsGUI_INCLUDES
    ${BulletRoboticsGUI_INCLUDES}
    ${BulletRoboticsGUI_ENET_INCLUDES}
	)



ENDIF(BUILD_ENET)


ADD_DEFINITIONS(-DPHYSICS_SERVER_DIRECT)


ADD_LIBRARY(BulletRoboticsGUI ${BulletRoboticsGUI_SRCS})

SET_TARGET_PROPERTIES(BulletRoboticsGUI PROPERTIES VERSION ${BULLET_VERSION})
SET_TARGET_PROPERTIES(BulletRoboticsGUI PROPERTIES SOVERSION ${BULLET_VERSION})

IF (BUILD_SHARED_LIBS)
	TARGET_LINK_LIBRARIES(BulletRoboticsGUI  BulletExampleBrowserLib BulletRobotics  BulletInverseDynamicsUtils BulletWorldImporter BulletFileLoader BulletSoftBody BulletDynamics BulletCollision BulletInverseDynamics LinearMath Bullet3Common)
ENDIF (BUILD_SHARED_LIBS)

  

	INSTALL(TARGETS
		BulletRoboticsGUI
		LIBRARY DESTINATION lib${LIB_SUFFIX}
		ARCHIVE DESTINATION lib${LIB_SUFFIX}
	)
		
  IF (APPLE AND BUILD_SHARED_LIBS AND FRAMEWORK)
    SET_TARGET_PROPERTIES(BulletRobotics  PROPERTIES FRAMEWORK true)
    SET_TARGET_PROPERTIES(BulletRobotics PROPERTIES PUBLIC_HEADER "PhysicsClientC_API.h" )
  ENDIF (APPLE AND BUILD_SHARED_LIBS AND FRAMEWORK)	

IF(NOT MSVC)
    SET(PKGCONFIG_INSTALL_PREFIX "lib${LIB_SUFFIX}/pkgconfig/" CACHE STRING "Base directory for pkgconfig files")
    CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/bullet_robotics_gui.pc.cmake
                   ${CMAKE_CURRENT_BINARY_DIR}/bullet_robotics_gui.pc @ONLY)
  	INSTALL(
      FILES
      ${CMAKE_CURRENT_BINARY_DIR}/bullet_robotics_gui.pc
      DESTINATION
      ${PKGCONFIG_INSTALL_PREFIX}
    )
ENDIF(NOT MSVC)

