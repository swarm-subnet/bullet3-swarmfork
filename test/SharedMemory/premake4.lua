project ("Test_SharedMemoryPhysicsClient")

		language "C++"
		kind "ConsoleApp"

		includedirs {"../../src", "../../examples"}
		links {
			"BulletFileLoader",
			"Bullet3Common", 
			"LinearMath"
		}
		defines {"PHYSICS_SHARED_MEMORY"}
			
		files {
			"test.c",
			"../../src/SharedMemory/PhysicsClient.cpp",
			"../../src/SharedMemory/PhysicsClient.h",
			"../../src/SharedMemory/PhysicsClientSharedMemory.cpp",
			"../../src/SharedMemory/PhysicsClientSharedMemory.h",
			"../../src/SharedMemory/PhysicsClientSharedMemory_C_API.cpp",
			"../../src/SharedMemory/PhysicsClientSharedMemory_C_API.h",
			"../../src/SharedMemory/PhysicsClientC_API.cpp",
			"../../src/SharedMemory/PhysicsClientC_API.h",
			"../../src/SharedMemory/Win32SharedMemory.cpp",
			"../../src/SharedMemory/Win32SharedMemory.h",
			"../../src/SharedMemory/PosixSharedMemory.cpp",
			"../../src/SharedMemory/PosixSharedMemory.h",
			"../../examples/Utils/b3Clock.cpp",
			"../../examples/Utils/b3Clock.h",
			"../../examples/Utils/b3ResourcePath.cpp",
			"../../examples/Utils/b3ResourcePath.h",
		}

project ("Test_PhysicsClientUDP")

                language "C++"
                kind "ConsoleApp"

                includedirs {
                "../../src", 
                "../../examples",
                "../../examples/ThirdPartyLibs/enet/include"
                }
                links {
												"enet",
                        "BulletFileLoader",
                        "Bullet3Common",
                        "LinearMath"
                }
		if os.is("Windows") then
                	defines { "WIN32" }
        	        links {"Ws2_32","Winmm"}
	        end
		if os.is("Linux") then
			links {"pthread"}
		end

                defines {"PHYSICS_UDP"}

                files {
									"test.c",
									"../../src/SharedMemory/PhysicsClient.cpp",
									"../../src/SharedMemory/PhysicsClient.h",
									"../../src/SharedMemory/PhysicsClientSharedMemory.cpp",
									"../../src/SharedMemory/PhysicsClientSharedMemory.h",
									"../../src/SharedMemory/PhysicsClientSharedMemory_C_API.cpp",
									"../../src/SharedMemory/PhysicsClientSharedMemory_C_API.h",
									"../../src/SharedMemory/PhysicsClientUDP.cpp",
									"../../src/SharedMemory/PhysicsClientUDP.h",
									"../../src/SharedMemory/PhysicsClientUDP_C_API.cpp",
									"../../src/SharedMemory/PhysicsClientUDP_C_API.h",
									"../../src/SharedMemory/PhysicsClientSharedMemory_C_API.h",	
									"../../src/SharedMemory/PhysicsClientC_API.cpp",
									"../../src/SharedMemory/PhysicsClientC_API.h",
									"../../src/SharedMemory/Win32SharedMemory.cpp",
									"../../src/SharedMemory/Win32SharedMemory.h",
									"../../src/SharedMemory/PosixSharedMemory.cpp",
									"../../src/SharedMemory/PosixSharedMemory.h",
									"../../examples/Utils/b3ResourcePath.cpp",
									"../../examples/Utils/b3ResourcePath.h",
									"../../src/SharedMemory/PhysicsDirect.cpp",
									"../../examples/Utils/b3Clock.cpp",
									"../../examples/MultiThreading/b3PosixThreadSupport.cpp",
									"../../examples/MultiThreading/b3Win32ThreadSupport.cpp",
									"../../examples/MultiThreading/b3ThreadSupportInterface.cpp",
            }


project ("Test_PhysicsClientTCP")

                language "C++"
                kind "ConsoleApp"

                includedirs {
                "../../src", 
                "../../examples",
                "../../examples/ThirdPartyLibs/clsocket/src"
                }
                links {
												"clsocket",
                        "BulletFileLoader",
                        "Bullet3Common",
                        "LinearMath"
                }
		if os.is("Windows") then
                	defines { "WIN32" }
        	        links {"Ws2_32","Winmm"}
	        end

		if os.is("Windows") then
                	defines { "WIN32","_WINSOCK_DEPRECATED_NO_WARNINGS" }
                	end
                if os.is("Linux") then
                 defines {"_LINUX"}
                end
                if os.is("MacOSX") then
                 defines {"_DARWIN"}
                end

                defines {"PHYSICS_TCP"}

                files {
									"test.c",
									"../../src/SharedMemory/PhysicsClient.cpp",
									"../../src/SharedMemory/PhysicsClient.h",
									"../../src/SharedMemory/PhysicsClientSharedMemory.cpp",
									"../../src/SharedMemory/PhysicsClientSharedMemory.h",
									"../../src/SharedMemory/PhysicsClientSharedMemory_C_API.cpp",
									"../../src/SharedMemory/PhysicsClientSharedMemory_C_API.h",
									"../../src/SharedMemory/PhysicsClientTCP.cpp",
									"../../src/SharedMemory/PhysicsClientTCP.h",
									"../../src/SharedMemory/PhysicsClientTCP_C_API.cpp",
									"../../src/SharedMemory/PhysicsClientTCP_C_API.h",
									"../../src/SharedMemory/PhysicsClientSharedMemory_C_API.h",	
									"../../src/SharedMemory/PhysicsClientC_API.cpp",
									"../../src/SharedMemory/PhysicsClientC_API.h",
									"../../src/SharedMemory/Win32SharedMemory.cpp",
									"../../src/SharedMemory/Win32SharedMemory.h",
									"../../src/SharedMemory/PosixSharedMemory.cpp",
									"../../src/SharedMemory/PosixSharedMemory.h",
									"../../examples/Utils/b3ResourcePath.cpp",
									"../../examples/Utils/b3ResourcePath.h",
									"../../src/SharedMemory/PhysicsDirect.cpp",
									"../../examples/Utils/b3Clock.cpp",
            }

		
project ("Test_PhysicsServerLoopBack")

		language "C++"
		kind "ConsoleApp"

		includedirs {"../../src", "../../examples",
		"../../examples/ThirdPartyLibs"}
		defines {"PHYSICS_LOOP_BACK", "SKIP_SOFT_BODY_MULTI_BODY_DYNAMICS_WORLD"}
		links {
			"BulletInverseDynamicsUtils",
			"BulletInverseDynamics",
			"BulletFileLoader",
			"BulletWorldImporter",
			"Bullet3Common",
			"BulletDynamics", 
			"BulletCollision", 
			"BussIK",
			"LinearMath"
		}
        if os.is("Linux") then
            links{"dl"}
        end
			
		files {
			"test.c",
			"../../src/SharedMemory/b3RobotSimulatorClientAPI_NoDirect.cpp",
			"../../src/SharedMemory/b3RobotSimulatorClientAPI_NoDirect.h",
			"../../src/SharedMemory/plugins/collisionFilterPlugin/collisionFilterPlugin.cpp",
			"../../src/SharedMemory/plugins/pdControlPlugin/pdControlPlugin.cpp",
			"../../src/SharedMemory/plugins/pdControlPlugin/pdControlPlugin.h",
			"../../src/SharedMemory/IKTrajectoryHelper.cpp",
			"../../src/SharedMemory/IKTrajectoryHelper.h",
			"../../src/SharedMemory/PhysicsClient.cpp",
			"../../src/SharedMemory/PhysicsClient.h",
			"../../src/SharedMemory/PhysicsServer.cpp",
			"../../src/SharedMemory/PhysicsServer.h",
			"../../src/SharedMemory/PhysicsServerSharedMemory.cpp",
			"../../src/SharedMemory/PhysicsServerSharedMemory.h",
			"../../src/SharedMemory/PhysicsServerCommandProcessor.cpp",
			"../../src/SharedMemory/PhysicsServerCommandProcessor.h",
			"../../src/SharedMemory/b3PluginManager.cpp",
			"../../src/SharedMemory/PhysicsDirect.cpp",
			"../../src/SharedMemory/PhysicsLoopBack.cpp",
			"../../src/SharedMemory/PhysicsLoopBack.h",
			"../../src/SharedMemory/PhysicsLoopBackC_API.cpp",
			"../../src/SharedMemory/PhysicsLoopBackC_API.h",
			"../../src/SharedMemory/PhysicsClientSharedMemory_C_API.cpp",
			"../../src/SharedMemory/PhysicsClientSharedMemory_C_API.h",
			"../../src/SharedMemory/PhysicsClientSharedMemory.cpp",
			"../../src/SharedMemory/PhysicsClientSharedMemory.h",
			"../../src/SharedMemory/PhysicsClientC_API.cpp",
			"../../src/SharedMemory/PhysicsClientC_API.h",
			"../../src/SharedMemory/Win32SharedMemory.cpp",
			"../../src/SharedMemory/Win32SharedMemory.h",
			"../../src/SharedMemory/PosixSharedMemory.cpp",
			"../../src/SharedMemory/PosixSharedMemory.h",
			"../../src/SharedMemory/plugins/tinyRendererPlugin/tinyRendererPlugin.cpp",
			"../../src/SharedMemory/plugins/tinyRendererPlugin/TinyRendererVisualShapeConverter.cpp",
			"../../examples/OpenGLWindow/SimpleCamera.cpp",
			"../../examples/OpenGLWindow/SimpleCamera.h",
			"../../src/TinyRenderer/geometry.cpp",
			"../../src/TinyRenderer/model.cpp",
			"../../src/TinyRenderer/tgaimage.cpp",
			"../../src/TinyRenderer/our_gl.cpp",
			"../../src/TinyRenderer/TinyRenderer.cpp",
			"../../examples/Utils/b3ResourcePath.cpp",
			"../../examples/Utils/b3ResourcePath.h",
			"../../examples/Utils/RobotLoggingUtil.cpp",
			"../../examples/Utils/RobotLoggingUtil.h",
			"../../examples/Utils/b3Clock.cpp",
			"../../examples/Utils/b3Clock.h",
			"../../examples/Utils/ChromeTraceUtil.cpp",
			"../../examples/Utils/ChromeTraceUtil.h",
			"../../examples/ThirdPartyLibs/tinyxml2/tinyxml2.cpp",
			"../../examples/ThirdPartyLibs/Wavefront/tiny_obj_loader.cpp",
			"../../examples/ThirdPartyLibs/Wavefront/tiny_obj_loader.h",
			"../../examples/Importers/ImportColladaDemo/LoadMeshFromCollada.cpp",
			"../../examples/Importers/ImportObjDemo/LoadMeshFromObj.cpp",
			"../../examples/Importers/ImportObjDemo/Wavefront2GLInstanceGraphicsShape.cpp",
			"../../examples/Importers/ImportMJCFDemo/BulletMJCFImporter.cpp",
			"../../examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp",
			"../../examples/Importers/ImportURDFDemo/MyMultiBodyCreator.cpp",
			"../../examples/Importers/ImportURDFDemo/URDF2Bullet.cpp",
			"../../examples/Importers/ImportURDFDemo/UrdfParser.cpp",
			"../../examples/Importers/ImportURDFDemo/urdfStringSplit.cpp",
			"../../examples/Importers/ImportMeshUtility/b3ImportMeshUtility.cpp",
			"../../examples/ThirdPartyLibs/stb_image/stb_image.cpp",
	}
	
if (_OPTIONS["enable_static_plugins"]) then
		files {"../../src/SharedMemory/plugins/vrSyncPlugin/vrSyncPlugin.cpp"}
end
		
		project ("Test_PhysicsServerDirect")

		language "C++"
		kind "ConsoleApp"

		includedirs {"../../src", "../../examples",
		"../../examples/ThirdPartyLibs"}
		defines {"PHYSICS_SERVER_DIRECT","SKIP_SOFT_BODY_MULTI_BODY_DYNAMICS_WORLD"}
		links {
			"BulletInverseDynamicsUtils",
			"BulletInverseDynamics",
			"BulletFileLoader",
			"BulletWorldImporter",
			"Bullet3Common",
			"BulletDynamics", 
			"BulletCollision",
			"BussIK",
			"LinearMath"
		}
        if os.is("Linux") then
            links{"dl"}
        end
			
		files {
			"test.c",
			"../../src/SharedMemory/b3RobotSimulatorClientAPI_NoDirect.cpp",
			"../../src/SharedMemory/b3RobotSimulatorClientAPI_NoDirect.h",
			"../../src/SharedMemory/plugins/collisionFilterPlugin/collisionFilterPlugin.cpp",
			"../../src/SharedMemory/plugins/pdControlPlugin/pdControlPlugin.cpp",
			"../../src/SharedMemory/plugins/pdControlPlugin/pdControlPlugin.h",
			"../../src/SharedMemory/IKTrajectoryHelper.cpp",
			"../../src/SharedMemory/IKTrajectoryHelper.h",
			"../../src/SharedMemory/PhysicsClient.cpp",
			"../../src/SharedMemory/PhysicsClient.h",
			"../../src/SharedMemory/PhysicsServer.cpp",
			"../../src/SharedMemory/PhysicsServer.h",
			"../../src/SharedMemory/PhysicsServerSharedMemory.cpp",
			"../../src/SharedMemory/PhysicsServerSharedMemory.h",
			"../../src/SharedMemory/PhysicsDirect.cpp",
			"../../src/SharedMemory/PhysicsDirect.h",
			"../../src/SharedMemory/PhysicsDirectC_API.cpp",
			"../../src/SharedMemory/PhysicsDirectC_API.h",
			"../../src/SharedMemory/PhysicsServerCommandProcessor.cpp",
			"../../src/SharedMemory/PhysicsServerCommandProcessor.h",
			"../../src/SharedMemory/b3PluginManager.cpp",
			"../../src/SharedMemory/PhysicsClientSharedMemory.cpp",
			"../../src/SharedMemory/PhysicsClientSharedMemory.h",
			"../../src/SharedMemory/PhysicsClientC_API.cpp",
			"../../src/SharedMemory/PhysicsClientC_API.h",
			"../../src/SharedMemory/Win32SharedMemory.cpp",
			"../../src/SharedMemory/Win32SharedMemory.h",
			"../../src/SharedMemory/PosixSharedMemory.cpp",
			"../../src/SharedMemory/PosixSharedMemory.h",
			"../../src/SharedMemory/plugins/tinyRendererPlugin/tinyRendererPlugin.cpp",
			"../../src/SharedMemory/plugins/tinyRendererPlugin/TinyRendererVisualShapeConverter.cpp",
			"../../src/TinyRenderer/geometry.cpp",
			"../../src/TinyRenderer/model.cpp",
			"../../src/TinyRenderer/tgaimage.cpp",
			"../../src/TinyRenderer/our_gl.cpp",
			"../../src/TinyRenderer/TinyRenderer.cpp",
			"../../examples/OpenGLWindow/SimpleCamera.cpp",
			"../../examples/OpenGLWindow/SimpleCamera.h",
			"../../examples/Utils/b3ResourcePath.cpp",
			"../../examples/Utils/b3ResourcePath.h",
			"../../examples/Utils/RobotLoggingUtil.cpp",
			"../../examples/Utils/RobotLoggingUtil.h",
			"../../examples/Utils/b3Clock.cpp",
			"../../examples/Utils/ChromeTraceUtil.cpp",
			"../../examples/Utils/ChromeTraceUtil.h",			
			"../../examples/ThirdPartyLibs/tinyxml2/tinyxml2.cpp",
			"../../examples/ThirdPartyLibs/Wavefront/tiny_obj_loader.cpp",
			"../../examples/ThirdPartyLibs/Wavefront/tiny_obj_loader.h",
			"../../examples/Importers/ImportColladaDemo/LoadMeshFromCollada.cpp",
			"../../examples/Importers/ImportObjDemo/LoadMeshFromObj.cpp",
			"../../examples/Importers/ImportObjDemo/Wavefront2GLInstanceGraphicsShape.cpp",
			"../../examples/Importers/ImportMJCFDemo/BulletMJCFImporter.cpp",
			"../../examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp",
			"../../examples/Importers/ImportURDFDemo/MyMultiBodyCreator.cpp",
			"../../examples/Importers/ImportURDFDemo/URDF2Bullet.cpp",
			"../../examples/Importers/ImportURDFDemo/UrdfParser.cpp",
			"../../examples/Importers/ImportURDFDemo/urdfStringSplit.cpp",
			"../../examples/Importers/ImportMeshUtility/b3ImportMeshUtility.cpp",
                        "../../examples/ThirdPartyLibs/stb_image/stb_image.cpp",     	
		}
if (_OPTIONS["enable_static_plugins"]) then
		files {"../../src/SharedMemory/plugins/vrSyncPlugin/vrSyncPlugin.cpp"}
end

project ("Test_PhysicsServerInProcessExampleBrowser")

		language "C++"
		kind "ConsoleApp"

		includedirs {"../../src", "../../examples",
		"../../examples/ThirdPartyLibs"}
		defines {"PHYSICS_IN_PROCESS_EXAMPLE_BROWSER", "SKIP_SOFT_BODY_MULTI_BODY_DYNAMICS_WORLD"}
--		links {
--			"BulletExampleBrowserLib",
--			"BulletFileLoader",
--			"BulletWorldImporter",
--			"Bullet3Common",
--			"BulletDynamics", 
--			"BulletCollision", 
--			"LinearMath"
--		}
	hasCL = findOpenCL("clew")

	links{"BulletExampleBrowserLib","gwen", "OpenGL_Window","BulletFileLoader","BulletWorldImporter","BulletSoftBody", "BulletInverseDynamicsUtils", "BulletInverseDynamics", "BulletDynamics","BulletCollision","LinearMath","BussIK","Bullet3Common"}
	initOpenGL()
	initGlew()

  	includedirs {
                ".",
                "../../src",
                "../ThirdPartyLibs",
                }


	if os.is("MacOSX") then
		links{"Cocoa.framework"}
	end

		if (hasCL) then
			links {
				"Bullet3OpenCL_clew",
				"Bullet3Dynamics",
				"Bullet3Collision",
				"Bullet3Geometry",
				"Bullet3Common",
			}
		end

if not _OPTIONS["no-clsocket"] then

                includedirs {"../../examples/ThirdPartyLibs/clsocket/src"}

                 if os.is("Windows") then
                        defines { "WIN32" }
                        links {"Ws2_32","Winmm"}
                 end
                if os.is("Linux") then
                        defines {"_LINUX"}
                end
                if os.is("MacOSX") then
                        defines {"_DARWIN"}
                end

                links {"clsocket"}

                files {
                        "../../src/SharedMemory/PhysicsClientTCP.cpp",
                        "../../src/SharedMemory/PhysicsClientTCP.h",
                        "../../src/SharedMemory/PhysicsClientTCP_C_API.cpp",
                        "../../src/SharedMemory/PhysicsClientTCP_C_API.h",
                }
                defines {"BT_ENABLE_CLSOCKET"}
        end


		files {
			"test.c",
			"../../src/SharedMemory/b3RobotSimulatorClientAPI_NoDirect.cpp",
			"../../src/SharedMemory/b3RobotSimulatorClientAPI_NoDirect.h",
			"../../src/SharedMemory/plugins/collisionFilterPlugin/collisionFilterPlugin.cpp",
			"../../src/SharedMemory/plugins/pdControlPlugin/pdControlPlugin.cpp",
			"../../src/SharedMemory/plugins/pdControlPlugin/pdControlPlugin.h",
			"../../src/SharedMemory/IKTrajectoryHelper.cpp",
			"../../src/SharedMemory/IKTrajectoryHelper.h",
			"../../src/SharedMemory/RemoteGUIHelper.cpp",
			"../../src/SharedMemory/RemoteGUIHelperTCP.cpp",
			"../../src/SharedMemory/GraphicsServerExample.cpp",
			"../../examples/ExampleBrowser/InProcessExampleBrowser.cpp",
			"../../src/SharedMemory/InProcessMemory.cpp",
			"../../src/SharedMemory/PhysicsClient.cpp",
			"../../src/SharedMemory/PhysicsClient.h",
			"../../src/SharedMemory/PhysicsServer.cpp",
			"../../src/SharedMemory/PhysicsServer.h",
			"../../src/SharedMemory/PhysicsServerExample.cpp",
			"../../src/SharedMemory/PhysicsServerExampleBullet2.cpp",
			"../../src/SharedMemory/SharedMemoryInProcessPhysicsC_API.cpp",
			"../../src/SharedMemory/PhysicsServerSharedMemory.cpp",
			"../../src/SharedMemory/PhysicsServerSharedMemory.h",
			"../../src/SharedMemory/PhysicsDirect.cpp",
			"../../src/SharedMemory/PhysicsDirect.h",
			"../../src/SharedMemory/PhysicsDirectC_API.cpp",
			"../../src/SharedMemory/PhysicsDirectC_API.h",
			"../../src/SharedMemory/PhysicsServerCommandProcessor.cpp",
			"../../src/SharedMemory/PhysicsServerCommandProcessor.h",
			"../../src/SharedMemory/b3PluginManager.cpp",
			"../../src/SharedMemory/PhysicsClientSharedMemory.cpp",
			"../../src/SharedMemory/PhysicsClientSharedMemory.h",
			"../../src/SharedMemory/PhysicsClientC_API.cpp",
			"../../src/SharedMemory/PhysicsClientC_API.h",
			"../../src/SharedMemory/Win32SharedMemory.cpp",
			"../../src/SharedMemory/Win32SharedMemory.h",
			"../../src/SharedMemory/PosixSharedMemory.cpp",
			"../../src/SharedMemory/PosixSharedMemory.h",
			"../../src/SharedMemory/plugins/tinyRendererPlugin/tinyRendererPlugin.cpp",
			"../../src/SharedMemory/plugins/tinyRendererPlugin/TinyRendererVisualShapeConverter.cpp",
			"../../src/TinyRenderer/geometry.cpp",
			"../../src/TinyRenderer/model.cpp",
			"../../src/TinyRenderer/tgaimage.cpp",
			"../../src/TinyRenderer/our_gl.cpp",
			"../../src/TinyRenderer/TinyRenderer.cpp",
			"../../examples/Utils/RobotLoggingUtil.cpp",
			"../../examples/Utils/RobotLoggingUtil.h",
			"../../examples/ThirdPartyLibs/tinyxml2/tinyxml2.cpp",
			"../../examples/ThirdPartyLibs/Wavefront/tiny_obj_loader.cpp",
			"../../examples/ThirdPartyLibs/Wavefront/tiny_obj_loader.h",
			"../../examples/Importers/ImportColladaDemo/LoadMeshFromCollada.cpp",
			"../../examples/Importers/ImportObjDemo/LoadMeshFromObj.cpp",
			"../../examples/Importers/ImportObjDemo/Wavefront2GLInstanceGraphicsShape.cpp",
			"../../examples/Importers/ImportMJCFDemo/BulletMJCFImporter.cpp",
			"../../examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp",
			"../../examples/Importers/ImportURDFDemo/MyMultiBodyCreator.cpp",
			"../../examples/Importers/ImportURDFDemo/URDF2Bullet.cpp",
			"../../examples/Importers/ImportURDFDemo/UrdfParser.cpp",
			"../../examples/Importers/ImportURDFDemo/urdfStringSplit.cpp",
			"../../examples/MultiThreading/b3PosixThreadSupport.cpp",
			"../../examples/MultiThreading/b3Win32ThreadSupport.cpp",
			"../../examples/MultiThreading/b3ThreadSupportInterface.cpp",
			"../../examples/Importers/ImportMeshUtility/b3ImportMeshUtility.cpp",
			"../../examples/ThirdPartyLibs/stb_image/stb_image.cpp",
	}
if (_OPTIONS["enable_static_vr_plugin"]) then
		files {"../../src/SharedMemory/plugins/vrSyncPlugin/vrSyncPlugin.cpp"}
end

	if os.is("Linux") then
       		initX11()
	end

	
