		

project ("BulletRobotics")
		language "C++"
		kind "StaticLib"
		
		includedirs {"../../src", "../../examples",
		"../../examples/ThirdPartyLibs"}
		defines {"PHYSICS_IN_PROCESS_EXAMPLE_BROWSER"}
	hasCL = findOpenCL("clew")

	links{"BulletExampleBrowserLib","gwen", "BulletFileLoader","BulletWorldImporter","OpenGL_Window","BulletSoftBody", "BulletInverseDynamicsUtils", "BulletInverseDynamics", "BulletDynamics","BulletCollision","LinearMath","BussIK", "Bullet3Common"}
	initOpenGL()
	initGlew()

  	includedirs {
                "../../src",
                "../../examples",
                "../../src/SharedMemory",
                "../ThirdPartyLibs",
                "../ThirdPartyLibs/enet/include",
                "../ThirdPartyLibs/clsocket/src",
                }

	if os.is("MacOSX") then
--		targetextension {"so"}
		links{"Cocoa.framework","Python"}
	end

	
if not _OPTIONS["no-enet"] then

		includedirs {"../../examples/ThirdPartyLibs/enet/include"}
	
		if os.is("Windows") then 
--			targetextension {"dylib"}
			defines { "WIN32" }
			links {"Ws2_32","Winmm"}
		end
		if os.is("Linux") then
		end
		if os.is("MacOSX") then
		end		
		
		links {"enet"}		

		files {
			"../../src/SharedMemory/PhysicsClientUDP.cpp",
			"../../src/SharedMemory/PhysicsClientUDP.h",
			"../../src/SharedMemory/PhysicsClientUDP_C_API.cpp",
			"../../src/SharedMemory/PhysicsClientUDP_C_API.h",
		}	
		defines {"BT_ENABLE_ENET"}
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
			"../../src/SharedMemory/RemoteGUIHelperTCP.cpp",
                        "../../src/SharedMemory/PhysicsClientTCP.cpp",
			"../../src/SharedMemory/GraphicsServerExample.cpp",
                        "../../src/SharedMemory/PhysicsClientTCP.h",
                        "../../src/SharedMemory/PhysicsClientTCP_C_API.cpp",
                        "../../src/SharedMemory/PhysicsClientTCP_C_API.h",
                }
                defines {"BT_ENABLE_CLSOCKET"}
        end


		files {
		"../../src/SharedMemory/plugins/collisionFilterPlugin/collisionFilterPlugin.cpp",
		"../../src/SharedMemory/plugins/pdControlPlugin/pdControlPlugin.cpp",
		"../../src/SharedMemory/plugins/pdControlPlugin/pdControlPlugin.h",
		"../../src/SharedMemory/b3RobotSimulatorClientAPI_NoGUI.cpp",
		"../../src/SharedMemory/b3RobotSimulatorClientAPI_NoGUI.h",
		"../../src/SharedMemory/b3RobotSimulatorClientAPI_NoDirect.cpp",
		"../../src/SharedMemory/b3RobotSimulatorClientAPI_NoDirect.h",
		"../../src/SharedMemory/IKTrajectoryHelper.cpp",
		"../../src/SharedMemory/IKTrajectoryHelper.h",
		"../../src/SharedMemory/plugins/tinyRendererPlugin/tinyRendererPlugin.cpp",
		"../../src/SharedMemory/plugins/tinyRendererPlugin/TinyRendererVisualShapeConverter.cpp",
		"../../src/SharedMemory/RemoteGUIHelper.cpp",
		"../../examples/OpenGLWindow/SimpleCamera.cpp",
		"../../examples/OpenGLWindow/SimpleCamera.h",
		"../../src/TinyRenderer/geometry.cpp",
		"../../src/TinyRenderer/model.cpp",
		"../../src/TinyRenderer/tgaimage.cpp",
		"../../src/TinyRenderer/our_gl.cpp",
		"../../src/TinyRenderer/TinyRenderer.cpp",
		"../../src/SharedMemory/InProcessMemory.cpp",
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
		"../../src/SharedMemory/b3PluginManager.h",
				
		"../../src/SharedMemory/PhysicsClientSharedMemory.cpp",
		"../../src/SharedMemory/PhysicsClientSharedMemory.h",
		"../../src/SharedMemory/PhysicsClientSharedMemory_C_API.cpp",
		"../../src/SharedMemory/PhysicsClientSharedMemory_C_API.h",
		"../../src/SharedMemory/PhysicsClientC_API.cpp",
	
		"../../src/SharedMemory/PhysicsClientC_API.h",
		"../../src/SharedMemory/SharedMemoryPublic.h",

		"../../src/SharedMemory/Win32SharedMemory.cpp",
		"../../src/SharedMemory/Win32SharedMemory.h",
		"../../src/SharedMemory/PosixSharedMemory.cpp",
		"../../src/SharedMemory/PosixSharedMemory.h",

		"../../examples/Utils/b3ResourcePath.cpp",
		"../../examples/Utils/b3ResourcePath.h",
		"../../examples/Utils/RobotLoggingUtil.cpp",
		"../../examples/Utils/RobotLoggingUtil.h",
		"../../examples/Utils/b3Clock.cpp",
		"../../examples/Utils/b3ResourcePath.cpp",
		"../../examples/Utils/b3ERPCFMHelper.hpp",
		"../../examples/Utils/b3ReferenceFrameHelper.hpp",
		"../../examples/Utils/ChromeTraceUtil.cpp",

		"../../examples/ThirdPartyLibs/tinyxml2/tinyxml2.cpp",

		"../../examples/ThirdPartyLibs/Wavefront/tiny_obj_loader.cpp",
		"../../examples/ThirdPartyLibs/Wavefront/tiny_obj_loader.h",

		"../../examples/ThirdPartyLibs/stb_image/stb_image.cpp",

		"../../examples/ThirdPartyLibs/BussIK/Jacobian.cpp",
		"../../examples/ThirdPartyLibs/BussIK/LinearR2.cpp",
		"../../examples/ThirdPartyLibs/BussIK/LinearR3.cpp",
		"../../examples/ThirdPartyLibs/BussIK/LinearR4.cpp",
		"../../examples/ThirdPartyLibs/BussIK/MatrixRmn.cpp",
		"../../examples/ThirdPartyLibs/BussIK/Misc.cpp",
		"../../examples/ThirdPartyLibs/BussIK/Node.cpp",
		"../../examples/ThirdPartyLibs/BussIK/Tree.cpp",
		"../../examples/ThirdPartyLibs/BussIK/VectorRn.cpp",

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

		"../../examples/MultiThreading/b3PosixThreadSupport.cpp",
		"../../examples/MultiThreading/b3Win32ThreadSupport.cpp",
		"../../examples/MultiThreading/b3ThreadSupportInterface.cpp",
			}
			
if (_OPTIONS["enable_static_vr_plugin"]) then
		files {"../../src/SharedMemory/plugins/vrSyncPlugin/vrSyncPlugin.cpp"}
end


	
