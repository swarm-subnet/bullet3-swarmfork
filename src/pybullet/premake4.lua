		

project ("pybullet")
		language "C++"
		kind "SharedLib"

		if _OPTIONS["enable_grpc"] then
				initGRPC()
				
				 files {
                  "../SharedMemory/PhysicsClientGRPC.cpp",
                  "../SharedMemory/PhysicsClientGRPC.h",
                  "../SharedMemory/PhysicsClientGRPC_C_API.cpp",
                  "../SharedMemory/PhysicsClientGRPC_C_API.h",
                }
		end
		
		includedirs {"..", "../../examples",
		"../../examples/ThirdPartyLibs",
		"../../Extras/VHACD/inc", "../../Extras/VHACD/public",
		}
		defines {"BT_ENABLE_VHACD"}
		
		defines {"PHYSICS_IN_PROCESS_EXAMPLE_BROWSER"}
		files 
		{
			"../../Extras/VHACD/test/src/main_vhacd.cpp",
			"../../Extras/VHACD/src/VHACD.cpp",
			"../../Extras/VHACD/src/vhacdICHull.cpp",
			"../../Extras/VHACD/src/vhacdManifoldMesh.cpp",
			"../../Extras/VHACD/src/vhacdMesh.cpp",
			"../../Extras/VHACD/src/vhacdVolume.cpp",
		}
		
		
	hasCL = findOpenCL("clew")

	links{ "BulletExampleBrowserLib","gwen", "BulletFileLoader","BulletWorldImporter","OpenGL_Window","BulletSoftBody", "BulletInverseDynamicsUtils", "BulletInverseDynamics", "BulletDynamics","BulletCollision","LinearMath","BussIK", "Bullet3Common"}
	initOpenGL()
	initGlew()

  	includedirs {
                ".",
                "..",
                "../../examples/ThirdPartyLibs",
                }

	if os.is("MacOSX") then
--		targetextension {"so"}
		links{"Cocoa.framework","Python"}
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
			"../SharedMemory/PhysicsClientUDP.cpp",
			"../SharedMemory/PhysicsClientUDP.h",
			"../SharedMemory/PhysicsClientUDP_C_API.cpp",
			"../SharedMemory/PhysicsClientUDP_C_API.h",
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
                        "../SharedMemory/PhysicsClientTCP.cpp",
                        "../SharedMemory/PhysicsClientTCP.h",
                        "../SharedMemory/PhysicsClientTCP_C_API.cpp",
                        "../SharedMemory/PhysicsClientTCP_C_API.h",
                }
                defines {"BT_ENABLE_CLSOCKET"}
        end


		files {
			"pybullet.c",
			"../SharedMemory/IKTrajectoryHelper.cpp",
			"../SharedMemory/IKTrajectoryHelper.h",
			"../../examples/ExampleBrowser/InProcessExampleBrowser.cpp",
			"../SharedMemory/plugins/tinyRendererPlugin/tinyRendererPlugin.cpp",
			"../SharedMemory/plugins/tinyRendererPlugin/tinyRendererPlugin.h",
			"../SharedMemory/plugins/tinyRendererPlugin/TinyRendererVisualShapeConverter.cpp",
			"../SharedMemory/plugins/tinyRendererPlugin/TinyRendererVisualShapeConverter.h",
			"../TinyRenderer/geometry.cpp",
			"../TinyRenderer/model.cpp",
			"../TinyRenderer/tgaimage.cpp",
			"../TinyRenderer/our_gl.cpp",
			"../TinyRenderer/TinyRenderer.cpp",
			"../SharedMemory/InProcessMemory.cpp",
			"../SharedMemory/b3RobotSimulatorClientAPI_NoDirect.cpp",
			"../SharedMemory/b3RobotSimulatorClientAPI_NoDirect.h",
			"../SharedMemory/PhysicsClient.cpp",
			"../SharedMemory/PhysicsClient.h",
			"../SharedMemory/PhysicsServer.cpp",
			"../SharedMemory/PhysicsServer.h",
			"../SharedMemory/PhysicsServerExample.cpp",
			"../SharedMemory/PhysicsServerExampleBullet2.cpp",
			"../SharedMemory/GraphicsClientExample.cpp",
      "../SharedMemory/GraphicsClientExample.h",
      "../SharedMemory/GraphicsServerExample.cpp",
    	"../SharedMemory/GraphicsServerExample.h",
 	   	"../SharedMemory/GraphicsSharedMemoryBlock.h",
   	 	"../SharedMemory/GraphicsSharedMemoryCommands.h",
    	"../SharedMemory/GraphicsSharedMemoryPublic.h",
    	"../SharedMemory/RemoteGUIHelper.cpp",
    	"../SharedMemory/RemoteGUIHelperTCP.cpp",
    	"../SharedMemory/RemoteGUIHelper.h",
			"../SharedMemory/SharedMemoryInProcessPhysicsC_API.cpp",
			"../SharedMemory/PhysicsServerSharedMemory.cpp",
			"../SharedMemory/PhysicsServerSharedMemory.h",
			"../SharedMemory/PhysicsDirect.cpp",
			"../SharedMemory/PhysicsDirect.h",
			"../SharedMemory/PhysicsDirectC_API.cpp",
			"../SharedMemory/PhysicsDirectC_API.h",
			"../SharedMemory/PhysicsServerCommandProcessor.cpp",
			"../SharedMemory/PhysicsServerCommandProcessor.h",
			"../SharedMemory/b3PluginManager.cpp",
			"../SharedMemory/b3PluginManager.h",
			"../SharedMemory/PhysicsClientSharedMemory.cpp",
			"../SharedMemory/PhysicsClientSharedMemory.h",
			"../SharedMemory/PhysicsClientSharedMemory_C_API.cpp",
			"../SharedMemory/PhysicsClientSharedMemory_C_API.h",
			"../SharedMemory/PhysicsClientC_API.cpp",
			"../SharedMemory/PhysicsClientC_API.h",
			"../SharedMemory/Win32SharedMemory.cpp",
			"../SharedMemory/Win32SharedMemory.h",
			"../SharedMemory/PosixSharedMemory.cpp",
			"../SharedMemory/PosixSharedMemory.h",
			"../SharedMemory/SharedMemoryCommands.h",
			"../SharedMemory/SharedMemoryPublic.h",
			"../../examples/Utils/RobotLoggingUtil.cpp",
			"../../examples/Utils/RobotLoggingUtil.h",
			"../../examples/ThirdPartyLibs/tinyxml2/tinyxml2.cpp",
			"../../examples/ThirdPartyLibs/Wavefront/tiny_obj_loader.cpp",
			"../../examples/ThirdPartyLibs/Wavefront/tiny_obj_loader.h",
			"../../examples/ThirdPartyLibs/stb_image/stb_image.cpp",
			"../../examples/ThirdPartyLibs/stb_image/stb_image_write.cpp",
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
			"../SharedMemory/plugins/collisionFilterPlugin/collisionFilterPlugin.cpp",
			"../SharedMemory/plugins/pdControlPlugin/pdControlPlugin.cpp",
			"../SharedMemory/plugins/pdControlPlugin/pdControlPlugin.h",
		}
		
	defines {"B3_ENABLE_FILEIO_PLUGIN", "B3_USE_ZIPFILE_FILEIO"}	
  files {
  	"../SharedMemory/plugins/fileIOPlugin/fileIOPlugin.cpp",
  	"../../examples/ThirdPartyLibs/minizip/ioapi.c",
    "../../examples/ThirdPartyLibs/minizip/unzip.c",
    "../../examples/ThirdPartyLibs/minizip/zip.c",
    "../../examples/ThirdPartyLibs/zlib/adler32.c",
    "../../examples/ThirdPartyLibs/zlib/compress.c",
    "../../examples/ThirdPartyLibs/zlib/crc32.c",
    "../../examples/ThirdPartyLibs/zlib/deflate.c",
    "../../examples/ThirdPartyLibs/zlib/gzclose.c",
    "../../examples/ThirdPartyLibs/zlib/gzlib.c",
    "../../examples/ThirdPartyLibs/zlib/gzread.c",
    "../../examples/ThirdPartyLibs/zlib/gzwrite.c",
    "../../examples/ThirdPartyLibs/zlib/infback.c",
    "../../examples/ThirdPartyLibs/zlib/inffast.c",
    "../../examples/ThirdPartyLibs/zlib/inflate.c",
    "../../examples/ThirdPartyLibs/zlib/inftrees.c",
    "../../examples/ThirdPartyLibs/zlib/trees.c",
    "../../examples/ThirdPartyLibs/zlib/uncompr.c",
    "../../examples/ThirdPartyLibs/zlib/zutil.c",
  }

	if _OPTIONS["enable_stable_pd"] then
		defines {"STATIC_LINK_SPD_PLUGIN"}
		files {
			"../SharedMemory/plugins/stablePDPlugin/SpAlg.cpp",
			"../SharedMemory/plugins/stablePDPlugin/SpAlg.h",
			"../SharedMemory/plugins/stablePDPlugin/Shape.cpp",
			"../SharedMemory/plugins/stablePDPlugin/Shape.h",
			"../SharedMemory/plugins/stablePDPlugin/RBDUtil.cpp",
			"../SharedMemory/plugins/stablePDPlugin/RBDUtil.h",
			"../SharedMemory/plugins/stablePDPlugin/RBDModel.cpp",
			"../SharedMemory/plugins/stablePDPlugin/RBDModel.h",
			"../SharedMemory/plugins/stablePDPlugin/MathUtil.cpp",
			"../SharedMemory/plugins/stablePDPlugin/MathUtil.h",
			"../SharedMemory/plugins/stablePDPlugin/KinTree.cpp",
			"../SharedMemory/plugins/stablePDPlugin/KinTree.h",
			"../SharedMemory/plugins/stablePDPlugin/BulletConversion.cpp",
			"../SharedMemory/plugins/stablePDPlugin/BulletConversion.h",
			}
		end
		
		
	if _OPTIONS["enable_physx"] then
  	defines {"BT_ENABLE_PHYSX","PX_PHYSX_STATIC_LIB", "PX_FOUNDATION_DLL=0"}
		
		configuration {"x64", "debug"}			
				defines {"_DEBUG"}
		configuration {"x86", "debug"}
				defines {"_DEBUG"}
		configuration {"x64", "release"}
				defines {"NDEBUG"}
		configuration {"x86", "release"}
				defines {"NDEBUG"}
		configuration{}

		includedirs {
                ".",
                "../../src/PhysX/physx/include",
						    "../../src/PhysX/physx/include/characterkinematic",
						    "../../src/PhysX/physx/include/common",
						    "../../src/PhysX/physx/include/cooking",
						    "../../src/PhysX/physx/include/extensions",
						    "../../src/PhysX/physx/include/geometry",
						    "../../src/PhysX/physx/include/geomutils",
						    "../../src/PhysX/physx/include/vehicle",
						    "../../src/PhysX/pxshared/include",
                }
		links {
				"PhysX",
			}
			
			files {
				"../SharedMemory/plugins/eglPlugin/eglRendererPlugin.cpp",
				"../SharedMemory/plugins/eglPlugin/eglRendererPlugin.h",
				"../SharedMemory/plugins/eglPlugin/eglRendererVisualShapeConverter.cpp",
				"../SharedMemory/plugins/eglPlugin/eglRendererVisualShapeConverter.h",
				"../SharedMemory/physx/PhysXC_API.cpp",
				"../SharedMemory/physx/PhysXServerCommandProcessor.cpp",
				"../SharedMemory/physx/PhysXUrdfImporter.cpp",
				"../SharedMemory/physx/URDF2PhysX.cpp",
				"../SharedMemory/physx/PhysXC_API.h",
				"../SharedMemory/physx/PhysXServerCommandProcessor.h",
				"../SharedMemory/physx/PhysXUrdfImporter.h",
				"../SharedMemory/physx/URDF2PhysX.h",
				"../SharedMemory/physx/PhysXUserData.h",
				}
  end
  			
if (_OPTIONS["enable_static_vr_plugin"]) then
		files {"../SharedMemory/plugins/vrSyncPlugin/vrSyncPlugin.cpp"}
end

	
	includedirs {
		_OPTIONS["python_include_dir"],
	}
	libdirs {
		_OPTIONS["python_lib_dir"]
	}
	
	if os.is("Linux") then
       		initX11()
	end

	
