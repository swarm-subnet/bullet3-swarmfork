project "App_BulletExampleBrowser"

        language "C++"

        kind "ConsoleApp"
        
        if os.is("Linux") then
	        buildoptions{"-fPIC"}
	    	end

				if _OPTIONS["enable_grpc"] then
					initGRPC()
					defines{"ENABLE_STATIC_GRPC_PLUGIN"}
					 files {
                  "../../src/SharedMemory/PhysicsClientGRPC.cpp",
                  "../../src/SharedMemory/PhysicsClientGRPC.h",
                  "../../src/SharedMemory/PhysicsClientGRPC_C_API.cpp",
                  "../../src/SharedMemory/PhysicsClientGRPC_C_API.h",
                  "../../src/SharedMemory/plugins/grpcPlugin/grpcPlugin.cpp",

                }
				end
		        
        hasCL = findOpenCL("clew")

        if (hasCL) then
            initOpenCL("clew")
        end

        links{"BulletExampleBrowserLib","gwen", "OpenGL_Window","BulletSoftBody", "BulletInverseDynamicsUtils", "BulletInverseDynamics", "BulletDynamics","BulletCollision","LinearMath","BussIK", "Bullet3Common"}
        initOpenGL()
        initGlew()

        includedirs {
                ".",
                "../../src",
		"../../src/SharedMemory",
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

	if _OPTIONS["audio"] then
			files {"../TinyAudio/*.cpp"}
			defines {"B3_ENABLE_TINY_AUDIO"}
			
			if os.is("Windows") then
				links {"winmm","Wsock32","dsound"}
				defines {"WIN32","__WINDOWS_MM__","__WINDOWS_DS__"}
			end
			
			if os.is("Linux") then initX11() 
			                defines  {"__OS_LINUX__","__LINUX_ALSA__"}
				links {"asound","pthread"}
			end


			if os.is("MacOSX") then
				links{"Cocoa.framework"}
				links{"CoreAudio.framework", "coreMIDI.framework", "Cocoa.framework"}
				defines {"__OS_MACOSX__","__MACOSX_CORE__"}
			end
		end
					
    if _OPTIONS["lua"] then
                includedirs{"../ThirdPartyLibs/lua-5.2.3/src"}
                links {"lua-5.2.3"}
                defines {"ENABLE_LUA"}
                files {"../LuaDemo/LuaPhysicsSetup.cpp"}
        end

	defines {"INCLUDE_CLOTH_DEMOS"}

        files {
        	
        "main.cpp",
        "ExampleEntries.cpp",
        "../InverseKinematics/*",
	"../BulletRobotics/FixJointBoxes.cpp",
	"../BulletRobotics/BoxStack.cpp",
	"../BulletRobotics/JointLimit.cpp",
	"../../src/TinyRenderer/geometry.cpp",
		"../../src/TinyRenderer/model.cpp",
		"../../src/TinyRenderer/tgaimage.cpp",
		"../../src/TinyRenderer/our_gl.cpp",
		"../../src/TinyRenderer/TinyRenderer.cpp",
		"../../src/SharedMemory/IKTrajectoryHelper.cpp",
		"../../src/SharedMemory/IKTrajectoryHelper.h",
		"../../src/SharedMemory/PhysicsClientC_API.cpp",
		"../../src/SharedMemory/PhysicsClientC_API.h",
		"../../src/SharedMemory/PhysicsServerExample.cpp",
		"../../src/SharedMemory/PhysicsServerExampleBullet2.cpp",
		"../../src/SharedMemory/PhysicsClientExample.cpp",
		"../../src/SharedMemory/PhysicsServer.cpp",
		"../../src/SharedMemory/PhysicsServerSharedMemory.cpp",
		"../../src/SharedMemory/PhysicsClientSharedMemory.cpp",
		"../../src/SharedMemory/PhysicsClientSharedMemory_C_API.cpp",
		"../../src/SharedMemory/PhysicsClientSharedMemory_C_API.h",
		"../../src/SharedMemory/PhysicsClientSharedMemory2.cpp",
		"../../src/SharedMemory/PhysicsClientSharedMemory2.h",
		"../../src/SharedMemory/PhysicsClientSharedMemory2_C_API.cpp",
		"../../src/SharedMemory/PhysicsClientSharedMemory2_C_API.h",
		"../../src/SharedMemory/SharedMemoryCommandProcessor.cpp",
		"../../src/SharedMemory/SharedMemoryCommandProcessor.h",
		"../../src/SharedMemory/SharedMemoryInProcessPhysicsC_API.cpp",
		"../../src/SharedMemory/GraphicsClientExample.cpp",
		"../../src/SharedMemory/GraphicsClientExample.h",
		"../../src/SharedMemory/GraphicsServerExample.cpp",
		"../../src/SharedMemory/GraphicsServerExample.h",
		"../../src/SharedMemory/GraphicsSharedMemoryBlock.h",
		"../../src/SharedMemory/GraphicsSharedMemoryCommands.h",
		"../../src/SharedMemory/GraphicsSharedMemoryPublic.h",
		"../../src/SharedMemory/RemoteGUIHelper.cpp",
		"../../src/SharedMemory/RemoteGUIHelper.h",
		"../../src/SharedMemory/PhysicsClient.cpp",
		"../../src/SharedMemory/PosixSharedMemory.cpp",
		"../../src/SharedMemory/Win32SharedMemory.cpp",
		"../../src/SharedMemory/InProcessMemory.cpp",
		"../../src/SharedMemory/PhysicsDirect.cpp",
		"../../src/SharedMemory/PhysicsDirect.h",
		"../../src/SharedMemory/PhysicsDirectC_API.cpp",
		"../../src/SharedMemory/PhysicsDirectC_API.h",
		"../../src/SharedMemory/PhysicsLoopBack.cpp",
		"../../src/SharedMemory/PhysicsLoopBack.h",
		"../../src/SharedMemory/PhysicsLoopBackC_API.cpp",
		"../../src/SharedMemory/PhysicsLoopBackC_API.h",
		"../../src/SharedMemory/PhysicsServerCommandProcessor.cpp",
		"../../src/SharedMemory/PhysicsServerCommandProcessor.h",
		"../../src/SharedMemory/b3PluginManager.cpp",		
		"../../src/SharedMemory/plugins/collisionFilterPlugin/collisionFilterPlugin.cpp",
		"../../src/SharedMemory/plugins/tinyRendererPlugin/TinyRendererVisualShapeConverter.cpp",
		"../../src/SharedMemory/plugins/tinyRendererPlugin/tinyRendererPlugin.cpp",
		"../../src/SharedMemory/plugins/pdControlPlugin/pdControlPlugin.cpp",
		"../../src/SharedMemory/plugins/pdControlPlugin/pdControlPlugin.h",
		"../../src/SharedMemory/SharedMemoryCommands.h",
		"../../src/SharedMemory/SharedMemoryPublic.h",
		"../../src/SharedMemory/b3RobotSimulatorClientAPI_NoGUI.cpp",
		"../../src/SharedMemory/b3RobotSimulatorClientAPI_NoGUI.h",		
		"../../src/SharedMemory/b3RobotSimulatorClientAPI_NoDirect.cpp",
		"../../src/SharedMemory/b3RobotSimulatorClientAPI_NoDirect.h",		
		"../MultiThreading/MultiThreadingExample.cpp",
		"../MultiThreading/b3PosixThreadSupport.cpp",
		"../MultiThreading/b3Win32ThreadSupport.cpp",
		"../MultiThreading/b3ThreadSupportInterface.cpp",
		"../InverseDynamics/InverseDynamicsExample.cpp",
		"../InverseDynamics/InverseDynamicsExample.h",
		"../RobotSimulator/b3RobotSimulatorClientAPI.cpp",
		"../RobotSimulator/b3RobotSimulatorClientAPI.h",		
		"../BasicDemo/BasicExample.*",
		"../Tutorial/*",
		"../ExtendedTutorials/*",
		"../Utils/RobotLoggingUtil.cpp",
		"../Utils/RobotLoggingUtil.h",
		"../Evolution/NN3DWalkers.cpp",
		"../Evolution/NN3DWalkers.h",
		"../Collision/*",
		"../RoboticsLearning/*",
		"../Collision/Internal/*",
		"../Benchmarks/*",
		"../MultiThreadedDemo/*",
		"../Heightfield/HeightfieldExample.*",
		"../CommonInterfaces/*.h",
		"../ForkLift/ForkLiftDemo.*",
		"../Importers/**",
		"../../Extras/Serialize/BulletWorldImporter/*",
		"../../Extras/Serialize/BulletFileLoader/*",	
		"../Planar2D/Planar2D.*",
		"../RenderingExamples/*",
		"../VoronoiFracture/*",
		"../SoftDemo/*",
		"../DeformableDemo/*",
                "../ReducedDeformableDemo/*",
		"../RollingFrictionDemo/*",
		"../rbdl/*",
		"../FractureDemo/*",
		"../DynamicControlDemo/*",
		"../Constraints/*",
		"../Vehicles/*",
		"../Raycast/*",
		"../MultiBody/MultiDofDemo.cpp",
		"../MultiBody/SerialChains.cpp",
		"../MultiBody/TestJointTorqueSetup.cpp",
		"../MultiBody/Pendulum.cpp",
		"../MultiBody/MultiBodySoftContact.cpp",
		"../MultiBody/MultiBodyConstraintFeedback.cpp",
		"../MultiBody/InvertedPendulumPDControl.cpp",
		"../MultiBody/KinematicMultiBodyExample.cpp",
		"../RigidBody/RigidBodySoftContact.cpp",
		"../RigidBody/KinematicRigidBodyExample.cpp",
		"../ThirdPartyLibs/stb_image/stb_image.cpp",
		"../ThirdPartyLibs/Wavefront/tiny_obj_loader.*",
		"../GyroscopicDemo/GyroscopicSetup.cpp",
		"../GyroscopicDemo/GyroscopicSetup.h",
    "../ThirdPartyLibs/tinyxml2/tinyxml2.cpp",
    "../ThirdPartyLibs/tinyxml2/tinyxml2.h",
        }
        
  if _OPTIONS["enable_stable_pd"] then
		defines {"STATIC_LINK_SPD_PLUGIN"}
		files {
			"../../src/SharedMemory/plugins/stablePDPlugin/SpAlg.cpp",
			"../../src/SharedMemory/plugins/stablePDPlugin/SpAlg.h",
			"../../src/SharedMemory/plugins/stablePDPlugin/Shape.cpp",
			"../../src/SharedMemory/plugins/stablePDPlugin/Shape.h",
			"../../src/SharedMemory/plugins/stablePDPlugin/RBDUtil.cpp",
			"../../src/SharedMemory/plugins/stablePDPlugin/RBDUtil.h",
			"../../src/SharedMemory/plugins/stablePDPlugin/RBDModel.cpp",
			"../../src/SharedMemory/plugins/stablePDPlugin/RBDModel.h",
			"../../src/SharedMemory/plugins/stablePDPlugin/MathUtil.cpp",
			"../../src/SharedMemory/plugins/stablePDPlugin/MathUtil.h",
			"../../src/SharedMemory/plugins/stablePDPlugin/KinTree.cpp",
			"../../src/SharedMemory/plugins/stablePDPlugin/KinTree.h",
			"../../src/SharedMemory/plugins/stablePDPlugin/BulletConversion.cpp",
			"../../src/SharedMemory/plugins/stablePDPlugin/BulletConversion.h",
			}
		end
if (hasCL and findOpenGL3()) then
			files {
				"../OpenCL/broadphase/*",
				"../OpenCL/CommonOpenCL/*",
				"../OpenCL/rigidbody/GpuConvexScene.cpp",
				"../OpenCL/rigidbody/GpuRigidBodyDemo.cpp",
			}
		end
		
if (_OPTIONS["enable_static_vr_plugin"]) then
		files {"../../src/SharedMemory/plugins/vrSyncPlugin/vrSyncPlugin.cpp"}
end

if os.is("Linux") then
        initX11()
end


	
project "BulletExampleBrowserLib"

		hasCL = findOpenCL("clew")
	
		if (hasCL) then

				-- project ("App_Bullet3_OpenCL_Demos_" .. vendor)

				initOpenCL("clew")

		end

		language "C++"
				
		kind "StaticLib"

  	includedirs {
                ".",
                "../../src",
                "../ThirdPartyLibs",
                }
                
        if os.is("Linux") then
            buildoptions{"-fPIC"}
        end

	if _OPTIONS["lua"] then
		includedirs{"../ThirdPartyLibs/lua-5.2.3/src"}
		links {"lua-5.2.3"}
		defines {"ENABLE_LUA"}
		files {"../LuaDemo/LuaPhysicsSetup.cpp"}
	end

	
	
			
		initOpenGL()
		initGlew()

		defines {"INCLUDE_CLOTH_DEMOS"}
			


		files {
		"OpenGLExampleBrowser.cpp",
		"OpenGLGuiHelper.cpp",
		"OpenGLExampleBrowser.cpp",
		"../Utils/b3Clock.cpp",
		"../Utils/b3Clock.h",
		"../Utils/ChromeTraceUtil.cpp",
		"../Utils/ChromeTraceUtil.h",
		"*.h",
		"GwenGUISupport/*.cpp",
		"GwenGUISupport/*.h",
		"CollisionShape2TriangleMesh.cpp",
		"CollisionShape2TriangleMesh.h",
		"../Utils/b3ResourcePath.*",
		"GL_ShapeDrawer.cpp",
		"InProcessExampleBrowser.cpp",
	
   

		}
		
		

if os.is("Linux") then 
	initX11()
end

			

