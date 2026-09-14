
project ("App_PhysicsServerSharedMemoryBridgeGRPC")

	language "C++"
			
	kind "ConsoleApp"
	
	includedirs {"../..",".."}
		
	initGRPC()

		links {
			"BulletFileLoader",
			"Bullet3Common", 
			"LinearMath"
		}
	
	files {
		"main.cpp",
		"../PhysicsClient.cpp",
		"../PhysicsClient.h",
		"../PhysicsDirect.cpp",
		"../PhysicsDirect.h",
		"../PhysicsCommandProcessorInterface.h",
		"../SharedMemoryCommandProcessor.cpp",
		"../SharedMemoryCommandProcessor.h",
		"../PhysicsClientC_API.cpp",
		"../PhysicsClientC_API.h",
		"../Win32SharedMemory.cpp",
		"../Win32SharedMemory.h",
		"../PosixSharedMemory.cpp",
		"../PosixSharedMemory.h",
		"../../../examples/Utils/b3ResourcePath.cpp",
		"../../../examples/Utils/b3ResourcePath.h",
		"../../../examples/Utils/b3Clock.cpp",
		"../../../examples/Utils/b3Clock.h",		
	}


project "App_PhysicsServerGRPC"

if _OPTIONS["ios"] then
	kind "WindowedApp"
else	
	kind "ConsoleApp"
end

defines { "NO_SHARED_MEMORY" }
						
includedirs {"..","../..", "../../../examples/ThirdPartyLibs","../../../examples/ThirdPartyLibs/clsocket/src"}

links {
	"clsocket","Bullet3Common","BulletInverseDynamicsUtils", "BulletInverseDynamics",	"BulletSoftBody", "BulletDynamics","BulletCollision", "LinearMath", "BussIK"
}


	initGRPC()


language "C++"

myfiles = 
{
	"../IKTrajectoryHelper.cpp",
	"../IKTrajectoryHelper.h",
	"../SharedMemoryCommands.h",
	"../SharedMemoryPublic.h",
	"../PhysicsServerCommandProcessor.cpp",
	"../PhysicsServerCommandProcessor.h",
	"../b3PluginManager.cpp",
	"../PhysicsDirect.cpp",
	"../PhysicsClientC_API.cpp",
	"../PhysicsClient.cpp",
	"../plugins/collisionFilterPlugin/collisionFilterPlugin.cpp",
	"../plugins/pdControlPlugin/pdControlPlugin.cpp",
	"../plugins/pdControlPlugin/pdControlPlugin.h",
	"../b3RobotSimulatorClientAPI_NoDirect.cpp",
	"../b3RobotSimulatorClientAPI_NoDirect.h",
  "../plugins/tinyRendererPlugin/tinyRendererPlugin.cpp",
	"../plugins/tinyRendererPlugin/TinyRendererVisualShapeConverter.cpp",
	"../../TinyRenderer/geometry.cpp",
	"../../TinyRenderer/model.cpp",
	"../../TinyRenderer/tgaimage.cpp",
	"../../TinyRenderer/our_gl.cpp",
	"../../TinyRenderer/TinyRenderer.cpp",
	"../../../examples/OpenGLWindow/SimpleCamera.cpp",
	"../../../examples/OpenGLWindow/SimpleCamera.h",
	"../../../examples/Importers/ImportURDFDemo/ConvertRigidBodies2MultiBody.h",
	"../../../examples/Importers/ImportURDFDemo/MultiBodyCreationInterface.h",
	"../../../examples/Importers/ImportURDFDemo/MyMultiBodyCreator.cpp",
	"../../../examples/Importers/ImportURDFDemo/MyMultiBodyCreator.h",
	"../../../examples/Importers/ImportMJCFDemo/BulletMJCFImporter.cpp",
	"../../../examples/Importers/ImportMJCFDemo/BulletMJCFImporter.h",
	"../../../examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp",
	"../../../examples/Importers/ImportURDFDemo/BulletUrdfImporter.h",
	"../../../examples/Importers/ImportURDFDemo/UrdfParser.cpp",
 	"../../../examples/Importers/ImportURDFDemo/urdfStringSplit.cpp",
	"../../../examples/Importers/ImportURDFDemo/UrdfParser.cpp",
	"../../../examples/Importers/ImportURDFDemo/UrdfParser.h",
	"../../../examples/Importers/ImportURDFDemo/URDF2Bullet.cpp",
	"../../../examples/Importers/ImportURDFDemo/URDF2Bullet.h",
	"../../../examples/Utils/b3ResourcePath.cpp",
	"../../../examples/Utils/b3Clock.cpp",
	"../../../examples/Utils/ChromeTraceUtil.cpp",
	"../../../examples/Utils/ChromeTraceUtil.h",
	"../../../examples/Utils/RobotLoggingUtil.cpp",
	"../../../examples/Utils/RobotLoggingUtil.h",
	"../../../Extras/Serialize/BulletWorldImporter/*",
	"../../../Extras/Serialize/BulletFileLoader/*",	
	"../../../examples/Importers/ImportURDFDemo/URDFImporterInterface.h",
	"../../../examples/Importers/ImportURDFDemo/URDFJointTypes.h",
	"../../../examples/Importers/ImportObjDemo/Wavefront2GLInstanceGraphicsShape.cpp",
	"../../../examples/Importers/ImportObjDemo/LoadMeshFromObj.cpp",
	"../../../examples/Importers/ImportSTLDemo/ImportSTLSetup.h",
	"../../../examples/Importers/ImportSTLDemo/LoadMeshFromSTL.h",
	"../../../examples/Importers/ImportColladaDemo/LoadMeshFromCollada.cpp",
	"../../../examples/Importers/ImportColladaDemo/ColladaGraphicsInstance.h",
	"../../../examples/ThirdPartyLibs/Wavefront/tiny_obj_loader.cpp",	
	"../../../examples/ThirdPartyLibs/tinyxml2/tinyxml2.cpp",
	"../../../examples/Importers/ImportMeshUtility/b3ImportMeshUtility.cpp",
	"../../../examples/ThirdPartyLibs/stb_image/stb_image.cpp",     
}

files {
	myfiles,
	"main.cpp",
}

