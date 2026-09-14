		

project ("pybullet_grpcPlugin")
		language "C++"
		kind "SharedLib"
		
		includedirs {".","../../..", "../../../../examples",
		"../../../../examples/ThirdPartyLibs"}
		defines {"PHYSICS_IN_PROCESS_EXAMPLE_BROWSER"}
	
	initGRPC()

	links{"BulletFileLoader", "Bullet3Common", "LinearMath"}


	if os.is("MacOSX") then
--		targetextension {"so"}
		links{"Cocoa.framework"}
	end


		files {
			"grpcPlugin.cpp",
			"../../PhysicsClient.cpp",
			"../../PhysicsClient.h",
			"../../PhysicsClientSharedMemory.cpp",
			"../../PhysicsClientSharedMemory.h",
			"../../PhysicsClientSharedMemory_C_API.cpp",
			"../../PhysicsClientSharedMemory_C_API.h",
			"../../PhysicsClientC_API.cpp",
			"../../PhysicsClientC_API.h",
			"../../Win32SharedMemory.cpp",
			"../../Win32SharedMemory.h",
			"../../PosixSharedMemory.cpp",
			"../../PosixSharedMemory.h",
			"../../../../examples/Utils/b3Clock.cpp",
			"../../../../examples/Utils/b3Clock.h",
			"../../../../examples/Utils/b3ResourcePath.cpp",
			"../../../../examples/Utils/b3ResourcePath.h",
				}
	
	
	
