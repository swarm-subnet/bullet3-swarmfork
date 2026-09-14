		

project ("pybullet_tinyRendererPlugin")
		language "C++"
		kind "SharedLib"
		
		includedirs {".","../../..", "../../../../examples",
		"../../../../examples/ThirdPartyLibs"}
		defines {"PHYSICS_IN_PROCESS_EXAMPLE_BROWSER"}
	hasCL = findOpenCL("clew")

	links{"BulletCollision", "Bullet3Common", "LinearMath"}

	if os.is("MacOSX") then
--		targetextension {"so"}
		links{"Cocoa.framework","Python"}
	end


		files {
			"tinyRendererPlugin.cpp",
			"tinyRendererPlugin.h",
			"TinyRendererVisualShapeConverter.cpp",
			"TinyRendererVisualShapeConverter.h",
			"../../../../examples/Importers/ImportColladaDemo/LoadMeshFromCollada.cpp",
			"../../../../examples/Importers/ImportColladaDemo/LoadMeshFromCollada.h",
			"../../../../examples/Importers/ImportMeshUtility/b3ImportMeshUtility.cpp",
			"../../../../examples/Importers/ImportMeshUtility/b3ImportMeshUtility.h",
			"../../../../examples/Importers/ImportObjDemo/LoadMeshFromObj.cpp",
			"../../../../examples/Importers/ImportObjDemo/LoadMeshFromObj.h",
			"../../../../examples/Importers/ImportObjDemo/Wavefront2GLInstanceGraphicsShape.cpp",
			"../../../../examples/Importers/ImportObjDemo/Wavefront2GLInstanceGraphicsShape.h",
			"../../../TinyRenderer/geometry.cpp",
			"../../../TinyRenderer/model.cpp",
			"../../../TinyRenderer/our_gl.cpp",
			"../../../TinyRenderer/tgaimage.cpp",
			"../../../TinyRenderer/TinyRenderer.cpp",
			"../../../../examples/ThirdPartyLibs/Wavefront/tiny_obj_loader.cpp",
			"../../../../examples/ThirdPartyLibs/Wavefront/tiny_obj_loader.h",
			"../../../../examples/ThirdPartyLibs/stb_image/stb_image.cpp",
			"../../../../examples/ThirdPartyLibs/stb_image/stb_image.h",
			"../../../../examples/ThirdPartyLibs/tinyxml2/tinyxml2.cpp",
			"../../../../examples/ThirdPartyLibs/tinyxml2/tinyxml2.h",
			"../../../../examples/OpenGLWindow/SimpleCamera.cpp",
			"../../../../examples/OpenGLWindow/SimpleCamera.h",
			"../../../../examples/Utils/b3Clock.cpp",
			"../../../../examples/Utils/b3Clock.h",
			"../../../../examples/Utils/b3ResourcePath.cpp",
			"../../../../examples/Utils/b3ResourcePath.h",
			}
	
	
	
