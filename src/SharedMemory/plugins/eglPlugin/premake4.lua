		

project ("pybullet_eglRendererPlugin")
		language "C++"
		kind "SharedLib"
		initEGL()
		
		includedirs {".","../../..", "../../../../examples",
		"../../../../examples/ThirdPartyLibs", "../../../../examples/ThirdPartyLibs/glad"}
		defines {"PHYSICS_IN_PROCESS_EXAMPLE_BROWSER", "STB_AGAIN"}
	hasCL = findOpenCL("clew")

	links{"BulletCollision", "Bullet3Common", "LinearMath"}

	initOpenGL()

	if os.is("Windows") then
		files {"../../../../examples/OpenGLWindow/Win32OpenGLWindow.cpp",
		"../../../../examples/OpenGLWindow/Win32Window.cpp",}
		
	end
	if os.is("MacOSX") then
--		targetextension {"so"}
		links{"Cocoa.framework"}
	end

  if os.is("Linux") then
	  files {"../../../../examples/OpenGLWindow/EGLOpenGLWindow.cpp"}

  end

		files {
			"eglRendererPlugin.cpp",
			"eglRendererPlugin.h",
			"eglRendererVisualShapeConverter.cpp",
			"eglRendererVisualShapeConverter.h",
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
			"../../../../examples/ThirdPartyLibs/glad/gl.c",
			"../../../../examples/ThirdPartyLibs/glad/egl.c",
			"../../../../examples/ThirdPartyLibs/Wavefront/tiny_obj_loader.cpp",
			"../../../../examples/ThirdPartyLibs/Wavefront/tiny_obj_loader.h",
			"../../../../examples/ThirdPartyLibs/stb_image/stb_image.cpp",
			"../../../../examples/ThirdPartyLibs/stb_image/stb_image.h",
			"../../../../examples/ThirdPartyLibs/stb_image/stb_image_write.cpp",
			"../../../../examples/ThirdPartyLibs/tinyxml2/tinyxml2.cpp",
			"../../../../examples/ThirdPartyLibs/tinyxml2/tinyxml2.h",
			"../../../../examples/OpenGLWindow/SimpleCamera.cpp",
			"../../../../examples/OpenGLWindow/SimpleCamera.h",
			"../../../../examples/OpenGLWindow/GLInstancingRenderer.cpp",
			"../../../../examples/OpenGLWindow/GLInstancingRenderer.h",
			"../../../../examples/OpenGLWindow/LoadShader.cpp",
			"../../../../examples/OpenGLWindow/LoadShader.h",
			"../../../../examples/OpenGLWindow/GLRenderToTexture.cpp",
			"../../../../examples/OpenGLWindow/GLRenderToTexture.h",
			"../../../../examples/Utils/b3Clock.cpp",
			"../../../../examples/Utils/b3Clock.h",
			"../../../../examples/Utils/b3ResourcePath.cpp",
			"../../../../examples/Utils/b3ResourcePath.h",
			}
	
	
	
