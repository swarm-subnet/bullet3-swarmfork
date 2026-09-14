SUBDIRS( HelloWorld BasicDemo)
IF(BUILD_BULLET3)
	SUBDIRS( ExampleBrowser RobotSimulator ThirdPartyLibs/Gwen ThirdPartyLibs/BussIK ThirdPartyLibs/clsocket OpenGLWindow TwoJoint )
ENDIF()
