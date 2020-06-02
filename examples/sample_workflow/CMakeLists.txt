set(LIBS a4md_cmn a4md_dtl a4md_ingest a4md_retrieve)

add_subdirectory(src) 

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/script/" 
	DESTINATION "${CMAKE_INSTALL_PREFIX}/examples/sample_workflow"
	USE_SOURCE_PERMISSIONS
	)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/dataspaces/" 
	DESTINATION "${CMAKE_INSTALL_PREFIX}/examples/sample_workflow"
	USE_SOURCE_PERMISSIONS
	)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/decaf/" 
	DESTINATION "${CMAKE_INSTALL_PREFIX}/examples/sample_workflow"
	USE_SOURCE_PERMISSIONS
	)