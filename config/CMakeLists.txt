configure_file(
	${CMAKE_CURRENT_SOURCE_DIR}/a4md_config.in
	${CMAKE_CURRENT_BINARY_DIR}/a4md_config
	@ONLY
)

install (FILES ${CMAKE_CURRENT_BINARY_DIR}/a4md_config DESTINATION bin)

install(CODE "
	execute_process(COMMAND chmod +x ${CMAKE_INSTALL_PREFIX}/bin/a4md_config RESULT_VARIABLE result)
	if(result)
		message(FATAL_ERROR \"Generating a4md_config returned \${result}, should be 0\")
	endif()
	"
	COMPONENT a4md_config
)