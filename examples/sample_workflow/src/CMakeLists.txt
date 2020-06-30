if (dtl_decaf)
	add_definitions(-DDTL_DECAF)
endif()

add_executable(producer producer.cxx)
add_executable(consumer consumer.cxx)
add_executable(intermediator intermediator.cxx)

target_link_libraries(producer ${LIBS})
target_link_libraries(consumer ${LIBS})
target_link_libraries(intermediator ${LIBS})

install (TARGETS producer DESTINATION "${CMAKE_INSTALL_PREFIX}/examples/sample_workflow")
install (TARGETS consumer DESTINATION "${CMAKE_INSTALL_PREFIX}/examples/sample_workflow")
install (TARGETS intermediator DESTINATION "${CMAKE_INSTALL_PREFIX}/examples/sample_workflow")