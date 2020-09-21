if (dtl_decaf)
	add_definitions(-DDTL_DECAF)
endif()

add_executable(chain_producer producer.cxx)
add_executable(chain_consumer consumer.cxx)
add_executable(chain_intermediator intermediator.cxx)

target_link_libraries(chain_producer ${LIBS})
target_link_libraries(chain_consumer ${LIBS})
target_link_libraries(chain_intermediator ${LIBS})

install (TARGETS chain_producer DESTINATION "${CMAKE_INSTALL_PREFIX}/examples/chain_workflow")
install (TARGETS chain_consumer DESTINATION "${CMAKE_INSTALL_PREFIX}/examples/chain_workflow")
install (TARGETS chain_intermediator DESTINATION "${CMAKE_INSTALL_PREFIX}/examples/chain_workflow")