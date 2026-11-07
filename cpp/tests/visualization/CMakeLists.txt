if (BUILD_GUI)
    target_sources(tests PRIVATE
        rendering/MaterialModifier.cpp
        rendering/GaussianSplatRender.cpp
    )

    # This smoke executable requires GUI support; it is not part of gtest.
    add_executable(TreeViewTest
        TreeViewTest.cpp
    )
    target_link_libraries(TreeViewTest PRIVATE Open3D::Open3D)
    set_target_properties(TreeViewTest PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endif()
