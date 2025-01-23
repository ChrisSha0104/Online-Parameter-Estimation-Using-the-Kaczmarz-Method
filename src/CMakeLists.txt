file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${param_id_SOURCE_DIR}/include/param_id/*.hpp")

add_library(${PROJECT_NAME} lib.cpp ${HEADER_LIST})

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)

# target_include_directories(${PROJECT_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/include")

target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
    $<BUILD_INTERFACE:${${PROJECT_NAME}_SOURCE_DIR}/include/param_id>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(param_id PRIVATE Eigen3::Eigen)
