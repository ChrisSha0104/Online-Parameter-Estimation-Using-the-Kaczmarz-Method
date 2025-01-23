file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${deka_SOURCE_DIR}/include/deka/*.hpp")

add_library(${PROJECT_NAME} lib.cpp ${HEADER_LIST})

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)

target_include_directories(
  ${PROJECT_NAME}
  PUBLIC
  $<BUILD_INTERFACE:${${PROJECT_NAME}_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)
