cmake_minimum_required(VERSION 3.15...3.31)

project(
  quadrotor_hover
  LANGUAGES CXX
)

find_package(param_id CONFIG REQUIRED)
message("param_id_DIR: ${param_id_DIR}")
message("param_id_INCLUDE_DIR: ${param_id_INCLUDE_DIR}")
include_directories(${param_id_INCLUDE_DIR})

add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} param_id::param_id)

add_custom_target(
  run
  COMMAND quadrotor_hover
  DEPENDS quadrotor_hover
  WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)
