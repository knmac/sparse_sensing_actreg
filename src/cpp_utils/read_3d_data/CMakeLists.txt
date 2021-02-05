cmake_minimum_required (VERSION 3.0)
project (lab)
 
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package( OpenCV REQUIRED )

add_executable (read_3d_data read_3d_data.cpp)
target_link_libraries (read_3d_data ${OpenCV_LIBS} )
