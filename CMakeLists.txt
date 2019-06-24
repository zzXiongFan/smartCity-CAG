cmake_minimum_required( VERSION 2.8 )
project( photo_location )

set( CMAKE_BUILD_TYPE "Release" )
# 设置编译模式，使用c++11
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
# 添加调试模式
set (CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -g")


# 添加opencv库,使用库版本opencv3.4.1
find_package( OpenCV REQUIRED )

# 词袋库,需要在本地先安装DBOW库
set( DBoW3_INCLUDE_DIRS "/usr/local/include" )
set( DBoW3_LIBS "/usr/local/lib/libDBoW3.a" )


# 将库链入include
include_directories( 
    ${OpenCV_INCLUDE_DIRS} 
)

# 生成对一个的可执行文件
add_executable( photoLocation src/photoLocation.cpp  )
target_link_libraries( photoLocation ${OpenCV_LIBS} ${DBoW3_LIBS})