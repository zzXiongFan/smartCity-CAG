cmake_minimum_required( VERSION 2.8 )
project( photo_location )

set( CMAKE_BUILD_TYPE "Release" )
# 设置编译模式，使用c++11,编译器优化模式
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
# 添加调试模式
set (CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -g")

# 编译设置
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/lib)           #设置库文件的输出目录
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/bin)       #设置可执行文件的输出目录


# 添加opencv库,使用库版本opencv3.4.1
find_package( OpenCV REQUIRED )

# 添加系统boost库
set( Boost_USE_STATIC_LIBS OFF )
set( Boost_USE_STATIC_RUNTIME OFF )
find_package( Boost REQUIRED COMPONENTS system filesystem )

# 词袋库,需要在本地先安装DBOW库
set( DBoW3_INCLUDE_DIRS "/usr/local/include" )
set( DBoW3_LIBS "/usr/local/lib/libDBoW3.a" )

# 添加源文件路径
include_directories(src)
include_directories(include())


# 生成插件库
add_library(utils src/utils/utils.cpp)
add_library(asift src/sift/ASifttDetector.cpp)


# 将库链入include
include_directories( 
    ${OpenCV_INCLUDE_DIRS} 
)

# 生成对一个的可执行文件
add_executable( photoLocation src/photoLocation.cpp  )
target_link_libraries( photoLocation utils asift ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${OpenCV_LIBS} ${OpenCV_LIBS} ${DBoW3_LIBS})

# 生成对应的demo
add_executable(Asift_demo src/sift/asift_demo.cpp)
target_link_libraries(Asift_demo utils asift ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${OpenCV_LIBS} )