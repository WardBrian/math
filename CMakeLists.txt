cmake_minimum_required(VERSION 3.20.2)
project(
    StanMath
    VERSION 4.6.2
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    StanMathConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

#################### dependencies ####################
# consider also local: https://cmake.org/cmake/help/latest/guide/using-dependencies/index.html#fetchcontent-and-find-package-integration
include(FetchContent)

# dependencies
FetchContent_Declare(
  TBB
  GIT_REPOSITORY https://github.com/oneapi-src/oneTBB
  GIT_TAG        v2021.7.0
#   FIND_PACKAGE_ARGS 2021.7.0 EXACT
)
option(TBB_TEST OFF)
option(TBB_FIND_PACKAGE ON)
FetchContent_MakeAvailable(tbb)

FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
)
option(EIGEN_BUILD_DOC OFF)
# note: To disable eigen tests,
# you should put this code in a add_subdirectory to avoid to change
# BUILD_TESTING for your own project too since variables are directory
# scoped
option(BUILD_TESTING OFF)
option(EIGEN_BUILD_PKGCONFIG OFF)
FetchContent_MakeAvailable(Eigen)

set(EXAMPLES_ENABLE_C OFF)
FetchContent_Declare(
  sundials
  GIT_REPOSITORY https://github.com/LLNL/sundials
  GIT_TAG        v6.5.1
)
FetchContent_MakeAvailable(sundials)


# TODO there should be a better way to do this
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_STATIC_RUNTIME ON)
set(BOOST_ENABLE_CMAKE ON)
FetchContent_Declare(
  boost
  URL https://boostorg.jfrog.io/artifactory/main/release/1.78.0/source/boost_1_78_0.tar.gz
)

FetchContent_GetProperties(boost)
if(NOT boost_POPULATED)
  FetchContent_Populate(boost)
endif()

# TODO opencl, MPI


#################### end dependencies ####################

add_library(stanmath INTERFACE)
target_include_directories(stanmath INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/stanmath>
    $<BUILD_INTERFACE:${boost_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/boost>

    )
target_compile_definitions(stanmath INTERFACE NO_FPRINTF_OUTPUT BOOST_DISABLE_ASSERTS TBB_INTERFACE_NEW _REENTRANT)

if (UNIX)
  target_compile_options(stanmath INTERFACE
    -Wno-deprecated-declarations
    -O3)
elseif (WIN32)
  # so far only tested with "-T ClangCL"
  target_compile_definitions(stanmath INTERFACE _BOOST_LGAMMA)
  target_compile_options(stanmath INTERFACE /bigobj /Z:_cplusplus)
  # TODO figure out way to "Install" TBB dlls
endif()

target_link_libraries(stanmath INTERFACE
  TBB::tbb
  Eigen3::Eigen
  sundials_kinsol
  sundials_cvodes)

install(TARGETS stanmath EXPORT StanMathTargets)
install(EXPORT StanMathTargets
    FILE StanMathTargets.cmake
    NAMESPACE StanMath::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/StanMath)
install(FILES "StanMathConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/StanMathConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/StanMath)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/stan/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/stanmath/stan)

install(DIRECTORY ${boost_SOURCE_DIR}/boost DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_subdirectory(examples)
