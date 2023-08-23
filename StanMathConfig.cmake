include(CMakeFindDependencyMacro)

find_dependency(TBB)
find_dependency(Eigen3)
find_dependency(SUNDIALS)

include(${CMAKE_CURRENT_LIST_DIR}/StanMathTargets.cmake)
