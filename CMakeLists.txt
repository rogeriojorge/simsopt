cmake_minimum_required(VERSION 3.5)
project(simsgeopp)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

if(SKBUILD)
  set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
  set(Python_INCLUDE_DIR "${PYTHON_INCLUDE_DIR}")
  set(Python_LIBRARY "${PYTHON_LIBRARY}")
  set(DUMMY "${PYTHON_VERSION_STRING}")  # Not needed, silences a warning
endif()


set(Python_FIND_IMPLEMENTATIONS CPython) #PyPy)  # PyPy requires 3.18
find_package(Python REQUIRED COMPONENTS Interpreter Development)

#find_package(PythonInterp 3 REQUIRED)
#find_package(PythonLibs 3 REQUIRED)
find_package(NumPy REQUIRED)
#find_package(OpenMP)


set(CMAKE_CXX_FLAGS "-O3 -march=native")


execute_process(
  COMMAND
    "${Python_EXECUTABLE}" -c
    "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE _tmp_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")

find_package(pybind11 CONFIG REQUIRED)

#add_subdirectory(thirdparty/pybind11)
#set(XTENSOR_USE_OPENMP 0)



pybind11_add_module(simsgeopp
    cpp/simsgeopp/python.cpp# simsgeopp/curve.cpp
    )
set_target_properties(simsgeopp
    PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON)
target_include_directories(simsgeopp PRIVATE "thirdparty/xtensor/include" "thirdparty/xtensor-python/include" "thirdparty/xtl/include" ${NUMPY_INCLUDE_DIRS})
target_link_libraries(simsgeopp PUBLIC ${PYTHON_LIBRARY})


#target_include_directories(hello_world PRIVATE  "xtensor/include" "xtensor-python/include" "xsimd/include" "xtl/include" "blaze" "cppplasmaopt" ${PYBIND11_INCLUDE_DIR} ${NUMPY_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIR})
#target_link_libraries(hello_world PUBLIC ${PYTHON_LIBRARY})


#if(OpenMP_CXX_FOUND)
#    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
#    target_link_libraries(profiling_biot_savart PRIVATE OpenMP::OpenMP_CXX)
#endif()

#get_cmake_property(_variableNames VARIABLES)
#list (SORT _variableNames)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()

install(TARGETS simsgeopp LIBRARY DESTINATION .)
