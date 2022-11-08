# Created by the script cgal_create_CMakeLists
# This is the CMake script for compiling a set of CGAL applications.

cmake_minimum_required(VERSION 3.1...3.23)

project( cgal_plane_est )


# CGAL and its components
find_package( CGAL QUIET COMPONENTS  )

if ( NOT CGAL_FOUND )

  message(STATUS "This project requires the CGAL library, and will not be compiled.")
  return()  

endif()


# Boost and its components
find_package( Boost REQUIRED )

if ( NOT Boost_FOUND )

  message(STATUS "This project requires the Boost library, and will not be compiled.")

  return()  

endif()

# Eigen3

find_package( Eigen3 REQUIRED )
if (EIGEN3_FOUND)
  include( CGAL_Eigen3_support )
endif()

# Activate concurrency?
#option(ACTIVATE_CONCURRENCY
#       "Enable concurrency"
#       ON)
#
#if( ACTIVATE_CONCURRENCY )
#  find_package( TBB REQUIRED )
#  if( TBB_FOUND )
#    include(${TBB_USE_FILE})
#    list(APPEND CGAL_3RD_PARTY_LIBRARIES ${TBB_LIBRARIES})
#  endif()
#endif()

find_package( TBB REQUIRED )
include( CGAL_TBB_support )

find_package( MLPACK REQUIRED )

# include for local directory

# include for local package

# Creating entries for all C++ files with "main" routine
# ##########################################################

#create_single_source_cgal_program( "main.cpp" )
add_executable(main main.cpp)
target_link_libraries(main CGAL::CGAL CGAL::TBB_support CGAL::Eigen3_support ${MLPACK_LIBRARY})


