# SPDX-License-Identifier: LicenseRef-CSSL-1.0

pkg_check_modules(zmq REQUIRED libzmq)
add_library(zmqscope SHARED zmqscope.cpp)
target_link_libraries(zmqscope PUBLIC ${zmq_LIBRARIES} ring_buffer)
