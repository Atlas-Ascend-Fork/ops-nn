# Compile and Run Samples

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T12:52:41.171Z -->

## Prerequisites

- To compile and execute operator APIs, ensure that the basic environment is set up, including the driver, firmware, CANN software package, and ops package.
- For details about the operator API invocation process and compilation and running operations, see "Single Operator Call > Single Operator API Execution > Example Code for Calling aclnn APIs" in [*Application Development (C&C++)*](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/others/acldevg/aclcppdevg_000006.html).

## Preparations Before Compilation

This chapter uses the case where the development and running environments are co-located as an example, that is, the machine with an AI processor serves as both the development environment and the running environment. In this case, code development and code running occur on the same machine. The following uses the **AddMatMul operator** as an example. The call logic, procedures, and build scripts of other operators are similar to those of the AddMatMul operator. Modify the API call script (\*.cpp) and build script (CMakeLists.txt) as required.

- **Sample Code**

   The AddMatMul operator implements tensor addition, with the formula: out = β self + α (mat1 @ mat2). You can obtain the sample code from the "Example" section in [aclnnAddmm&aclnnInplaceAddmm](../../../matmul/mat_mul_v3/docs/aclnnAddmm&aclnnInplaceAddmm_en.md) and name the code file “**test\_addmm.cpp**”.

- **CMakeLists File**

    The following is an example of the CMake file. Modify it based on the actual situation:

    ```bash
    # Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

    # CMake lowest version requirement
    cmake_minimum_required(VERSION 3.14)

    # Set the project name.
    project(ACLNN_EXAMPLE)

    # Compile options
    add_compile_options(-std=c++11)

    # Set the build options.
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "./bin")
    set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
    set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

    # Set the executable file name (for example, opapi_test) and specify the directory where the *.cpp file of the operator to run resides.
    add_executable(opapi_test
                   test_addmm.cpp)

    # Set ASCEND_PATH (CANN software package directory, modify it based on the actual path) and INCLUDE_BASE_DIR (header file directory).
    if(NOT "$ENV{ASCEND_CUSTOM_PATH}" STREQUAL "")
        set(ASCEND_PATH $ENV{ASCEND_CUSTOM_PATH})
    else()
        set(ASCEND_PATH "/usr/local/Ascend/cann")
    endif()
    set(INCLUDE_BASE_DIR "${ASCEND_PATH}/include")
    include_directories(
        ${INCLUDE_BASE_DIR}
        ${INCLUDE_BASE_DIR}/aclnn
    )

    # Set the path of the library files to be linked.
    target_link_libraries(opapi_test PRIVATE
                          ${ASCEND_PATH}/lib64/libascendcl.so
                          ${ASCEND_PATH}/lib64/libnnopbase.so
                          ${ASCEND_PATH}/lib64/libopapi_math.so
                          ${ASCEND_PATH}/lib64/libopapi_nn.so)

    # The executable file is generated in the bin directory under the directory where the CMakeLists file resides.
    install(TARGETS opapi_test DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

    ```

    Operators that integrate collective communication with MatMul computation and parallelism are collectively referred to as communication-computation fusion operators (MC2 operators for short), including AllGatherMatmul, AlltoAllAllGatherBatchMatMul, BatchMatMulReduceScatterAlltoAll, MatmulAllReduce, MatmulAllReduceAddRmsNorm, MatmulReduceScatter, and others. When calling APIs of such operators, multi-threading and HCCL (Huawei Collective Communication Library) are typically involved. Therefore, the CMake file must additionally import the following content; otherwise, compilation will fail.

  ```bash
  # Set the path of the library files to be linked.
  find_package(Threads REQUIRED)
  target_link_libraries(opapi_test PRIVATE
                        ${ASCEND_PATH}/lib64/libascendcl.so
                        ${ASCEND_PATH}/lib64/libnnopbase.so
                        ${ASCEND_PATH}/lib64/libopapi_math.so
                        ${ASCEND_PATH}/lib64/libopapi_nn.so
                        ${ASCEND_PATH}/lib64/libhccl.so      # HCCL (Huawei Collective Communication Library) file
                        ${CMAKE_THREAD_LIBS_INIT})           # Library files required for multithreading
  ```

  Here, "find_package(Threads REQUIRED)" is a CMake command for locating the thread library, which automatically links the header files that the thread library depends on or the indirectly dependent library files.

## Compile and Run

  1. Prepare the operator invocation code (\*.cpp) and the build script (CMakeLists.txt) in advance.
  2. Configure environment variables.

     After installing the CANN software, log in to the CANN runtime environment as the user and execute the following command to make the environment variables take effect.

        ```bash
        source ${INSTALL_DIR}/set_env.sh
        ```

     Where ${INSTALL_DIR} is the storage path of the CANN software after installation. Replace it with the actual path as needed.

  3. Compile and run.

        - Go to the directory where CMakeLists.txt is located and run the following command to create a build directory for storing the generated compilation files.

            ```bash
            mkdir -p build
            ```

        - Go to the build directory, run the cmake command to compile, and then run the make command to generate the executable file.

          ```bash
          cd build
          cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
          make
          ```

          After successful compilation, the opapi\_test executable file is generated in the bin folder under the build directory.

        - Go to the bin directory and run the executable file opapi_test.

          ```bash
          cd bin
          ./opapi_test
          ```

          Taking the AddMatMul operator as an example, the result after running is as follows:

          ```bash
          result[0] is: 7.200000
          result[1] is: 9.400000
          result[2] is: 11.600000
          result[3] is: 13.800000
          result[4] is: 18.400000
          result[5] is: 21.800000
          result[6] is: 25.200000
          result[7] is: 28.600000
          ```

          If an error occurs or the expected result is not obtained, call the aclGetRecentErrMsg API to obtain detailed error information.
          The following is an example of obtaining exception information when a call to aclnnAddmmGetWorkspaceSize fails:

          ```bash
          // self is nullptr
          ret = aclnnAddmmGetWorkspaceSize(self, mat1, mat2, beta, alpha, out, cubeMathType, &workspaceSize, &executor);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddmmGetWorkspaceSize failed. ERROR: %d\n[ERROR msg]%s", ret, aclGetRecentErrMsg()); return ret);
          ```

          The following is an example of obtaining error information for the null pointer construction problem above:

          ```bash
          aclnnAddmmGetWorkspaceSize failed. ERROR: 161001
          [ERROR msg][PID:xxxx] xxx(timestamp) AclNN_Parameter_Error(EZ1001): Expected a proper Tensor but got null for argument addmmTensor.self.
          ```
