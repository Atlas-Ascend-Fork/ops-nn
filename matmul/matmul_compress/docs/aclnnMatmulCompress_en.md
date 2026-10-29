# aclnnMatmulCompress

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T18:22:45.337Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Not supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: For l@r matrix multiplication, you can use the msModelSlim tool to compress the r matrix losslessly first to reduce its memory usage, and then call this API to perform lossless decompression, matrix multiplication, and dequantization.
- Formula:

$$
x2\_unzip = unzip(x2, compressIndex)\\
result=x1 @ x2\_unzip + bias
$$

where x2 denotes the one-dimensional data of the r matrix after being compressed by the msModelSlim tool, and compressIndex denotes the information related to the compression algorithm. $x2\_unzip$ is the data obtained after lossless decompression inside this API (identical to the original r matrix data). For detailed examples of compression and calling this API, see [Example](#example).

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnMatmulCompressGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the "aclnnMatmulCompress" API to perform the computation.

```cpp
aclnnStatus aclnnMatmulCompressGetWorkspaceSize(
  const aclTensor* x,
  const aclTensor* weight,
  const aclTensor* bias,
  const aclTensor* compressIndex,
  aclTensor*       out,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnMatmulCompress(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnMatmulCompressGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1494px"><colgroup>
  <col style="width: 155px">
  <col style="width: 123px">
  <col style="width: 425px">
  <col style="width: 251px">
  <col style="width: 124px">
  <col style="width: 122px">
  <col style="width: 147px">
  <col style="width: 147px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Left input of matrix multiplication, matrix x1 in the formula.</td>
      <td>-</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>Input</td>
      <td>Right input of the compressed matrix multiplication, matrix $x2\_unzip$ in the formula. It is the input compressed by the weight_compression module in the msModelSlim tool.</td>
      <td>-</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>Input</td>
      <td>Indicates the input of the bias, that is, the matrix bias in the formula. Only the FLOAT data type is supported. A null pointer can be passed in. Only the (1, n) or (n) shape is supported, where n is the n in the output shape (m, n).</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>compressIndex</td>
      <td>Input</td>
      <td>Indicates the compression index table for the right input of matrix multiplication.</td>
      <td>Obtained using the msModelSlim tool in the example.</td>
      <td>INT8</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output tensor.</td>
      <td>Obtained from the msModelSlim tool in the example.</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, containing the computation process of the operator.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 281px">
  <col style="width: 119px">
  <col style="width: 749px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The passed-in x, weight, or out is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data type or data format of x, weight, or out is not within the supported range.</td>
    </tr>
  </tbody>
  </table>

## aclnnMatmulCompress

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device side, obtained by calling the first-phase API aclnnMatmulCompressGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:

  <!-- npu="910,310p" id7 -->
  - <term>Atlas training products</term> and <term>Atlas inference products</term>: aclnnMatmulCompress defaults to a deterministic implementation.

  <!-- end id7 -->

## Example

1. Prepare data before compression

    The following example shows how to generate input data using the gen_data.py script, for reference only:

    ```python
    import numpy as np
    import os
    import sys
    from numpy import random

    def write2file(data, path):
      with open(path, 'wb') as f:
          data.tofile(f)

    if not os.path.exists("./data"):
        os.mkdir("./data")

    if len(sys.argv) != 4:
      print("Usage: python gen_data.py m k n")
      sys.exit(1)

    m = int(sys.argv[1])
    k = int(sys.argv[2])
    n = int(sys.argv[3])

    if m <= 0 or k <= 0 or n <= 0:
      print("Error: m, k and n must be positive integers.")
      sys.exit(1)

    # Randomly generate matrix mat1 with shape (m,k )
    mat1 = random.randn(m, k).astype(np.float16)
    write2file(mat1, "./data/mat1.bin")

    # Randomly generate matrix mat2 with shape (n, k)
    mat2 = random.randint(0, 100, size=(n, k)).astype(np.float16)
    mat2 = np.transpose(mat2, (1, 0)).copy()
    mat2 = mat2.view(np.int8)
    np.save("./data/weight.npy", {'weight': mat2})
    os.chmod("./data/weight.npy", 0o0640)

    # Generate output
    output = np.random.randn(m, n).astype(np.float16)
    write2file(output, "./data/output.bin")

    # Generate bias
    bias = random.randn(n).astype(np.float32)
    write2file(bias, "./data/bias.bin")

    ```

    Execute gen_data.py, assuming the shape parameters of mat1 and mat2 are m=512, k=1024, and n=1024.

    ```shell
    python3 gen_data.py 512 1024 1024
    ```

2. Preprocess the data

    **The original weight is compressed by the msModelSlim compression tool to generate the compressed x2, compressIndex, and compressInfo:**
    To use the following APIs, compile the msModelSlim compression tool in the CANN package. For details, see item 5, Basic Weight Compression Procedure, in foundation_model_compression.md in the docs/en/user_guide/traditional_quantization_v0 directory of the [Gitcode msmodelslim repository](https://gitcode.com/Ascend/msmodelslim).

    ```python
    from msmodelslim.pytorch.weight_compression import CompressConfig, Compressor

    compress_config = CompressConfig(do_pseudo_sparse=False, sparse_ratio=1)
    weight_path = "./data/weight.npy"  # Path of the weight file generated by gen_data.py
    compressor = Compressor(compress_config, weight_path=weight_path)

    compress_weight, compress_index, compress_info = compressor.run()
    # Compressed weight, corresponding to x2 of the aclnnMatmulCompressGetWorkspaceSize API
    compressor.export(compress_weight, './data/weight')
    # Index of the compressed weight, corresponding to compressIndex of the aclnnMatmulCompressGetWorkspaceSize API
    compressor.export(compress_index, './data/index')
    ```

3. Call the aclnn APIs to perform the computation

    The sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

    ```Cpp
    #include <iostream>
    #include <vector>
    #include <acl/acl.h>
    #include <aclnnop/aclnn_matmul_compress.h>
    #include <fstream>
    #include <unistd.h>
    #include <sys/stat.h>
    #include <stdio.h>
    #include <cstdlib>
    #include <string>

    #define CHECK_RET(cond, return_expr) \
      do {                               \
        if (!(cond)) {                   \
          return_expr;                   \
        }                                \
      } while (0)

    #define LOG_PRINT(message, ...)     \
      do {                              \
        printf(message, ##__VA_ARGS__); \
      } while (0)

    int64_t GetShapeSize(const std::vector<int64_t>& shape) {
      int64_t shapeSize = 1;
      for (auto i : shape) {
        shapeSize *= i;
      }
      return shapeSize;
    }

    int Init(int32_t deviceId, aclrtStream* stream) {
      // Boilerplate: resource initialization.
      auto ret = aclInit(nullptr);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
      ret = aclrtSetDevice(deviceId);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
      ret = aclrtCreateStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
      return 0;
    }

    int ReadBinFileNNop(std::string filePath, void* buffer, size_t bufferSize)
    {
        struct stat sBuf;
        int fileStatus = stat(filePath.data(), &sBuf);
        CHECK_RET(fileStatus == ACL_SUCCESS, LOG_PRINT("Failed to get file %s\n", filePath); return -1);

        std::ifstream file;
        file.open(filePath, std::ios::binary);
        CHECK_RET(file.is_open(), LOG_PRINT("Open file failed.\n"); return -1);

        file.seekg(0, file.end);
        uint64_t binFileBufferLen = file.tellg();
        CHECK_RET(binFileBufferLen > 0,
            std::cout<<"File size is 0.\n";
            file.close();
            return -1);

        file.seekg(0, file.beg);
        file.read(static_cast<char *>(buffer), binFileBufferLen);
        file.close();
        return ACL_SUCCESS;
    }

    int CreateAclTensor(std::string filePath, const std::vector<int64_t>& shape, int typeSize,
                        void** deviceAddr, aclDataType dataType, aclTensor** tensor) {
      auto size = GetShapeSize(shape) * typeSize;
      // Call aclrtMalloc to allocate device memory.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

      // Call aclrtMallocHost to allocate host memory.
      void* binBufferHost = nullptr;
      ret = aclrtMallocHost(&binBufferHost, size);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMallocHost failed. ERROR: %d\n", ret); return ret);

      // Read the file.
      ret = ReadBinFileNNop(filePath, binBufferHost, size);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ReadBinFileNNop failed. ERROR: %d\n", ret); return ret);

      // Call aclrtMemcpy to copy data from the host to the device.
      ret = aclrtMemcpy(*deviceAddr, size, binBufferHost, size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Calculate the strides of a contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
      }

      // Call aclCreateTensor to create an aclTensor.
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
      return 0;
    }

    int main(int argc, char* argv[]) {
      // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual.
      // Set deviceId based on the actual device.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      if (argc != 6) {
        std::cerr << "Error: Invalid number of arguments. Usage: <program> m k n wCompressedSize indexSize" << std::endl;
        return -1;
      }

      // 2. Construct the input and output. Custom construction is required based on the API.
      int m = atoi(argv[1]);
      int k = atoi(argv[2]);
      int n = atoi(argv[3]);
      // wShape is the size of the compressed data of the right matrix.
      int wCompressedSize = atoi(argv[4]);
      // indexShape is the size of the compressed index data.
      int indexSize = atoi(argv[5]);

      if (m <= 0 || k <= 0 || n <= 0 || wCompressedSize <= 0 || indexSize <= 0) {
        std::cerr << "Error: m, k, n, wCompressedSize and indexSize must be positive integers." << std::endl;
        return -1;
      }

      std::vector<int64_t> mat1Shape = {m, k};
      std::vector<int64_t> mat2CompressedShape = {wCompressedSize};
      std::vector<int64_t> indexShape = {indexSize};
      std::vector<int64_t> biasShape = {n};
      std::vector<int64_t> outputShape = {m, n};

      void* mat1DeviceAddr = nullptr;
      void* mat2CompressedDeviceAddr = nullptr;
      void* indexDeviceAddr = nullptr;
      void* biasDeviceAddr = nullptr;
      void* outputDeviceAddr = nullptr;

      aclTensor* mat1 = nullptr;
      aclTensor* mat2Compressed = nullptr;
      aclTensor* index = nullptr;
      aclTensor* bias = nullptr;
      aclTensor* output = nullptr;

      std::string rootPath = "./data/";

      // Create mat1 aclTensor.
      std::string mat1FilePath = rootPath + "mat1.bin";
      ret = CreateAclTensor(mat1FilePath, mat1Shape, sizeof(uint16_t), &mat1DeviceAddr, aclDataType::ACL_FLOAT16, &mat1);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create mat1 tensor failed. ERROR: %d\n", ret); return ret);
      // Create mat2Compressed aclTensor.
      std::string mat2FilePath = rootPath + "weight/weight.dat";
      ret = CreateAclTensor(mat2FilePath, mat2CompressedShape, sizeof(int8_t), &mat2CompressedDeviceAddr,
                            aclDataType::ACL_FLOAT16, &mat2Compressed);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create mat2 tensor failed. ERROR: %d\n", ret); return ret);
      // Create index aclTensor.
      std::string indexFilePath = rootPath + "index/weight.dat";
      ret = CreateAclTensor(indexFilePath, indexShape, sizeof(int8_t), &indexDeviceAddr, aclDataType::ACL_INT8, &index);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create index tensor failed. ERROR: %d\n", ret); return ret);
      // Create bias aclTensor.
      std::string biasFilePath = rootPath + "bias.bin";
      ret = CreateAclTensor(biasFilePath, biasShape, sizeof(float), &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create bias tensor failed. ERROR: %d\n", ret); return ret);
      // Create out aclTensor.
      std::string outputFilePath = rootPath + "output.bin";
      ret = CreateAclTensor(outputFilePath, outputShape, sizeof(uint16_t), &outputDeviceAddr, aclDataType::ACL_FLOAT16, &output);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create output tensor failed. ERROR: %d\n", ret); return ret);

      int32_t offsetX = 0;

      // 3. Call the CANN operator library API, and change it to the specific API name.
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;
      // Call the first-phase API of aclnnMatmulCompress.
      ret = aclnnMatmulCompressGetWorkspaceSize(mat1, mat2Compressed, bias, index, output, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulCompressGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // Allocate device memory based on the workspaceSize calculated by the first-phase API.
      void* workspaceAddr = nullptr;
      if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // Call the second-phase API of aclnnMatmulCompress.
      ret = aclnnMatmulCompress(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulCompress failed. ERROR: %d\n", ret); return ret);

      // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value and copy the result from the device side to the host side. Modify the code based on the specific API definition.
      auto size = GetShapeSize(outputShape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outputDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
      }

      // 6. Release aclTensor and aclScalar. Modify the code based on the specific API definition.
      aclDestroyTensor(mat1);
      aclDestroyTensor(mat2Compressed);
      aclDestroyTensor(index);
      aclDestroyTensor(bias);
      aclDestroyTensor(output);

      // 7. Release hardware resources. Modify the code based on the specific API definition.
      aclrtFree(mat1DeviceAddr);
      aclrtFree(mat2CompressedDeviceAddr);
      aclrtFree(indexDeviceAddr);
      aclrtFree(biasDeviceAddr);
      aclrtFree(outputDeviceAddr);
      if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
      }
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
      return 0;
    }
    ```
