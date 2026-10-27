# aclnnSparse4to2QuantMatmulWeightNz

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：不支持
<!-- end id1 -->

<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->

<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id3 -->

<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->

<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->

<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：完成4选2稀疏量化矩阵乘计算。sparseWeight和index由[aclnnTransSparse4to2Para](aclnnTransSparse4to2Para.md)生成。

- 计算公式：

  $$
  out = (x @ sparseWeight) * sparseWeightScale * xScale + bias
  $$

  其中，`@`表示根据index执行的稀疏矩阵乘法，结果的shape为（m，n）。sparseWeightScale按列缩放，xScale按行缩放。bias对应biasOptional，未传入时按0处理。

## 函数原型

本算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用aclnnSparse4to2QuantMatmulWeightNzGetWorkspaceSize接口获取计算所需workspace大小以及包含算子计算流程的执行器，再调用aclnnSparse4to2QuantMatmulWeightNz接口执行计算。

```cpp
aclnnStatus aclnnSparse4to2QuantMatmulWeightNzGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* sparseWeight,
    const aclTensor* index,
    const aclTensor* xScale,
    const aclTensor* sparseWeightScale,
    const aclTensor* biasOptional,
    aclTensor*       out,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnSparse4to2QuantMatmulWeightNz(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
```

## aclnnSparse4to2QuantMatmulWeightNzGetWorkspaceSize

- **参数说明**

  <table style="table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 340px">
  <col style="width: 100px">
  <col style="width: 230px">
  <col style="width: 390px">
  <col style="width: 120px">
  <col style="width: 120px">
  <col style="width: 100px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
   <tbody>
      <tr>
        <td>x（const aclTensor*）</td>
        <td>输入</td>
        <td>矩阵乘运算中的左矩阵。</td>
        <td><ul><li>仅支持2维输入，shape为（m，k），不支持空Tensor。</li><li>k不超过65535。</li><li>不支持通过转置stride表示的输入。</li></ul></td>
        <td>INT8</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
      </tr>
      <tr>
        <td>sparseWeight（const aclTensor*）</td>
        <td>输入</td>
        <td>aclnnTransSparse4to2Para返回的压缩权重。</td>
        <td><ul><li>不支持空Tensor。ViewShape为（n，k_half），其中k_half=ceil（k / 8）* 8 / 2。</li><li>StorageShape为（ceil（k_half / 32），ceil（n / 16），16，32），由aclnnTransSparse4to2Para的sparseWeightDims和sparseWeightDimsNum返回。</li><li>数据须按FRACTAL_NZ布局存储。可按示例使用ND格式创建aclTensor，并设置上述ViewShape和StorageShape。接口不进行数据重排。</li></ul></td>
        <td>INT8</td>
        <td>FRACTAL_NZ</td>
        <td>2</td>
        <td>×</td>
      </tr>
      <tr>
        <td>index（const aclTensor*）</td>
        <td>输入</td>
        <td>aclnnTransSparse4to2Para返回的稀疏权重索引。</td>
        <td><ul><li>不支持空Tensor。ViewShape为（n，ceil（k / 8））。</li><li>StorageShape为（ceil（k_half / 32），ceil（n / 16），16，8），由aclnnTransSparse4to2Para的indexDims和indexDimsNum返回。</li><li>保持预处理接口返回的数据排列，Tensor须连续。</li></ul></td>
        <td>UINT8</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
      </tr>
      <tr>
        <td>xScale（const aclTensor*）</td>
        <td>输入</td>
        <td>x的逐行反量化缩放因子。</td>
        <td><ul><li>不支持空指针或空Tensor。</li><li>shape为（m，）。</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>×</td>
      </tr>
      <tr>
        <td>sparseWeightScale（const aclTensor*）</td>
        <td>输入</td>
        <td>sparseWeight的反量化缩放因子，对应输出的每一列。</td>
        <td><ul><li>不支持空指针或空Tensor。</li><li>shape为（n，）。</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>×</td>
      </tr>
      <tr>
        <td>biasOptional（const aclTensor*）</td>
        <td>可选输入</td>
        <td>偏置，对应公式中的bias。</td>
        <td><ul><li>传入空指针时不添加偏置。</li><li>非空时shape为（n，），不支持空Tensor。</li></ul></td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>1</td>
        <td>×</td>
      </tr>
      <tr>
        <td>out（aclTensor*）</td>
        <td>输出</td>
        <td>计算结果，对应公式中的out。</td>
        <td><ul><li>shape为（m，n），不支持空Tensor。</li></ul></td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
      </tr>
      <tr>
        <td>workspaceSize（uint64_t*）</td>
        <td>输出</td>
        <td>返回需要在Device侧申请的workspace大小，单位为字节。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor（aclOpExecutor**）</td>
        <td>输出</td>
        <td>返回op执行器，包含了算子计算流程。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
  </tbody>
  </table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口会完成入参校验，出现以下场景时报错：

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 300px">
  <col style="width: 150px">
  <col style="width: 550px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的x、sparseWeight、index、xScale、sparseWeightScale或out任一为空指针。</td>
    </tr>
    <tr>
      <td rowspan="10">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="10">161002</td>
      <td>输入或输出的数据类型不符合参数说明。</td>
    </tr>
    <tr>
      <td>x、index、xScale、sparseWeightScale、out或非空的biasOptional的数据格式不是ND。</td>
    </tr>
    <tr>
      <td>x、sparseWeight、index或out的ViewShape不是2维，或xScale、sparseWeightScale及非空的biasOptional不是1维。</td>
    </tr>
    <tr>
      <td>x的k维超过65535。</td>
    </tr>
    <tr>
      <td>x和sparseWeight的k维不匹配，或xScale、sparseWeightScale、biasOptional及out的shape不符合参数说明。</td>
    </tr>
    <tr>
      <td>sparseWeight的StorageShape不符合参数说明。</td>
    </tr>
    <tr>
      <td>index的ViewShape或StorageShape不符合参数说明，或index为非连续Tensor。</td>
    </tr>
    <tr>
      <td>输入为空Tensor，即m、n、k任一为0。</td>
    </tr>
    <tr>
      <td>x通过转置stride表示。</td>
    </tr>
    <tr>
      <td>当前运行平台不在支持范围内。</td>
    </tr>
  </tbody></table>

## aclnnSparse4to2QuantMatmulWeightNz

- **参数说明**

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 300px">
  <col style="width: 100px">
  <col style="width: 600px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace（void*）</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t）</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnSparse4to2QuantMatmulWeightNzGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor*）</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream（aclrtStream）</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- sparseWeight和index须对应同一份权重。按各自的StorageShape申请内存，并拷贝完整数据。
- 输入与输出的shape须满足参数表中的m、n、k对应关系，不支持batch维度。
- 确定性计算：

  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：aclnnSparse4to2QuantMatmulWeightNz默认确定性实现。
  <!-- end id7 -->

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

对应工程样例：[test_aclnn_sparse4to2quant_matmul.cpp](../examples/test_aclnn_sparse4to2quant_matmul.cpp)。

<!-- npu="A3,910b" id8 -->
- <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：

  ```cpp
  #include <iostream>
  #include <memory>
  #include <vector>
  #include <stdlib.h>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_sparse4to2quant_matmul_weight_nz.h"

  #define CHECK_RET(cond, return_expr) \
      do {                             \
          if (!(cond)) {               \
              return_expr;             \
          }                            \
      } while (0)

  #define CHECK_FREE_RET(cond, return_expr) \
      do {                                  \
          if (!(cond)) {                    \
              Finalize(deviceId, stream);   \
              return_expr;                  \
          }                                 \
      } while (0)

  #define LOG_PRINT(message, ...)         \
      do {                                \
          printf(message, ##__VA_ARGS__); \
      } while (0)

  #define CREATE_TENSOR(hostData, shape, deviceAddr, dtype, tensor)                                        \
      ret = CreateAclTensor(hostData, shape, &deviceAddr, dtype, &tensor);                                 \
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> tensor##Ptr(tensor, aclDestroyTensor); \
      std::unique_ptr<void, aclError (*)(void*)> deviceAddr##Ptr(deviceAddr, aclrtFree);                  \
      CHECK_RET(ret == ACL_SUCCESS, return ret)

  #define CREATE_SPARSE_TENSOR(hostData, weightShape, storageShape, deviceAddr, dataType, tensor)          \
      ret = CreateSparseTensor(hostData, weightShape, storageShape, &deviceAddr, dataType, &tensor);       \
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> tensor##Ptr(tensor, aclDestroyTensor); \
      std::unique_ptr<void, aclError (*)(void*)> deviceAddr##Ptr(deviceAddr, aclrtFree);                  \
      CHECK_RET(ret == ACL_SUCCESS, return ret)

  int64_t GetShapeSize(const std::vector<int64_t>& shape)
  {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  int Init(int32_t deviceId, aclrtStream* stream)
  {
      // 固定写法，资源初始化
      auto ret = aclInit(nullptr);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
      ret = aclrtSetDevice(deviceId);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
      ret = aclrtCreateStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
      return 0;
  }

  template <typename T>
  int CreateSparseTensor(
      const T* sparseWeightData, const std::vector<int64_t>& viewShape, const std::vector<int64_t>& storageShape,
      void** deviceAddr, aclDataType dataType, aclTensor** tensor)
  {
      auto size = static_cast<uint64_t>(GetShapeSize(storageShape)) * sizeof(T);

      // 调用aclrtMalloc申请device侧内存
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
      ret = aclrtMemcpy(*deviceAddr, size, sparseWeightData, size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // 计算连续tensor的strides
      std::vector<int64_t> strides(viewShape.size(), 1);
      for (int64_t i = viewShape.size() - 2; i >= 0; i--) {
          strides[i] = viewShape[i + 1] * strides[i + 1];
      }

      // 调用aclCreateTensor接口创建aclTensor
      *tensor = aclCreateTensor(
          viewShape.data(), viewShape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, storageShape.data(),
          storageShape.size(), *deviceAddr);
      return 0;
  }

  template <typename T>
  int CreateAclTensor(
      const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
      aclTensor** tensor)
  {
      auto size = GetShapeSize(shape) * sizeof(T);
      // 调用aclrtMalloc申请device侧内存
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
      if (hostData.size() > 0) {
          ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
      }

      // 计算连续tensor的strides
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      // 调用aclCreateTensor接口创建aclTensor
      *tensor = aclCreateTensor(
          shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
          *deviceAddr);
      return 0;
  }

  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  }

  void GenRandomMask(std::vector<size_t>& masks)
  {
      masks[0] = random() % 4;
      masks[1] = random() % 4;
      while (masks[1] == masks[0]) {
          masks[1] = random() % 4;
      }
  }

  void GenRandomSparseData(std::vector<int8_t>& weightHostData)
  {
      srandom(233U);
      std::vector<size_t> masks(2, 0UL);
      constexpr size_t step = 4UL;
      for (size_t i = 0; i < weightHostData.size(); i += step) {
          GenRandomMask(masks);
          for (auto mask : masks) {
              weightHostData[i + mask] = 0;
          }
      }
  }

  std::vector<int64_t> GenStorageShape(int64_t* dims, uint64_t dimsNum)
  {
      std::vector<int64_t> storageShape;
      for (uint64_t i = 0UL; i < dimsNum; i++) {
          storageShape.push_back(dims[i]);
      }
      return storageShape;
  }

  int aclnnSparse4to2QuantMatmulWeightNzTest(int32_t deviceId, aclrtStream& stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. 构造输入与输出，需要根据API的接口自定义构造
      int64_t m = 64L;
      int64_t k = 512L;
      int64_t n = 128L;
      std::vector<int64_t> xShape = {m, k};
      std::vector<int64_t> weightShape = {n, k};
      std::vector<int64_t> indexShape = {n, (k + 7) / 8};
      std::vector<int64_t> biasShape = {n};
      std::vector<int64_t> xScaleShape = {m};
      std::vector<int64_t> weightScaleShape = {n};
      std::vector<int64_t> outShape = {m, n};
      void* xDeviceAddr = nullptr;
      void* sparseWeightDeviceAddr = nullptr;
      void* indexDeviceAddr = nullptr;
      void* biasDeviceAddr = nullptr;
      void* xScaleDeviceAddr = nullptr;
      void* weightScaleDeviceAddr = nullptr;
      void* outDeviceAddr = nullptr;
      aclTensor* x = nullptr;
      aclTensor* sparseWeight = nullptr;
      aclTensor* index = nullptr;
      aclTensor* bias = nullptr;
      aclTensor* xScale = nullptr;
      aclTensor* weightScale = nullptr;
      aclTensor* out = nullptr;
      std::vector<int8_t> xHostData(GetShapeSize(xShape), 1);
      std::vector<int8_t> weightHostData(GetShapeSize(weightShape), 1);
      std::vector<uint16_t> biasHostData(GetShapeSize(biasShape), 1); // 实际上是bfloat16半精度方式
      std::vector<float> xScaleHostData(GetShapeSize(xScaleShape), 1);
      std::vector<float> weightScaleHostData(GetShapeSize(weightScaleShape), 1);
      GenRandomSparseData(weightHostData);

      int8_t* sparseWeightHostData = nullptr;
      uint8_t* indexHostData = nullptr;
      int64_t* sparseWeightDims = nullptr;
      uint64_t sparseWeightDimsNum = 0UL;
      int64_t* indexDims = nullptr;
      uint64_t indexDimsNum = 0UL;
      aclIntArray* weightShapeArray = aclCreateIntArray(weightShape.data(), weightShape.size());
      CHECK_RET(weightShapeArray != nullptr, LOG_PRINT("aclCreateIntArray failed.\n"); return 1);
      ret = aclnnTransSparse4to2Para(
          weightHostData.data(), weightShapeArray, &sparseWeightHostData, &sparseWeightDims, &sparseWeightDimsNum,
          &indexHostData, &indexDims, &indexDimsNum);
      aclDestroyIntArray(weightShapeArray);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransSparse4to2Para failed. ERROR: %d\n", ret); return ret);
      std::unique_ptr<int8_t, decltype(&free)> sparseWeightHostDataPtr(sparseWeightHostData, free);
      std::unique_ptr<uint8_t, decltype(&free)> indexHostDataPtr(indexHostData, free);
      std::unique_ptr<int64_t, decltype(&free)> sparseWeightDimsPtr(sparseWeightDims, free);
      std::unique_ptr<int64_t, decltype(&free)> indexDimsPtr(indexDims, free);

      CREATE_TENSOR(xHostData, xShape, xDeviceAddr, aclDataType::ACL_INT8, x);

      weightShape.back() = (weightShape.back() + 7) / 8 * 8 / 2; // 4选2后K轴向上8对齐后减半
      auto sparseWeightStorageShape = GenStorageShape(sparseWeightDims, sparseWeightDimsNum);
      CREATE_SPARSE_TENSOR(
          sparseWeightHostData, weightShape, sparseWeightStorageShape, sparseWeightDeviceAddr, aclDataType::ACL_INT8,
          sparseWeight);

      auto indexStorageShape = GenStorageShape(indexDims, indexDimsNum);
      CREATE_SPARSE_TENSOR(indexHostData, indexShape, indexStorageShape, indexDeviceAddr, aclDataType::ACL_UINT8, index);
      CREATE_TENSOR(biasHostData, biasShape, biasDeviceAddr, aclDataType::ACL_BF16, bias);
      CREATE_TENSOR(xScaleHostData, xScaleShape, xScaleDeviceAddr, aclDataType::ACL_FLOAT, xScale);
      CREATE_TENSOR(weightScaleHostData, weightScaleShape, weightScaleDeviceAddr, aclDataType::ACL_FLOAT, weightScale);
      CREATE_TENSOR(std::vector<uint16_t>(), outShape, outDeviceAddr, aclDataType::ACL_BF16, out);

      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;
      void* workspaceAddr = nullptr;

      // 调用aclnnSparse4to2QuantMatmulWeightNz第一段接口
      ret = aclnnSparse4to2QuantMatmulWeightNzGetWorkspaceSize(
          x, sparseWeight, index, xScale, weightScale, bias, out, &workspaceSize, &executor);

      CHECK_RET(
          ret == ACL_SUCCESS, LOG_PRINT("aclnnSparse4to2QuantMatmulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret);
          return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtrTrans(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrTrans.reset(workspaceAddr);
      }
      // 调用aclnnSparse4to2QuantMatmulWeightNz第二段接口
      ret = aclnnSparse4to2QuantMatmulWeightNz(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSparse4to2QuantMatmulWeightNz failed. ERROR: %d\n", ret); return ret);

      // 4.（固定写法）同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
      auto size = GetShapeSize(outShape);
      // C语言中无法直接打印bf16的数据，需要用uint16读出来，自行通过二进制转成bf16
      std::vector<uint16_t> resultData(size, 0);
      ret = aclrtMemcpy(
          resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
          ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %u\n", i, resultData[i]);
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1.（固定写法）device/stream初始化，参考acl API手册
      // 根据自己的实际device填写deviceId
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = aclnnSparse4to2QuantMatmulWeightNzTest(deviceId, stream);
      CHECK_FREE_RET(
          ret == ACL_SUCCESS, LOG_PRINT("aclnnSparse4to2QuantMatmulWeightNzTest failed. ERROR: %d\n", ret); return ret);

      Finalize(deviceId, stream);
      return 0;
  }

  ```

<!-- end id8 -->
