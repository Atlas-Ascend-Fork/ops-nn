# aclnnTransSparse4to2Para

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

以上为配套接口[aclnnSparse4to2QuantMatmulWeightNz](aclnnSparse4to2QuantMatmulWeightNz.md)的产品支持范围。本接口在Host侧执行。

## 功能说明

- 接口功能：压缩4选2稀疏权重weight，生成FRACTAL_NZ布局的sparseWeight和索引index，供[aclnnSparse4to2QuantMatmulWeightNz](aclnnSparse4to2QuantMatmulWeightNz.md)使用。
- 处理过程：weight沿k轴每4个元素至少有2个零，压缩后每组保留2个元素。每8个元素生成4个INT8权重和1个UINT8索引，尾部不足8个时补零。输出按StorageShape补齐并存储。

## 函数原型

本接口同步执行，返回成功后即可读取压缩结果。

```cpp
aclnnStatus aclnnTransSparse4to2Para(
    const int8_t* weight,
    aclIntArray*  shape,
    int8_t**      sparseWeight,
    int64_t**     sparseWeightDims,
    uint64_t*     sparseWeightDimsNum,
    uint8_t**     index,
    int64_t**     indexDims,
    uint64_t*     indexDimsNum)
```

## aclnnTransSparse4to2Para

- **参数说明**

  所有参数均不可为空指针。

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
        <td>weight（const int8_t*）</td>
        <td>输入</td>
        <td>Host侧INT8稀疏权重的首地址。</td>
        <td>按（n，k）行优先连续存储，至少包含n*k个int8_t元素。每行沿k轴每4个元素至少有2个零，不足一组时补零后判断。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>shape（aclIntArray*）</td>
        <td>输入</td>
        <td>Host侧未压缩权重的shape。</td>
        <td>通过aclCreateIntArray创建，包含n、k两个正整数。使用完后调用aclDestroyIntArray释放。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>sparseWeight（int8_t**）</td>
        <td>输出</td>
        <td>返回压缩权重的Host侧地址。</td>
        <td>输出为INT8数据，元素个数为sparseWeightDims各元素的乘积。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>sparseWeightDims（int64_t**）</td>
        <td>输出</td>
        <td>返回压缩权重的StorageShape数组。</td>
        <td>数组内容为（ceil（k / 64），ceil（n / 16），16，32）。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>sparseWeightDimsNum（uint64_t*）</td>
        <td>输出</td>
        <td>返回压缩权重StorageShape数组的元素个数。</td>
        <td>成功时返回4。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>index（uint8_t**）</td>
        <td>输出</td>
        <td>返回索引数据的Host侧地址。</td>
        <td>输出为UINT8数据，元素个数为indexDims各元素的乘积。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>indexDims（int64_t**）</td>
        <td>输出</td>
        <td>返回索引数据的StorageShape数组。</td>
        <td>数组内容为（ceil（k / 64），ceil（n / 16），16，8）。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>indexDimsNum（uint64_t*）</td>
        <td>输出</td>
        <td>返回索引数据StorageShape数组的元素个数。</td>
        <td>成功时返回4。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
  </tbody>
  </table>

  调用前，将`*sparseWeight`、`*sparseWeightDims`、`*index`和`*indexDims`初始化为nullptr。调用成功后，这四个指针指向接口分配的Host内存，使用完后分别调用free释放。

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  本接口会完成入参校验，出现以下场景时报错：

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
      <td>传入的weight、shape、sparseWeight、sparseWeightDims、sparseWeightDimsNum、index、indexDims或indexDimsNum任一为空指针。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>shape的size不为2、n或k不大于0、原始或压缩后的元素总数超出int64_t范围；或weight沿k轴的分组数据不满足每4个元素至少2个为零。</td>
    </tr>
  </tbody></table>

## 约束说明

- 原始权重和压缩后权重的元素总数均不能超过int64_t的最大值。
- 将压缩结果用于aclnnSparse4to2QuantMatmulWeightNz时，k不能超过65535，压缩权重和索引须配套使用。
- 确定性计算：

  <!-- npu="A3,910b" id7 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：aclnnTransSparse4to2Para默认确定性实现。
  <!-- end id7 -->

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

示例先在Host侧压缩权重，再调用aclnnSparse4to2QuantMatmulWeightNz计算。对应工程样例：[test_aclnn_sparse4to2quant_matmul.cpp](../examples/test_aclnn_sparse4to2quant_matmul.cpp)。

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

  int aclnnSparse4to2QuantMatmulTest(int32_t deviceId, aclrtStream& stream)
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
      auto ret = aclnnSparse4to2QuantMatmulTest(deviceId, stream);
      CHECK_FREE_RET(
          ret == ACL_SUCCESS, LOG_PRINT("aclnnSparse4to2QuantMatmulTest failed. ERROR: %d\n", ret); return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```
