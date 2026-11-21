# aclnnSituGluGroupGrad

## 产品支持情况

<!-- npu="950" id1 -->

- <term>Ascend 950PR&950DT系列产品</term>：支持

<!-- end id1 -->

<!-- npu="A3" id2 -->

- <term>Atlas A3系列产品</term>：不支持

<!-- end id2 -->

<!-- npu="910b" id3 -->

- <term>Atlas A2系列产品</term>：不支持

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

- 接口功能：SiTU门控线性单元（SiTU Gated Linear Unit）激活函数的反向梯度计算。根据上游梯度gradY和前向输入x，计算输入x的梯度gradX与weight的梯度gradWeight；传入可选输入weight时按行对gradX加权并计算gradWeight，未传入weight时gradWeight为空Tensor（shape [0]）；传入可选输入groupIndex时仅计算前realBs = min(总行数, 行数)行，总行数由groupListType决定（cumsum模式取groupIndex[-1]，count模式取sum(groupIndex)）。仅Ascend 950PR&950DT系列产品支持。

- 计算公式：

  对给定的输入张量x（维度为[..., cut]，cut为偶数，令h = cut // 2）和上游梯度gradY（维度为[..., h]），aclnnSituGluGroupGrad对其执行以下计算：

  1. 根据activateLeft切分x为gate和up（各shape为[..., h]）。

  2. 计算前向中间量：

     $$
     t = \tanh\left(\frac{gate}{\beta}\right), \quad s = \sigma(gate), \quad situ\_a = \beta \cdot t \cdot s
     $$

     当linearBeta > 0时：

     $$
     up' = linearBeta \cdot \tanh\left(\frac{up}{linearBeta}\right)
     $$

     否则up' = up。

  3. 乘积法则：

     $$
     grad\_situ\_a = gradY \cdot up', \quad grad\_up' = gradY \cdot situ\_a
     $$

  4. gate梯度（w为可选输入weight对应行的标量，不传weight时w = 1）：

     $$
     grad\_gate = grad\_situ\_a \cdot s \cdot \left[ (1 - t^2) + \beta \cdot t \cdot (1 - s) \right] \cdot w
     $$

  5. up梯度：

     - linearBeta > 0时：

     $$
     grad\_up = grad\_up' \cdot \left(1 - \tanh^2\left(\frac{up}{linearBeta}\right)\right) \cdot w
     $$

     - linearBeta ≤ 0时：

     $$
     grad\_up = grad\_up' \cdot w
     $$

  6. 根据activateLeft将grad_gate、grad_up拼接为grad_x（shape与x一致）。

  7. weight梯度（输出grad_weight，未传入weight时grad_weight为空Tensor），对每行n沿h求和（注意不乘w）：

     $$
     grad\_w_n = \sum_{h} gradY_{n,h} \cdot situ\_a_{n,h} \cdot up'_{n,h}
     $$

  8. 行掩码（可选输入groupIndex）：realBs = min(总行数, 行数)，仅前realBs行被计算并写回；行号 ≥ realBs的行不读不写（输出buffer对应行保持调用方原值）。总行数由groupListType决定：cumsum模式（0，默认）为groupIndex[-1]，count模式（1）为sum(groupIndex)。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用"aclnnSituGluGroupGradGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnSituGluGroupGrad"接口执行计算。

```cpp
aclnnStatus aclnnSituGluGroupGradGetWorkspaceSize(
    const aclTensor *gradY,
    const aclTensor *x,
    const aclTensor *weightOptional,
    const aclTensor *groupIndexOptional,
    int64_t dim,
    double beta,
    double linearBeta,
    bool activateLeft,
    int64_t groupListType,
    const aclTensor *gradXOut,
    const aclTensor *gradWeightOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
```

```cpp
aclnnStatus aclnnSituGluGroupGrad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
```

## aclnnSituGluGroupGradGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 320px">
  <col style="width: 250px">
  <col style="width: 120px">
  <col style="width: 140px">
  <col style="width: 140px">
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
      <td>gradY（aclTensor*）</td>
      <td>输入</td>
      <td>公式中的输入gradY，表示上游梯度。</td>
      <td><ul><li>支持空Tensor。</li><li>shape在入参dim对应的维度上为x的一半，其他维度与x一致。</li><li>数据类型与x一致。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x（aclTensor*）</td>
      <td>输入</td>
      <td>公式中的输入x，表示前向输入。</td>
      <td><ul><li>支持空Tensor（输出为对应空Tensor）。</li><li>shape必须在入参dim对应的维度上为偶数。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weightOptional（aclTensor*）</td>
      <td>输入</td>
      <td>公式中的可选输入weight，前向的每行路由权重。</td>
      <td><ul><li>仅Ascend 950PR&950DT系列产品支持，可为nullptr。</li><li>元素个数必须等于x在dim轴之前各维度的乘积。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>groupIndexOptional（aclTensor*）</td>
      <td>输入</td>
      <td>公式中的可选输入groupIndex，各组token数，含义由groupListType决定。</td>
      <td><ul><li>仅Ascend 950PR&950DT系列产品支持，可为nullptr。</li><li>必须为一维非空Tensor，元素非负。</li><li>仅前min(总行数, 行数)行被计算并写回，总行数由groupListType决定。</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dim（int64_t）</td>
      <td>输入</td>
      <td>公式中的输入dim，表示对x进行合轴以及切分的维度序号。</td>
      <td>取值范围为[-x.dim(), x.dim()-1]。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>beta（double）</td>
      <td>输入</td>
      <td>公式中的输入beta，SiTU门控部分的缩放系数。</td>
      <td>建议为非零值。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>linearBeta（double）</td>
      <td>输入</td>
      <td>公式中的输入linearBeta，up路径线性tanh的缩放系数。小于等于0时up路径不进行变换。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activateLeft（bool）</td>
      <td>输入</td>
      <td>公式中的输入activateLeft，表示切分x时gate是否为前半部分。</td>
      <td>true表示gate为前半、up为后半；false表示gate为后半、up为前半。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupListType（int64_t）</td>
      <td>输入</td>
      <td>公式中的输入groupListType，表示groupIndex的分组表示方式。</td>
      <td><ul><li>取值只能为0或1，传入其他值会返回参数校验失败。</li><li>0为cumsum模式，groupIndex为各组行数的累计和，总行数为groupIndex[-1]。</li><li>1为count模式，groupIndex为各组行数，总行数为sum(groupIndex)。</li><li>默认值为0。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradXOut（aclTensor*）</td>
      <td>输出</td>
      <td>公式中的输出grad_x，表示输入梯度。</td>
      <td><ul><li>支持空Tensor。</li><li>shape与x一致。</li><li>数据类型与x一致。</li><li>传入groupIndex时，行号 ≥ realBs的行不写（保持buffer原值）。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>gradWeightOut（aclTensor*）</td>
      <td>输出</td>
      <td>公式中的输出grad_weight，表示weight的梯度。</td>
      <td><ul><li>仅Ascend 950PR&950DT系列产品支持。</li><li>传入weight时shape与weight一致，元素个数等于x在dim轴之前各维度的乘积；未传入weight时须为空Tensor（shape [0]）。</li><li>传入groupIndex时，行号 ≥ realBs的行不写（保持buffer原值）。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
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
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
  <col style="width: 300px">
  <col style="width: 150px">
  <col style="width: 650px">
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
      <td>gradY、x、gradXOut存在空指针。</td>
    </tr>
    <tr>
      <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="9">161002</td>
      <td>gradY、x或gradXOut的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>gradY、x或gradXOut的shape维度不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>dim不在指定的取值范围内或x在dim对应维度上不能被2整除。</td>
    </tr>
    <tr>
      <td>gradY的维度数与x不一致，或gradY在dim对应维度上的大小不等于x的一半。</td>
    </tr>
    <tr>
      <td>gradY的数据类型与x不匹配。</td>
    </tr>
    <tr>
      <td>weightOptional的数据类型不是FLOAT，或其元素个数不等于x在dim轴之前各维度的乘积。</td>
    </tr>
    <tr>
      <td>groupIndexOptional的数据类型不是INT64、不是一维或为空Tensor。</td>
    </tr>
    <tr>
      <td>groupListType的取值不是0或1。</td>
    </tr>
    <tr>
      <td>传入weightOptional时gradWeightOut元素个数不等于x在dim轴之前各维度的乘积；或未传入weightOptional时gradWeightOut非空。</td>
    </tr>
  </tbody></table>

## aclnnSituGluGroupGrad

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
  <col style="width: 200px">
  <col style="width: 130px">
  <col style="width: 770px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnSituGluGroupGradGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性说明：aclnnSituGluGroupGrad默认确定性实现。
- 仅Ascend 950PR&950DT系列产品支持（weightOptional、groupIndexOptional为可选输入，gradWeightOut为必选输出）。
- 未传入weightOptional时，gradWeightOut必须为空Tensor（shape [0]），此时不产生weight梯度。
- 传入groupIndexOptional时，gradXOut与gradWeightOut中行号 ≥ realBs的行不会被写入，调用方如需严格零值请自行清零buffer。
- groupListType必须从[0, 1]中取值。
- groupIndexOptional的数值需要满足以下条件，否则无法保证输出是否符合预期：
  - 当groupListType为0时，groupIndexOptional必须为非负单调非递减数列，表示分组后每组大小的cumsum结果（累计和），最后一个值应小于等于行数。
  - 当groupListType为1时，groupIndexOptional必须为非负数列，表示分组后每组大小，数值的总和应小于等于行数。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

  ```cpp
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_situ_glu_group_grad.h"

  #define CHECK_RET(cond, return_expr) \
      do {                             \
          if (!(cond)) {               \
              return_expr;             \
          }                            \
      } while (0)

  #define LOG_PRINT(message, ...)         \
      do {                                \
          printf(message, ##__VA_ARGS__); \
      } while (0)

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
      // 固定写法，acl初始化
      auto ret = aclInit(nullptr);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
      ret = aclrtSetDevice(deviceId);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
      ret = aclrtCreateStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
      return 0;
  }

  template <typename T>
  int CreateAclTensor(
      const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
      aclTensor** tensor)
  {
      auto size = GetShapeSize(shape) * sizeof(T);
      // grad_weight在未传入weight时为空Tensor（shape [0]），size为0；为可移植性至少申请一个block
      auto allocSize = size > 0 ? size : static_cast<decltype(size)>(32);
      // 调用aclrtMalloc申请device侧内存
      auto ret = aclrtMalloc(deviceAddr, allocSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
      if (size > 0) {
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

  int main()
  {
      // 1. （固定写法）device/stream初始化，参考acl API手册
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. 构造输入与输出
      std::vector<int64_t> xShape = {2, 32};
      std::vector<int64_t> gradYShape = {2, 16};
      std::vector<int64_t> gradXShape = {2, 32};
      std::vector<int64_t> gradWeightShape = {0}; // 未传入weight，grad_weight为必选的空Tensor
      void* gradYDeviceAddr = nullptr;
      void* xDeviceAddr = nullptr;
      void* gradXDeviceAddr = nullptr;
      void* gradWeightDeviceAddr = nullptr;
      aclTensor* gradY = nullptr;
      aclTensor* x = nullptr;
      aclTensor* gradX = nullptr;
      aclTensor* gradWeight = nullptr;
      std::vector<float> gradYHostData = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                          16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
      std::vector<float> xHostData = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                      16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                      32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                                      48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
      std::vector<float> gradXHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

      int64_t dim = -1;
      double beta = 1.0;
      double linearBeta = 0.0;
      bool activateLeft = true;
      int64_t groupListType = 0;
      // 创建gradY aclTensor
      ret = CreateAclTensor(gradYHostData, gradYShape, &gradYDeviceAddr, aclDataType::ACL_FLOAT, &gradY);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建x aclTensor
      ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建gradX aclTensor
      ret = CreateAclTensor(gradXHostData, gradXShape, &gradXDeviceAddr, aclDataType::ACL_FLOAT, &gradX);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建gradWeight aclTensor（grad_weight为必选输出；此处未传入weight，故传入空Tensor）
      std::vector<float> gradWeightHostData;
      ret = CreateAclTensor(gradWeightHostData, gradWeightShape, &gradWeightDeviceAddr, aclDataType::ACL_FLOAT,
                            &gradWeight);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3. 调用CANN算子库API
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;
      // 调用aclnnSituGluGroupGrad第一段接口（weightOptional/groupIndexOptional传nullptr表示不使用；
      // gradWeightOut为必选输出，须传空Tensor）
      ret = aclnnSituGluGroupGradGetWorkspaceSize(gradY, x, nullptr, nullptr, dim, beta, linearBeta, activateLeft,
                                              groupListType, gradX, gradWeight, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSituGluGroupGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      void* workspaceAddr = nullptr;
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // 调用aclnnSituGluGroupGrad第二段接口
      ret = aclnnSituGluGroupGrad(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSituGluGroupGrad failed. ERROR: %d\n", ret); return ret);

      // 4. （固定写法）同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧
      auto size = GetShapeSize(gradXShape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(
          resultData.data(), resultData.size() * sizeof(resultData[0]), gradXDeviceAddr, size * sizeof(resultData[0]),
          ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
      }
      // 6. 释放aclTensor
      aclDestroyTensor(gradY);
      aclDestroyTensor(x);
      aclDestroyTensor(gradX);
      aclDestroyTensor(gradWeight);
      // 7. 释放device资源
      aclrtFree(gradYDeviceAddr);
      aclrtFree(xDeviceAddr);
      aclrtFree(gradXDeviceAddr);
      aclrtFree(gradWeightDeviceAddr);
      if (workspaceSize > 0) {
          aclrtFree(workspaceAddr);
      }
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
      return 0;
  }
  ```
