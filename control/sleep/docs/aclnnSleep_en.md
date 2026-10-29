# aclnnSleep

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T16:38:40.605Z -->

## Applicable Products
<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
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
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: Puts the NPU device to sleep for a specified number of clock cycles. The operator implements precise delay control through SIMT `clock()` busy-waiting, with semantics consistent with CUDA's `spin_kernel`/`torch.cuda._sleep`.

- Formula:

  For the given number of clock cycles, cycles, aclnnSleep performs the following computation:

  1. Obtain the current clock counter value as the start time:

     $$
     start = clock()
     $$

  2. Busy-wait until the number of elapsed cycles reaches the specified value:

     $$
     while(clock() - start < cycles): spin
     $$

  3. The actual sleep time (in seconds) depends on the clock frequency:

     $$
     t_{sleep} = \frac{cycles}{f_{clock}}
     $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnSleepGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor containing the operator computation flow, and then call the "aclnnSleep" API to perform the computation.

> **NOTE**: The `cycles` parameter is of the `aclIntArray*` type. Pass the value directly. The array contains one element, which is the number of sleep cycles.

```Cpp
aclnnStatus aclnnSleepGetWorkspaceSize(
  const aclIntArray* cycles,
  uint64_t*          workspaceSize,
  aclOpExecutor**    executor)
```

```Cpp
aclnnStatus aclnnSleep(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnSleepGetWorkspaceSize

- **Parameters**

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
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Shape</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>cycles (aclIntArray*)</td>
      <td>Input</td>
      <td>Number of clock cycles for the sleep. Due to the declared value dependency, the aclnn API layer converts it to aclIntArray* before passing it in.</td>
      <td><ul><li>Must be a positive integer (cycles &gt; 0).</li><li>Empty arrays are not supported.</li><li>Due to the AICore timeout limit, the maximum value of cycles is approximately 1.782e12 (Ascend 950PR&950DT products with a main frequency of 1.65 GHz, about 18 minutes). For details, see Constraints.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>[1]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the Device.</td>
      <td>The current implementation returns 0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
  <col style="width: 300px">
  <col style="width: 150px">
  <col style="width: 650px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The input data type is unsupported (not INT64).</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="2">561002</td>
      <td>cycles is an empty array, and data fails to be obtained in the tiling phase.</td>
    </tr>
    <tr>
      <td>The cycles parameter is non-positive (cycles &lt;= 0).</td>
    </tr>
  </tbody></table>

## aclnnSleep

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
  <col style="width: 200px">
  <col style="width: 130px">
  <col style="width: 770px">
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
      <td>Memory address of the workspace requested on the Device. The current implementation does not require a workspace, so nullptr can be passed in.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace requested on the Device, obtained by the first-phase API aclnnSleepGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor that contains the computation flow of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

aclnnStatus: return status code, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Determinism: aclnnSleep defaults to a deterministic implementation.
- Input restriction: cycles must be a positive integer (cycles &gt; 0); otherwise, ACLNN_ERR_INNER_TILING_ERROR (561002) is returned.
- cycles is passed in as aclIntArray*, and the array contains 1 element, which specifies the number of sleep cycles.
- Empty arrays are not supported: if cycles is an empty array, ACLNN_ERR_INNER_TILING_ERROR (561002) is returned.
- Inputs containing ±inf or nan are not supported: cycles is of the INT64 type and does not have inf/nan itself. If an upstream passes in a value converted from a floating-point type (such as PyTorch's `.to(torch.int64)`), `+inf` is converted to INT64_MAX; the cycles check passes, but an AICore timeout is triggered at runtime. `-inf` and `nan` are converted to non-positive values, which are rejected in the tiling phase and return ACLNN_ERR_INNER_TILING_ERROR (561002).
- AICore timeout limit: the default AICore execution timeout is 18 minutes. On Ascend 950PR&950DT products (core frequency: 1.65GHz), the maximum value of cycles is approximately 1.782e12 (= 1080s × 1.65GHz). If this value is exceeded, the operator may be forcibly terminated. Set cycles properly based on the actual use case. For a longer sleep time, adjust this limit by modifying the AICore timeout configuration through the `aclrtSetOpExecuteTimeOut` API.

## Example

The following sample code is for reference only. For details about how to compile and run the sample, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

<term>Ascend 950PR&950DT products</term>:

```Cpp

#include <iostream>
#include "acl/acl.h"
#include "aclnnop/aclnn_sleep.h"

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

int Init(int32_t deviceId, aclrtStream* stream) {
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  int64_t cyclesValue = 1000000;
  LOG_PRINT("Testing aclnnSleep with cycles = %ld\n", cyclesValue);

  aclIntArray* cycles = aclCreateIntArray(&cyclesValue, 1);
  CHECK_RET(cycles != nullptr, LOG_PRINT("aclCreateIntArray failed.\n"); return ACL_ERROR_BAD_ALLOC);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnSleepGetWorkspaceSize(cycles, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSleepGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnSleep(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSleep failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  LOG_PRINT("aclnnSleep test passed.\n");

  aclDestroyIntArray(cycles);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
