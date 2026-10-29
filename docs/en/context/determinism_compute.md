# Deterministic Computation

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T12:52:58.620Z -->

## Introduction

During application development, due to differences such as CANN versions or NPU models, results of multiple runs of the same operator may not be exactly the same. Even under the same conditions with the same random seed, results of multiple NPU executions may differ. This difference usually arises from asynchronous multi-threaded execution in the operator implementation, which changes the accumulation order of floating-point numbers.

In some special use cases (such as experiments, debugging, and regression tests), an operator is required to produce the same output over multiple runs, to locate problems or analyze algorithms. This solution is commonly known as deterministic computation.

## Notes

- Generally, do not enable determinism, because deterministic computation of the same operator usually executes more slowly than non-deterministic computation and may degrade the performance of a single model run. However, in use cases such as experiments, debugging, and regression tests, where consistent results across multiple runs are required to locate problems or analyze algorithms, deterministic computation improves efficiency.

- Thread notes: The deterministic state can be set only once in a thread. If it is set multiple times, the last valid setting prevails. A valid setting means that an operator task is actually dispatched after the deterministic state is set. If the state is set but no operator task is dispatched, the deterministic variable is enabled but not delivered to the operator, and the operator will not execute deterministically. Setting determinism multiple times in one thread is not recommended currently. This issue exists whether the binary is enabled or disabled, and will be resolved in later versions.

## Usage

Currently, CANN operators are mainly called through the aclnn API or PyTorch API (torch_extension). Some operator APIs default to a deterministic implementation, and some operator APIs default to a non-deterministic implementation. For operators that are non-deterministic by default, deterministic computation can be manually configured for some of them.

- **Calling the aclnn API**

  In this use case, enable determinism through the "Runtime Configuration > aclrtSetSysParamOpt" API (process-level) in [*Runtime API*](https://hiascend.com/en/document/redirect/CannCommunityRuntimeApi). Specifically, set `ACL_OPT_DETERMINISTIC=1` to enable determinism.

- **Calling the PyTorch API**

  In this use case, enable determinism through the native PyTorch switch (`torch.use_deterministic_algorithms`).

For operator APIs of different frameworks, the default implementation strategy for deterministic computation is subject to the specific aclnn API documentation or PyTorch API documentation.
