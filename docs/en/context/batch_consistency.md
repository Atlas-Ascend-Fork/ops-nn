# Batch Consistency

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T12:49:48.727Z -->

## Introduction

During application development, partial operators, in pursuit of higher performance, may produce computation result deviations for the same Token in use cases with different batch sizes or different positions within the same batch.
Currently, for partial operators, under the same running environment conditions, you can configure the computation process to use the Batch consistency algorithm, so that the computation results remain completely consistent no matter how the inputs are combined.
**Batch consistency algorithm**: For a given Token, regardless of its position in the batch, the batch size, or which other Tokens are batched with it, the output must be bitwise identical.

## Notes

- Generally, it is recommended not to enable Batch consistency computation, because enabling it for an operator causes some performance degradation in related operators, and therefore the single-run performance of the model may decrease. However, in use cases such as experiments, debugging, and regression testing, where the results of multiple runs must be identical to locate problems and experiment with algorithms, Batch consistency computation improves efficiency.

- The current configuration is a process-level switch configuration.

- **Version constraints**: TorchNPU 26.2.0 or later, and CANN 9.2.0 or later.

## Usage

Currently, CANN operators are mainly called through the aclnn API or the PyTorch API (torch_extension). Partial operator APIs implement Batch consistency by default, and partial operator APIs implement non-Batch consistency by default. For operators with non-Batch consistency implementation, some can be manually configured to enable Batch consistency.

- **Calling the aclnn API**

  In this use case, enable Batch consistency through the "Runtime Configuration>aclrtSetSysParamOpt" API (process-level) in *Runtime API*. Specifically, set `ACL_OPT_DETERMINISTIC=3` to enable Batch consistency computation.

- **Calling the PyTorch API**

  In this use case, enable Batch consistency computation through the `torch_npu.npu.set_deterministic_level` API in *TorchNPU Custom APIs*.

For operator APIs of different frameworks, the default implementation strategy of Batch consistency computation is subject to the description in the specific aclnn API documentation or PyTorch API documentation.
