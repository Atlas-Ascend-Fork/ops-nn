# Two-Phase API

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T12:54:40.950Z -->

When calling an operator API based on the single-operator API execution mode, the process typically follows a "two-phase" pattern, as shown below:

```Cpp
aclnnStatus aclxxXxxGetWorkspaceSize(const aclTensor *src, ..., aclTensor *out, ..., uint64_t *workspaceSize, aclOpExecutor **executor);
aclnnStatus aclxxXxx(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
```

You must first call the first-phase API aclxxXxxGetWorkspaceSize to calculate the amount of workspace memory required for this API call. After obtaining the required workspaceSize, allocate NPU memory based on workspaceSize, and then call the second-phase API aclxxXxx to perform the computation.

Here, "aclxx" represents the operator API prefix, such as aclnn, and "Xxx" represents the corresponding operator type, such as the Add operator.

> NOTE:
>
> - workspace is the temporary memory required by the operator to complete computation on the NPU, excluding input/output. workspaceSize indicates the size of the temporary memory.
> - The second-phase API aclxxXxx(...) cannot be called repeatedly. The following call sequence causes an exception:
>
>   ```Cpp
>   aclxxXxxGetWorkspaceSize(...)
>   aclxxXxx(...)
>   aclxxXxx(...)
>   ```
