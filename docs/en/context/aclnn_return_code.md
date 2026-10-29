# aclnn Return Codes

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T12:50:43.314Z -->

When calling the aclnn API, the common API return codes are shown in [Table 1](#table1).
For exception status code values, obtain the exception information using the aclGetRecentErrMsg API (see *Runtime API*). Troubleshoot the problem based on the error message or contact technical support.

**Table 1**  Return Status Codes <a id="table1"></a>

| Status Code Name | Status Code Value | Status Code Description |
| ----- | ----- | ------ |
| ACLNN_SUCCESS | 0 | Succeeded. |
| ACLNN_ERR_PARAM_NULLPTR | 161001 | Parameter validation error: the parameters contain an invalid nullptr. |
| ACLNN_ERR_PARAM_INVALID | 161002 | Parameter validation error: for example, the data types of the two inputs do not satisfy the input type deduction relationship. |
| ACLNN_ERR_RUNTIME_ERROR | 361001 | An error occurs when the API calls the npu runtime API internally. |
| ACLNN_ERR_INNER_XXX | 561xxx | An internal exception occurs in the API. |

For more information about status codes of the ACLNN_ERR_INNER_XXX type, see [Table 2](#table2).

**Table 2**  Exception Status Codes <a id="table2"></a>

| Status Code Name | Status Code Value | Status Code Description |
| ----- | ----- | ------ |
| ACLNN_ERR_INNER | 561000 | Internal Exception: An internal exception occurs in the API. |
| ACLNN_ERR_INNER_INFERSHAPE_ERROR | 561001 | Internal Exception: An error occurs when the API infers the output shape internally. |
| ACLNN_ERR_INNER_TILING_ERROR | 561002 | Internal Exception: An exception occurs when the API performs tiling for the npu kernel internally. |
| ACLNN_ERR_INNER_FIND_KERNEL_ERROR | 561003 | Internal Exception: The API fails to find the npu kernel internally (possibly because the operator binary package is not installed). |
| ACLNN_ERR_INNER_CREATE_EXECUTOR | 561101 | Internal Exception: The API fails to create aclOpExecutor internally (possibly because of an operating system exception). |
| ACLNN_ERR_INNER_NOT_TRANS_EXECUTOR | 561102 | Internal Exception: The API does not call uniqueExecutor ReleaseTo internally. |
| ACLNN_ERR_INNER_NULLPTR | 561103 | Internal Exception: A nullptr exception occurs in the aclnn API. |
| ACLNN_ERR_INNER_WRONG_ATTR_INFO_SIZE | 561104 | Internal Exception: The number of operator attributes is abnormal in the aclnn API. |
| ACLNN_ERR_INNER_KEY_CONFILICT (Deprecated) | 561105 | **Deprecated. Use the latest ACLNN_ERR_INNER_KEY_CONFLICT instead.** |
| ACLNN_ERR_INNER_KEY_CONFLICT | 561105 | Internal Exception: A hash key conflict occurs during operator kernel matching in the aclnn API. |
| ACLNN_ERR_INNER_INVALID_IMPL_MODE | 561106 | Internal Exception: The implementation mode parameter of the operator is incorrect in the aclnn API. |
| ACLNN_ERR_INNER_OPP_PATH_NOT_FOUND | 561107 | Internal Exception: The required environment variable ASCEND_OPP_PATH is not detected in the aclnn API. |
| ACLNN_ERR_INNER_LOAD_JSON_FAILED | 561108 | Internal Exception: The aclnn API fails to load the operator information json file in the operator kernel library. |
| ACLNN_ERR_INNER_JSON_VALUE_NOT_FOUND | 561109 | Internal Exception: The aclnn API fails to load a field of the operator information json file in the operator kernel library. |
| ACLNN_ERR_INNER_JSON_FORMAT_INVALID | 561110 | Internal Exception: The format of the operator information json file in the operator kernel library is set to an invalid value in the aclnn API. |
| ACLNN_ERR_INNER_JSON_DTYPE_INVALID | 561111 | Internal Exception: The dtype of the operator information json file in the operator kernel library is set to an invalid value in the aclnn API. |
| ACLNN_ERR_INNER_OPP_KERNEL_PKG_NOT_FOUND | 561112 | Internal Exception: The binary kernel library of the operator is not loaded in the aclnn API. |
| ACLNN_ERR_INNER_OP_FILE_INVALID | 561113 | Internal Exception: An exception occurs in the aclnn API when loading the fields of the operator json file. |
| ACLNN_ERR_INNER_ATTR_NUM_OUT_OF_BOUND | 561114 | Internal Exception: In the aclnn API, the number of operator attributes is inconsistent with the operator information json and exceeds the number of attrs specified in the json. |
| ACLNN_ERR_INNER_ATTR_LEN_NOT_ENOUGH | 561115 | Internal Exception: In the aclnn API, the number of operator attributes is inconsistent with the operator information json and is less than the number of attrs specified in the json. |
| ACLNN_ERR_INNER_INPUT_NUM_IN_JSON_TOO_LARGE | 561116 | Internal Exception: The number of operator inputs exceeds the limit of 32 in the aclnn API. |
| ACLNN_ERR_INNER_INPUT_JSON_IS_NULL | 561117 | Internal Exception: Description information is missing in the operator information json file in the aclnn API. |
| ACLNN_ERR_INNER_STATIC_WORKSPACE_INVALID | 561118 | Internal Exception: An exception occurs in the aclnn API when parsing the workspace information in the static binary json file. |
| ACLNN_ERR_INNER_STATIC_BLOCK_DIM_INVALID | 561119 | Internal Exception: An exception occurs in the aclnn API when parsing the core usage information in the static binary json file. |
