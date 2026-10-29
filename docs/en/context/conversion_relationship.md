# Conversion Relationships

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T12:49:32.966Z -->

When the **output aclTensor data type** of an API (such as aclnnAdd, aclnnMul, etc.) differs from the data type deduced from the **input data type** after computation type deduction, the API internally converts the computation result to the data type corresponding to the output type.

Data type conversion must satisfy the following rules. Conversions that do not meet these rules are not supported, and a parameter validation failure will occur when the API is called.

- Floating-point types: ACL_FLOAT16, ACL_FLOAT, ACL_DOUBLE, ACL_BF16.
- Integer types: ACL_INT8, ACL_UINT8, ACL_INT16, ACL_UINT16, ACL_INT32, ACL_UINT32, ACL_INT64, ACL_UINT64.
- Complex types: ACL_COMPLEX64, ACL_COMPLEX128.
- Integer types can be converted among themselves, and conversion to floating-point and complex types is also supported.
- Floating-point types can be converted among themselves, and conversion to complex types is also supported.
- Complex types can be converted among themselves.
- BOOL supports conversion to integer, floating-point, and complex types.

Conversions in scenarios other than those described above are not supported.
