# Deduction Relationship

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T13:10:18.389Z -->

## Tensor-Tensor Deduction Relationship

### Deduction Rules

When the **input aclTensor data types** of an API (such as aclnnAdd, aclnnMul, etc.) are inconsistent, the API internally deduces a data type and converts the input data to that data type for computation.

For the data types supported by aclTensor, see [data types](data_type.md). Some of these types follow the derivation rules below, and the deduction principle is similar to [Type Promotion](https://pytorch.org/docs/stable/tensor_attributes.html#type-promotion-doc) in PyTorch.

The type deduction rules are as follows:

> **NOTE**
>
>- For ease of description, the data types used in the table are in **abbreviated form**, with the following meanings: ACL\_FLOAT\(f32\), ACL\_FLOAT16\(f16\), ACL\_DOUBLE\(f64\), ACL\_BF16\(bf16\), ACL\_INT8\(s8\), ACL\_UINT8\(u8\), ACL\_INT16\(s16\), ACL\_UINT16\(u16\), ACL\_INT32\(s32\), ACL\_UINT32\(u32\), ACL\_INT64\(s64\), ACL\_UINT64\(u64\), ACL\_BOOL\(bool\), ACL\_COMPLEX32\(c32\), ACL\_COMPLEX64\(c64\), ACL\_COMPLEX128\(c128\).
>- The table header and the leftmost column represent the two input data types to be deduced, and the corresponding cell in the table indicates the deduced data type.
>- A cross (×) in the table indicates that the two types cannot be used for deduction computation.

**Table 1**  Data type deduction relationship

| Data Type | f32  | f16  | f64  | bf16 |  s8  |  u8  | s16  | u16  | s32  | u32  | s64  | u64  | bool | c32  | c64  | c128 |
| :------: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| **f32**  | f32  | f32  | f64  | f32  | f32  | f32  | f32  |  ×   | f32  |  ×   | f32  |  ×   | f32  | c64  | c64  | c128 |
| **f16**  | f32  | f16  | f64  | f32  | f16  | f16  | f16  |  ×   | f16  |  ×   | f16  |  ×   | f16  | c32  | c64  | c128 |
| **f64**  | f64  | f64  | f64  | f64  | f64  | f64  | f64  |  ×   | f64  |  ×   | f64  |  ×   | f64  | c128 | c128 | c128 |
| **bf16** | f32  | f32  | f64  | bf16 | bf16 | bf16 | bf16 |  ×   | bf16 |  ×   | bf16 |  ×   | bf16 | c32  | c64  | c128 |
|  **s8**  | f32  | f16  | f64  | bf16 |  s8  | s16  | s16  |  ×   | s32  |  ×   | s64  |  ×   |  s8  | c32  | c64  | c128 |
|  **u8**  | f32  | f16  | f64  | bf16 | s16  |  u8  | s16  |  ×   | s32  |  ×   | s64  |  ×   |  u8  | c32  | c64  | c128 |
| **s16**  | f32  | f16  | f64  | bf16 | s16  | s16  | s16  |  ×   | s32  |  ×   | s64  |  ×   | s16  | c32  | c64  | c128 |
| **u16**  |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   | u16  |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   |
| **s32**  | f32  | f16  | f64  | bf16 | s32  | s32  | s32  |  ×   | s32  |  ×   | s64  |  ×   | s32  | c32  | c64  | c128 |
| **u32**  |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   | u32  |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   |
| **s64**  | f32  | f16  | f64  | bf16 | s64  | s64  | s64  |  ×   | s64  |  ×   | s64  |  ×   | s64  | c32  | c64  | c128 |
| **u64**  |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   | u64  |  ×   |  ×   |  ×   |  ×   |
| **bool** | f32  | f16  | f64  | bf16 |  s8  |  u8  | s16  |  ×   | s32  |  ×   | s64  |  ×   | bool | c32  | c64  | c128 |
| **c32**  | c64  | c32  | c128 | c32  | c32  | c32  | c32  |  ×   | c32  |  ×   | c32  |  ×   | c32  | c32  | c64  | c128 |
| **c64**  | c64  | c64  | c128 | c64  | c64  | c64  | c64  |  ×   | c64  |  ×   | c64  |  ×   | c64  | c64  | c64  | c128 |
| **c128** | c128 | c128 | c128 | c128 | c128 | c128 | c128 |  ×   | c128 |  ×   | c128 |  ×   | c128 | c128 | c128 | c128 |

### Deduction Examples

- When calling the aclnnAdd API, if the input parameters have inconsistent data types — one being float16 and the other float32 — the API internally converts the float16 data type to float32 and then performs the computation.
- When calling the aclnnAdd API, if the input parameters have inconsistent data types — one being float32 and the other bool — the API internally converts the bool data type to float32 and then performs the computation.

## Deduction Relationship between Tensor and Scalar

### Deduction Rules

When the **input Tensor data type** and the **input Scalar data type** of an API (such as aclnnAdds, aclnnMuls, etc.) are inconsistent, the API internally deduces a data type and converts the input data to that data type for computation.

For the data types supported by aclTensor, see [data types](data_type.md). Some of these types follow the derivation rules below, and the deduction principle is similar to [Type Promotion](https://pytorch.org/docs/stable/tensor_attributes.html#type-promotion-doc) in PyTorch.

The type deduction rules are as follows:

> **NOTE**
>
>- For ease of description, the data types used in the table are abbreviated. Their meanings are: ACL\_FLOAT\(f32\), ACL\_FLOAT16\(f16\), ACL\_DOUBLE\(f64\), ACL\_BF16\(bf16\), ACL\_INT8\(s8\), ACL\_UINT8\(u8\), ACL\_INT16\(s16\), ACL\_UINT16\(u16\), ACL\_INT32\(s32\), ACL\_UINT32\(u32\), ACL\_INT64\(s64\), ACL\_UINT64\(u64\), ACL\_BOOL\(bool\), ACL\_COMPLEX32\(c32\), ACL\_COMPLEX64\(c64\), ACL\_COMPLEX128\(c128\).
>- In the table, the header row represents the input Tensor data type to be deduced, the leftmost column represents the input Scalar data type to be deduced, and the corresponding cell in the table indicates the deduced data type.
>- A cross mark (×) in the table indicates that the two types cannot undergo deduction computation.

**Table 2**  Data type deduction relationship

| Data Type | f32  | f16  | f64  | bf16 |  s8  |  u8  | s16  | u16  | s32  | u32  | s64  | u64  | bool | c32  | c64  | c128 |
| :------: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| **f32**  | f32  | f16  | f64  | bf16 | f32  | f32  | f32  |  ×   | f32  |  ×   | f32  |  ×   | f32  | c32  | c64  | c128 |
| **f16**  | f32  | f16  | f64  | bf16 | f32  | f32  | f32  |  ×   | f32  |  ×   | f32  |  ×   | f32  | c32  | c64  | c128 |
| **f64**  | f32  | f16  | f64  | bf16 | f32  | f32  | f32  |  ×   | f32  |  ×   | f32  |  ×   | f32  | c128 | c128 | c128 |
| **bf16** | f32  | f16  | f64  | bf16 | f32  | f32  | f32  |  ×   | f32  |  ×   | f32  |  ×   | f32  | c32  | c64  | c128 |
|  **s8**  | f32  | f16  | f64  | bf16 |  s8  |  u8  | s16  | u16  | s32  | u32  | s64  | u64  |  s8  | c32  | c64  | c128 |
|  **u8**  | f32  | f16  | f64  | bf16 |  s8  |  u8  | s16  | u16  | s32  | u32  | s64  | u64  |  u8  | c32  | c64  | c128 |
| **s16**  | f32  | f16  | f64  | bf16 |  s8  |  u8  | s16  | u16  | s32  | u32  | s64  | u64  | s16  | c32  | c64  | c128 |
| **u16**  | f32  | f16  | f64  | bf16 |  s8  |  u8  | s16  | u16  | s32  | u32  | s64  | u64  |  ×   | c32  | c64  | c128 |
| **s32**  | f32  | f16  | f64  | bf16 |  s8  |  u8  | s16  | u16  | s32  | u32  | s64  | u64  | s32  | c32  | c64  | c128 |
| **u32**  | f32  | f16  | f64  | bf16 |  s8  |  u8  | s16  | u16  | s32  | u32  | s64  | u64  |  ×   | c32  | c64  | c128 |
| **s64**  | f32  | f16  | f64  | bf16 |  s8  |  u8  | s16  | u16  | s32  | u32  | s64  | u64  | s64  | c32  | c64  | c128 |
| **u64**  | f32  | f16  | f64  | bf16 |  s8  |  u8  | s16  | u16  | s32  | u32  | s64  | u64  |  ×   | c32  | c64  | c128 |
| **bool** | f32  | f16  | f64  | bf16 |  s8  |  u8  | s16  | u16  | s32  | u32  | s64  | u64  | bool | c32  | c64  | c128 |
| **c32**  | c64  | c32  | c128 | c64  | c64  | c64  | c64  | c64  | c64  | c64  | c64  | c64  | c64  | c32  | c64  | c128 |
| **c64**  | c64  | c32  | c128 | c64  | c64  | c64  | c64  | c64  | c64  | c64  | c64  | c64  | c64  | c32  | c64  | c128 |
| **c128** | c64  | c32  | c128 | c64  | c64  | c64  | c64  | c64  | c64  | c64  | c64  | c64  | c64  | c32  | c64  | c128 |

### Deduction Examples

- If the data type of the input Tensor is float16 and the data type of the input Scalar is float32, the API internally converts the float32 data type of the input Scalar to float16 before performing the computation.
- If the data type of the input Tensor is bool and the data type of the input Scalar is float32, the API internally converts the bool data type of the input Tensor to float32 before performing the computation.
