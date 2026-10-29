# Data Formats

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T12:51:04.423Z -->

A data format describes the business semantics of the axes of a multidimensional tensor, indicating the physical layout format of data, such as 1D, 2D, 3D, 4D, and 5D. It is typically required in CNN (Convolutional Neural Networks) APIs to specify the exact format.

For the **full range of data formats** supported by aclTensor, see "Data Types and Operation APIs > aclFormat" in [*Runtime API*](https://hiascend.com/en/document/redirect/CannCommunityRuntimeApi).

For an introduction to the **data format layout principles**, see "Concept Principles and Terminology > Neural Networks and Operators > Data Layout Format" in [*Ascend C Operator Development Guide*](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/programug/Ascendcopdevg/docs/en/guide/getting_started/ascend_c_overview_and_learning_path.md).

## Usage Instructions

Currently, most operator APIs support the ND data format. For example, the aclnnAdd interface indicates that the supported data format is ND (i.e., the rule of low-dimension-first contiguous layout for multi-dimensional tensors). For aclnnConvolution, which is a CNN-type API, the input aclTensor is required to be set with a format that carries business semantics, rather than the ND format. Such operators need to understand the business semantics within the tensor during computation to perform the corresponding calculations. For instance, in 2D convolution, it is necessary to know the correspondence between the Batch dimension, Channel dimension, Height dimension, Width dimension, and the tensor dimensions.

>**NOTE:**
>
>- In the parameter description of a two-phase API, for brevity, **the original data format “ACL\_FORMAT\_XXXX_” is abbreviated as "_XXXX_"**.
>- Meaning of each dimension in a data format: N (Batch) indicates the batch size, H (Height) indicates the feature map height, W (Width) indicates the feature map width, C (Channels) indicates the feature map channels, D (Depth) indicates the feature map depth, and L (Length) indicates the feature map length.

## Common Data Formats

When creating an aclTensor using the **aclCreateTensor** interface, you need to set the data format according to the API service requirements. The currently **supported data formats** are:

ACL\_FORMAT\_ND, ACL\_FORMAT\_NCHW, ACL\_FORMAT\_NHWC, ACL\_FORMAT\_HWCN, ACL\_FORMAT\_NDHWC, ACL\_FORMAT\_NCDHW, ACL\_FORMAT\_NC, ACL\_FORMAT\_NCL.

For non-ND tensors, the dimension requirements of the tensor must be consistent with the format representation. For example:

- 5D tensor: ACL_FORMAT_NCDHW, ACL_FORMAT_NDHWC, or ACL_FORMAT_ND is required (if ND format support is not specified in the API parameter description, setting the ND format will cause the API to report a validation error).
- 4D tensor: ACL_FORMAT_NCHW, ACL_FORMAT_NHWC, ACL_FORMAT_HWCN, or ACL_FORMAT_ND is required.
- 3D tensor: ACL_FORMAT_NCL or ACL_FORMAT_ND is required.
- 2D tensor: ACL_FORMAT_NC or ACL_FORMAT_ND is required.
- Tensors of other dimensions: ACL_FORMAT_ND is required.

## Private Data Formats

In addition to the common data formats described above, there are other data formats, such as ACL_FORMAT_NC1HWC0, ACL_FORMAT_FRACTAL_Z, ACL_FORMAT_NC1HWC0_C04, ACL_FORMAT_FRACTAL_NZ, ACL_FORMAT_NDC1HWC0, and ACL_FORMAT_FRACTAL_Z_3D.

These formats are proprietary NPU formats, and most current aclnn APIs do not support them. If an individual API declares supported data formats, refer to the actual description of that API.
