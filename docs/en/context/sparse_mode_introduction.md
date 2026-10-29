# Introduction to Sparse Mode

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T12:54:08.069Z -->

In the field of large models, sparseMode (sparse mode) generally refers to the sparsity design of parameters or activations in model architectures or computational formulas, as opposed to DenseMode.

This section introduces commonly used sparseMode values and their corresponding scenario descriptions.

| sparseMode | Meaning                                  | Note               |
| ---------- | --------------------- | ------------------ |
| 0          | defaultMask mode.                     | -    |
| 1          | allMask mode.                         | -    |
| 2          | leftUpCausal mode.                    | -    |
| 3          | rightDownCausal mode.                 | -    |
| 4          | band mode.                            | -    |
| 5          | prefix uncompressed mode.                    | Not supported in varlen scenarios. |
| 6          | prefix compressed mode.                      | -       |
| 7          | varlen circumscribed scenario, rightDownCausal mode. | Supported only in varlen scenarios. |
| 8          | varlen circumscribed scenario, leftUpCausal mode.    | Supported only in varlen scenarios. |

The working principle of attenMask is to mask the values of the transposed matrix product of query (Q) and key (K) at positions where the Mask is True, as shown as follows:

![schematic diagram](../figures/QK_transpose_diagram.png)

The $QK^T$ matrix is occluded at positions where attenMask is True, with the effect shown as follows:

![schematic diagram](../figures/masked_QK_diagram.png)

## sparseMode=0

When sparseMode is 0, it represents the defaultMask mode.

- No mask passed: If attenMask is not passed, no mask operation is performed. attenMask takes the value None, and the values of preTokens and nextTokens are ignored. The masked $QK^T$ matrix is shown as follows:

  ![schematic diagram](../figures/sparsemode_0_masked_matrix.png)

- nextTokens is set to 0 and preTokens is greater than or equal to Sq, indicating a causal sparse scenario. attenMask should be passed as a lower triangular matrix. In this case, the portion between preTokens and nextTokens needs to be computed. The masked $QK^T$ matrix is shown as follows:

  ![schematic diagram](../figures/sparsemode_0_masked_matrix_1.png)

  attenMask should be passed as a lower triangular matrix, as shown below:

  ![schematic diagram](../figures/attenmask_lower_triangle.png)

- When preTokens is less than Sq, nextTokens is less than Skv, and both are greater than or equal to 0, this indicates a band scenario, where the portion between preTokens and nextTokens needs to be computed. The masked $QK^T$ matrix is shown as follows:

  ![schematic diagram](../figures/sparsemode_0_masked_matrix_2.png)

  attenMask should be passed as a band-shaped matrix, as shown below:

  ![schematic diagram](../figures/attenmask_band_matrix.png)

- When nextTokens is negative, taking preTokens=9 and nextTokens=-3 as an example, the portion between preTokens and nextTokens needs to be computed. The masked $QK^T$ is shown as follows:

  **NOTE: When nextTokens is negative, the value of preTokens must be greater than or equal to the absolute value of nextTokens, and the absolute value of nextTokens must be less than Skv.**

  ![schematic diagram](../figures/sparsemode_0_masked_matrix_3.png)

- When preTokens is a negative number, taking nextTokens=7 and preTokens=-3 as an example, the portion between preTokens and nextTokens needs to be computed. The masked $QK^T$ is shown as follows:

  **NOTE: When preTokens is a negative number, the value of nextTokens must be greater than or equal to the absolute value of preTokens, and the absolute value of preTokens must be less than Sq.**

  ![schematic diagram](../figures/sparsemode_0_masked_matrix_4.png)

## sparseMode=1

When sparseMode is 1, it represents allMask, meaning the complete attenMask matrix is passed in.

In this scenario, the values of nextTokens and preTokens are ignored. The masked $QK^T$ matrix diagram is shown as follows:

![schematic diagram](../figures/sparsemode_1_masked_matrix.png)

## sparseMode=2

When sparseMode is 2, it represents the mask of the leftUpCausal mode, corresponding to the lower triangular scenario partitioned by the upper-left vertex (the parameter start is the upper-left corner).

In this scenario, the values of preTokens and nextTokens are ignored. The masked $QK^T$ matrix diagram is shown as follows:

![schematic diagram](../figures/sparsemode_2_masked_matrix.png)

The passed attenMask is an optimized compressed lower triangular matrix (2048\*2048). The compressed lower triangular matrix diagram is shown as follows (the same below):

![schematic diagram](../figures/attenmask_compressed_lower_triangle.png)

## sparseMode=3

When sparseMode is 3, it represents the mask of the rightDownCausal mode, corresponding to the lower triangular scenario partitioned by the bottom-right vertex (the parameter start is the bottom-right corner).

In this scenario, the values of preTokens and nextTokens are ignored. attenMask is an optimized compressed lower triangular matrix (2048*2048). The masked $QK^T$ matrix diagram is shown as follows:

![schematic diagram](../figures/sparsemode_3_masked_matrix.png)

## sparseMode=4

When sparseMode is 4, it represents the band scenario, where the portion between preTokens and nextTokens is computed. The parameter start is at the lower right corner, and there must be an intersection between preTokens and nextTokens. attenMask is an optimized compressed lower triangular matrix (2048*2048). The masked $QK^T$ matrix is shown as follows:

![Schematic diagram](../figures/sparsemode_4_masked_matrix.png)

## sparseMode=5

When sparseMode is 5, it represents the prefix non-compressed scenario, where a matrix of length Sq and width N is added to the left on top of rightDownCausal. The value of N is obtained from the optional input prefix. For example, in the figure below, which shows a batch=2 scenario, the prefix input array is [4,5]. The N value for each batch axis can differ, and the parameter start is at the upper-left corner.

In this scenario, the values of preTokens and nextTokens are ignored. The attenMask matrix data format must be BNSS or B1SS. The Masked $QK^T$ matrix diagram is shown as follows:

![schematic diagram](../figures/sparsemode_5_masked_matrix.png)

The matrix diagram that attenMask should pass in is shown as follows:

![schematic diagram](../figures/attenmask_matrix.png)

## sparseMode=6

When sparseMode is 6, it represents the prefix compression scenario. In this scenario, attenMask is an optimized compressed lower triangular + rectangular matrix (3072\*2048): the upper part is a [2048, 2048] lower triangular matrix, and the lower part is a [1024, 2048] rectangular matrix. The left half of the rectangular matrix is all zeros, and the right half is all ones. The attenMask to be passed in is shown in the matrix diagram below. In this scenario, the values of preTokens and nextTokens are ignored.

![schematic diagram](../figures/sparsemode_6_masked_matrix.png)

## sparseMode=7

When sparseMode is set to 7, it indicates that the varlen mode is used and the long sequence is cut externally (that is, the long sequence is used to cut the query sequence length of multiple cards in the model script). You need to ensure that sparseMode 3 is used before the external cutting. In this mode, you need to set preTokens and nextTokens (the start point is the lower right vertex) and ensure that the parameters are correct. Otherwise, accuracy problems may occur.

The masked $QK^T$ matrix is shown as follows: in the second batch, the query is sliced while the key and value are not sliced. The 4×6 mask matrix is split into 2×6 and 2×6 masks, which are computed on card 1 and card 2 respectively:

- The last mask block on card 1 is a band-type mask, with preTokens=6 (ensuring it is greater than or equal to the last Skv), nextTokens=-2, actual_seq_qlen should be passed as {3,5}, and actual_seq_kvlen should be passed as {3,9}.
- The mask type on card 2 remains unchanged after slicing, with sparseMode=3, actual_seq_qlen should be passed as {2,7,11}, and actual_seq_kvlen should be passed as {6,11,15}.

![schematic diagram](../figures/sparsemode_7_masked_matrix.png)

**Description**

- sparseMode=7: band indicates the sparse type of the last non-empty tensor batch. If there is only one batch, the user must configure parameters according to the requirements of the band mode. When sparseMode=7, the user needs to input a 2048x2048 lower triangular mask as the input of this fusion operator.
- The parameters of the band mode deduced by circumscribing based on sparseMode=3 must meet the following conditions:
  - preTokens >= last_Skv.
  - last_Sq-last_Skv <= nextTokens <= 0.
  - The optional input pse is not supported in the current mode.
- Batches in non-band mode must satisfy: Sq <= Skv.

## sparseMode=8

When sparseMode is 8, it indicates a varlen and long-sequence multi-card splitting scenario. Users must ensure that the scenario before splitting uses sparseMode 2. In this mode, users need to set preTokens and nextTokens (with the lower right vertex as the starting point) and must ensure the parameters are correct; otherwise, precision issues may occur.

The masked $QK^T$ matrix diagram is shown as follows. In the second batch, the query is split while the key and value are not split. The 5×4 mask matrix is split into 2×4 and 3×4 masks, which are computed on card 1 and card 2 respectively:

- After splitting, the mask type on card 1 remains unchanged, with sparseMode set to 2. actual_seq_qlen should be passed as {3,5}, and actual_seq_kvlen should be passed as {3,7}.
- The first mask on card 2 is a band-type mask, configured with preTokens=4 (ensuring the value is greater than or equal to the first Skv) and nextTokens=1. actual_seq_qlen should be passed as {3,8,12}, and actual_seq_kvlen should be passed as {4,9,13}.

![Schematic Diagram](../figures/sparsemode_8_masked_matrix.png)

**NOTE**

- When sparseMode=8, band indicates the sparse type of the Batch corresponding to the first non-empty tensor. If there is only one batch, configure the parameters according to the requirements of the band mode. When sparseMode=8, a 2048x2048 lower triangular mask must be provided as the input to this fusion operator.

- The parameters of the band-mode sparse generated by clipping based on sparseMode=2 must meet the following conditions:
  - preTokens >= first_Skv.
  - nextTokens >= first_Sq - first_Skv, configure based on the actual situation.
  - The optional input pse is not supported in the current mode.
