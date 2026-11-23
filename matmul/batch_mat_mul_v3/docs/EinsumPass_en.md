# EinsumPass

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-11-11T12:41:48.322Z pushedAt=2026-11-16T06:28:26.366Z nl=2 -->

## Description

Constructs the corresponding computation graph based on the expression of the Einsum operator. In the computation graph shown below, the dashed nodes are operators that may be added based on the actual expression.

- The **static shape scenario** includes the following two cases:

  Static shape, single input:

  ![EinsumPass_1](../../../docs/en/figures/EinsumPass_1.png)

  After:

  ![EinsumPass_2](../../../docs/en/figures/EinsumPass_2.png)

  Static shape, dual-input case (depending on the specific expression, the order of the two inputs of the MatMul-type operator may be swapped):

  ![EinsumPass_3](../../../docs/en/figures/EinsumPass_3.png)

  After:

  ![EinsumPass_4](../../../docs/en/figures/EinsumPass_4.png)

- In the **dynamic shape** scenario, the graph to which the Einsum operator belongs before fusion is always a dual-input case, as shown in the following figure:

  ![EinsumPass_5](../../../docs/en/figures/EinsumPass_5.png)

  The dynamic shape scenario currently supports 28 expression forms. The graphs after fusion are as follows:

 1. The Einsum expression form is "abc,cde-\>abde":

    ![EinsumPass_6](../../../docs/en/figures/EinsumPass_6.png)

 2. The Einsum expression form is "abcd,aecd-\>aceb" (the two inputs of the BatchMatMul operator are swapped):

    ![EinsumPass_7](../../../docs/en/figures/EinsumPass_7.png)

 3. The Einsum expression form is "abcd,adbe-\>acbe":

    ![EinsumPass_8](../../../docs/en/figures/EinsumPass_8.png)

 4. The Einsum expression form is "abcd,cde-\>abe":

    ![EinsumPass_9](../../../docs/en/figures/EinsumPass_9.png)

 5. The Einsum expression form is "abc,cd-\>abd":

    ![EinsumPass_10](../../../docs/en/figures/EinsumPass_10.png)

 6. The Einsum expression form is "abc,dc-\>abd":

    ![EinsumPass_11](../../../docs/en/figures/EinsumPass_11.png)

 7. The Einsum expression form is "abc,abd-\>dc" (the two inputs of the MatMulV2 operator are swapped):

    ![EinsumPass_12](../../../docs/en/figures/EinsumPass_12.png)

 8. The Einsum expression form is "abc,dec-\>abde":

    ![EinsumPass_13](../../../docs/en/figures/EinsumPass_13.png)

 9. The Einsum expression form is "abc,abde-\>dec":

    When TensorB has a static shape: ![EinsumPass_14](../../../docs/en/figures/EinsumPass_14.png)

    When TensorB has a dynamic shape:

    ![EinsumPass_15](../../../docs/en/figures/EinsumPass_15.png)

10. The Einsum expression form is "abcd,aecd-\>acbe":

    ![EinsumPass_16](../../../docs/en/figures/EinsumPass_16.png)

11. The Einsum expression form is "abcd,acbe-\>aecd":

    ![EinsumPass_17](../../../docs/en/figures/EinsumPass_17.png)

12. The Einsum expression form is "abcd,ecd-\>abe":

    ![EinsumPass_18](../../../docs/en/figures/EinsumPass_18.png)

13. The Einsum expression form is as follows "abcd,abe-\>ecd":

    When TensorA has a static shape:

    ![EinsumPass_19](../../../docs/en/figures/EinsumPass_19.png)

    When TensorA has a dynamic shape:

    ![EinsumPass_20](../../../docs/en/figures/EinsumPass_20.png)

14. The Einsum expression form is as follows "abcd,acbe-\>adbe":

    ![EinsumPass_21](../../../docs/en/figures/EinsumPass_21.png)

    Based on whether the input shape is dynamic, the operators contained in the **Reshape dym/stc** node in the above diagram are as follows:

    ![EinsumPass_22](../../../docs/en/figures/EinsumPass_22.png)

    Based on whether the input shape is dynamic, the operators contained in the **Transpose dym/stc** node in the above diagram are as follows:

    ![EinsumPass_23](../../../docs/en/figures/EinsumPass_23.png)

15. Einsum expression form "abcd, abde-\>abce"":

    ![EinsumPass_24](../../../docs/en/figures/EinsumPass_24.png)

16. Einsum expression form "abcd, abce-\>abde"":

    ![EinsumPass_25](../../../docs/en/figures/EinsumPass_25.png)

    Based on whether the input shape is dynamic, the operators contained in the **Transpose dym/stc** node in the above diagram are as follows:

    ![EinsumPass_26](../../../docs/en/figures/EinsumPass_26.png)

17. Einsum expression form "abcd, aebd-\>aebc"":

    ![EinsumPass_27](../../../docs/en/figures/EinsumPass_27.png)

    Based on whether the input shape is dynamic, the operators contained in the **Reshape** node in the above diagram are as follows:

    ![EinsumPass_28](../../../docs/en/figures/EinsumPass_28.png)

18. The Einsum expression takes the form of "abcd, abce-\>acde"":

    ![EinsumPass_29](../../../docs/en/figures/EinsumPass_29.png)

    Based on whether the input shape is dynamic, the operators contained in the **Reshape** node in the above diagram are as follows:

    ![EinsumPass_30](../../../docs/en/figures/EinsumPass_30.png)

19. The Einsum expression takes the form of "abc, abd-\>acd"":

    ![EinsumPass_31](../../../docs/en/figures/EinsumPass_31.png)

    Based on whether the input shape is dynamic, the operators contained in the **Transpose dym/stc** node in the above diagram are as follows:

    ![EinsumPass_32](../../../docs/en/figures/EinsumPass_32.png)

20. The Einsum expression takes the form of "ab, cb-\>ac"":

    ![EinsumPass_33](../../../docs/en/figures/EinsumPass_33.png)

    Based on whether the input shape is dynamic, the **Transpose dym/stc** node in the above diagram contains the following operators:

    ![EinsumPass_34](../../../docs/en/figures/EinsumPass_34.png)

21. The Einsum expression is in the form of "abc, acd-\>abd":

    ![EinsumPass_35](../../../docs/en/figures/EinsumPass_35.png)

22. The Einsum expression is in the form of "abc, adc-\>abd":

    ![EinsumPass_36](../../../docs/en/figures/EinsumPass_36.png)

    Based on whether the input shape is dynamic, the **Transpose dym/stc** node in the above diagram contains the following operators:

    ![EinsumPass_37](../../../docs/en/figures/EinsumPass_37.png)

23. The Einsum expression is in the form of "abcd, ced-\>abce":

    ![EinsumPass_38](../../../docs/en/figures/EinsumPass_38.png)

    Based on whether the input shape is dynamic, the **Reshape** node in the above diagram contains the following operators:

    ![EinsumPass_39](../../../docs/en/figures/EinsumPass_39.png)

    Based on whether the input shape is dynamic, the **Transpose dym/stc** node in the above diagram contains the following operators:

    ![EinsumPass_40](../../../docs/en/figures/EinsumPass_40.png)

24. The Einsum expression is in the form of "abcd, ebcd-\>bcae"":

    ![EinsumPass_41](../../../docs/en/figures/EinsumPass_41.png)

    Based on whether the input shape is dynamic, the **Reshape** node in the above diagram contains the following operators:

    ![EinsumPass_42](../../../docs/en/figures/EinsumPass_42.png)

    Based on whether the input shape is dynamic, the **Transpose dym/stc** node in the above diagram contains the following operators:

    ![EinsumPass_43](../../../docs/en/figures/EinsumPass_43.png)

25. The Einsum expression is in the form of "abcd, dabe-\>cabe":

    ![EinsumPass_44](../../../docs/en/figures/EinsumPass_44.png)

    Based on whether the input shape is dynamic, the **Reshape** node in the above diagram contains the following operators:

    ![EinsumPass_45](../../../docs/en/figures/EinsumPass_45.png)

26. The Einsum expression is in the form of "a, b-\>ab":

    ![EinsumPass_46](../../../docs/en/figures/EinsumPass_46.png)

27. The Einsum expression is in the form of "abcd, aecd-\>eb":

    ![EinsumPass_47](../../../docs/en/figures/EinsumPass_47.png)

    Based on whether the input shape is dynamic, the **Reshape** node in the above diagram contains the following operators:

    ![EinsumPass_48](../../../docs/en/figures/EinsumPass_48.png)

    Based on whether the input shape is dynamic, the operators contained in the **Transpose dym/stc** node in the above diagram are as follows:

    ![EinsumPass_49](../../../docs/en/figures/EinsumPass_49.png)

28. The Einsum expression form is "abcd, eb-\>aecd"":

  ![EinsumPass_50](../../../docs/en/figures/EinsumPass_50.png)

  Based on whether the input shape is dynamic, the operators contained in the **Reshape** node in the above diagram are as follows:

  ![EinsumPass_51](../../../docs/en/figures/EinsumPass_51.png)

1. Other dynamic shape cases include the following two situations:
    1. Single-input case:

        ![EinsumPass_52](../../../docs/en/figures/EinsumPass_52.png)

        After fusion

        ![EinsumPass_53](../../../docs/en/figures/EinsumPass_53.png)

    2. Dual-input scenario:
        1. Use case 1

            ![EinsumPass_54](../../../docs/en/figures/EinsumPass_54.png)

            After fusion

            ![EinsumPass_55](../../../docs/en/figures/EinsumPass_55.png)

            Unsqueeze, ReduceSum, and Squeeze are all optional, and each can appear multiple times.

        2. Use case 2

            ![EinsumPass_56](../../../docs/en/figures/EinsumPass_56.png)

            After fusion

            ![EinsumPass_57](../../../docs/en/figures/EinsumPass_57.png)

            Unsqueeze, ReduceSum, Unsqueeze/FlattenV2, Reshape, Transpose, and Squeeze are all optional, and there can be multiple ReduceSum and Unsqueeze/FlattenV2 operators.

## Constraints

- This fusion pattern is enabled by default and cannot be disabled.
- The Einsum operator can have only 1 or 2 inputs, and the number of elements in the input tensors must not overflow the int64 representation range.
- In static shape scenarios, only single-input single-output or dual-input single-output Einsum expressions are supported.
- In static shape scenarios, repeated dimension labels in the input or output expressions are not supported.
- In static shape scenarios, when the fused operator is BatchMatMul, the input tensor data types support only float16 and float32.
- In scenarios with dynamic shape, apart from the general decomposition scenario, only the following Einsum expression forms are supported (the specific dimension labels are only examples), 28 in total:"abc,cde-\>abde", "abcd,aecd-\>aceb", "abcd,adbe-\>acbe", "abcd,cde-\>abe", "abc,cd-\>abd", "abc,dc-\>abd", "abc,abd-\>dc", "abc,dec-\>abde", "abc,abde-\>dec", "abcd,aecd-\>acbe", "abcd,acbe-\>aecd", "abcd,ecd-\>abe", "abcd,abe-\>ecd", "abcd,acbe-\>adbe", "abc,acd-\>abd", "abc,adc-\>abd", "abc,abd-\>acd", "ab,cb-\>ac", "a,b-\>ab", "abcd,abde-\>abce", "abcd,abce-\>abde", "abcd,abce-\>acde", "abcd,aebd-\>aebc", "abcd,ced-\>abce", "abcd,aecd-\>eb", "abcd,eb-\>aecd", "abcd,ebcd-\>bcae", "abcd,dabe-\>cabe".
- In the case of dynamic shapes, the number of input dimensions must be consistent with the number of dimensions in the corresponding Einsum input expression.
- In dynamic shape scenarios, Einsum expressions with single-input single-output, dual-input single-output, or empty output are supported. Duplicate dimension labels in the input or output expressions are not supported, and the output dimension labels must appear in the inputs.
- In dynamic shape scenarios, only input tensors with up to four dimensions are supported.

## Applicable Products

<!-- npu="910b" id1 -->
Atlas A2 products
<!-- end id1 -->

<!-- npu="A3" id2 -->
Atlas A3 products
<!-- end id2 -->

<!-- npu="950" id3 -->
Ascend 950PR&950DT products
<!-- end id3 -->
