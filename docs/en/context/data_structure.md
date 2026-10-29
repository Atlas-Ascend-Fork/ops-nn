# Data Structures

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-23T12:50:35.561Z -->

This chapter provides the basic data structures required for calling CANN operator APIs. **Developers do not need to focus on their internal implementation and can use them directly.**

Note that this basic data structure can be created using the public APIs in [*Operator Library APIs*](https://hiascend.com/en/document/redirect/CannCommunityOplist), such as aclCreateTensor.

- **aclTensor**

  A structure defined by the framework for managing and storing tensor data (such as vectors, matrices, and other multi-dimensional data). This object can be created through the **aclCreateTensor** interface.

  ```bash
  typedef struct aclTensor aclTensor
  ```

- **aclScalar**

  A framework-defined structure for managing and storing scalar data (i.e., a single numeric value). The object can be created through the **aclCreateScalar** interface.

  ```bash
  typedef struct aclScalar aclScalar
  ```

- **aclIntArray**

  A framework-defined array structure for managing and storing integer data. The object can be created through the **aclCreateIntArray** interface.

  ```bash
  typedef struct aclIntArray aclIntArray
  ```

- **aclFloatArray**

  An array structure defined by the framework for managing and storing float32 data. This object can be created through the **aclCreateFloatArray** interface.

  ```bash
  typedef struct aclFloatArray aclFloatArray
  ```

- **aclBoolArray**

  An array structure defined by the framework for managing and storing Boolean data. This object can be created through the **aclCreateBoolArray** interface.

  ```bash
  typedef struct aclBoolArray aclBoolArray
  ```

- **aclTensorList**

  An array structure defined by the framework for managing and storing multiple tensor data. This object can be created through the **aclCreateTensorList** interface.

  ```bash
  typedef struct aclTensorList aclTensorList
  ```

- **aclScalarList**

  An array structure defined by the framework for managing and storing scalar data. This object can be created through the **aclCreateScalarList** interface.

  ```bash
  typedef struct aclScalarList aclScalarList
  ```

- **aclOpExecutor**

  An executor data structure defined by the framework, serving as a container for executing operator computations.

  Typically, when calling the one-stage operator interface aclxxXxxGetWorkspaceSize, the framework automatically creates an aclOpExecutor; after calling the two-stage interface aclxxXxx, the object is automatically released.

  ```bash
  typedef struct aclOpExecutor aclOpExecutor
  ```

- **aclrtStream**

  A stream processing data structure defined by the framework, used to manage and maintain the execution order of some asynchronous operations.

  ```bash
  typedef void *aclrtStream
  ```
