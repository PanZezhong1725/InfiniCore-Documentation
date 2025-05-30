# `SPMV`

`SPMV`，即**稀疏矩阵向量乘法**算子。计算公式为：

$$ y = A * x $$

其中：

- `A` 为稀疏矩阵，通过CSR稀疏格式存储。
- `x` 为输入向量。
- `y` 为输出向量，结果由稀疏矩阵向量乘法规则确定。

## 接口

### 计算

```c
infiniStatus_t infiniopSpMV(
    infiniopSpMVDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    const void *values,
    const void *row_ptr,
    const void *col_indices,
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  已使用 `infiniopCreateSpMVDescriptor()` 初始化的算子描述符。
- `workspace`:
  指向算子计算所需的额外工作空间。
- `workspace_size`:
  `workspace` 的大小，单位：字节。
- `y`:
  计算输出向量。张量限制见[创建算子描述](#创建算子描述)部分。
- `x`:
  输入向量。张量限制见[创建算子描述](#创建算子描述)部分。
- `values`:
  稀疏矩阵的非零元素值。张量限制见[创建算子描述](#创建算子描述)部分。
- `row_ptr`:
  稀疏矩阵的行偏移。张量限制见[创建算子描述](#创建算子描述)部分。
- `col_indices`:
  稀疏矩阵的列索引。张量限制见[创建算子描述](#创建算子描述)部分。
- `stream`:
  计算流/队列。

<div style="background-color: lightblue; padding: 1px;">  返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`].

### 创建算子描述

```c
infiniStatus_t infiniopCreateSpMVDescriptor(
    infiniopHandle_t handle,
    infiniopSpMVDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t values_desc,
    infiniopTensorDescriptor_t row_ptr_desc,
    infiniopTensorDescriptor_t col_indices_desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`:
  `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]
- `desc_ptr`:
  指向将被初始化的算子描述符地址；
- `y_desc` - { float32 | (M,) | (~) }:
  算子计算参数 `y` 的张量描述。
- `x_desc` - { float32 | (N,) | (~) }:
  算子计算参数 `x` 的张量描述。
- `values_desc` - { float32 | (nnz,) | (~) }:
  算子计算参数 `values` 的张量描述。
- `row_ptr_desc` - { int32 | (M+1,) | (~) }:
  算子计算参数 `row_indices` 的张量描述。
- `col_indices_desc` - { int32 | (nnz,) | (~) }:
  算子计算参数 `col_indices` 的张量描述。

<div style="background-color: lightblue; padding: 1px;"> 参数限制：</div>

参数限制：

- `M`: M > 0（输出向量维度）；
- `N`: N > 0（输入向量维度）；
- `nnz`: nnz > 0（非零元素个数）；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 计算额外工作空间

```c
infiniStatus_t infiniopGetSpMVWorkspaceSize(
    infiniopSpMVDescriptor_t desc,
    size_t *size
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`:
  已使用 `infiniopCreateSpMVDescriptor()` 初始化的算子描述符。
- `size`:
  额外空间大小的计算结果的写入地址；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 销毁算子描述符

```c
infiniStatus_t infiniopDestroySpMVDescriptor(
    infiniopSpMVDescriptor_t desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  输入。待销毁的算子描述符；

<div style="background-color: lightblue; padding: 1px;"> 返回值： </div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

<!-- 链接 -->
[`InfiniopHandle_t`]: /handle/README.md

[`INFINI_STATUS_SUCCESS`]:/common/status/README.md#INFINI_STATUS_SUCCESS
[`INFINI_STATUS_BAD_PARAM`]:/common/status/README.md#INFINI_STATUS_BAD_PARAM
[`INFINI_STATUS_INSUFFICIENT_WORKSPACE`]:/common/status/README.md#INFINI_STATUS_INSUFFICIENT_WORKSPACE
[`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`]:/common/status/README.md#INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED
[`INFINI_STATUS_INTERNAL_ERROR`]:/common/status/README.md#INFINI_STATUS_INTERNAL_ERROR
[`INFINI_STATUS_BAD_TENSOR_SHAPE`]:/common/status/README.md#INFINI_STATUS_BAD_TENSOR_SHAPE
[`INFINI_STATUS_BAD_TENSOR_DTYPE`]:/common/status/README.md#INFINI_STATUS_BAD_TENSOR_DTYPE
[`INFINI_STATUS_BAD_TENSOR_STRIDES`]:/common/status/README.md#INFINI_STATUS_BAD_TENSOR_STRIDES
