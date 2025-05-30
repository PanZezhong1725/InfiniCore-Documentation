# `SPMM`

`SPMM`，即**稀疏矩阵-矩阵乘法**算子。计算公式为：

$$ dense A' = (sparse) A * (dense) B $$

其中：

- `A` 为稀疏矩阵，通过CSR稀疏格式存储。
- `B` 为稠密矩阵。
- `A'` 为输出稠密矩阵。

## 接口

### 计算

```c
infiniStatus_t infiniopSpMM_csr(
    infiniopHandle_t handle,
    infiniopSpMVDescriptor_t desc,
    void *dense_rec,
    const void *dense_B,
    size_t M,
    size_t N,
    const void *sparse_A_val,
    const void *row_ptr,
    const void *col_indices,
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `handle`:
  `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]。
- `desc`:
  已使用 `infiniopCreateSpMVDescriptor()` 初始化的算子描述符。
- `dense_rec`:
  计算输出稠密矩阵。张量限制见[创建算子描述](#创建算子描述)部分。
- `dense_B`:
  输入稠密矩阵B。张量限制见[创建算子描述](#创建算子描述)部分。
- `M`
  输入稀疏矩阵A的行数。
- `N`
  输入稀疏矩阵A的列数。
- `sparse_A_val`:
  稀疏矩阵的非零元值。张量限制见[创建算子描述](#创建算子描述)部分。
- `row_ptr`:
  稀疏矩阵的行偏移。张量限制见[创建算子描述](#创建算子描述)部分。
- `col_indices`:
  稀疏矩阵的列索引。张量限制见[创建算子描述](#创建算子描述)部分。
- `stream`:
  计算流/队列。

<div style="background-color: lightblue; padding: 1px;">  返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`].



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
