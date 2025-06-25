# `SPMV`

`SPMV`，即**稀疏矩阵向量乘法**算子。计算公式为：

$$ y = A * x $$

其中：

- `A` 为稀疏矩阵。
- `x` 为输入稠密向量。
- `y` 为输出的稠密向量。

## 接口

### 计算

```c
infiniStatus_t infiniopSpMV(
    infiniopSpMVDescriptor_t desc,
    void *y,
    const void *x,
    const void *values,
    const void *row_indices,
    const void *col_indices,
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  已使用 `infiniopCreateSpMVDescriptor()` 初始化的算子描述符。
- `y`:
  计算输出向量。
- `x`:
  输入向量。
- `values`:
  稀疏矩阵的非零元素值数组。
- `row_indices`:
  稀疏矩阵的行偏移数组。
- `col_indices`:
  稀疏矩阵的列索引数组。
- `stream`:
  计算流/队列。

<div style="background-color: lightblue; padding: 1px;">  返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`].

### 创建算子描述

```c
infiniStatus_t infiniopCreateSpMVDescriptor(
    infiniopHandle_t handle,
    infiniopSpMVDescriptor_t *desc_ptr,
    size_t num_cols,
    size_t num_rows,
    size_t nnz,
    infiniDtype_t dtype
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`:
  `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]
- `desc_ptr`:
  指向将被初始化的算子描述符地址。
- `num_cols`:
  稀疏矩阵的列数。
- `num_rows`:
  行偏移数组长度。
- `nnz`:
  非零元素数量。
- `dtype`:
  数据类型（当前仅支持 `Float32`）。

<div style="background-color: lightblue; padding: 1px;"> 参数限制：</div>

- `num_cols` > 0；
- `num_rows` > 0；
- `nnz` > 0；
- `dtype`: 当前仅支持 `Float32`；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 销毁算子描述符

```c
infiniStatus_t infiniopDestroySpMVDescriptor(
    infiniopSpMVDescriptor_t desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  输入。待销毁的算子描述符。

<div style="background-color: lightblue; padding: 1px;"> 返回值： </div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

<!-- 链接 -->
[`InfiniopHandle_t`]: /infiniop/handle/README.md

[`INFINI_STATUS_SUCCESS`]:/common/status/README.md#INFINI_STATUS_SUCCESS
[`INFINI_STATUS_BAD_PARAM`]:/common/status/README.md#INFINI_STATUS_BAD_PARAM
[`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`]:/common/status/README.md#INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED
[`INFINI_STATUS_INTERNAL_ERROR`]:/common/status/README.md#INFINI_STATUS_INTERNAL_ERROR
[`INFINI_STATUS_BAD_TENSOR_DTYPE`]:/common/status/README.md#INFINI_STATUS_BAD_TENSOR_DTYPE