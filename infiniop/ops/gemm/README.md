
# `GEMM`

`GEMM`，即**通用矩阵乘法**算子。计算公式为：

$$ C = α ⋅ (A * B) + β ⋅ C $$

其中：

- `A` 为左输入张量，形状为 `( [batch,] M, K )`。
- `B` 为右输入张量，形状为 `( [batch,] K, N )`。
- `C` 的形状由矩阵乘法规则确定，形状为 `( [batch,] M, N )`。
- `α` 为缩放因子，`β` 为累加系数；

## 接口

### 计算

```c
infiniStatus_t infiniopGemm(
    infiniopGemmDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *c,
    const void *a,
    const void *b,
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  已使用 `infiniopCreateGemmDescriptor()` 初始化的算子描述符。
- `workspace`:
  指向算子计算所需的额外工作空间。
- `workspace_size`:
  `workspace` 的大小，单位：字节。
- `c`:
  计算输出结果。张量限制见[创建算子描述](#创建算子描述)部分。
- `a`:
  左输入张量。张量限制见[创建算子描述](#创建算子描述)部分。
- `b`:
  右输入张量。张量限制见[创建算子描述](#创建算子描述)部分。
- `stream`:
  计算流/队列。

<div style="background-color: lightblue; padding: 1px;">  返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`].

### 创建算子描述

```c
infiniStatus_t infiniopCreateGemmDescriptor(
    infiniopHandle_t handle,
    infiniopGemmDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    float alpha,
    float beta
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`:、
  `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]
- `desc_ptr`:
  指向将被初始化的算子描述符地址；
- `c_desc` - { dT | ( [batch,] , M, N) | (~) }:
  算子计算参数 `c` 的张量描述。
- `a_desc` - { dT | ( [batch,] , M, K) | (~) }:
  算子计算参数 `a` 的张量描述。
- `b_desc` - { dT | ( [batch,] , K, N) | (~) }:
  算子计算参数 `b` 的张量描述。
- `alpha` - float:
  算子计算缩放因子。
- `beta` - float:
  算子计算累加系数。

<div style="background-color: lightblue; padding: 1px;"> 参数限制：</div>

参数限制：

- `dT`: (`Float16`, `Float32`, `BFloat16`) 之一；
- `[batch,]`: `batch ≥ 1`（可选）；
- `M`: M > 0；
- `N`: N > 0；
- `K`: K > 0；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 计算额外工作空间

```c
infiniStatus_t infiniopGetGemmWorkspaceSize(
    infiniopGemmDescriptor_t desc,
    size_t *size
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`:
  已使用 `infiniopCreateMatmulDescriptor()` 初始化的算子描述符。
- `size`:
  额外空间大小的计算结果的写入地址；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 销毁算子描述符

```c
infiniStatus_t infiniopDestroyGemmDescriptor(
    infiniopGemmDescriptor_t desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  输入。待销毁的算子描述符；

<div style="background-color: lightblue; padding: 1px;"> 返回值： </div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

<!-- 链接 -->
[`InfiniopHandle_t`]: /infiniop/handle/README.md

[`INFINI_STATUS_SUCCESS`]:/common/status/README.md#INFINI_STATUS_SUCCESS
[`INFINI_STATUS_BAD_PARAM`]:/common/status/README.md#INFINI_STATUS_BAD_PARAM
[`INFINI_STATUS_INSUFFICIENT_WORKSPACE`]:/common/status/README.md#INFINI_STATUS_INSUFFICIENT_WORKSPACE
[`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`]:/common/status/README.md#INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED
[`INFINI_STATUS_INTERNAL_ERROR`]:/common/status/README.md#INFINI_STATUS_INTERNAL_ERROR
[`INFINI_STATUS_BAD_TENSOR_SHAPE`]:/common/status/README.md#INFINI_STATUS_BAD_TENSOR_SHAPE
[`INFINI_STATUS_BAD_TENSOR_DTYPE`]:/common/status/README.md#INFINI_STATUS_BAD_TENSOR_DTYPE
[`INFINI_STATUS_BAD_TENSOR_STRIDES`]:/common/status/README.md#INFINI_STATUS_BAD_TENSOR_STRIDES
