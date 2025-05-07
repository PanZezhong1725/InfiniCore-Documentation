# `Conv`

`Conv`，即**卷积**算子。计算过程可表示为对输入特征图应用卷积核进行空间域的滑动窗口计算，是深度学习中的基础操作。

$$ Y = \text{Conv}(X, W) $$

其中：

- `X` 为输入特征图，形状为 `(N, C_in, D_1, D_2, ...)`。
- `W` 为卷积核权重，形状为 `(C_out, C_in/groups, K_1, K_2, ...)`。
- `Y` 为输出特征图，形状为 `(N, C_out, D_out_1, D_out_2, ...)`。
- 相关参数包括 `pads`, `strides`, `dilations` 等控制卷积操作细节。

## 接口

### 计算

```c
infiniStatus_t infiniopConv(
    infiniopConvDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    const void *w,
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  已使用 `infiniopCreateConvDescriptor()` 初始化的算子描述符。
- `workspace`:
  指向算子计算所需的额外工作空间。
- `workspace_size`:
  `workspace` 的大小，单位：字节。
- `y`:
  计算输出结果。张量限制见创建算子描述部分。
- `x`:
  输入特征图。张量限制见创建算子描述部分。
- `w`:
  卷积权重。张量限制见创建算子描述部分。
- `stream`:
  计算流/队列。

<div style="background-color: lightblue; padding: 1px;">  返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`].

### 创建算子描述

```c
infiniStatus_t infiniopCreateConvDescriptor(
    infiniopHandle_t handle,
    infiniopConvDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    infiniopTensorDescriptor_t w,
    void *pads,
    void *strides,
    void *dilations,
    size_t n
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`:
  `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]
- `desc_ptr`:
  指向将被初始化的算子描述符地址；
- `y` - { dT | (N, C_out, D_out_1, D_out_2, ...) | (~) }:
  算子计算参数 `y` 的张量描述。
- `x` - { dT | (N, C_in, D_1, D_2, ...) | (~) }:
  算子计算参数 `x` 的张量描述。
- `w` - { dT | (C_out, C_in/groups, K_1, K_2, ...) | (~) }:
  算子计算参数 `w` 的张量描述。
- `pads` - void*:
  输入特征图填充大小的指针，数组长度为 `2*n`，表示每个空间维度的前后填充。
- `strides` - void*:
  卷积滑动步长的指针，数组长度为 `n`。
- `dilations` - void*:
  卷积膨胀率的指针，数组长度为 `n`。
- `n` - size_t:
  空间维度数量。

<div style="background-color: lightblue; padding: 1px;"> 参数限制：</div>

参数限制：

- `dT`: (`Float16`, `Float32`) 之一；
- `N`: 批量大小，N > 0；
- `C_in`: 输入通道数，C_in > 0；
- `C_out`: 输出通道数，C_out > 0；
- `D_i`: 输入特征图的第i个维度，D_i > 0；
- `K_i`: 卷积核的第i个维度，K_i > 0；
- `n`: 空间维度数量，通常为 2（2D卷积）或 3（3D卷积）；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 计算额外工作空间

```c
infiniStatus_t infiniopGetConvWorkspaceSize(
    infiniopConvDescriptor_t desc,
    size_t *size
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`:
  已使用 `infiniopCreateConvDescriptor()` 初始化的算子描述符。
- `size`:
  额外空间大小的计算结果的写入地址；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 销毁算子描述符

```c
infiniStatus_t infiniopDestroyConvDescriptor(
    infiniopConvDescriptor_t desc
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