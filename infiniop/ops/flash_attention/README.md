# `Flash Attention`

`Flash Attention`，是一种高效的**自注意力机制实现**，在加速注意力计算的同时也减少了内存占用。其核心原理是通过将输入分块，并在每个块上分别执行注意力计算操作，从而减少对 HBM 的读写。

示意图如下：

![alt text](image.png)

## 接口

### 计算

```c
infiniStatus_t infiniopFlashAttention(
    infiniopFlashAttentionDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *q,
    const void *k,
    const void *v,
    const void *mask,
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  已使用 `infiniopCreateFlashAttentionDescriptor()` 初始化的算子描述符；
- `workspace`:
  指向算子计算所需的额外工作空间；
- `workspace_size`:
  `workspace` 的大小，单位：字节；
- `out`:
  注意力计算结果地址。张量限制见[创建算子描述](#创建算子描述)部分。
- `q`:
  查询（Query）张量数据指针。张量限制见[创建算子描述](#创建算子描述)部分。
- `k`:
  键（Key）张量数据指针。张量限制见[创建算子描述](#创建算子描述)部分。
- `v`:
  值（Value）张量数据指针。张量限制见[创建算子描述](#创建算子描述)部分。
- `mask`:
  注意力掩码的数据指针，为空时，则不使用 mask。张量限制见[创建算子描述](#创建算子描述)部分。
- `stream`:
  计算流/队列。

<div style="background-color: lightblue; padding: 1px;">  返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_NULL_POINTER`], [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`].

### 创建算子描述

```c
infiniStatus_t infiniopCreateFlashAttentionDescriptor(
    infiniopHandle_t handle,
    infiniopFlashAttentionDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t mask_desc) 
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`:
  `infiniopHandle_t`类型的硬件控柄。详见 [`InfiniopHandle_t`]。
- `desc_ptr`:
  `infiniopFlashAttentionDescriptor_t` 指针，指向将被初始化的算子描述符地址。
- `out_desc` - { dT | ((batch_size,) num_heads, seq_len, head_dim) | ($\ldots, 1$)}:
  算子计算参数 `out` 的张量描述，四维或者三维，最后一维连续。
- `q_desc` - { dT | ((batch_size,) num_heads, seq_len, head_dim) | ($\ldots, 1$)}:
  算子计算参数 `q` 的张量描述，形状与 `out_desc` 一致，最后一维连续。
- `k_desc` - { dT | ((batch_size,) num_heads, seq_len, head_dim) | ($\ldots, 1$)}:
  算子计算参数 `k` 的张量描述，形状与 `out_desc` 一致，最后一维连续。
- `v_desc` - { dT | ((batch_size,) num_heads, seq_len, head_dim) | ($\ldots, 1$)}:
  算子计算参数 `v` 的张量描述，形状与 `out_desc` 一致，最后一维连续。
- `mask_desc` - { dM | ((batch_size,) seq_len) | ($\ldots, 1$)}:
  算子计算参数 `mask` 的张量描述，二维或者一维，最后一维连续。

参数限制：

- `dT`: `Float16`。
- `dM`: (`Bool`, 任意整数类型) 之一。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`],  [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 计算额外工作空间

```c
infiniStatus_t infiniopGetFlashAttentionWorkspaceSize(
    infiniopFlashAttentionDescriptor_t desc, 
    size_t *size
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`:
  已使用 `infiniopCreateFlashAttentionDescriptor()` 初始化的算子描述符；
- `size`:
  额外空间大小的计算结果的写入地址；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 销毁算子描述符

```c
infiniStatus_t infiniopDestoryFlashAttentionDescriptor(
    infiniopFlashAttentionDescriptor_t desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  输入。 待销毁的算子描述符；

<div style="background-color: lightblue; padding: 1px;"> 返回值： </div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

<!-- 链接 -->
[`InfiniopHandle_t`]: /infiniop/handle/README.md

[`INFINI_STATUS_SUCCESS`]: /common/status/README.md#INFINI_STATUS_SUCCESS
[`INFINI_STATUS_BAD_PARAM`]: /common/status/README.md#INFINI_STATUS_BAD_PARAM
[`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`]: /common/status/README.md#INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED
[`INFINI_STATUS_BAD_TENSOR_SHAPE`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_SHAPE
[`INFINI_STATUS_BAD_TENSOR_DTYPE`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_DTYPE
[`INFINI_STATUS_BAD_TENSOR_STRIDES`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_STRIDES
[`INFINI_STATUS_NULL_POINTER`]:/common/status/README.md#INFINI_STATUS_NULL_POINTER
[`INFINI_STATUS_INSUFFICIENT_WORKSPACE`]:/common/status/README.md#INFINI_STATUS_INSUFFICIENT_WORKSPACE
[`INFINI_STATUS_INTERNAL_ERROR`]:/common/status/README.md#INFINI_STATUS_INTERNAL_ERROR