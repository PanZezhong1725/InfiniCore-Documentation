
# `RoPE`

`RoPE (Rotary Position Embedding)`, 即**旋转位置编码**算子。

对于一个位置为 $m$ 的 token 的嵌入向量 $x$ 中的每两个相邻元素，即第 $2i$ 和 $2i+1$ 个元素，其旋转后的嵌入向量 $y$ 可以表示为：

  $$
  y[2i] = x[2i] \cdot \cos(m\cdot \theta_i) - x[2i+1] \cdot \sin(m\cdot \theta_i)
  $$
  $$
  y[2i+1] = x[2i] \cdot \sin(m\cdot \theta_i) + x[2i+1] \cdot \cos(m\cdot \theta_i)
  $$

其中：

$$
\theta_{i} = \text{base}^{\frac{-2i}{d}}
$$

- `base`：控制旋转速率的超参数，常用 `base` = 10000。
- `d` ：嵌入向量长度，须为2的倍数。
- `i`：嵌入维度的双步长索引，满足 $i \in [0, ..., d/2 - 1]$。
- `m`：位置

公式中的正弦和余弦值 ( $\sin(m\cdot \theta_i)$ 和 $\cos(m\cdot \theta_i)$ ) 只与位置 `m` 和向量长度 `d` 有关，而与嵌入向量的数值无关。本接口支持用户以数表的形式传入正弦和余弦值，避免冗余计算。

## 接口

### 计算

```c
infiniStatus_t infiniopRoPE(
    infiniopRoPEDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    const void *pos_ids,
    const void *sin_table,
    const void *cos_table,
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  已使用 `infiniopCreateRoPEDescriptor()` 初始化的算子描述符。
- `workspace`:
  额外工作空间。
- `workspace_size`:
  `workspace` 的大小，单位：字节。
- `y`:
  计算结果地址，当与 `x` 相同时为原地操作。
- `x`:
  嵌入向量矩阵（比如自注意力层中的 query 或 key）的数据指针，可与 `y` 相同。
- `pos_ids`:
  位置数值的地址，代表 `x` 中每个向量对应的 cos、sin 表中的序号。计算时，会根据序号选择表中的数值进行位置编码。用户应自行保证序号不会超过 cos、sin 表的长度。
- `sin_table`:
  sin 表指针。
- `cos_table`:
  cos 表指针。
- `stream`:
  计算流/队列。

<div style="background-color: lightblue; padding: 1px;">  返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_NULL_POINTER`], [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`].

### 创建算子描述

```c
infiniStatus_t infiniopCreateRoPEDescriptor(
    infiniopHandle_t handle,
    infiniopRoPEDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    infiniopTensorDescriptor_t pos_ids,
    infiniopTensorDescriptor_t sin_table,
    infiniopTensorDescriptor_t cos_table
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`
 : 硬件控柄。详见 [`InfiniopHandle_t`]。
- `desc_ptr`:
  `infiniopRoPEDescriptor_t` 指针，指向将被初始化的算子描述符地址。
- `y` - { dT | (seq_len, num_head, head_dim) | (..., 1) }:
  三维张量。最后一维必须连续，且长度 `head_dim` 为2的倍数。
- `x` - { dT | (seq_len, num_head, head_dim) | (..., 1) }:
  限制与 `y` 相同。
- `pos_ids` - { dI | (seq_len) | (~) }:
  位置信息张量描述。张量必须为一维连续张量，长度为 `seq_len` 。用户需自行保证位置数据中所有数值小于 `max_seq_len`。
- `sin_table` - { dT | (max_seq_len, head_dim/2) | (~) }:
  sin 值表的张量描述，二维连续张量。
- `cos_table` - { dT | (max_seq_len, head_dim/2) | (~) }:
  cos 值表的张量描述，要求与 sin 表相同。

参数限制：

- `dT`:  (`Float16`, `Float32`, `Float64`) 之一。
- `dI`: 任意整数类型。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`],  [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 计算额外工作空间

```c
infiniStatus_t infiniopGetRoPEWorkspaceSize(
    infiniopRoPEDescriptor_t desc,
    size_t *size
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`:
  已使用 `infiniopCreateRoPEDescriptor()` 初始化的算子描述符。
- `size`:
  额外空间大小的计算结果的写入地址。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 销毁算子描述符

```c
infiniStatus_t infiniopDestroyRoPEDescriptor(
    infiniopRoPEDescriptor_t desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  待销毁的算子描述符。

<div style="background-color: lightblue; padding: 1px;"> 返回值： </div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

## 已知问题

无

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
