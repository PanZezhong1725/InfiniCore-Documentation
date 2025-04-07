
# `Rotary Position Embedding`

`Rotary Position Embedding`, 即**旋转位置编码**算子：

旋转频率 ( $\theta$ )

计算公式为：

$$
\theta_{i} = \text{base}^{\frac{-2i}{d}}
$$

其中：

- `base`：控制旋转速率的超参数，通常为 `base` = 10000；
- `i`：当前嵌入维度的索引；
- `d` ：嵌入的总维度；

每个 token 的嵌入向量的旋转变换，对于第 `k` 个维度（每个 token 的两个部分）：

**获取原始嵌入**：

  $$
  a = \text{t[2k]}
  $$

  $$
  b = \text{t[2k+1]}
  $$

**获取对应位置的正弦和余弦值**：

  $$
  \sin_0 = \sin[2k], \quad \cos_0 = \cos[2k]
  $$
  $$
  \sin_1 = \sin[2k + 1], \quad \cos_1 = \cos[2k + 1]
  $$

**旋转嵌入向量**：

  $$
  t[2k] = a \cdot \cos_0 - b \cdot \sin_0
  $$
  $$
  t[2k+1] = a \cdot \sin_1 + b \cdot \cos_1
  $$

## 接口

### 计算

```c
infiniStatus_t infiniopRoPE(
    infiniopRoPEDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *t,
    const void *pos_ids,
    const float *sin_table,
    const float *cos_table,
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  已使用 `infiniopCreateRoPEDescriptor()` 初始化的算子描述符；
- `workspace`:
  指向算子计算所需的额外工作空间；
- `workspace_size`:
  `workspace` 的大小，单位：字节；
- `t`:
  （比如 attention 层中的 query 或 key）张量数据指针。张量限制见[创建算子描述](#创建算子描述)部分；
- `pos_ids`:
  位置数值张量指针，代表 data 中每个向量对应的 cos、sin 表中的序号。计算时，会根据序号选择表中的数值进行位置编码。用户应自行保证序号不会超过 cos、sin 表的长度。张量限制见[创建算子描述](#创建算子描述)部分；
- `sin_table`:
  sin 表指针。张量限制见[创建算子描述](#创建算子描述)部分；
- `cos_table`:
  cos 表指针。张量限制见[创建算子描述](#创建算子描述)部分；
- `stream`:
  计算流/队列。

<div style="background-color: lightblue; padding: 1px;">  返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`STATUS_MEMORY_NOT_ALLOCATED`], [`STATUS_BAD_TENSOR_SHAPE`], [`STATUS_BAD_TENSOR_STRIDES`], [`STATUS_BAD_TENSOR_DTYPE`].

- 当 `t`、`pos_ids`、`sin_table`、或 `cos_table` 参数张量不统一返回 `INFINI_STATUS_BAD_PARAM`；
- 当 `t`、`pos_ids`、`sin_table`、或 `cos_table` 张量形状不符合要求返回 `INFINI_STATUS_BAD_TENSOR_SHAPE`；
- 当 `t`、`pos_ids`、`sin_table`、或 `cos_table` 输入输出张量类型不被支持返回 `INFINI_STATUS_BAD_TENSOR_DTYPE`；

### 创建算子描述

```c
infiniStatus_t infiniopCreateRoPEDescriptor(
    infiniopHandle_t handle,
    infiniopRoPEDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t t,
    infiniopTensorDescriptor_t pos_ids,
    infiniopTensorDescriptor_t sin_table,
    infiniopTensorDescriptor_t cos_table
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`
 : `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]；
- `desc_ptr`:
  `infiniopRoPEDescriptor_t` 指针，指向将被初始化的算子描述符地址；
- `t` - { dT | `(seq_len, num_head, head_dim)` | (..., 1) }:
  张量必须为三维：`(seq_len, num_head, head_dim)`。最后一维数据必须连续，即步长为1，且长度`(head_dim)` 为2的倍数；
- `pos_ids` - { dI | `(seq_len)` | (~) }:
  位置信息张量描述。张量必须为一维连续张量，长度为 `seq_len` 。用户需自行保证位置数据中所有数值小于 `max_seq_len`；
- `sin_table` - { float | `(max_seq_len, head_dim/2)` | (~) }:
  sin 值表的张量描述，二维连续张量，形状为 `(max_seq_len, head_dim/2)`；
- `cos_table` - { float | `(max_seq_len, head_dim/2)` | (~) }:
  cos 值表的张量描述，要求与 sin 表相同；

参数限制：

- `dT`:  (`Float16`, `Float32`, `Float64`) 之一；
- `dI`: (`Uint8`, `Uint16`, `Uint32`, `Uint64`) 之一；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`],  [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_BAD_DEVICE`].

### 计算额外工作空间

```c
infiniStatus_t infiniopGetRoPEWorkspaceSize(
    infiniopRoPEDescriptor_t desc,
    size_t *size
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`:
  已使用 `infiniopCreateRoPEDescriptor()` 初始化的算子描述符；
- `size`:
  额外空间大小的计算结果的写入地址；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_DEVICE`].

### 销毁算子描述符

```c
infiniStatus_t infiniopDestroyRoPEDescriptor(
    infiniopRoPEDescriptor_t desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  待销毁的算子描述符；

<div style="background-color: lightblue; padding: 1px;"> 返回值： </div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_DEVICE`].

## 已知问题

无

<!-- 链接 -->
[`InfiniopHandle_t`]: /infiniop/handle/README.md

[`INFINI_STATUS_SUCCESS`]: /common/status/README.md#INFINI_STATUS_SUCCESS
[`INFINI_STATUS_BAD_PARAM`]: /common/status/README.md#INFINI_STATUS_BAD_PARAM
[`INFINI_STATUS_BAD_DEVICE`]: /common/status/README.md#INFINI_STATUS_BAD_DEVICE
[`INFINI_STATUS_BAD_TENSOR_SHAPE`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_SHAPE
[`INFINI_STATUS_BAD_TENSOR_DTYPE`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_DTYPE
[`INFINI_STATUS_BAD_TENSOR_STRIDES`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_STRIDES
[`STATUS_MEMORY_NOT_ALLOCATED`]:/common/status/README.md#STATUS_MEMORY_NOT_ALLOCATED
[`STATUS_BAD_TENSOR_SHAPE`]:/common/status/README.md#STATUS_BAD_TENSOR_SHAPE
[`STATUS_BAD_TENSOR_STRIDES`]:/common/status/README.md#STATUS_BAD_TENSOR_STRIDES
[`STATUS_BAD_TENSOR_DTYPE`]:/common/status/README.md#STATUS_BAD_TENSOR_DTYP
