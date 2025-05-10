
# `EmbeddingBackward`

`EmbeddingBackward`，即[**嵌入**算子](/infiniop/ops/embedding/README.md)的反向算子。用于训练大模型的词嵌入和加性位置嵌入。

`EmbeddingBackward` 算子支持 1 个或 2 个相同的步骤，根据“号码”从将输出的梯度叠加到嵌入表的梯度，其公式表述为：

$$ \begin{equation} d_{table1} = \alpha_1 \cdot dy[i_1] \end{equation} $$

$$ \begin{equation} d_{table2} = \alpha_2 \cdot dy[i_2] \end{equation} $$

- 通常 $α$ 为 1；
- $table2$ 可以不使用，则公式 $(2)$ 不存在；

## 接口

### 计算

```c
infiniStatus_t infiniopEmbeddingBackward(
    infiniopEmbeddingBackwardDescriptor_t desc,
    void *dtable1,
    void *dtable2,
    const void *dy,
    const void *i1,
    const void *i2,
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  已使用 `infiniopEmbeddingBackwardDescriptor_t()` 初始化的算子描述符；
- `dtable1`:
  第 1 个嵌入表的梯度；
- `dtable2`:
  第 2 个嵌入表的梯度，不使用则为空；
- `dy`:
  输出结果的梯度；
- `i1`:
  第 1 个嵌入序号；
- `i2`:
  第 2 个嵌入序号，不使用则为空；
- `stream`:
  计算流/队列；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_DEVICE`], [`INFINI_STATUS_EXECUTION_FAILED`].

### 创建算子描述

```c
infiniStatus_t infiniopCreateEmbeddingBackwardDescriptor(
    infiniopHandle_t handle,
    infiniopEmbeddingBackwardDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t dtable1_desc,
    infiniopTensorDescriptor_t dtable2_desc,
    infiniopTensorDescriptor_t dy_desc,
    infiniopTensorDescriptor_t i1_desc,
    infiniopTensorDescriptor_t i2_desc,
    float alpha1,
    float alpha2,
    char dtable1_acc,
    char dtable2_acc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`:
  `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]
- `desc_ptr`:
  `infiniopCreateEmbeddingBackwardDescriptor` 指针，指向将被初始化的算子描述符地址；
- `dtable1_desc` - { dT | (N1, D) | (..., 1) }:
  算子输入 `table1` 的张量描述；
- `dtable2_desc` - { dT | (N2, D) | (..., 1) }:
  算子输入 `table2` 的张量描述；
- `dy_desc` - { dT | (N, D) | (..., 1) }:
  算子输出 `y` 的张量描述；
- `i1_desc` - { dI | (N) | (1) }:
  算子输入 `i1` 的张量描述；
- `i2_desc` - { dI | (N) | (1) }:
  算子输入 `i2` 的张量描述，为空表示不使用 $ table2 $, `alpha2` 必须同时为 0；
- `alpha1` - float:
  第 1 项嵌入的缩放因子；
- `alpha2` - float:
  第 2 项嵌入的缩放因子，取 0 表示不使用 $ table2 $，`i2_desc` 必须同时为空；
- `dtable1_acc` - char:
  第 1 项嵌入是否叠加梯度，0 表示不叠加；
- `dtable2_acc` - float:
  第 2 项嵌入是否叠加梯度，0 表示不叠加；

<div style="background-color: lightblue; padding: 1px;"> 参数限制：</div>

- $dT$: 任意代数类型；
- $dT_i$: 任意整型；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_BAD_DEVICE`].

### 销毁算子描述符

```c
infiniStatus_t infiniopDestroyEmbeddingBackwardDescriptor(
    infiniopEmbeddingBackwardDescriptor_t desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  输入。待销毁的算子描述符；

<div style="background-color: lightblue; padding: 1px;"> 返回值： </div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_DEVICE`].

<!-- 链接 -->
[`InfiniopHandle_t`]: /infiniop/handle/README.md

[`INFINI_STATUS_SUCCESS`]:/common/status/README.md#INFINI_STATUS_SUCCESS
[`INFINI_STATUS_BAD_PARAM`]:/common/status/README.md#INFINI_STATUS_BAD_PARAM
[`INFINI_STATUS_BAD_DEVICE`]:/common/status/README.md#INFINI_STATUS_BAD_DEVICE
[`INFINI_STATUS_EXECUTION_FAILED`]:/common/status/README.md#INFINI_STATUS_EXECUTION_FAILED
[`INFINI_STATUS_BAD_TENSOR_SHAPE`]:/common/status/README.md#INFINI_STATUS_BAD_TENSOR_SHAPE
[`INFINI_STATUS_BAD_TENSOR_DTYPE`]:/common/status/README.md#INFINI_STATUS_BAD_TENSOR_DTYPE
[`INFINI_STATUS_BAD_TENSOR_STRIDES`]:/common/status/README.md#INFINI_STATUS_BAD_TENSOR_STRIDES
