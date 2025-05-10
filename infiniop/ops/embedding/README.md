
# `Embedding`

`Embedding`，即**嵌入**算子。用于对大模型进行词嵌入和加性位置嵌入。

`Embedding` 算子支持 1 个或 2 个相同的步骤，根据“号码”从嵌入表中获取嵌入向量并叠加到输出，其公式表述为：

$$ Y = \alpha_1 \cdot table_1[i_1] + \alpha_2 \cdot table_2[i_2] $$

- 通常 $α$ 为 1；
- $table2$ 可以不使用，则公式变为 $Y = \alpha_1 \cdot table_1[i_1]$；

## 接口

### 计算

```c
infiniStatus_t infiniopEmbedding(
    infiniopEmbeddingDescriptor_t desc,
    void *y,
    const void *table1,
    const void *table2,
    const void *i1,
    const void *i2,
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  已使用 `infiniopCreateEmbeddingDescriptor()` 初始化的算子描述符；
- `y`:
  计算输出结果；
- `table1`:
  第 1 个嵌入表；
- `table2`:
  第 2 个嵌入表，不使用则为空；
- `i1`:
  第 1 个嵌入序号；
- `i2`:
  第 2 个嵌入序号，不使用则为空；
- `stream`:
  计算流/队列；

<div style="background-color: lightblue; padding: 1px;">  返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_DEVICE`], [`INFINI_STATUS_EXECUTION_FAILED`].

### 创建算子描述

```c
infiniStatus_t infiniopCreateEmbeddingDescriptor(
    infiniopHandle_t handle,
    infiniopEmbeddingDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t table1_desc,
    infiniopTensorDescriptor_t table2_desc,
    infiniopTensorDescriptor_t i1_desc,
    infiniopTensorDescriptor_t i2_desc,
    float alpha1,
    float alpha2
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`:
  `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]
- `desc_ptr`:
  `infiniopCreateEmbeddingDescriptor` 指针，指向将被初始化的算子描述符地址；
- `y_desc` - { dT | (N, D) | (..., 1) }:
  算子输出 `y` 的张量描述；
- `table1_desc` - { dT | (N1, D) | (..., 1) }:
  算子输入 `table1` 的张量描述；
- `table2_desc` - { dT | (N2, D) | (..., 1) }:
  算子输入 `table2` 的张量描述；
- `i1_desc` - { dI | (N) | (1) }:
  算子输入 `i1` 的张量描述；
- `i2_desc` - { dI | (N) | (1) }:
  算子输入 `i2` 的张量描述，为空表示不使用 $ table2 $, `alpha2` 必须同时为 0；
- `alpha1` - float:
  第 1 项嵌入的缩放因子；
- `alpha2` - float:
  第 2 项嵌入的缩放因子，取 0 表示不使用 $ table2 $，`i2_desc` 必须同时为空；

参数限制：

- $dT$: 任意代数类型；
- $dT_i$: 任意整型；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_BAD_DEVICE`].

### 销毁算子描述符

```c
infiniStatus_t infiniopDestroyEmbeddingDescriptor(
    infiniopEmbeddingDescriptor_t desc
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
