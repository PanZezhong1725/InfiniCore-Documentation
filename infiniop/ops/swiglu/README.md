
# `SwiGLU`

`SwiGLU (Switched Gated Linear Unit)` , 即**门控线性单元激活函数**算子，计算公式为：

$$
c_{i} = a_{i} \circ SiLU(b_{i})
$$

$$
SiLU(b_{i}) = \frac {b_{i}}{1 + e^{-b_{i}}}
$$

其中：

- $c_{i}$: 输出张量第 `i` 个元素。
- $a_{i}$: 输入张量第 `i` 个元素。
- $b_{i}$: 门控输入张量第 `i` 个元素。

## 接口

### 计算

```c
infiniStatus_t infiniopSwiGLU(
    infiniopSwiGLUDescriptor_t desc,
    void *c,
    const void *a,
    const void *b,
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  已使用 `infiniopCreateSwiGLUDescriptor()` 初始化的算子描述符。
- `c`:
  输出张量，张量限制见[创建算子描述](#创建算子描述)部分。
- `a`:
  输入张量，张量限制见[创建算子描述](#创建算子描述)部分。
- `b`:
  门控输入张量，张量限制见[创建算子描述](#创建算子描述)部分。
- `stream`:
  计算流/队列。

<div style="background-color: lightblue; padding: 1px;">  返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`STATUS_MEMORY_NOT_ALLOCATED`], [`STATUS_BAD_TENSOR_SHAPE`], [`STATUS_BAD_TENSOR_STRIDES`], [`STATUS_BAD_TENSOR_DTYPE`].

### 创建算子描述

```c
infiniStatus_t infiniopCreateSwiGLUDescriptor(
    infiniopHandle_t handle,
    infiniopSwiGLUDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c,
    infiniopTensorDescriptor_t a,
    infiniopTensorDescriptor_t b
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`
 : `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]。
- `desc_ptr`:
  `infiniopSwiGLUDescriptor_t` 指针，指向将被初始化的算子描述符地址。
- `c` - { dT | (...) | (...) }:
  算子计算参数 `c` 的张量描述, 支持原位计算。
- `a` - { dT | (...) | (...) }:
  算子计算参数 `a` 的张量描述，支持原位计算。
- `b` - { dT | (...) | (...) }:
  算子计算参数 `b` 的张量描述，支持原位计算。

参数限制：

- `dT`:  (`Float16`, `Float32`, `Float64`, `Bfloat16`) 之一
- `c`, `a`, `b` 均支持不连续步长

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`],  [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 销毁算子描述符

```c
infiniStatus_t infiniopDestroySwiGLUDescriptor(
    infiniopSwiGLUDescriptor_t desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  输入。待销毁的算子描述符。

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
[`STATUS_MEMORY_NOT_ALLOCATED`]:/common/status/README.md#STATUS_MEMORY_NOT_ALLOCATED
[`STATUS_BAD_TENSOR_SHAPE`]:/common/status/README.md#STATUS_BAD_TENSOR_SHAPE
[`STATUS_BAD_TENSOR_STRIDES`]:/common/status/README.md#STATUS_BAD_TENSOR_STRIDES
[`STATUS_BAD_TENSOR_DTYPE`]:/common/status/README.md#STATUS_BAD_TENSOR_DTYPE
