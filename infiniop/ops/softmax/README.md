
# `Softmax`

归一化指数函数，常用于讲输入数据转换为概率分布。对于长度为 $N$ 的一维张量 $x$ ，其公式为：

$$ y_i = \frac{e^{x_i}}{\sum_{i=0}^{N - 1} e^{x_i}} $$

对于多维输入的情况，则在指定的维度 axis 上执行归一化计算。

## 接口

### 计算

```c
infiniStatus_t infiniopSoftmax(
    infiniopSoftmaxDescriptor_t desc, 
    void *y, 
    const void *x, 
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`：已使用 `infiniopCreateSoftmaxDescriptor()` 初始化的算子描述符。
- `y`：输出指针。
- `x`：输入指针，可以与 `y` 相同。
- `stream`：计算流/队列。

<div style="background-color: lightblue; padding: 1px;">  返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`]

---

### 创建算子描述

```c
infiniStatus_t infiniopCreateSoftmaxDescriptor(
    infiniopHandle_t handle, 
    infiniopSoftmaxDescriptor_t *desc_ptr, 
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x, 
    int axis
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`: 硬件控柄。详情请看 [`InfiniopHandle_t`]
- `desc_ptr`: `infiniopSoftmaxDescriptor_t` 指针，指向将被初始化的算子描述符地址。
- `y` - { dT | (d1, $\ldots$ , dr) | ($\ldots$) } : 输出 `y` 的张量描述。
- `x` - { dT | (d1, $\ldots$ , dr) | ($\ldots$) }: 算子计算参数 `x` 的张量描述，张量形状和 `y` 保持一致。
- `axis`: 执行规约的维度。可选择范围是 $[-\text{r}, \text{r} - 1]$，其中 $\text{r}$ 是 `x` 的秩。当 `axis` 为负数时，表示倒数第 `axis` 个维度。

参数限制：

- **`dT`**:  (`Float16`, `Float32`, `Float64`) 之一。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`],  [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

---

### 销毁算子描述符

```c
infiniStatus_t infiniopDestroySoftmaxDescriptor(
    infiniopSoftmaxDescriptor_t desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`
     : 待销毁的算子描述符。

<div style="background-color: lightblue; padding: 1px;"> 返回值： </div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

## 已知问题

### 平台限制

- 寒武纪中 tensor.to(device) 的 tensor 不支持 uint64 或者是 int64 数据类型。

<!-- 链接 -->
[`InfiniopHandle_t`]: /infiniop/handle/README.md

[`INFINI_STATUS_SUCCESS`]: /common/status/README.md#INFINI_STATUS_SUCCESS
[`INFINI_STATUS_BAD_PARAM`]: /common/status/README.md#INFINI_STATUS_BAD_PARAM
[`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`]: /common/status/README.md#INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED
[`INFINI_STATUS_BAD_TENSOR_SHAPE`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_SHAPE
[`INFINI_STATUS_BAD_TENSOR_DTYPE`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_DTYPE
[`INFINI_STATUS_BAD_TENSOR_STRIDES`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_STRIDES
