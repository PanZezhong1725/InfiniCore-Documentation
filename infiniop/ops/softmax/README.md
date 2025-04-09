
# `Softmax`

`Softmax` 对张量做指数变换，进而做归一化得到另一个张量。
对于长度为 $N$ 的一维张量 $x$ ，数学变换为：

$$ y_i = \frac{e^{x_i}}{\sum_{i=0}^{N - 1} e^{x_i}} $$

高维张量的 softmax 只需要在指定 axis 做上述数学变换即可。例如，形状为 $[N,C,H,W]$ 的四维向量 $x$ ，axis = 1，对应数学变换如下所示：

$$ y_{i,j,k,s} = \frac{e^{x_{i,j,k,s}}}{\sum_{j=0}^{C - 1} e^{x_{i,j,k,s}}} $$

## 接口

### 计算

```c
infiniStatus_t infiniopSoftmax(
    infiniopSoftmaxDescriptor_t desc, 
    void *output, 
    const void *input, 
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`
     : 已使用 `infiniopCreateSoftmaxDescriptor()` 初始化的算子描述符。
- `input`
     : 输入指针，张量限制见[创建算子描述](#创建算子描述)部分。
- `output`
     : 输出指针，张量限制见[创建算子描述](#创建算子描述)部分。
- `stream`
     : 计算流/队列。

<div style="background-color: lightblue; padding: 1px;">  返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`]

---

### 创建算子描述

```c
infiniStatus_t infiniopCreateSoftmaxDescriptor(
    infiniopHandle_t handle, 
    infiniopSoftmaxDescriptor_t *desc_ptr, 
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc, 
    int axis
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`: `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]
- `desc_ptr`: `infiniopSoftmaxDescriptor_t` 指针，指向将被初始化的算子描述符地址。
- `input_desc` - { dT | ($\ldots$) | ($\ldots$) }: 算子计算参数 `input_desc` 的张量描述，数据为 $r$  维张量，其中 $r$ 是任意正整数。
- `axis`: 默认值是 -1 ，表示操作维度是最后一维，可选择范围是 $[-r, r - 1]$ 。
- `output_desc` - { dT | ($\ldots$) | ($\ldots$) } : 算子计算参数 `output_desc` 的张量描述，张量形状和 `input_desc` 保持一致。

参数限制：

- **`dT`**:  (`Float16`, `Float32`, `Double`, `Bfloat16`) 之一。
- `input_desc` 和 `output_desc` 都支持不连续步长。

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
