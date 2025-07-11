
# `Causal Softmax`

`Causal Softmax` 是使用二维因果掩码的 softmax 函数，常用于各类因果类模型。其公式如下：

$$ y = softmax(x + mask) $$

其中

$$ softmax(v_i) = \frac{e^{v_i}}{\sum_{k=0}^{N - 1} e^{v_k}} $$

对于一个形状为 $(M,N)$ 的二维输入（一般 $M\leq N$），因果掩码定义如下：

$$
mask_{i,j} =
\begin{cases}
0 & \text{if } j \leq N - M + i \\
-\infty & \text{otherwise}
\end{cases}
$$

以形状为 $[4, 7]$ 的输入为例，最终计算结果会有如下形式：

$$ \left[\begin{gathered}
     y_{0,0} & y_{0,1} & y_{0, 2} & y_{0, 3} & 0 & 0 & 0\\
     y_{1,0} & y_{1,1} & y_{1, 2} & y_{1, 3} & y_{1, 4} & 0 & 0\\
     y_{2,0} & y_{2,1} & y_{2, 2} & y_{2, 3} & y_{2, 4} & y_{2,5} & 0\\
     y_{3,0} & y_{3,1} & y_{3, 2} & y_{3, 3} & y_{3, 4} & y_{3,5} & y_{3, 6}
    \end{gathered}\right] $$

### 计算

```c
infiniStatus_t infiniopCausalSoftmax(
    infiniopCausalSoftmaxDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream
);
```
<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

 - `desc`: 使用 `infiniopCreateCausalSoftmaxDescriptor()` 初始化的算子描述符。
 - `workspace`: 算子计算所需的额外工作空间。
 - `workspace_size`: `workspace` 的大小，单位：字节（byte）。
 - `y`: 计算结果的数据地址。张量限制见[创建算子描述](#创建算子描述)部分。
 - `x`: 输入数据地址，可以与 `y` 相同。张量限制见[创建算子描述](#创建算子描述)部分。
 - `stream`: 计算流/队列。

<div style="background-color: lightblue; padding: 1px;">  返回值：</div>

 - [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`]

---

### 创建算子描述

```c
infiniStatus_t infiniopCreateCausalSoftmaxDescriptor(
    infiniopHandle_t handle,
    infiniopCausalSoftmaxDescriptor_t *desc_ptr,  
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc
);
```
<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

 - `handle`: `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]
 - `desc_ptr`: 存放将被初始化的算子描述符的地址。
 - `y_desc` - { dT | ((batch,) total, seqlen) | ($\ldots,1$) }:
     算子计算参数 `y` 的张量描述，三维或者两维，最后一维连续。
 - `x_desc` - { dT | ((batch,) total, seqlen) | ($\ldots,1$) }:
     算子计算参数 `x` 的张量描述，形状与 `y_desc` 一致，最后一维连续。

参数限制：

 - **`dT`**:  (`Float16`, `Float32`, `Float64`, `Bfloat16`) 之一

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

 - [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`],  [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

---

### 计算额外工作空间

```c
infiniStatus_t infiniopGetCausalSoftmaxWorkspaceSize(
    infiniopCausalSoftmaxDescriptor_t desc,
    size_t *size
);
```
<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

 - `desc`: 使用 `infiniopCreateCausalSoftmaxDescriptor()` 初始化的算子描述符。
 - `size`: 存放额外空间大小的计算结果的地址。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

 - [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

---

### 销毁算子描述符

```c
infiniopStatus_t infiniopDestroyCausalSoftmaxDescriptor(
    infiniopCausalSoftmaxDescriptor_t desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

 - `desc`: 待销毁的算子描述符。

<div style="background-color: lightblue; padding: 1px;"> 返回值： </div>

 - [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

<!-- 链接 -->
[`InfiniopHandle_t`]: /infiniop/handle/README.md

[`INFINI_STATUS_SUCCESS`]: /common/status/README.md#INFINI_STATUS_SUCCESS
[`INFINI_STATUS_BAD_PARAM`]: /common/status/README.md#INFINI_STATUS_BAD_PARAM
[`INFINI_STATUS_INSUFFICIENT_WORKSPACE`]: /common/status/README.md#INFINI_STATUS_INSUFFICIENT_WORKSPACE
[`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`]: /common/status/README.md#INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED
[`INFINI_STATUS_INTERNAL_ERROR`]: /common/status/README.md#INFINI_STATUS_INTERNAL_ERROR
[`INFINI_STATUS_BAD_TENSOR_SHAPE`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_SHAPE
[`INFINI_STATUS_BAD_TENSOR_DTYPE`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_DTYPE
[`INFINI_STATUS_BAD_TENSOR_STRIDES`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_STRIDES
