
# `Causal Softmax`

`Causal Softmax` 是使用 causal mask 的 softmax 函数，其中指数变换的操作维度限定在最后一维，适用于各类因果类模型。
在 `Softmax` 的基础上引入 mask ，对于形状为 $[s_0,\ldots, s_{r-1}]$ 的输入张量 $x$ 来说，mask = $s_{r - 1} - s_{r - 2} \geq 0$ 。
以形状为 $[4, 7]$ 的张量 $x$ 举例，mask 变换如下所示：

$$ \left[\begin{gathered}
     x_{0,0} & x_{0,1} & x_{0, 2} & x_{0, 3} & x_{0, 4} & x_{0,5} & x_{0, 6}\\
     x_{1,0} & x_{1,1} & x_{1, 2} & x_{1, 3} & x_{1, 4} & x_{1,5} & x_{1, 6}\\
     x_{2,0} & x_{2,1} & x_{2, 2} & x_{2, 3} & x_{2, 4} & x_{2,5} & x_{2, 6}\\
     x_{3,0} & x_{3,1} & x_{3, 2} & x_{3, 3} & x_{3, 4} & x_{3,5} & x_{3, 6}
    \end{gathered}\right]  \Rightarrow $$
$$ \left[\begin{gathered}
     x_{0,0} & x_{0,1} & x_{0, 2} & x_{0, 3} & 0 & 0 & 0\\
     x_{1,0} & x_{1,1} & x_{1, 2} & x_{1, 3} & x_{1, 4} & 0 & 0\\
     x_{2,0} & x_{2,1} & x_{2, 2} & x_{2, 3} & x_{2, 4} & x_{2,5} & 0\\
     x_{3,0} & x_{3,1} & x_{3, 2} & x_{3, 3} & x_{3, 4} & x_{3,5} & x_{3, 6}
    \end{gathered}\right] $$

经过 mask 变换以后针对最后一维做 softmax 变换即可，一维向量的 softmax 变换参考：

$$ y_i = \frac{e^{x_i}}{\sum_{i=0}^{N - 1} e^{x_i}} $$  

高维向量的 `Causal Softmax` 只需要考虑最后两维即可。

### 计算

```c
infiniStatus_t infiniopCausalSoftmax(
    infiniopCausalSoftmaxDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *data,
    void *stream
);
```
<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

 - `desc`:
     使用 `infiniopCreateCausalSoftmaxDescriptor()` 初始化的算子描述符。
 - `workspace`:
     算子计算所需的额外工作空间。
 - `workspace_size`:
     `workspace` 的大小，单位：字节（byte）。
 - `data`:
     输入以及计算结果的数据地址。张量限制见[创建算子描述](#创建算子描述)部分。
 - `stream`:
     计算流/队列。

参数限制：

 - `data` 仅支持原地计算。

<div style="background-color: lightblue; padding: 1px;">  返回值：</div>

 - [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_DEVICE`], [`INFINI_STATUS_EXECUTION_FAILED`]

---

### 创建算子描述

```c
infiniStatus_t infiniopCreateCausalSoftmaxDescriptor(
    infiniopHandle_t handle,
    infiniopCausalSoftmaxDescriptor_t *desc_ptr,  
    infiniopTensorDescriptor_t t_desc
);
```
<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

 - `handle`:
     `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]
 - `desc_ptr`:
     存放将被初始化的算子描述符的地址。
 - `t_desc` - { dT | ((batch,) total, seqlen) | ($\ldots,1$) }:
     算子计算参数 `t_desc` 的张量描述，三维或者两维，最后一维连续。

参数限制：

 - **`dT`**:  (`Float16`, `Float32`, `Double`, `Bfloat16`) 之一

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

 - [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`],  [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_BAD_DEVICE`].

---

### 计算额外工作空间

```c
infiniStatus_t infiniopGetCausalSoftmaxWorkspaceSize(
    infiniopCausalSoftmaxDescriptor_t desc,
    size_t *size
);
```
<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

 - `desc`:
     使用 `infiniopCreateCausalSoftmaxDescriptor()` 初始化的算子描述符。
 - `size`:
     存放额外空间大小的计算结果的地址。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

 - [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_DEVICE`].

---

### 销毁算子描述符

```c
infiniopStatus_t infiniopDestroyCausalSoftmaxDescriptor(
    infiniopCausalSoftmaxDescriptor_t desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

 - `desc`:
     待销毁的算子描述符。

<div style="background-color: lightblue; padding: 1px;"> 返回值： </div>

 - [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_DEVICE`].

[`InfiniopHandle_t`]: /

[`INFINI_STATUS_SUCCESS`]: /
[`INFINI_STATUS_BAD_PARAM`]: /
[`INFINI_STATUS_INSUFFICIENT_WORKSPACE`]: /
[`INFINI_STATUS_BAD_DEVICE`]: /
[`INFINI_STATUS_EXECUTION_FAILED`]: /
[`INFINI_STATUS_BAD_TENSOR_SHAPE`]: /
[`INFINI_STATUS_BAD_TENSOR_DTYPE`]: /
[`INFINI_STATUS_BAD_TENSOR_STRIDES`]: /
