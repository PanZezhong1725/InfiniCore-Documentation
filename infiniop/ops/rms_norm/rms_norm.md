# `RMS Norm`

`RMS Norm`，即 **Root Mean Square Normalization** 算子，用于对输入张量进行归一化处理。它通过计算输入张量在指定维度上的均方根值（Root Mean Square, RMS），并将其缩放到单位范数。该算子广泛应用于神经网络中，尤其是在处理具有不同尺度的输入数据时。


对于输入张量 $X$ 和归一化维度 $D$，RMS Norm 的计算公式为：

$$
Y = \frac{X \cdot W}{\sqrt{\frac{1}{N} \sum_{i=1}^{N} X_i^2 + \epsilon}}
$$

其中：
- $N$ 是在归一化维度 $D$ 上的元素数量。
- $\epsilon$ 是一个小的常数，用于避免除以零，通常取值为 $10^{-6}$ 或 $10^{-8}$。
- $W$ 是可选的权重张量，用于对归一化后的结果进行缩放。
- $Y$ 表示归一化后的输出结果


## 接口

### 计算

```c
infiniopStatus_t infiniopRMSNorm(
    infiniopRMSNormDescriptor_t desc,
    void *workspace,
    uint64_t workspace_size,
    void *y,
    const void *x,
    const void *w,
    void *stream
);
```
<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`  
	 : 输入。已使用 `infiniopCreateRMSNormDescriptor()` 初始化的算子描述符。
- `workspace`  
	 : 输入。Device 指针，指向算子计算所需的额外工作空间。
- `workspace_size`  
	 : 输入。`workspace` 的大小，单位：字节（byte）。
- `y`  
	 : 输出。Device 指针，归一化后的结果张量。张量限制见[创建算子描述](#创建算子描述)部分。
- `x`  
	 : 输入。Device 常量指针，输入张量。张量限制见[创建算子描述](#创建算子描述)部分。
- `w`  
	 : 输入。Device 常量指针，权重张量（可选）。如果权重张量为 `NULL`，则不进行缩放。
- `stream`  
	 : 输入。计算流/队列。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINIOP_STATUS_SUCCESS`](), [`INFINIOP_STATUS_BAD_PARAM`](), [`INFINIOP_STATUS_INSUFFICIENT_WORKSPACE`](), [`INFINIOP_STATUS_BAD_DEVICE`](), [`INFINIOP_STATUS_EXECUTION_FAILED`]()

- 当 `epsilon` 超出范围时返回 `INFINIOP_STATUS_BAD_PARAM`。

---

### 创建算子描述

```c
infiniopStatus_t infiniopCreateRMSNormDescriptor(
    infiniopHandle_t handle,
    infiniopRMSNormDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    float epsilon
);
```
<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`  
  : 输入。`infiniopHandle_t` 类型的硬件控柄。详情请看：[InfiniopHandle_t]()
- `desc_ptr`  
  : 输出。Host `infiniopRMSNormDescriptor_t` 指针，指向将被初始化的算子描述符地址。
- `y_desc` - { dT | ($\ldots$) | ($\ldots, 1$) }
  : 输入。算子计算参数 `y` 的张量描述。
- `x_desc` - { dT | ($\ldots$) | ($\ldots, 1$) }
  : 输入。算子计算参数 `x` 的张量描述，形状和 `y_desc` 保持一致。
- `w_desc` - { dW | ($\ldots$) | ($\ldots, 1$) }
  : 输入。权重张量的描述。`w_desc` 为一维张量，长度和归一化维度的长度保持一致。如果权重张量为 `NULL`，则不进行缩放。
- `epsilon`  
  : 输入。用于避免除以零的小常数，范围是 (0, 1]。

参数限制：

- **`dT`**: `Float16`
- **`dW`**: (`Float16`, `Float32`) 之一

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINIOP_STATUS_SUCCESS`](), [`INFINIOP_STATUS_BAD_PARAM`](), [`INFINIOP_STATUS_BAD_TENSOR_SHAPE`](), [`INFINIOP_STATUS_BAD_TENSOR_DTYPE`](), [`INFINIOP_STATUS_BAD_TENSOR_STRIDES`](), [`INFINIOP_STATUS_BAD_DEVICE`]()

---

### 计算额外工作空间

```c
infiniopStatus_t infiniopGetRMSNormWorkspaceSize(
    infiniopRMSNormDescriptor_t desc,
    uint64_t *size
);
```
<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`  
  : 输入。已使用 `infiniopCreateRMSNormDescriptor()` 初始化的算子描述符。
- `size`  
  : 输出。Host 指针，额外空间大小的计算结果的写入地址。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINIOP_STATUS_SUCCESS`](), [`INFINIOP_STATUS_BAD_PARAM`](), [`INFINIOP_STATUS_BAD_DEVICE`]()

---

### 销毁算子描述符

```c
infiniopStatus_t infiniopDestroyRMSNormDescriptor(
    infiniopRMSNormDescriptor_t desc
);
```
<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`  
  : 输入。待销毁的算子描述符。

<div style="background-color: lightblue; padding: 1px;"> 返回值： </div>

- [`INFINIOP_STATUS_SUCCESS`](), [`INFINIOP_STATUS_BAD_DEVICE`]()

---

## 已知问题

无