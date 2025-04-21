
# `MatmulGptq`

`MatmulGptq (Matmul Gradient Pre-Trained Quantization)`, 是一种针对大语言模型的高效后量化方法，旨在将模型权重从高精度（如 f16）量化为低精度（如 int4），同时最小化量化误差对模型性能的影响。

其中量化过程如下所示：

  $$
  q_{k,n} = clip\left( \left\lfloor \frac{w_{k,n}}{s_{g,n}} + z_{g,n} \right\rfloor, -8, 7 \right)
  $$

- `Scale` 是一个形状为 `( num_groups, N )` 的张量，$s_{g,n}$ 是 Scale 的第 $(g, n)$ 个元素。 
- `Zero` 是一个形状为 `( num_groups, N )` 的张量，$z_{g,n}$ 是 Zero 的第 $(g, n)$ 个元素。
- `W` 是一个形状为 `( K, N )` 的张量，上面这个公式对于 $k \in [g \times group\_{}size, (g + 1) \times group\_{}size)$成立，其中 $group\_{}size = 128, K = num\_{}groups \times group\_{}size$ 。

反量化过程如下所示：

  $$
  \hat{w}_{k,n} = (q_{k,n} - z_{g,n}) \times s_{g,n}
  $$

最终得到计算结果，其中 $\hat{W}$ 是 $\hat{w}_{k,n}$ 构成的新矩阵。

  $$
  C = A * \hat{W}
  $$

- `A` 为左输入张量，形状为 `( M, K )`。
- `C` 为输出张量，形状为 `( M, N )`。

实际操作过程中会将量化以后的结果以 int4 的方式存储在一个形状为 $(num\_{}groups, 2 \times N)$ ，数据类型为 int32_t 的中间矩阵中。
## 接口

### 计算

```c
infiniStatus_t infiniopMatmulGptq(
    infiniopMatmulGptqDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *c,
    const void *a,
    const void *b,
    const void *b_scale,
    const void *zero,
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  已使用 `infiniopCreateMatmulGptqDescriptor()` 初始化的算子描述符。
- `workspace`:
  额外工作空间。
- `workspace_size`:
  `workspace` 的大小，单位：字节。
- `c`:
  计算输出结果。张量限制见[创建算子描述](#创建算子描述)部分。
- `a`:
  左输入张量。张量限制见[创建算子描述](#创建算子描述)部分。
- `b`:
  右输入张量。张量限制见[创建算子描述](#创建算子描述)部分。
- `b_scale`:
  缩放因子张量。张量限制见[创建算子描述](#创建算子描述)部分。
- `zero`:
  零点张量。张量限制见[创建算子描述](#创建算子描述)部分。
- `stream`:
  计算流/队列。

<div style="background-color: lightblue; padding: 1px;">  返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_NULL_POINTER`], [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`].

### 创建算子描述

```c
infiniStatus_t infiniopCreateMatmulGptqDescriptor(
    infiniopHandle_t handle,
    infiniopMatmulGptqDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t b_scale_desc,
    infiniopTensorDescriptor_t zero_desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`
 : 硬件控柄。详见 [`InfiniopHandle_t`]。
- `desc_ptr`:
  `infiniopMatmulGptqDescriptor_t` 指针，指向将被初始化的算子描述符地址。
- `c_desc` - { dT | ( M, N) | (~) }:
  算子计算参数 `c` 的张量描述。
- `a_desc` - { dT | ( M, K) | (~) }:
  算子计算参数 `a` 的张量描述。
- `b_desc` - { dT | ( K, N) | (~) }:
  算子计算参数 `b` 的张量描述。
- `b_scale_desc` - { dT | ( num_groups, N) | (~) }:
  算子计算参数 `b_scale` 的张量描述。
- `zero_desc` - { dT | ( num_groups, N) | (~) }:
  算子计算参数 `zero` 的张量描述。

参数限制：

- `dT`:  (`Float16`, `Float32`) 之一。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`],  [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 计算额外工作空间

```c
infiniStatus_t infiniopGetMatmulGptqWorkspaceSize(
    infiniopMatmulGptqDescriptor_t desc, 
    size_t *size
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`:
  已使用 `infiniopCreateMatmulGptqDescriptor()` 初始化的算子描述符。
- `size`:
  额外空间大小的计算结果的写入地址。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 销毁算子描述符

```c
infiniStatus_t infiniopDestroyMatmulGptqDescriptor(
    infiniopMatmulGptqDescriptor_t desc
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
