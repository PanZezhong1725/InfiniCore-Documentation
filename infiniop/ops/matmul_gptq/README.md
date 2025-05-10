
# `MatmulGptq`

`MatmulGptq (Matmul Gradient Pre-Trained Quantization)`, 是一种针对大语言模型的高效后量化方法，旨在将模型权重从高精度（如 f16）量化为低精度（如 int4），同时最小化量化误差对模型性能的影响。

其中量化过程如下所示：

  $$
  q_{n, k} = clip\left( \left\lfloor \frac{w_{n, k}}{s_{n, g}} + z_{n, g} \right\rfloor, 0, 15 \right)
  $$

- `Scale` 是一个形状为 `( N, num_groups )` 的张量， $s_{n, g}$ 是 `Scale` 的第 $(n, g)$ 个元素。 
- `Zero` 是一个形状为 `( N, num_groups )` 的张量， $z_{n, g}$ 是 `Zero` 的第 $(n, g)$ 个元素。
- `W` 是一个形状为 `( N, K )` 的权重张量，如果 num_groups > 1 ，上面这个公式对于 $g \times$ group_size $\leq k < (g + 1) \times$ group_size 成立，其中 group_size = K / num_groups 。当 num_groups = 1 时，上述公式对于 $0 \leq k \leq K - 1$ 成立。
- `Q` 是一个形状为 `( N, K / 8 )` ，数据类型为 int32_t 的张量，一个元素存储 8 个 int4 类型的量化结果 $q_{n, k}$ 。


`Scale` ， `Zero` 是根据权重 `W` 生成的缩放因子和零点， `Scale` 和 `Zero` 的生成方式大体如下所示：         

  $$
  s_{n, g} = \frac{\max_{k} \{w_{n, k} \} - \min_{k} \{w_{n, k} \}}{15}, \\
  z_{n, g} = \left\lfloor \frac{- \min_{k} \{w_{n, k}\}}{s_{n, g}}  \right\rfloor
  $$

关于一些细节的补充可以参考 https://zhuanlan.zhihu.com/p/692338716 ，源代码参考 https://github.com/IST-DASLab/gptq 。

这个算法希望找到一个量化过的权重 $\hat{W}$ ，使得新权重和旧权重之间输出结果差别最小，即：

  $$
  \arg \min_{\hat{W}} || \hat{W}X - WX||^2
  $$

其中 $\hat{W}$ 是 $(q_{n,k} - z_{n,g}) \times s_{n,g}$ 构成的量化权重张量。

- `X` 为输入张量，形状为 `( K, M )`。
- `C` 为输出张量，存储计算结果 $\hat{W}X$ ，形状为 `( N, M )`。

## 接口

### 量化

```c
infiniStatus_t infiniopMatmulQuant(
    infiniopMatmulGptqDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *packed_weights,
    void *b_scale,
    void *zero,
    const void *a,
    const void *b,
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
- `packed_weights`:
  输出量化结果张量。张量限制见[创建算子描述](#创建算子描述)部分。
- `b_scale`:
  输出缩放因子张量。张量限制见[创建算子描述](#创建算子描述)部分。
- `zero`:
  输出零点张量。张量限制见[创建算子描述](#创建算子描述)部分。
- `a`:
  输入张量。张量限制见[创建算子描述](#创建算子描述)部分。
- `b` - { dT | ( N, K) | (~) }:
  输入权重张量。
- `stream`:
  计算流/队列。

### 计算

```c
infiniStatus_t infiniopMatmulGptq(
    infiniopMatmulGptqDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *c,
    const void *a,
    void *packed_weights,
    void *b_scale,
    void *zero,
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
- `packed_weights`:
  输入量化结果张量。张量限制见[创建算子描述](#创建算子描述)部分。
- `b_scale`:
  输入缩放因子张量。张量限制见[创建算子描述](#创建算子描述)部分。
- `zero`:
  输入零点张量。张量限制见[创建算子描述](#创建算子描述)部分。
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
    infiniopTensorDescriptor_t packed_weights_desc,
    infiniopTensorDescriptor_t b_scale_desc,
    infiniopTensorDescriptor_t zero_desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`
 : 硬件控柄。详见 [`InfiniopHandle_t`]。
- `desc_ptr`:
  `infiniopMatmulGptqDescriptor_t` 指针，指向将被初始化的算子描述符地址。
- `c_desc` - { dT | ( N, M) | (~) }:
  算子计算参数 `c` 的张量描述。
- `a_desc` - { dT | ( K, M) | (~) }:
  算子计算参数 `a` 的张量描述。
- `packed_weights_desc` - { dI | ( N, K / 8) | (~) }:
  算子计算参数 `q` 的张量描述。
- `b_scale_desc` - { dT | ( N, num_groups) | (~) }:
  算子计算参数 `b_scale` 的张量描述。
- `zero_desc` - { dT | ( N, num_groups) | (~) }:
  算子计算参数 `zero` 的张量描述。

参数限制：

- `dT`:  (`Float16`, `Float32`) 之一。
- `dI`:  `Int4` 。

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
