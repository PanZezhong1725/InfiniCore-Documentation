
# `Matmul`

`Matmul`，即**矩阵乘法**算子。计算公式为：

$$ C = α ⋅ (A * B) + β ⋅ C $$

其中：

- `A` 为左输入张量，形状为 `( [batch,] M, K )`；
- `B` 为右输入张量，形状为 `( [batch,] K, N )`；
- `C` 的形状由矩阵乘法规则确定，形状为 `( [batch,] M, N )`；
- `α` 为缩放因子，`β` 为累加系数；

## 接口

### 计算

```c
infiniopStatus_t infiniopMatmul(
    infiniopMatmulDescriptor_t desc,
    void *workspace,
    uint64_t workspace_size,
    void *c,
    const void *a,
    const void *b,
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  输入。已使用 `infiniopCreateMatmulDescriptor()` 初始化的算子描述符；
- `workspace`:
  输入。Device 指针，指向算子计算所需的额外工作空间；
- `workspace_size`:
  输入。`workspace` 的大小，单位：字节；
- `c`:
  输出。Device 指针，计算输出结果。张量限制见[创建算子描述](#创建算子描述)部分；
- `a`:
  输入。Device 常量指针，左输入张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `b`:
  输入。Device 常量指针，右输入张量。张量限制见[创建算子描述](#创建算子描述)部分；
- `stream`:
  输入。计算流/队列；

<div style="background-color: lightblue; padding: 1px;">  返回值：</div>

- [`INFINIOP_STATUS_SUCCESS`], [`INFINIOP_STATUS_BAD_PARAM`], [`INFINIOP_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINIOP_STATUS_BAD_DEVICE`], [`INFINIOP_STATUS_EXECUTION_FAILED`].

### 创建算子描述

```c
infiniopStatus_t infiniopCreateMatmulDescriptor(
    infiniopHandle_t handle,
    infiniopMatmulDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    float alpha,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    float beta
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`:、
  输入。`infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]
- `desc_ptr`:
  输出。Host `infiniopCreateMatmulDescriptor` 指针，指向将被初始化的算子描述符地址；
- `c_desc` - { dT | ( [batch,] , M, N) | (~) }:
  输出。算子计算参数 `c` 的张量描述；
- `alpha` - float:
  输入。算子计算缩放因子；
- `a_desc` - { dT | ( [batch,] , M, K) | (~) }:
  输入。算子计算参数 `a` 的张量描述；
- `b_desc` - { dT | ( [batch,] , K, N) | (~) }:
  输入。算子计算参数 `b` 的张量描述；
- `beta` - float:
  输入。算子计算累加系数；

<div style="background-color: lightblue; padding: 1px;"> 参数限制：</div>

参数限制：

- `dT`: (`Float16`, `Float32`) 之一；
- `[batch,]`: `batch ≥ 1`（可选）；
- `M`: M > 0；
- `N`: N > 0；
- `K`: K > 0；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINIOP_STATUS_SUCCESS`], [`INFINIOP_STATUS_BAD_PARAM`], [`INFINIOP_STATUS_BAD_TENSOR_SHAPE`], [`INFINIOP_STATUS_BAD_TENSOR_DTYPE`], [`INFINIOP_STATUS_BAD_TENSOR_STRIDES`], [`INFINIOP_STATUS_BAD_DEVICE`].

### 计算额外工作空间

```c
infiniopStatus_t infiniopGetMatmulWorkspaceSize(
    infiniopMatmulDescriptor_t desc,
    uint64_t *size
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`:
  输入。已使用 `infiniopCreateMatmulDescriptor()` 初始化的算子描述符；
- `size`:
  输出。Host 指针，额外空间大小的计算结果的写入地址；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINIOP_STATUS_SUCCESS`], [`INFINIOP_STATUS_BAD_PARAM`], [`INFINIOP_STATUS_BAD_DEVICE`].

### 销毁算子描述符

```c
infiniopStatus_t infiniopDestroyMatmulDescriptor(
    infiniopMatmulDescriptor_t desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  输入。待销毁的算子描述符；

<div style="background-color: lightblue; padding: 1px;"> 返回值： </div>

- [`INFINIOP_STATUS_SUCCESS`], [`INFINIOP_STATUS_BAD_DEVICE`].

[`InfiniopHandle_t`]:/

[`INFINIOP_STATUS_SUCCESS`]:/
[`INFINIOP_STATUS_BAD_PARAM`]:/
[`INFINIOP_STATUS_INSUFFICIENT_WORKSPACE`]:/
[`INFINIOP_STATUS_BAD_DEVICE`]:/
[`INFINIOP_STATUS_EXECUTION_FAILED`]:/
[`INFINIOP_STATUS_BAD_TENSOR_SHAPE`]:/
[`INFINIOP_STATUS_BAD_TENSOR_DTYPE`]:/
[`INFINIOP_STATUS_BAD_TENSOR_STRIDES`]:/
