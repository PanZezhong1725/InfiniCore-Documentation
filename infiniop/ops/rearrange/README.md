
# `Rearrange`

`Rearrange`，即**布局重整**算子。用于在数据类型和形状相同、布局可能不同的张量之间拷贝数据。

`Rearrange` 算子关心 2 个张量，称为 `src` 和 `dst`，它们的数据类型和形状相同，但布局可能不同。由于两个张量形状相同，二者的数据元素是一一对应的。算子将 `src` 中的数据拷贝到 `dst` 中的对应位置。

例如，要转置 $2 \times 3$ 矩阵：

> - 符号表示数据元素。
> - 下标表示数据在存储空间中的位置。

$$
src:
\left(
    \begin{gathered}
    a_0, b_1, c_2\\
    d_3, e_4, f_5\\
    \end{gathered}
\right)
\stackrel{T}{\rightarrow}
dst:
\left(
    \begin{gathered}
    a_0, d_1\\
    b_2, e_3\\
    c_4, f_5\\
    \end{gathered}
\right)
$$

这个操作可以通过**张量布局变换**和**数据重整** 2 步完成：

$$
src:
\left(
    \begin{gathered}
    a_0, b_1, c_2\\
    d_3, e_4, f_5\\
    \end{gathered}
\right)
\stackrel{T_{meta}}{\rightarrow}
\left(
    \begin{gathered}
    a_0, d_3\\
    b_1, e_4\\
    c_2, f_5\\
    \end{gathered}
\right)
\stackrel{R}{\rightarrow}
dst:
\left(
    \begin{gathered}
    a_0, d_1\\
    b_2, e_3\\
    c_4, f_5\\
    \end{gathered}
\right)
$$

| 操作 | 解释
|:-:|:-
| $T$ | 转置，既转置形状布局，也移动数据位置
| $T_{meta}$ | 不改变数据在存储空间中的排布，但将形状从 $2 \times 3$ 改为 $3 \times 2$，步长从 $1,3$ 改为 $3,1$
| $R$ | 调用算子 $rearrange(shape=(3,2), strides_{dst}=(1,2),strides_{src}=(3,1))$

## 接口

### 计算

```c
infiniStatus_t infiniopRearrange(
    infiniopRearrangeDescriptor_t desc,
    void *dst,
    const void *src,
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  已使用 `infiniopCreateRearrangeDescriptor()` 初始化的算子描述符。
- `dst`:
  计算输出结果。
- `src`:
  输入张量。
- `stream`:
  计算流/队列。

<div style="background-color: lightblue; padding: 1px;">  返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`].

### 创建算子描述

```c
infiniStatus_t infiniopCreateRearrangeDescriptor(
    infiniopHandle_t handle,
    infiniopRearrangeDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t dst_desc,
    infiniopTensorDescriptor_t src_desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`:
  `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]
- `desc_ptr`:
  `infiniopCreateRearrangeDescriptor` 指针，指向将被初始化的算子描述符地址。
- `dst_desc` - $\{ dT | shape | strides_{dst} \}$:
  算子输出 `dst` 的张量描述。
- `src_desc` - $\{ dT | shape | strides_{src} \}$:
  算子计算参数 `src` 的张量描述。

<div style="background-color: lightblue; padding: 1px;"> 参数限制：</div>

参数限制：

- $dT$: 任意类型。
- $shape$: 任意形状。
- $strides_{dst}$: 任意布局，但长度不为 1 的维度不能含有 0 步长（广播）。
- $strides_{src}$: 任意布局。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 销毁算子描述符

```c
infiniStatus_t infiniopDestroyRearrangeDescriptor(
    infiniopRearrangeDescriptor_t desc
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

[`INFINI_STATUS_SUCCESS`]:/common/status/README.md#INFINI_STATUS_SUCCESS
[`INFINI_STATUS_BAD_PARAM`]:/common/status/README.md#INFINI_STATUS_BAD_PARAM
[`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`]:/common/status/README.md#INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED
[`INFINI_STATUS_INTERNAL_ERROR`]:/common/status/README.md#INFINI_STATUS_INTERNAL_ERROR
[`INFINI_STATUS_BAD_TENSOR_SHAPE`]:/common/status/README.md#INFINI_STATUS_BAD_TENSOR_SHAPE
[`INFINI_STATUS_BAD_TENSOR_DTYPE`]:/common/status/README.md#INFINI_STATUS_BAD_TENSOR_DTYPE
[`INFINI_STATUS_BAD_TENSOR_STRIDES`]:/common/status/README.md#INFINI_STATUS_BAD_TENSOR_STRIDES
