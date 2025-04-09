# 张量描述（`InfiniopTensorDescriptor_t`）

张量（Tensor）是一种表示多维数据的数据结构，可用来表示算子的输入与输出。一个张量由描述它数据形态的元信息和实际数据两部分组成。在 *InfiniOP* 中，张量的元信息和数据是解耦的。算子描述符（`InfiniopTensorDescriptor_t`）被用来保存某一张量的元信息。

算子描述符会记录张量的三个重要信息：数据类型（data type）、形状（shape）、步长（strides）。一个张量中的所有数据的类型必须相同，支持的数据类型详见 [`InfiniDtype_t`]。形状和步长是两个长度相同的数组。形状包含每个维度的元素个数，步长包含每个维度从一个元素到下一个元素的元素数间隔。

## 张量数据的排布与索引

`InfiniOP` 的张量描述比起一些传统的张量表示方式（如 ONNX）额外引入了步长信息，可以表示更复杂的张量数据排布方式。对于一些不涉及数学计算的操作，如转置（transpose）、切片（slice）等，可以通过只变换元信息来表示，无需移动数据。这为用户提供了更多的性能优化空间。当用户确实需要重新排布数据时，可以使用 [`Rearrange`] 算子。

引入步长信息后，张量数据的索引方式如下：假设一个张量的数据起始地址为 `addr`，数据类型为 `T`, 每个元素的数据大小为 `size(T)` 个字节，形状为 `(d1, d2)`，步长为 `(s1, s2)`，那么该张量第3行第2列的数据地址的计算方式为：

```c
addr + ((3 -1) * s1 + (2 - 1) * s2) * size(T)
```

当形状和步长的长度为0时，代表标量。当某一个维的步长为0且长度不为1时，则该维度为广播维度，同样的数据会在此维度上重复出现。当一个维度的步长为负数时，此维度的元素索引方向为反向。

## 算子对张量的限制

不同的算子对输入输出的张量的数据类型、形状和步长往往有各种各样的限制。在文档中，会使用 `{数据类型; 形状; 步长}` 的写法来对编程接口所涉及的张量进行描述和限定。

示例：

- `{ float32 | (d1, d2) | (..., 1) }`：类型为 `float32`，维度为两维（用`d1`、`d2`表示），最后一维步长为 `1`（即该维度数据连续排布），其他维度步长不做限制。

- `{ dT | ((b,) h, w, c) | (...) }`：类型可能支持多种，用 `dT` 表示（后续可增加限制说明），维度可能为三维或四维（`b` 维度可有可无），步长不做限制。

- `{ dT | ((...,) m, n) | (~) }`：类型可能支持多种，维度为二维或以上，步长需满足所有维度全部连续。

## 接口

### 创建张量描述

```c
infiniStatus_t infiniopCreateTensorDescriptor(
    infiniopTensorDescriptor_t *desc_ptr, 
    size_t ndim, 
    const size_t *shape, 
    const ptrdiff_t *strides, 
    infiniDtype_t dtype
);
```

- `desc_ptr`: 存储将被创建的张量描述的地址。
- `ndim`: 张量的维度数量，维度数量为 $0$ 时则为标量。
- `shape`: 形状数组地址，标量可传入 `nullptr`，其他情况用户需要传入长度足够的有效地址。
- `strides`: 步长数组地址，若传入 `nullptr` 则默认数据完全连续，其他情况用户需要转入长度足够的有效地址，步长支持负数。
- `dtype`: 数据类型，详见 [`InfiniDtype_t`]。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_NULL_POINTER`]

### 销毁张量描述符

```c
infiniStatus_t infiniopDestroyTensorDescriptor(
    infiniopTensorDescriptor_t desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`: 待销毁的张量描述。

<div style="background-color: lightblue; padding: 1px;"> 返回值： </div>

- [`INFINI_STATUS_SUCCESS`]

<!-- 链接 -->
[`InfiniDtype_t`]: /common/dtype/README.md
[`INFINI_STATUS_SUCCESS`]: /common/status/README.md#INFINI_STATUS_SUCCESS
[`INFINI_STATUS_NULL_POINTER`]: /common/status/README.md#INFINI_STATUS_NULL_POINTER
[`Rearrange`]: /infiniop/ops/rearrange/README.md
