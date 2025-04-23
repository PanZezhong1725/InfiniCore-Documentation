# `Clip`

`Clip`，即**裁剪**算子。用于将输入张量的元素值限制在指定的最小值和最大值范围内。对于超出范围的值，将被裁剪到范围边界。

对于输入张量 $x$，以及两个标量参数 $min\_val$ 和 $max\_val$，输出张量 $y$ 中的每个元素按以下规则计算：

$$
y_i = \begin{cases}
min\_val & \text{if } x_i < min\_val \\
max\_val & \text{if } x_i > max\_val \\
x_i & \text{otherwise}
\end{cases}
$$

例如，对于输入张量 $x = [-1.5, 0.5, 2.5]$，$min\_val = -1.0$，$max\_val = 2.0$，输出将是 $y = [-1.0, 0.5, 2.0]$。

## 接口

### 计算

```c
infiniStatus_t infiniopClip(
    infiniopClipDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  已使用 `infiniopCreateClipDescriptor()` 初始化的算子描述符。
- `workspace`:
  算子执行所需的工作空间。
- `workspace_size`:
  工作空间大小，以字节为单位。
- `output`:
  计算输出结果。
- `input`:
  输入张量。
- `stream`:
  计算流/队列。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`].

### 创建算子描述符

```c
infiniStatus_t infiniopCreateClipDescriptor(
    infiniopHandle_t handle,
    infiniopClipDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    float min_val,
    float max_val
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`:
  `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]
- `desc_ptr`:
  `infiniopClipDescriptor_t` 指针，指向将被初始化的算子描述符地址。
- `output_desc` - $\{ dT | shape | strides_{out} \}$:
  算子输出的张量描述。
- `input_desc` - $\{ dT | shape | strides_{in} \}$:
  算子输入的张量描述。
- `min_val`:
  裁剪的最小值。
- `max_val`:
  裁剪的最大值。

<div style="background-color: lightblue; padding: 1px;"> 参数限制：</div>

- $dT$: 支持 `INFINI_DTYPE_F16`, `INFINI_DTYPE_F32`, `INFINI_DTYPE_F64`。
- $shape$: 任意形状。
- $strides_{out}$: 任意布局。
- $strides_{in}$: 任意布局。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 查询工作空间大小

```c
infiniStatus_t infiniopGetClipWorkspaceSize(
    infiniopClipDescriptor_t desc,
    size_t *workspace_size
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  输入。已初始化的算子描述符。
- `workspace_size`:
  输出。算子执行所需的工作空间大小，以字节为单位。

<div style="background-color: lightblue; padding: 1px;"> 返回值： </div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`].

### 销毁算子描述符

```c
infiniStatus_t infiniopDestroyClipDescriptor(
    infiniopClipDescriptor_t desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  输入。待销毁的算子描述符。

<div style="background-color: lightblue; padding: 1px;"> 返回值： </div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`].

## 实现细节

Clip 算子是一个 elementwise 操作，它利用 InfiniCore 的 elementwise 基建实现。主要组件包括：

- 描述符类 (Descriptor)：
  - 继承自 InfiniopDescriptor
  - 存储 min_val 和 max_val 参数
  - 包含 calculate 方法实现计算逻辑
- 操作符类 (ClipOp)：
  - 定义 operator() 函数实现元素级操作
  - 支持不同数据类型（如 float, double, half）
  - 对于 CUDA 实现，包含针对 half2 类型的优化

## 已知问题

- 当 min_val > max_val 时，行为未定义，建议用户确保 min_val <= max_val。
- 对于非常小的值（接近浮点精度限制），可能会出现舍入误差。

<!-- 链接 -->
[`InfiniopHandle_t`]: /infiniop/handle/README.md
[`INFINI_STATUS_SUCCESS`]: /common/status/README.md#INFINI_STATUS_SUCCESS
[`INFINI_STATUS_BAD_PARAM`]: /common/status/README.md#INFINI_STATUS_BAD_PARAM
[`INFINI_STATUS_BAD_TENSOR_SHAPE`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_SHAPE
[`INFINI_STATUS_BAD_TENSOR_DTYPE`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_DTYPE
[`INFINI_STATUS_BAD_TENSOR_STRIDES`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_STRIDES
[`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`]: /common/status/README.md#INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED
[`INFINI_STATUS_INTERNAL_ERROR`]: /common/status/README.md#INFINI_STATUS_INTERNAL_ERROR