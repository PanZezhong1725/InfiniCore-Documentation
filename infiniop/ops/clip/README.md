# `Clip`

`Clip`，即**裁剪**算子。用于将输入张量的元素值限制在指定的最小值和最大值范围内。对于超出范围的值，将被裁剪到范围边界。

对于输入张量 $x$，以及两个标量参数 $min\_val$和 $max\_val$，输出张量 $y$ 中的每个元素按以下规则计算：

$$
y_i = \begin{cases}
min\_ val & \text{if } x_i < min\_ val \\
max\_ val & \text{if } x_i > max\_ val \\
x_i & \text{otherwise}
\end{cases}
$$

例如，对于输入张量 $x = [-1.5, 0.5, 2.5]$，$min\_ val = -1.0$，$max\_ val = 2.0$，输出将是 $y = [-1.0, 0.5, 2.0]$。

## 接口

### 计算

```c
infiniStatus_t infiniopClip(
    infiniopClipDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    float min_val,                      
    float max_val, 
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
- `min_val`:
  裁剪的最小值。
- `max_val`:
  裁剪的最大值。


### 创建算子描述符

```c
infiniStatus_t infiniopCreateClipDescriptor(
    infiniopHandle_t handle,
    infiniopClipDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`:
  `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]
- `desc_ptr`:
  `infiniopClipDescriptor_t` 指针，指向将被初始化的算子描述符地址。
- `y` - $\{ dT | shape | strides_{dst} \}$:
  算子输出的张量描述。
- `x` - $\{ dT | shape | strides_{src} \}$:
  算子输入的张量描述。


<div style="background-color: lightblue; padding: 1px;"> 参数限制：</div>

- $dT$: `INFINI_DTYPE_F16`, `INFINI_DTYPE_F32`, `INFINI_DTYPE_F64`。
- $shape$: 任意形状。
- $strides_{out}$: 任意布局。
- $strides_{in}$: 任意布局。



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

### 销毁算子描述符

```c
infiniStatus_t infiniopDestroyClipDescriptor(
    infiniopClipDescriptor_t desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  输入。待销毁的算子描述符。

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

- 暂无
<!-- 链接 -->
[`InfiniopHandle_t`]: /infiniop/handle/README.md
[`INFINI_STATUS_SUCCESS`]: /common/status/README.md#INFINI_STATUS_SUCCESS
[`INFINI_STATUS_BAD_PARAM`]: /common/status/README.md#INFINI_STATUS_BAD_PARAM
[`INFINI_STATUS_BAD_TENSOR_SHAPE`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_SHAPE
[`INFINI_STATUS_BAD_TENSOR_DTYPE`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_DTYPE
[`INFINI_STATUS_BAD_TENSOR_STRIDES`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_STRIDES
[`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`]: /common/status/README.md#INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED
[`INFINI_STATUS_INTERNAL_ERROR`]: /common/status/README.md#INFINI_STATUS_INTERNAL_ERROR