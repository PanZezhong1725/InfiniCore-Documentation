# `Clip`

`Clip`，即**裁剪**算子。用于将输入张量的元素值限制在指定的最小值和最大值范围内。对于超出范围的值，将被裁剪到范围边界。

对于输入张量 $x$，以及两个与输入相同形状的张量参数 $minval$和 $maxval$，输出张量 $y$ 中的每个元素按以下规则计算：

$$
y_i = \begin{cases}
minval_i & \text{if } x_i < minval_i \\
maxval_i & \text{if } x_i > maxval_i \\
x_i & \text{otherwise}
\end{cases}
$$

每个元素都会与输入张量的对应元素进行比较。例如，对于输入张量 $x = [-1.5, 0.5, 2.5]$，minval = [-2.0, 0.0, 1.0]，maxval = [0.0, 1.0, 2.0]，输出将是 $y = [-1.5, 0.5, 2.0]$。

## 接口

### 计算

```c
infiniStatus_t infiniopClip(
    infiniopClipDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    const void *min_val,
    const void *max_val,
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
- `y`:
  计算输出结果。
- `x`:
  输入张量。
- `min_val`:
  裁剪的最小值，必须是与输入张量形状相同的张量。
- `max_val`:
  裁剪的最大值，必须是与输入张量形状相同的张量。在实现中使用了 `return max(min(x, max_val), min_val)` 的方式，这会导致当 min_val > max_val 时，输出总是等于 min_val。
- `stream`:
  计算流/队列。
<div style="background-color: lightblue; padding: 1px;"> 返回值： </div>

- [`INFINI_STATUS_SUCCESS`]
- [`INFINI_STATUS_BAD_TENSOR_DTYPE`]
- [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`]
- [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`]


### 创建算子描述符

```c
infiniStatus_t infiniopCreateClipDescriptor(
    infiniopHandle_t handle,
    infiniopClipDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    infiniopTensorDescriptor_t min_val,
    infiniopTensorDescriptor_t max_val
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`:
  `infiniopHandle_t` 类型的硬件控柄。详情请看：[`InfiniopHandle_t`]
- `desc_ptr`:
  `infiniopClipDescriptor_t` 指针，指向将被初始化的算子描述符地址。
- `y` - {dT | (d1, ..., dn) | (...)} :
  算子输出的张量描述。
- `x` - {dT | (d1, ..., dn) | (...)} :
  算子输入的张量描述。
- `min_val` - {dT | (d1, ..., dn) | (...)} :
  裁剪的最小值张量描述，必须是与输入张量形状相同的张量。
- `max_val` - {dT | (d1, ..., dn) | (...)} :
  裁剪的最大值张量描述，必须是与输入张量形状相同的张量。

参数限制：

- `dT`:  (`Float16`, `Float32`, `Float64`, `BFloat16`) 之一。
- 输入 `a` 与 `b` 的形状需与 `c` 相同。`a` 与 `b` 涉及多向广播时需调整步长以匹配多向广播的映射关系。
- 支持原位计算，即计算时 `c` 可以和 `a` 或 `b` 指向同一地址。
- 计算输出参数 `c` 不能进行广播（`c` 的步长不能涉及广播设置，即步长不能有 0）

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`]
- [`INFINI_STATUS_BAD_TENSOR_DTYPE`]
- [`INFINI_STATUS_BAD_TENSOR_SHAPE`]
- [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`]



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

- [`INFINI_STATUS_SUCCESS`]: 成功获取工作空间大小。
- [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`]: 当设备类型不受支持时。

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

- [`INFINI_STATUS_SUCCESS`]
- [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`]


<!-- 链接 -->
[`InfiniopHandle_t`]: /infiniop/handle/README.md
[`INFINI_STATUS_SUCCESS`]: /common/status/README.md#INFINI_STATUS_SUCCESS
[`INFINI_STATUS_BAD_PARAM`]: /common/status/README.md#INFINI_STATUS_BAD_PARAM
[`INFINI_STATUS_BAD_TENSOR_SHAPE`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_SHAPE
[`INFINI_STATUS_BAD_TENSOR_DTYPE`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_DTYPE
[`INFINI_STATUS_BAD_TENSOR_STRIDES`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_STRIDES
[`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`]: /common/status/README.md#INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED
[`INFINI_STATUS_INTERNAL_ERROR`]: /common/status/README.md#INFINI_STATUS_INTERNAL_ERROR
[`INFINI_STATUS_INSUFFICIENT_WORKSPACE`]: /common/status/README.md#INFINI_STATUS_INSUFFICIENT_WORKSPACE