
# `SwiGLU`

`SwiGLU (Switched Gated Linear Unit)`, 即**门控线性单元激活函数**算子，计算公式为：

$$
c_{i} = a_{i} \circ SiLU(b_{i})
$$

$$
SiLU(b_{i}) = \frac {b_{i}}{1 + e^{-b_{i}}}
$$

  其中：
  - `c_{i}`：输出张量第 `i` 个元素
  - `a_{i}`：输入张量第 `i` 个元素
  - `b_{i}` ：门控输入张量第 `i` 个元素
  

## 接口

### 计算

```c
infiniopStatus_t infiniopSwiGLU(
  infiniopSwiGLUDescriptor_t desc,
  void *c,
  const void *a,
  const void *b,
  void* stream
);
```
<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

 - `desc`
	 : 输入。已使用 `infiniopCreateSwiGLUDescriptor()` 初始化的算子描述符。 
 - `c`
	 : 输出。Device 指针，输出张量，张量限制见[创建算子描述](#创建算子描述)部分。
 - `a`
	 : 输入。Device 常量指针，输入张量，张量限制见[创建算子描述](#创建算子描述)部分。
 - `b`
	 : 输入。Device 常量指针，门控输入张量，张量限制见[创建算子描述](#创建算子描述)部分。
 - `stream`
	 : 输入。计算流/队列。

<div style="background-color: lightblue; padding: 1px;">  返回值：</div>

 - [`INFINIOP_STATUS_SUCCESS`](), [`STATUS_MEMORY_NOT_ALLOCATED`](), [`STATUS_BAD_TENSOR_SHAPE`](), [`STATUS_BAD_TENSOR_STRIDES`](), [`STATUS_BAD_TENSOR_DTYPE`]()

---

### 创建算子描述

```c
infiniopStatus_t infiniopCreateSwiGLUDescriptor(
    infiniopHandle_t handle,
    infiniopSwiGLUDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c,
    infiniopTensorDescriptor_t a,
    infiniopTensorDescriptor_t b
);
```
<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

 - `handle`
	: 输入。`infiniopHandle_t` 类型的硬件控柄。详情请看：[InfiniopHandle_t]()
 - `desc_ptr`
	 : 输出。Host `infiniopSwiGLUDescriptor_t` 指针，指向将被初始化的算子描述符地址。
 - `c` - {dT|(...)|(...)}
	 : 输出。算子计算参数 `c` 的张量描述, 支持原位计算。
 - `a` - {dT|(...)|(...)}
	 : 输入。算子计算参数 `a` 的张量描述，支持原位计算。
 - `b` - {dT|(...)|(...)}
	 : 输入。算子计算参数 `b` 的张量描述，支持原位计算。

参数限制：

 - **`dT`**:  (`Float16`, `Float32`, `Double`, `Bfloat16`) 之一
 - `c`, `a`, `b` 支持不连续步长
    
<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

 - [`INFINIOP_STATUS_SUCCESS`](), [`INFINIOP_STATUS_BAD_PARAM`](),  [`INFINIOP_STATUS_BAD_TENSOR_SHAPE`](), [`INFINIOP_STATUS_BAD_TENSOR_DTYPE`](), [`INFINIOP_STATUS_BAD_TENSOR_STRIDES`](), [`INFINIOP_STATUS_BAD_DEVICE`]().

---

### 销毁算子描述符

```c
infiniopStatus_t infiniopDestroySwiGLUDescriptor(
	infiniopSwiGLUDescriptor_t desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

 - `desc`
	 : 输入。 待销毁的算子描述符。 

<div style="background-color: lightblue; padding: 1px;"> 返回值： </div>

 - [`INFINIOP_STATUS_SUCCESS`](), [`INFINIOP_STATUS_BAD_DEVICE`]().

## 已知问题

无