
# `Softmax`

$\bullet$ `Softmax` 对张量做指数变换，进而做归一化得到另一个张量。    
$\bullet$ 对于长度为 $N$ 的一维张量 $x$ ，数学变换为：$ y_i = \frac{e^{x_i}}{\sum_{i=0}^{N - 1} e^{x_i}} $ 。    
$\bullet$ 高维张量的 softmax 只需要在指定 axis 做上述数学变换即可。例如，形状为 $[N,C,H,W]$ 的四维向量 $x$ ，axis = 1，对应数学变换如下所示：

$$ y_{i,j,k,s} = \frac{e^{x_{i,j,k,s}}}{\sum_{j=0}^{C - 1} e^{x_{i,j,k,s}}} $$



## 接口

### 创建算子描述

```c
infiniopStatus_t infiniopCreateSoftmaxDescriptor(
    infiniopHandle_t handle, 
    infiniopSoftmaxDescriptor_t *desc_ptr, 
    infiniopTensorDescriptor_t input_desc, 
    int axis, 
    infiniopTensorDescriptor_t output_desc
);
```
<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

 - `handle`    
     : 输入。`infiniopHandle_t` 类型的硬件控柄。详情请看：[InfiniopHandle_t]()
 - `desc_ptr`    
     : 输出。Host `infiniopSoftmaxDescriptor_t` 指针，指向将被初始化的算子描述符地址。
 - `input_desc` - {dT}          
     : 输入。算子计算参数 `input_desc` 的张量描述，数据为 $r$维张量，其中 $r$ 是任意正整数。
 - `axis` ：int      
     : 输入。默认值是-1，可选择范围是 $[-r, r - 1]$。
 - `output_desc` - {dT}      
     : 输入。算子计算参数 `output_desc` 的张量描述，张量形状和 `input_desc` 保持一致。

参数限制：

 - **`dT`**:  (`Float16`, `Float32`, `Double`, `Bfloat16`) 之一。
 - 支持不连续步长。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

 - [`INFINIOP_STATUS_SUCCESS`](), [`INFINIOP_STATUS_BAD_PARAM`](),  [`INFINIOP_STATUS_BAD_TENSOR_SHAPE`](), [`INFINIOP_STATUS_BAD_TENSOR_DTYPE`](), [`INFINIOP_STATUS_BAD_TENSOR_STRIDES`](), [`INFINIOP_STATUS_BAD_DEVICE`]().

---

### 计算

```c
infiniopStatus_t infiniopSoftmax(
    infiniopSoftmaxDescriptor_t desc, 
    void* const input, 
    void* output, 
    void* stream
);
```
<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

 - `desc`      
     : 输入。已使用 `infiniopCreateSoftmaxDescriptor()` 初始化的算子描述符。 
 - `input`    
     : 输入。Device 常量指针，张量限制见[创建算子描述](#创建算子描述)部分。
 - `output`    
     : 输出。Device 指针，张量限制见[创建算子描述](#创建算子描述)部分。
 - `stream`    
     : 输入。计算流/队列。

<div style="background-color: lightblue; padding: 1px;">  返回值：</div>

 - [`INFINIOP_STATUS_SUCCESS`](), [`INFINIOP_STATUS_BAD_PARAM`](), [`INFINIOP_STATUS_BAD_DEVICE`](), [`INFINIOP_STATUS_EXECUTION_FAILED`]()


---

### 销毁算子描述符

```c
infiniopStatus_t infiniopDestroySoftmaxDescriptor(
    infiniopSoftmaxDescriptor_t desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

 - `desc`    
     : 输入。 待销毁的算子描述符。 

<div style="background-color: lightblue; padding: 1px;"> 返回值： </div>

 - [`INFINIOP_STATUS_SUCCESS`](), [`INFINIOP_STATUS_BAD_DEVICE`]().

## 已知问题

### 平台限制

- 寒武纪中 tensor.to(device) 的 tensor 不支持uint64或者是int64数据类型。

### 