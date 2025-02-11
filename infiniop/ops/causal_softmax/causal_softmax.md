
# `Causal Softmax`

$\bullet$ `Causal Softmax`是使用causal mask的softmax函数，适用于各类因果类模型。     
$\bullet$ 在`Softmax`的基础上引入mask，其中指数变换的操作维度限定在最后一维。    
$\bullet$ 对于形状为$[s_0,\ldots, s_{r-1}]$的输入张量$x$来说，mask = $s_{r - 1} - s_{r - 2} \geq 0$，以形状为$[M, N], N \geq M$的张量$x$举例，`Causal Softmax`的数学变换如下所示：

$$ \left[\begin{gathered}
     x_{0,0} & \ldots & x_{0, mask} & x_{0, mask + 1} & x_{0, mask + 2} & \ldots & x_{0, N - 1}\\
     x_{1,0} & \ldots & x_{1, mask} & x_{1, mask + 1} & x_{1, mask + 2} & \ldots & x_{1, N - 1}\\
     \vdots \\
     x_{M - 2,0} & \ldots & x_{M - 2, mask} & x_{M - 2, mask + 1} & x_{M - 2, mask + 2}& \ldots & x_{M - 2, N - 1} \\
      x_{M - 1,0} & \ldots & x_{M - 1, mask} & x_{M - 1, mask + 1} & x_{M - 1, mask + 2}& \ldots & x_{M - 1, N - 1}
    \end{gathered}\right]  \Rightarrow $$
$$ \left[\begin{gathered}
     x_{0,0} & \ldots & x_{0, mask} & 0 & 0 & \ldots & 0\\
     x_{1,0} & \ldots & x_{1, mask} & x_{1, mask + 1} & 0 & \ldots & 0\\
     \vdots \\
     x_{M - 2,0} & \ldots & x_{M - 2, mask} & x_{M - 2, mask + 1} & x_{M - 2, mask + 2}& \ldots & 0 \\
     x_{M - 1,0} & \ldots & x_{M - 1, mask} & x_{M - 1, mask + 1} & x_{M - 1, mask + 2}& \ldots & x_{M - 1, N - 1}
    \end{gathered}\right]  \Rightarrow $$
$$ \left[\begin{gathered}
     \frac{e^{x_{0,0}}}{\sum_{i=0}^{mask} e^{x_{0, i}}} & \ldots & \frac{e^{x_{0,mask}}}{\sum_{i=0}^{mask} e^{x_{0, i}}} & 0 & 0 & \ldots & 0\\
     \frac{e^{x_{1,0}}}{\sum_{i=0}^{mask + 1} e^{x_{1, i}}} & \ldots & \frac{e^{x_{1,mask}}}{\sum_{i=0}^{mask + 1} e^{x_{1, i}}} & \frac{e^{x_{1,mask + 1}}}{\sum_{i=0}^{mask + 1} e^{x_{1, i}}} & 0 & \ldots & 0\\
     \vdots \\
     \frac{e^{x_{M - 2,0}}}{\sum_{i=0}^{N - 2} e^{x_{M - 2, i}}} & \ldots & \frac{e^{x_{M - 2,mask}}}{\sum_{i=0}^{N - 2} e^{x_{M - 2, i}}} & \frac{e^{x_{M - 2,mask + 1}}}{\sum_{i=0}^{N - 2} e^{x_{M - 2, i}}} & \frac{e^{x_{M - 2,mask + 2}}}{\sum_{i=0}^{N - 2} e^{x_{M - 2, i}}}& \ldots & 0 \\
     \frac{e^{x_{M - 1,0}}}{\sum_{i=0}^{N - 1} e^{x_{M - 1, i}}} & \ldots & \frac{e^{x_{M - 1,mask}}}{\sum_{i=0}^{N - 1} e^{x_{M - 1, i}}} & \frac{e^{x_{M - 1,mask + 1}}}{\sum_{i=0}^{N - 1} e^{x_{M - 1, i}}} & \frac{e^{x_{M - 1,mask + 2}}}{\sum_{i=0}^{N - 1} e^{x_{M - 1, i}}}& \ldots & \frac{e^{x_{M - 1,N - 1}}}{\sum_{i=0}^{N - 1} e^{x_{M - 1, i}}}
    \end{gathered}\right] $$

$\bullet$ 对于其他维度的causal softmax变换，只需要将最后两个维度做上述mask变换即可。



## 接口

### 创建算子描述

```c
infiniopStatus_t infiniopCreateCausalSoftmaxDescriptor(
    infiniopHandle_t handle, 
    infiniopCausalSoftmaxDescriptor_t *desc_ptr,  
    infiniopTensorDescriptor_t t_desc);
```
<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

 - `handle`    
     : 输入，`infiniopHandle_t` 类型的硬件控柄。详情请看：[InfiniopHandle_t]()
 - `desc_ptr`    
     : 输出，Host `infiniopCausalSoftmaxDescriptor_t` 指针，指向将被初始化的算子描述符地址。
 - `t_desc` ：dT         
     : 输入，算子计算参数 `t_desc` 的张量描述，数据为$r$维张量，其中$r \geq 2$。

参数限制：

 - **`dT`**:  (`Float16`, `Float32`, `Double`, `Bfloat16`) 之一
 - 支持不连续步长

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

 - [`INFINIOP_STATUS_SUCCESS`](), [`INFINIOP_STATUS_BAD_PARAM`](),  [`INFINIOP_STATUS_BAD_TENSOR_SHAPE`](), [`INFINIOP_STATUS_BAD_TENSOR_DTYPE`](), [`INFINIOP_STATUS_BAD_TENSOR_STRIDES`](), [`INFINIOP_STATUS_BAD_DEVICE`]().

---

### 计算额外工作空间

```c
infiniopStatus_t infiniopGetCausalSoftmaxWorkspaceSize(
    infiniopCausalSoftmaxDescriptor_t desc, 
    uint64_t *size);
```
<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

 - `desc`  
	 : 输入。已使用 `infiniopCreateCausalSoftmaxDescriptor()` 初始化的算子描述符。 
 - `size`   
	 : 输出。Host 指针，额外空间大小的计算结果的写入地址。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

 - [`INFINIOP_STATUS_SUCCESS`](), [`INFINIOP_STATUS_BAD_PARAM`](), [`INFINIOP_STATUS_BAD_DEVICE`]().

---

### 计算

```c
infiniopStatus_t infiniopCausalSoftmax(
    infiniopCausalSoftmaxDescriptor_t desc, 
    void *workspace, 
    uint64_t workspace_size, 
    void* data, 
    void *stream);
```
<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

 - `desc`      
     : 输入，已使用 `infiniopCreateCausalSoftmaxDescriptor()` 初始化的算子描述符。 
 - `workspace`   
	 : 输入。Device 指针，指向算子计算所需的额外工作空间。
 - `workspace_size`   
	 : 输入。`workspace` 的大小，单位：字节（byte）。
 - `data`      
     : 既是输入，也是输出，Device 常量指针，张量限制见[创建算子描述](#创建算子描述)部分。
 - `stream`    
     : 输入，计算流/队列。

<div style="background-color: lightblue; padding: 1px;">  返回值：</div>

 - [`INFINIOP_STATUS_SUCCESS`](), [`INFINIOP_STATUS_BAD_PARAM`](), [`INFINIOP_STATUS_BAD_DEVICE`](), [`INFINIOP_STATUS_EXECUTION_FAILED`]()


---

### 销毁算子描述符

```c
infiniopStatus_t infiniopDestroyCausalSoftmaxDescriptor(
    infiniopCausalSoftmaxDescriptor_t desc);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

 - `desc`  
     : 输入。 待销毁的算子描述符。 

<div style="background-color: lightblue; padding: 1px;"> 返回值： </div>

 - [`INFINIOP_STATUS_SUCCESS`](), [`INFINIOP_STATUS_BAD_DEVICE`]().

## 已知问题

### 平台限制

- 寒武纪不支持uint64或者是int64计算，传入shape或者stride的时候需要使用int32数据类型。

### 
