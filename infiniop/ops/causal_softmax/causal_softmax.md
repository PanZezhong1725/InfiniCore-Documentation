
# `Causal Softmax`

`Causal Softmax` 是使用 causal mask 的 softmax 函数，其中指数变换的操作维度限定在最后一维，适用于各类因果类模型。     
在 `Softmax` 的基础上引入 mask ，对于形状为 $[s_0,\ldots, s_{r-1}]$ 的输入张量 $x$ 来说，mask = $s_{r - 1} - s_{r - 2} \geq 0$ 。    
以形状为 $[4, 7]$ 的张量 $x$ 举例，mask 变换如下所示：


$$ \left[\begin{gathered}
     x_{0,0} & x_{0,1} & x_{0, 2} & x_{0, 3} & x_{0, 4} & x_{0,5} & x_{0, 6}\\
     x_{1,0} & x_{1,1} & x_{1, 2} & x_{1, 3} & x_{1, 4} & x_{1,5} & x_{1, 6}\\
     x_{2,0} & x_{2,1} & x_{2, 2} & x_{2, 3} & x_{2, 4} & x_{2,5} & x_{2, 6}\\
     x_{3,0} & x_{3,1} & x_{3, 2} & x_{3, 3} & x_{3, 4} & x_{3,5} & x_{3, 6}
    \end{gathered}\right]  \Rightarrow $$
$$ \left[\begin{gathered}
     x_{0,0} & x_{0,1} & x_{0, 2} & x_{0, 3} & 0 & 0 & 0\\
     x_{1,0} & x_{1,1} & x_{1, 2} & x_{1, 3} & x_{1, 4} & 0 & 0\\
     x_{2,0} & x_{2,1} & x_{2, 2} & x_{2, 3} & x_{2, 4} & x_{2,5} & 0\\
     x_{3,0} & x_{3,1} & x_{3, 2} & x_{3, 3} & x_{3, 4} & x_{3,5} & x_{3, 6}
    \end{gathered}\right] $$

经过 mask 变换以后针对最后一维做 softmax 变换即可，一维向量的 softmax 变换参考：

$$ y_i = \frac{e^{x_i}}{\sum_{i=0}^{N - 1} e^{x_i}} $$  

高维向量的 `Causal Softmax` 只需要考虑最后两维即可。



## 接口

### 计算

```c
infiniopStatus_t infiniopCausalSoftmax(
    infiniopCausalSoftmaxDescriptor_t desc, 
    void *workspace, 
    size_t workspace_size, 
    void *data, 
    void *stream
);
```
<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

 - `desc`      
     : 输入。已使用 `infiniopCreateCausalSoftmaxDescriptor()` 初始化的算子描述符。 
 - `workspace`   
	 : 输入。Device 指针，指向算子计算所需的额外工作空间。
 - `workspace_size`   
	 : 输入。`workspace` 的大小，单位：字节（byte）。
 - `data`      
     : 既是输入，也是输出。Device 指针，仅支持原地计算，张量限制见[创建算子描述](#创建算子描述)部分。
 - `stream`    
     : 输入。计算流/队列。

参数限制：

 - `data` 仅支持原地计算。

<div style="background-color: lightblue; padding: 1px;">  返回值：</div>

 - [`INFINIOP_STATUS_SUCCESS`](), [`INFINIOP_STATUS_BAD_PARAM`](), [`INFINIOP_STATUS_BAD_DEVICE`](), [`INFINIOP_STATUS_EXECUTION_FAILED`]()


---

### 创建算子描述

```c
infiniopStatus_t infiniopCreateCausalSoftmaxDescriptor(
    infiniopHandle_t handle, 
    infiniopCausalSoftmaxDescriptor_t *desc_ptr,  
    infiniopTensorDescriptor_t t_desc
);
```
<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

 - `handle`    
     : 输入。`infiniopHandle_t` 类型的硬件控柄。详情请看：[InfiniopHandle_t]()
 - `desc_ptr`    
     : 输出。Host `infiniopCausalSoftmaxDescriptor_t` 指针，指向将被初始化的算子描述符地址。
 - `t_desc` - { dT | ($\ldots$) | ($\ldots,1$) }       
     : 输入。算子计算参数 `t_desc` 的张量描述，数据为 $r$ 维张量，其中 $r \geq 2$ 。

参数限制：

 - **`dT`**:  (`Float16`, `Float32`, `Double`, `Bfloat16`) 之一
 - `t_desc` 支持不连续步长

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

 - [`INFINIOP_STATUS_SUCCESS`](), [`INFINIOP_STATUS_BAD_PARAM`](),  [`INFINIOP_STATUS_BAD_TENSOR_SHAPE`](), [`INFINIOP_STATUS_BAD_TENSOR_DTYPE`](), [`INFINIOP_STATUS_BAD_TENSOR_STRIDES`](), [`INFINIOP_STATUS_BAD_DEVICE`]().

---

### 计算额外工作空间

```c
infiniopStatus_t infiniopGetCausalSoftmaxWorkspaceSize(
    infiniopCausalSoftmaxDescriptor_t desc, 
    size_t *size
);
```
<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

 - `desc`  
	 : 输入。已使用 `infiniopCreateCausalSoftmaxDescriptor()` 初始化的算子描述符。 
 - `size`   
	 : 输出。Host 指针，额外空间大小的计算结果的写入地址。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

 - [`INFINIOP_STATUS_SUCCESS`](), [`INFINIOP_STATUS_BAD_PARAM`](), [`INFINIOP_STATUS_BAD_DEVICE`]().

---

### 销毁算子描述符

```c
infiniopStatus_t infiniopDestroyCausalSoftmaxDescriptor(
    infiniopCausalSoftmaxDescriptor_t desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

 - `desc`  
     : 输入。 待销毁的算子描述符。 

<div style="background-color: lightblue; padding: 1px;"> 返回值： </div>

 - [`INFINIOP_STATUS_SUCCESS`](), [`INFINIOP_STATUS_BAD_DEVICE`]().

## 已知问题

### 平台限制

- 寒武纪中 tensor.to(device) 的 tensor 不支持 uint64 或者是 int64 数据类型

### 
