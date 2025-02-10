
# `Random Sample`

`Random Sample`, 即**随机采样**算子：给定一个代表离散概率分布（或权重）的序列：

$$ P = \{ p_0, p_1, \dots, p_{n-1} \}, \quad \sum_{i=0}^{n-1} p_i = 1 $$

以及一个随机数种子 $r$ 服从均匀分布：

$$ r \sim U(0,1) $$

从分布中采样出一个样本，样本为0到概率分布总长度减1之间的一个整数。支持 top-k、top-p 采样策略，支持temperature 随机性调整。
温度缩放 (Temperature Scaling) 给定原始概率分布 $P$ 和 `temperature` 参数 $T$，调整后的概率分布 $P'$ 为：

$$
P^{\prime} = \{ p_0^{\prime}, p_1^{\prime}, \dots, p_{n-1}^{\prime} \}, \quad p_i^{\prime} = \frac{e^{\frac{p_i}{T}}}{\sum_{j=0}^{n-1} e^{\frac{p_j}{T}}}
$$

其中 $p_i$ 是第 $i$ 个元素的原始概率。
Top-k 选择概率最高的前 `k` 。
Top-p 选择累计概率和达到设定阈值 `p` 的最小子集。


## 接口

### 计算

```c
infiniopStatus_t infiniopRandomSample(
	infiniopRandomSampleDescriptor_t desc,
	void *workspace,
	uint64_t workspace_size,
	void *result,
	void const *probs,
	float random_val,
	float topp,
	uint64_t topk,
	float temperature,
	void *stream
);
```
<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

 - `desc`
	 : 输入。已使用 `infiniopCreateRandomSampleDescriptor()` 初始化的算子描述符。 
 - `workspace`
	 : 输入。Device 指针，指向算子计算所需的额外工作空间。
 - `workspace_size`
	 : 输入。`workspace` 的大小，单位：字节（byte）。
 - `result`
	 : 输出。Device 指针，采样输出结果。张量限制见[创建算子描述](#创建算子描述)部分。
 - `probs`
	 : 输入。Device 常量指针，概率分布数据。张量限制见[创建算子描述](#创建算子描述)部分。
 - `random_val`
	 : 输入。随机数种子，一般通过 Uniform 分布产生，范围是 $[0,1]$。
 - `topp`
	 : 输入。top-p 采样阈值，使得采样只从靠前的概率和为 `topp` 的范围内进行，范围是 $[0,1]$ 。当 `topp` 为0时，采样退化为 **Argmax** 算子。当 `topp` 大于等于1时，不设置 top-p 阈值。
 - `topk`
	 : 输入。top-k 采样阈值，使得采样只从靠前的 `topk` 项里进行，范围是 $[0,\infty)$ 。当 `topk` 为1时，采样退化为 **Argmax** 算子；当 `topk` 大于等于概率分布长度或为0时，不设置 top-k 阈值。
 - `temperature`
	 : 输入。概率分布随机性度，范围是 $[0,\infty)$ 。
	 : - 当 `temperature` 为1时，采样结果与原始概率分布一致；
	 : - 当 `temperature` 大于1时，采样结果越靠近原始概率分布，越可能被选中；
	 : - 当 `temperature` 小于1时，采样结果越远离原始概率分布，越可能被选中；
	 : - 当 `temperature` 为0时，采样退化为 **Argmax** 算子。
 - `stream`
	 : 输入。计算流/队列。

<div style="background-color: lightblue; padding: 1px;">  返回值：</div>

 - [`INFINIOP_STATUS_SUCCESS`](), [`INFINIOP_STATUS_BAD_PARAM`](), [`INFINIOP_STATUS_INSUFFICIENT_WORKSPACE`](), [`INFINIOP_STATUS_BAD_DEVICE`](), [`INFINIOP_STATUS_EXECUTION_FAILED`]()

 - 当 `random_val`、`topp`、`topk`、或 `temperature` 超出范围返回 `INFINIOP_STATUS_BAD_PARAM`

---

### 创建算子描述

```c
infiniopStatus_t infiniopCreateRandomSampleDescriptor(
	infiniopHandle_t handle,
	infiniopRandomSampleDescriptor_t *desc_ptr,
	infiniopTensorDescriptor_t result,
	infiniopTensorDescriptor_t probs
);
```
<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

 - `handle`
	: 输入。`infiniopHandle_t` 类型的硬件控柄。详情请看：[InfiniopHandle_t]()
 - `desc_ptr`
	 : 输出。Host `infiniopRandomSampleDescriptor_t` 指针，指向将被初始化的算子描述符地址。
 - `result` - { dOut | (1,) | (1,) }
	 : 输入。 算子计算参数 `result` 的张量描述。
 - `probs` - { dT | (N,) | (1,) }
	 : 输入。算子计算参数 `probs` 的张量描述。目前仅支持连续一维张量。

参数限制：

 - **`dT`**:  (`Float16`, `Float32`) 之一
 
 - **`dOut`**: `Uint64`
    
 - **`N`**: N > 0

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

 - [`INFINIOP_STATUS_SUCCESS`](), [`INFINIOP_STATUS_BAD_PARAM`](),  [`INFINIOP_STATUS_BAD_TENSOR_SHAPE`](), [`INFINIOP_STATUS_BAD_TENSOR_DTYPE`](), [`INFINIOP_STATUS_BAD_TENSOR_STRIDES`](), [`INFINIOP_STATUS_BAD_DEVICE`]().

---

### 计算额外工作空间

```c
infiniopStatus_t infiniopGetRandomSampleWorkspaceSize(
	infiniopRandomSampleDescriptor_t desc,
	uint64_t *size
);
```
<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

 - `desc`
	 : 输入。已使用 `infiniopCreateRandomSampleDescriptor()` 初始化的算子描述符。 
 - `size`
	 : 输出。Host 指针，额外空间大小的计算结果的写入地址。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

 - [`INFINIOP_STATUS_SUCCESS`](), [`INFINIOP_STATUS_BAD_PARAM`](), [`INFINIOP_STATUS_BAD_DEVICE`]().

---

### 销毁算子描述符

```c
infiniopStatus_t infiniopDestroyRandomSampleDescriptor(
	infiniopRandomSampleDescriptor_t desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

 - `desc`
	 : 输入。 待销毁的算子描述符。 

<div style="background-color: lightblue; padding: 1px;"> 返回值： </div>

 - [`INFINIOP_STATUS_SUCCESS`](), [`INFINIOP_STATUS_BAD_DEVICE`]().

## 已知问题

### 平台限制

- 昇腾目前只支持 Argmax 的情况

### 