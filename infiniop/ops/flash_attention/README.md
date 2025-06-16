# `Flash Attention`

`Flash Attention`，是一种高效的**自注意力机制实现**，在加速注意力计算的同时也减少了内存占用。其核心原理是通过将输入分块，并在每个块上分别执行注意力计算操作，从而减少对 HBM 的读写。

标准 Attention 的计算为：
1. 计算 $QK^T$，得到初步的注意力分数 attention_score；
2. 添加位置编码以区别不同位置的元素，并乘以缩放系数；
3. 根据注意力掩码屏蔽对应位置的元素，得到 masked_attention_score；
4. 最后经过 softmax 计算，得到最终的注意力分数 attention_out。

而 FlashAttention 主要针对的是最后一步 softmax 的计算：
1. 提前对输入的 QKV 进行分块；
2. 分块 softmax 的增量计算，并维护两个全局变量： `max_score`：已处理块的最大值； `exp_sum`：已处理块的指数和
    1. 若 $i=0$，计算当前的局部 softmax，
        $$\mathrm{out}[0]=e^{S_{0}-m_{0}},\quad m_{0}=\max(S_{0}),\quad\mathrm{exp\_sum}_{0}=\sum e^{S_{0}-m_{0}}$$
        并更新全局变量
        $$\mathrm{max\_score}=m_{0},\quad\mathrm{exp\_sum}=\mathrm{exp\_sum}_{0}$$
    2. 从 $i\ge1$ 开始，首先计算当前块的局部最大值 $m_i$ 和 $\mathrm{exp\_sum}_i$，并且有 
        $$m_\mathrm{old}=\mathrm{max\_score},\quad m_\mathrm{new}=\max(m_\mathrm{old}, m_i)$$
        然后根据该计算结果调整旧结果，并加入当前块结果 
        $$\mathrm{out}[0:i]=\mathrm{out}[0:i]\cdot e^{m_\mathrm{old}-m_\mathrm{new}},\quad \mathrm{out}[i]=e^{S_i-m_\mathrm{new}}$$
        并更新全局变量 
        $$\mathrm{max\_score}=m_\mathrm{new},\quad\mathrm{exp\_sum}=\mathrm{exp\_sum}\cdot e^{m_{\mathrm{old}}-m_{\mathrm{new}}}+\mathrm{exp\_sum}_{i}$$
    3. 最后，当所有块都处理完成后，可得 
        $$\text{attention\_out}=\frac{\mathrm{out}}{\exp\_\mathrm{sum}}$$

## 接口

### 计算

```c
infiniStatus_t infiniopFlashAttention(
    infiniopFlashAttentionDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *q,
    const void *k,
    const void *v,
    void *mask,
    void *mask_type,
    void *stream
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  已使用 `infiniopCreateFlashAttentionDescriptor()` 初始化的算子描述符；
- `workspace`:
  指向算子计算所需的额外工作空间；
- `workspace_size`:
  `workspace` 的大小，单位：字节；
- `out`:
  注意力计算结果地址。张量限制见[创建算子描述](#创建算子描述)部分。
- `q`:
  查询（Query）张量数据指针。张量限制见[创建算子描述](#创建算子描述)部分。
- `k`:
  键（Key）张量数据指针。张量限制见[创建算子描述](#创建算子描述)部分。
- `v`:
  值（Value）张量数据指针。张量限制见[创建算子描述](#创建算子描述)部分。
- `mask`:
  注意力掩码的数据指针，取值为 `0` 或者 `false` 表示保留对应位置的元素（参与计算）；取值为 `1` 或者 `true` 表示屏蔽对应位置的元素（即跳过，不参与计算）。张量限制见[创建算子描述](#创建算子描述)部分。
- `mask_type`:
  注意力类型，取值为 0 ~ 3。张量限制见[创建算子描述](#创建算子描述)部分。
- `stream`:
  计算流/队列。

<div style="background-color: lightblue; padding: 1px;">  返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_NULL_POINTER`], [`INFINI_STATUS_INSUFFICIENT_WORKSPACE`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`].

### 创建算子描述

```c
infiniStatus_t infiniopCreateFlashAttentionDescriptor(
    infiniopHandle_t handle,
    infiniopFlashAttentionDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t mask_desc,
    infiniopTensorDescriptor_t mask_type_desc) 
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `handle`:
  `infiniopHandle_t`类型的硬件控柄。详见 [`InfiniopHandle_t`]。
- `desc_ptr`:
  `infiniopFlashAttentionDescriptor_t` 指针，指向将被初始化的算子描述符地址。
- `out_desc` - { dT | ((batch_size,) seq_len_q, num_heads_q, head_dim) | ($\ldots, 1$)}:
  算子计算参数 `out` 的张量描述，四维或者三维，最后一维连续。
- `q_desc` - { dT | ((batch_size,) seq_len_q, num_heads_q, head_dim) | ($\ldots, 1$)}:
  算子计算参数 `q` 的张量描述，形状与 `out_desc` 一致，最后一维连续。
- `k_desc` - { dT | ((batch_size,) seq_len_kv, num_heads_kv, head_dim) | ($\ldots, 1$)}:
  算子计算参数 `k` 的张量描述，形状与 `out_desc` 一致，最后一维连续。
- `v_desc` - { dT | ((batch_size,) seq_len_kv, num_heads_kv, head_dim) | ($\ldots, 1$)}:
  算子计算参数 `v` 的张量描述，形状与 `out_desc` 一致，最后一维连续。
- `mask_desc` - { dM | (seq_len_q, seq_len_kv) | ($\ldots, 1$)}:
  算子计算参数 `mask` 的张量描述，二维或者一维，最后一维连续。
- `mask_type_desc` - int
  算子计算参数 `mask_type` 的张量描述，取值为 0 ~ 3 的整数。

参数限制：

- `dT`: `Float16`, `Float32` 或 `BFloat16`。
- `dM`: `Bool` 或 `Uint8`。
- `seq_len_q` 与 `seq_len_kv` 可以不同。
- `num_heads_q` 与 `num_heads_kv` 可以不同，但需满足前者是后者的整数倍（非0整数）。
  - 当 $N_q/N_{kv}=1$ 时，即为 MQA (multi-query attention)
  - 当 $N_q/N_{kv}>1$ 时，即为 GQA (grouped-query attention)
- `mask_type` 的四种类型：
  - `0`: 不使用注意力掩码，忽略 `mask` 取值；
  - `1`: 使用完整 mask 矩阵，此时 `mask` 不能为空；
  - `2`: 代表leftUpCausal模式的mask，对应以左上顶点划分的下三角场景；
  - `3`: 代表rightDownCausal模式的mask，对应以右下顶点划分的下三角场景。

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`],  [`INFINI_STATUS_BAD_TENSOR_SHAPE`], [`INFINI_STATUS_BAD_TENSOR_DTYPE`], [`INFINI_STATUS_BAD_TENSOR_STRIDES`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 计算额外工作空间

```c
infiniStatus_t infiniopGetFlashAttentionWorkspaceSize(
    infiniopFlashAttentionDescriptor_t desc, 
    size_t *size
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数：</div>

- `desc`:
  已使用 `infiniopCreateFlashAttentionDescriptor()` 初始化的算子描述符；
- `size`:
  额外空间大小的计算结果的写入地址；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_BAD_PARAM`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

### 销毁算子描述符

```c
infiniStatus_t infiniopDestoryFlashAttentionDescriptor(
    infiniopFlashAttentionDescriptor_t desc
);
```

<div style="background-color: lightblue; padding: 1px;"> 参数： </div>

- `desc`:
  输入。 待销毁的算子描述符；

<div style="background-color: lightblue; padding: 1px;"> 返回值： </div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`].

<!-- 链接 -->
[`InfiniopHandle_t`]: /infiniop/handle/README.md

[`INFINI_STATUS_SUCCESS`]: /common/status/README.md#INFINI_STATUS_SUCCESS
[`INFINI_STATUS_BAD_PARAM`]: /common/status/README.md#INFINI_STATUS_BAD_PARAM
[`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`]: /common/status/README.md#INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED
[`INFINI_STATUS_BAD_TENSOR_SHAPE`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_SHAPE
[`INFINI_STATUS_BAD_TENSOR_DTYPE`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_DTYPE
[`INFINI_STATUS_BAD_TENSOR_STRIDES`]: /common/status/README.md#INFINI_STATUS_BAD_TENSOR_STRIDES
[`INFINI_STATUS_NULL_POINTER`]:/common/status/README.md#INFINI_STATUS_NULL_POINTER
[`INFINI_STATUS_INSUFFICIENT_WORKSPACE`]:/common/status/README.md#INFINI_STATUS_INSUFFICIENT_WORKSPACE
[`INFINI_STATUS_INTERNAL_ERROR`]:/common/status/README.md#INFINI_STATUS_INTERNAL_ERROR