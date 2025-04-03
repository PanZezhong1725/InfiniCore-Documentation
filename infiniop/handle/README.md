# 硬件控柄（`infiniopHandle_t`）

*InfiniOP* 是一个支持多种硬件的跨平台算子库，用户在创建算子时，需要首先创建并传入对应的硬件控柄（`infiniopHandle_t`），以便于算子构建出正确的计算函数。每一个硬件控柄会唯一对应一个硬件，例如在一个一机多卡的环境中，每张卡都需要有自己独立的硬件控柄才能在对应的卡上执行计算。在同一个硬件上构建多个算子时，推荐重复使用同一个硬件控柄以减少开销。硬件控柄是线程安全的。

用户创建硬件控柄时，需要预先使用 *InfiniRT* 运行时库中的 [`infinirtSetDevice`] 接口来指定硬件设备。需要注意的是，使用硬件控柄（包括算子计算）时，算子库不会自动切换硬件的运行时上下文（Context）。当用户在一个线程上对多个硬件进行操作（包括构建算子描述和执行计算）时，需自行调用 [`infinirtSetDevice`] 接口切换设备，以保证算子库接口在指定的硬件设备上的正确运行。推荐用户使用“一个线程对应一个硬件”的程序设计避免复杂的硬件管理。

## 接口

### 创建硬件控柄

```c
infiniStatus_t infiniopCreateHandle(infiniopHandle_t *handle_ptr);
```

- `handle_ptr`: 存储将被创建的硬件控柄的地址；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_NULL_POINTER`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`], [`INFINI_STATUS_INTERNAL_ERROR`]

### 销毁硬件控柄

```c
infiniStatus_t infiniopDestroyHandle(infiniopHandle_t handle);
```

- `handle`: 将被销毁的硬件控柄；

<div style="background-color: lightblue; padding: 1px;"> 返回值：</div>

- [`INFINI_STATUS_SUCCESS`], [`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`]

<!-- 链接 -->
[`infinirtSetDevice`]: /
[`INFINI_STATUS_SUCCESS`]: /common/status/README.md#INFINI_STATUS_SUCCESS
[`INFINI_STATUS_NULL_POINTER`]: /common/status/README.md#INFINI_STATUS_NULL_POINTER
[`INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED`]: /common/status/README.md#INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED
[`INFINI_STATUS_INTERNAL_ERROR`]: /common/status/README.md#INFINI_STATUS_INTERNAL_ERROR
