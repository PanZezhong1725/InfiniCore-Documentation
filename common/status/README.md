# `接口调用结果代码（infiniStatus_t）`

`InfiniCore` 库中所有的API均会返回接口调用结果，用来表示调用是否成果以及错误类型。各类返回值以及含义如下：

<a id="INFINI_STATUS_SUCCESS"></a>
<div style="background-color: grey; padding: 1px;"> INFINI_STATUS_SUCCESS = 0 </div>

    运行成功。

<a id="INFINI_STATUS_INTERNAL_ERROR"></a>
<div style="background-color: grey; padding: 1px;"> INFINI_STATUS_INTERNAL_ERROR = 1 </div>

    硬件内部/底层功能错误。如果错误来源是硬件平台提供的编程库，会打印出该库的错误代码，详细错误原因需要去相应厂商提供的文档中查询。

<a id="INFINI_STATUS_NOT_IMPLEMENTED"></a>
<div style="background-color: grey; padding: 1px;"> INFINI_STATUS_NOT_IMPLEMENTED = 2 </div>

    接口功能尚未实现。

<a id="INFINI_STATUS_BAD_PARAM"></a>
<div style="background-color: grey; padding: 1px;"> INFINI_STATUS_BAD_PARAM = 3 </div>

    传入的参数（组合）不符合接口限制。

<a id="INFINI_STATUS_NULL_POINTER"></a>
<div style="background-color: grey; padding: 1px;"> INFINI_STATUS_NULL_POINTER = 4 </div>

    需要有效的地址的参数被传入了空指针。

<a id="INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED"></a>
<div style="background-color: grey; padding: 1px;"> INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED = 5 </div>

    接口不支持用户所指定的硬件类型。  

<a id="INFINI_STATUS_DEVICE_NOT_FOUND"></a>
<div style="background-color: grey; padding: 1px;"> INFINI_STATUS_DEVICE_NOT_FOUND = 6 </div>

    无法找到用户指定的硬件类型与硬件序号所指向的设备。  

<a id="INFINI_STATUS_DEVICE_NOT_INITIALIZED"></a>
<div style="background-color: grey; padding: 1px;"> INFINI_STATUS_DEVICE_NOT_INITIALIZED = 7 </div>

    设备未被初始化。请检查硬件驱动是否正常，以及是否调用了相应计算库的初始化接口。  

<a id="INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED"></a>
<div style="background-color: grey; padding: 1px;"> INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED = 8 </div>

    未被支持的硬件的架构。一些硬件类型包含不同型号以及硬件特性的设备，该型号不被接口支持。  

<a id="INFINI_STATUS_BAD_TENSOR_DTYPE"></a>
<div style="background-color: grey; padding: 1px;"> INFINI_STATUS_BAD_TENSOR_DTYPE = 10 </div>

    用户传入的设备类型不支持。  

<a id="INFINI_STATUS_BAD_TENSOR_SHAPE"></a>
<div style="background-color: grey; padding: 1px;"> INFINI_STATUS_BAD_TENSOR_SHAPE = 11 </div>

    用户传入的张量形状不支持。  

<a id="INFINI_STATUS_BAD_TENSOR_STRIDES"></a>
<div style="background-color: grey; padding: 1px;"> INFINI_STATUS_BAD_TENSOR_STRIDES = 12 </div>

    用户传入的张量步长不支持。  

<a id="INFINI_STATUS_INSUFFICIENT_WORKSPACE"></a>
<div style="background-color: grey; padding: 1px;"> INFINI_STATUS_INSUFFICIENT_WORKSPACE = 13 </div>

    用户传入的工作空间大小不足。
