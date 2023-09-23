# 08.30~09.08 初期构思


### 主要文件
1. MemoryManager: 负责memory的malloc/free等。to_cpu函数负责malloc&memset（TODO需要考虑Backend，MemoryManager应被移除相关功能移至Backend的子类中，负责该backend的内存管理）。

2. Tensor: 基础数据类型，包含NCHW四个为维度。数据存储在私有变量data_（vector类型，TODO需要考虑Backend）。
构造函数不进行malloc, 通过cpu_data函数进行malloc(现阶段只考虑cpu，TODO该函数需要考虑Backend)。

3. Op: 算子基础，作为不同backend的op的父类。setUp函数负责1）输入输出Tensor的malloc；2）输出Tensor的reshape；3）该算子的weight的malloc。execute负责执行。（TODO考虑Backend，setUp/execute函数利用Tensor的malloc功能的函数）。

4. Graph: 网络图？，根据NetParamter参数构建（TODO通过文件构建NetParamter，通过类似MNN的方式构建Net）。setUp函数调用Op的setUp同时对tensors进行malloc&memeset（TODO从文件读取weights并memset）。forward调用Op的execute。

5. Net:convert()将NetParamter转成多个Graph。 Run()利用Graph.setUp/forward进行pipeline，交给用户。

6. Backend（TODO）: 需要负责管理后端设备（CPU/OPENCL/NNAPI等）。~~主要函数应有:aligend_malloc, memset, mem_free。Backend需要作为不同backend的父类。~~

### 文件夹
1. src/backends（TODO）:XX表示backend名（CPU等） 包含1）对应的XXBackend.h/cpp; 2）每个Op的XXOp.c/hpp；3）对应的Tensor的常见计算（待决定？）

2. ~~src/nets（TODO）:负责不同模型（llama等），继承自Net。~~

### TODO
~~第一阶段：不考虑Backend~~
1. ~~实现Tensor/Op中的主要函数（能满足基础功能，tensor计算的相关函数待定）。~~
2. ~~完善NetParamter。~~
3. ~~Net中init/setUp/execute实现。~~
4. ~~完善backends/cpu/CPUMatmul.cpp。~~
5. ~~自己初始化一个简单的有两层Matmul算子构成的NetParamter, 来初始化Net并跑通，检查setUp/forward功能。~~

~~第二阶段：考虑Backend~~
1. ~~构建Backend类，将MemoryManager中功能移入并删除MemoryManager类。~~
2. ~~修改Tensor/Op中的MemoryManager相关函数。~~
3. ~~构建CPUBackend类继承Backend类。~~
4. ~~实现CPUxxop.c/hpp，实现CPU的算子。~~

~~第三阶段：构建transformer.~~

1. ~~利用CPU算子跑通llama.~~




# 09.08 会议决定

1. 把Net.h/cpp改为Graph.h/cpp. 

2. 构建Net.h/cpp
convert()函数：将根据Netparam得到多个subgraph(Graph类)。 需要用到backend：CheckSupportOp
run()函数：调用每一个subgraph的setUp&forward函数。（先setUp后forward//pipline）可用户重构。

3. 删除MemoryManager.h/cpp，构建MemoryManager.h/cpp。
MemoryManager中管理CPU内存。包含函数malloc/memset/memfree。类似memorypool的功能。
删除CPUMemory.h/cpp。
不同的backend的Op.setUp调用MemoryManager的malloc/memset/memfree。

4. Backend中添加CheckSupport：检查该backend支持的算子，用于Net.convert()。

# 09.08~09.12 疑问

1. 全局的MemoryManager，所有backend需要host(cpu)内存时向该全局MemoryManager申请。（考虑到需要用到多个backend）
Tensor类初始化时加入参数MemoryManager,将全局MemoryManager的指针传入Tensor类中负责内存.
Op类同上.

2. ModelConverter添加到Net类的init(), 模型的load是否可以使用protobuf(like MNN)。
 

# 09.12 会议决定

1. 类图见<https://bwlvw7nk5ty.feishu.cn/wiki/JVZFwtfKBin09lkOtq5cQqaHnef>


# 09.22 会议决定

1. 使用clang-tidy

2. subgraph后续改成自动划分
