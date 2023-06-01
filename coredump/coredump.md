

https://blog.51cto.com/u_15471709/4868198

### 简介

core 文件通常包含了程序运行时的内存、寄存器状态、堆栈指针、内存管理信息、各种函数调用堆栈信息等。许多程序出错的时候都会产生一个 core 文件，通过 gdb 工具分析这个文件，可以定位到程序异常退出时对应的堆栈调用等信息。

### 使用流程

__1. 更改 coredump 文件的存储位置，并使产生的 core 文件带有崩溃程序的 filename、以及 process id__

```shell
echo "/home/neal/coredump/core-%e-%p" > /proc/sys/kernel/core_pattern
```

__2. 设置当前会话的 core 大小为无限__

```shell
ulimit -c unlimited
```

__3. 确认生成文件是否为 core 文件格式，并启用 gdb 进行调试__

```shell
file core-coremain-19879

gdb ~/projects/qtros/devel/lib/qtros/ros_qt_demo ~/coredump/core-coremain-19879
```

__4. ROS 包下的 CMakeLists.txt 中添加 gdb 调试信息__

```cmake
SET(CMAKE_BUILD_TYPE "Debug")
# -O0 关闭所有代码优化选项；-Wall 开启大部分告警提示；-g 包含调试信息；-ggdb 在可执行文件中包含可供 gdb 使用的调试信息
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
# -O3 开启第三级别优化，在 -O2 基础上增加产生 inline 函数、使用寄存器等优化技术
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
```

__5. 几个常用的 gdb 命令__

```shell
l(list)      # 显示源代码，并且可以看到对应的行号
b(break) x   # x 是行号，表示在对应的行号位置设置断点
p(print) x   # x 是变量名，表示打印变量 x 的值
r(run)       # 表示继续执行到断点的位置
n(next)      # 表示执行下一步
c(continue)  # 表示继续执行
q(quit)      # 表示退出 gdb
```

bt



