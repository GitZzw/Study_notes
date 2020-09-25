（）## [参考文献]:()

# 1.什么是cmake
> cmake和autotools是makefile的上层工具。它们的目的正是为了产生可移植的makefile，并简化自己动手写makefile时的巨大工作量。
> makefile通常依赖于你当前的编译平台，而且编写makefile的工作量比较大，解决依赖关系时也容易出错。
> cmake这种项目构建工具能够帮我们在不同平台上更好地组织和管理我们的代码及其编译过程，这是我们使用的主要原因。

# 2.cmake基础使用方法

## 单个.c文件
* 为项目代码建立目录ex1，与此项目有关的所有代码和文档都位于此目录下

* 在ex1目录下建立一个main.c文件

* 在ex1目录下建立一个新的文件CMakeLists.txt，它就是 cmake所处理的"代码"。
  在CMakeLists.txt文件中输入下面的代码(#后面的内容为代码行注释)：
```
        #cmake最低版本需求，不加入此行会收到警告信息
        CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
        PROJECT(EX1) #项目名称
        #生成应用程序ex1 (在windows下会自动生成ex1.exe)
        ADD_EXECUTABLE(ex1 main.c)
```

* 编译项目
在当前目录执行`cmake . `，得到 Makefile 后再使用 make 命令编译得到 ex1 可执行文件

## 多个.c文件
### 同一目录下的多个.c文件
* 为项目代码建立目录ex2

* 在ex2目录下建立一个CMakeList.txt文件，一个main.c文件，一个func1.c文件，一个func.h文件，其中main.c包含func1.h文件(参考study_makefile_notes.md)，调用func1.c文件中的函数

* 此时CMakeList.txt的内容应该是
   ```
          # CMake 最低版本号要求
          cmake_minimum_required (VERSION 2.8)
          # 项目信息
          project (EX2)
          # 指定生成目标
          add_executable(ex2 main.cc MathFunctions.cc)
  ```

  >Tips:如果源文件很多，把所有源文件的名字都加进去很麻烦，`aux_source_directory(<dir> <variable>)`，该命令会查找指定目录dir下的所有源文件，然后将结果存进指定变量名variable
  
  可以修改CMakeList.txt如下
  ```
        # CMake 最低版本号要求
        cmake_minimum_required (VERSION 2.8)
        # 项目信息
        project (EX2)
        # 查找当前目录下的所有源文件,并将名称保存到 DIR_SRCS 变量
        aux_source_directory(. DIR_SRCS)
        # 生成可执行文件
        add_executable(ex2 ${DIR_SRCS})
  ```
  
  ### 不同目录下的多个.c文件
* 为项目代码建立目录ex3

* 在ex3目录下建立一个CMakeList.txt文件，一个main.c文件，一个func文件夹，func文件夹里有一个func1.c文件，一个func.h文件，其中main.c包含func1.h文件(参考study_makefile_notes.md)，调用func1.c文件中的函数

* 此时CMakeList.txt的内容应该是
   ```
          # CMake 最低版本号要求
          cmake_minimum_required (VERSION 2.8)
          # 项目信息
          project (EX2)
          # 指定生成目标
          add_executable(ex2 main.cc MathFunctions.cc)
  ```

  >Tips:如果源文件很多，把所有源文件的名字都加进去很麻烦，`aux_source_directory(<dir> <variable>)`，该命令会查找指定目录dir下的所有源文件，然后将结果存进指定变量名variable
  
  可以修改CMakeList.txt如下
  ```
        # CMake 最低版本号要求
        cmake_minimum_required (VERSION 2.8)
        # 项目信息
        project (EX2)
        # 查找当前目录下的所有源文件,并将名称保存到 DIR_SRCS 变量
        aux_source_directory(. DIR_SRCS)
        # 生成可执行文件
        add_executable(ex2 ${DIR_SRCS})
  ```
