## [参考文献]:(https://github.com/GitZzw/Master_Degree/blob/master/Makefile/cmake.pdf)

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

* 在ex3目录下建立一个CMakeList.txt文件，一个main.c文件，一个src文件夹，src文件夹里有一个func1.c文件，一个func.h文件,一个CMakeList.txt文件，其中main.c包含func1.h文件(参考study_makefile_notes.md)，调用func1.c文件中的函数

* 对于这种情况，需要在func文件夹中也建立一个CMakeList.txt文件，目的是将src中的文件编译成静态库由main.c调用。add_library 将路径中的源文件编译为静态链接库

* src文件夹中CMakeList.txt的内容应该是
   ```
        # 查找当前目录下的所有源文件，并将名称保存到 DIR_LIB_SRCS 变量
        aux_source_directory(. DIR_LIB_SRCS)
        # 生成链接库
        add_library (function ${DIR_LIB_SRCS})
  ```
  
* ex3目录下CMakeList.txt的内容应该是，target_link_libraries指明可执行文件ex3需要连接一个名为function的链接库；add_subdirectory 指明包含一个子目录src，这样src目录下的 CMakeLists.txt 文件和源代码也会被处理。执行 cmake 的过程中,首先解析ex3目录下的 CMakeLists.txt ,当程序执行命令 ADD_SUBDIRECTORY( src ) 时进入目录 src 对其中的 CMakeLists.txt 进行解析
  ```
        # CMake 最低版本号要求
        cmake_minimum_required (VERSION 2.8)
        # 项目信息
        project (EX3)
        # 查找当前目录下的所有源文件,并将名称保存到 DIR_SRCS 变量
        aux_source_directory(. DIR_SRCS)
        # 添加子目录func
        add_subdirectory(src)
        # 添加链接库生成可执行文件
        add_executable(ex3 main.cc)
        target_link_libraries(ex3 function)
  ```

  >Tips:如果源文件很多，把所有源文件的名字都加进去很麻烦，`aux_source_directory(<dir> <variable>)`，该命令会查找指定目录dir下的所有源文件，然后将结果存进指定变量名variable
  
  
  # 3.外部编译方式
  
  ## 以单个.c文件为例
  
  * 为项目代码建立目录ex4，与此项目有关的所有代码和文档都位于此目录下

  * 在ex4目录下建立一个build文件夹和一个main.c文件

  * 在ex4目录下建立一个CMakeLists.txt
   
  * 之前的步骤除了多了一个空的build文件夹其他都一样，现在进入build文件夹，命令行`cmake ..`，如下：
```
        $ ls
        ex4
        $ cd ex4/build/
        $ ls
        $ cmake ..
        – The C compiler identification is GNU
        – The CXX compiler identification is GNU
        – Check for working C compiler: /usr/bin/gcc
        – Check for working C compiler: /usr/bin/gcc — works
        – Detecting C compiler ABI info
         – Detecting C compiler ABI info - done
        – Check for working CXX compiler: /usr/bin/c++
        – Check for working CXX compiler: /usr/bin/c++ — works
        – Detecting CXX compiler ABI info
        – Detecting CXX compiler ABI info - done
        – Configuring done
        – Generating done
        – Build files have been written to: /home/zzw/cmake/ex4/build
        
        $ make
        Scanning dependencies of target ex4
        [100%] Building C object CMakeFiles/ex4.dir/main.c.o
        Linking C executable ex4
        [100%] Built target ex4
        $ ls
        CMakeCache.txt CMakeFiles cmake_install.cmake ex4 Makefile
 ```
  < Tips:与之前cmake直接在当前目录进行编译，不建立build目录相比，这种方法建立的所有的中间文件都会生成在build目录下，需要删除时直接清空该目录即可。
  
  
  # 4.查找并使用其他程序库
> 如果需要用到一些别人的函数库,这些函数库在不同的系统中安装的位置可能不同,编译的时候需要首先找到这些软件包的头文件以及链接库所在的目录以便生成编译选项。
例如一个需要使用博克利数据库项目,需要头文件db_cxx.h 和链接库 libdb_cxx.so ,现在该项目中有一个源代码文件 main.cpp ，放在项目ex4的根目录中。

* 在项目ex5目录下创建目录cmake/modules/，然后在ex5/cmake/modules/下创建文件Findlibdb_cxx.cmake，该cmake文件的语法与 CMakeLists.txt 相同，其中内容如下
```
    #命令 MESSAGE 将参数的内容输出到终端
    MESSAGE(STATUS "Using bundled Findlibdb.cmake...")
    #命令 FIND_PATH 指明头文件查找的路径，意思是在 /usr/include/和/usr/local/include/ 中查找文件db_cxx.h ,并将db_cxx.h 所在的路径保存在 LIBDB_CXX_INCLUDE_DIR中。
    FIND_PATH(
    LIBDB_CXX_INCLUDE_DIR
    db_cxx.h 
    /usr/include/ 
    /usr/local/include/ 
    )
    #命令 FIND_LIBRARYz指明链接库查找的路径，意思是在目录 /usr/lib/ 和 /usr/local/lib/ 中寻找名称为 db_cxx 的链接库,并将结果保存在 LIBDB_CXX_LIBRARIES。
    FIND_LIBRARY(
    LIBDB_CXX_LIBRARIES NAMES  db_cxx
    PATHS /usr/lib/ /usr/local/lib/
   )
```


* 在目录ex5创建CMakeList.txt，内容如下
```
    # CMake 最低版本号要求
    cmake_minimum_required (VERSION 2.8)
    PROJECT(EX5)
    #SET命令用来显式地定义变量，将当前目录存在CMAKE_SOURCE_DIR变量中
    SET(CMAKE_SOURCE_DIR .)
    #CMAKE_MODULE_PATH代表后面两个地址，一个是当前ex5工程的modules，一个是根目录下modules
    SET(CMAKE_MODULE_PATH ${CMAKE_ROOT}/Modules ${CMAKE_SOURCE_DIR}/cmake/modules)
    # 查找当前目录下的所有源文件,并将名称保存到 DIR_SRCS 变量
    AUX_SOURCE_DIRECTORY(. DIR_SRCS)
    # 生成可执行文件
    ADD_EXECUTABLE(ex5 ${DIR_SRCS})
    
    # FIND_PACKAGE命令进行查找,这条命令执行后 CMake 会到变量 CMAKE_MODULE_PATH 指示的目录中查找文件 Findlibdb_cxx.cmake 并执行,即在ex5/cmake/modules/中查找到文件 Findlibdb_cxx.cmake
    FIND_PACKAGE(libdb_cxx REQUIRED)
    MARK_AS_ADVANCED(
    LIBDB_CXX_INCLUDE_DIR   #头文件查找到的路径，cmake文件中定义
    LIBDB_CXX_LIBRARIES     #链接库查找到的路径，cmake文件中定义
    )
    
    # 如果成功查找到头文件和链接库
    IF (LIBDB_CXX_INCLUDE_DIR AND LIBDB_CXX_LIBRARIES)  
    
    将参数的内容输出到终端
    MESSAGE(STATUS "Found libdb libraries")
 
    # 设置编译时到 LIBDB_CXX_INCLUDE_DIR 寻找头文件
    INCLUDE_DIRECTORIES(${LIBDB_CXX_INCLUDE_DIR})
    
    # 将参数的内容输出到终端
    MESSAGE( ${LIBDB_CXX_LIBRARIES} ) 
    # 设置可执行文件 ex5 需要与链接库 LIBDB_CXX_LIBRARIES 进行链接
    TARGET_LINK_LIBRARIES(ex5 ${LIBDB_CXX_LIBRARIES})
    ENDIF (LIBDB_CXX_INCLUDE_DIR AND LIBDB_CXX_LIBRARIES)
```

* cmake . 生成makefile，再make即可






