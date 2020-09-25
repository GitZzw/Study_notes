## [参考资料1，关于makefile]:(https://tieba.baidu.com/p/591519800?pn=0)
或者[pdf版本]:(https://github.com/GitZzw/Master_Degree/blob/master/Makefile/Makefile%E8%AF%A6%E8%A7%A3.pdf)

# 什么是makefile
> makefile关系到了整个工程的编译规则。
> 一个工程中的源文件不计数，其按类型、功能、模块分别放在若干个目录中
> makefile定义了一系列的规则来指定，哪些文件需要先编译，哪些文件需要后编译，哪些文件需要重新编译，甚至于进行更复杂的功能操作，
> 因为makefile就像一个Shell 脚本一样，其中也可以执行操作系统的命令

# makefile的好处
> makefile带来的好处就是——“自动化编译”，一旦写好，只需要一个make命令，整个工程完全自动编译，极大的提高了软件开发的效率。
> make命令是一个命令工具，是一个解释makefile中指令的命令工具，一般来说，大多数的IDE都有这个命令，比如：Delphi的make，Visual C++的nmake，Linux下GNU的make

# 程序的编译和链接

### 1.编译
要把源文件编译成中间代码文件，在Windows下也就是 .obj 文件，UNIX下是 .o 文件，即 Object File，这个动作叫做编译（compile）。

### 2.链接
把大量的Object File合成执行文件，这个动作叫作链接（link）。

> 编译时，编译器需要的是语法的正确，函数与变量的声明的正确。对于后者，通常是你需要告诉编译器头文件的所在位置（头文件中应该只是声明，而定义应该放在 C/C++文件中）
只要所有的语法正确，编译器就可以编译出中间目标文件。一般来说，每个源文件都应该对应于一个中间目标文件（O文件或是OBJ文件）。

> 链接时，主要是链接函数和全局变量，所以，我们可以使用这些中间目标文件（O文件或是OBJ文件）来链接我们的应用程序。链接器并不管函数所在的源文件，只管函数的中间目标文件（Object File）
> 在大多数时候，由于源文件太多，编译生成的中间目标文件太多，而在链接时需要明显地指出中间目标文件名，很不方便
> 所以，我们要给中间目标文件打个包，在Windows下这种包叫“库文件”（Library File)，也就是 .lib 文件，在UNIX下，是Archive File，也就是 .a 文件。

### 3.补充
> 假设有三个文件**a.h，b.c，main.c(#include "a.h")**
> 其中`(#include "a.h")`表示将**a.h**中的内容(即声明)拷贝过来，使得编译能够成功，编译只关心编译器需要的是语法、函数与变量的声明的正确，并不关心**a.h**中函数如何实现或在哪个源文件中实现。

> **[参考文档：头文件和源文件的关系]：(https://github.com/GitZzw/Master_Degree/blob/master/Makefile/%E5%A4%B4%E6%96%87%E4%BB%B6%E5%92%8C%E6%BA%90%E6%96%87%E4%BB%B6%E7%9A%84%E5%85%B3%E7%B3%BB.pdf)**


> 编译时将**main.c**和**b.c**编译为**main.o**和**b.o**，要将**main.o**生成为可执行文件需要链接相应的库，此时在makefile文件中指明**main.o**链接**b.o**，使得链接能够成功。
> makefile指明链接时相应的库,或者在所以obj中建立符号表，在符号表中找对应的函数和变量实现。


# makefile文件的作用
> make命令执行时，需要一个 Makefile 文件，以告诉make命令需要怎么样的去编译和链接程序。

### 1.makefile的规则
···
target ... : prerequisites ...
command
...
...
···

> target也就是一个目标文件，可以是Object File，也可以是执行文件；prerequisites就是，要生成那个target所需要的文件或是目标；command也就是make需要执行的命令（任意的Shell命令）
> 这是一个文件的依赖关系，也就是说，target这一个或多个的目标文件依赖于prerequisites中的文件，其生成规则定义在command中。
> 就是说，prerequisites中如果有一个以上的文件比target文件要新的话，command所定义的命令就会被执行。这就是 Makefile的规则。也就是Makefile中最核心的内容。

### 2.一个示例
> 如果一个工程有3个头文件，和8个C文件，我们为了完成前面所述的那三个规则，我们的Makefile应该是下面的这个样子的：
```
edit : main.o kbd.o command.o display.o insert.o search.o files.o utils.o
cc -o edit main.o kbd.o command.o display.o \
insert.o search.o files.o utils.o

main.o : main.c defs.h
cc -c main.c

kbd.o : kbd.c defs.h command.h
cc -c kbd.c

command.o : command.c defs.h command.h
cc -c command.c

display.o : display.c defs.h buffer.h
cc -c display.c

insert.o : insert.c defs.h buffer.h
cc -c insert.c

search.o : search.c defs.h buffer.h
cc -c search.c

files.o : files.c defs.h buffer.h command.h
cc -c files.c

utils.o : utils.c defs.h
cc -c utils.c

clean :
rm edit main.o kbd.o command.o display.o \
insert.o search.o files.o utils.o
command.o : defs.h command.h
display.o : defs.h buffer.h
insert.o : defs.h buffer.h
search.o : defs.h buffer.h
files.o : defs.h buffer.h command.h
utils.o : defs.h

.PHONY : clean
clean :
rm edit $(objects)
```

### 3.另类风格的makefile
> 即然我们的make可以自动推导命令，那么我看到那堆[.o]和[.h]的依赖就有点不爽，那么多的重复的[.h]，能不能把其收拢起来
```
objects = main.o kbd.o command.o display.o \
insert.o search.o files.o utils.o

edit : $(objects)
cc -o edit $(objects)

$(objects) : defs.h
kbd.o command.o files.o : command.h
display.o insert.o search.o files.o : buffer.h

.PHONY : clean
clean :
rm edit $(objects)
```
