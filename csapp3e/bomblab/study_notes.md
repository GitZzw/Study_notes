# 反汇编方法(linux)
## gdb调试器方法

> [gdb](https://man.linuxde.net/objdump)
```
>gdb bomb 
>break 37
>run
>disas
```
> 首先进入gdb调试工具，然后设置断点(对应bomb.c的37行)，运行，disas反汇编当前函数(此时为main)
```
>disas phase_1
```
> 反汇编phase_1函数，拆除bomb1

## objdump命令使用

> [objdump](https://man.linuxde.net/objdump)

> [gcc](https://man.linuxde.net/gcc)

> objdump命令是Linux下的反汇编目标文件或者可执行文件的命令，它以一种可阅读的格式让你更多地了解二进制文件可能带有的附加信息
```
>objdump -d bomb>bomb.asm
```

> 反汇编bomb可执行文件，生成bomb.asm汇编文件



### 1.phase_1

```
0000000000400ee0 <phase_1>:
  400ee0:   48 83 ec 08             sub    $0x8,%rsp
  400ee4:   be 00 24 40 00          mov    $0x402400,%esi
  400ee9:   e8 4a 04 00 00          callq  401338 <strings_not_equal>
  400eee:   85 c0                   test   %eax,%eax
  400ef0:   74 05                   je     400ef7 <phase_1+0x17>
  400ef2:   e8 43 05 00 00          callq  40143a <explode_bomb>
  400ef7:   48 83 c4 08             add    $0x8,%rsp
  400efb:   c3                      retq   
```

```
> x/s 0x402400 
```

> strings_not_equal函数接受两个参数rsi和rdi,rsi保存地址0x402400处的值,rdi接受phase_1的输入参数,test比较。 

> x/s 查看地址处的字符串
