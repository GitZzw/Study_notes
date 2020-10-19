# 反汇编方法(linux)
## gdb调试器方法

> [gdb](http://csapp.cs.cmu.edu/3e/docs/gdbnotes-x86-64.pdf)
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



### 2.phase_2

```
0000000000400efc <phase_2>:
  400efc:   55                      push   %rbp
  400efd:   53                      push   %rbx
  400efe:   48 83 ec 28             sub    $0x28,%rsp
  400f02:   48 89 e6                mov    %rsp,%rsi
  400f05:   e8 52 05 00 00          callq  40145c <read_six_numbers>　　　　;读入六个数
  400f0a:   83 3c 24 01             cmpl   $0x1,(%rsp)　　　　　　　　　　  ;第一个数和1比较
  400f0e:   74 20                   je     400f30 <phase_2+0x34>                        ；等于１跳转
  400f10:   e8 25 05 00 00          callq  40143a <explode_bomb>                      ;否则爆炸
  400f15:   eb 19                   jmp    400f30 <phase_2+0x34>
  400f17:   8b 43 fc                mov    -0x4(%rbx),%eax                     ;取出rbx-4处的值赋给eax
  400f1a:   01 c0                   add    %eax,%eax                               ; eax = eax *2
  400f1c:   39 03                   cmp    %eax,(%rbx)                                        
　　　　　　　　　;比较eax*2和rbx处的值,注意:eax是ebx-4处的值，即将rbx和前一个数的两倍比较
  400f1e:   74 05                   je     400f25 <phase_2+0x29>
　　　　　　                                                ；如果相等就跳转，而跳转处的代码是将rbx+4
  400f20:   e8 15 05 00 00          callq  40143a <explode_bomb>    ;否则爆炸
  400f25:   48 83 c3 04             add    $0x4,%rbx         ; 将rbx+4
  400f29:   48 39 eb                cmp    %rbp,%rbx      
                  ;将加4后的值和rbp比较，注意rbp是rsp+24,而rsp是第一个数，一个数四个字节。那么rbp就应该是
                  后那个数后面那个地址，即rbp是个循环哨兵
  400f2c:   75 e9                   jne    400f17 <phase_2+0x1b>   ；不等就继续跳转去循环
  400f2e:   eb 0c                   jmp    400f3c <phase_2+0x40>  ;　相等就结束跳转到函数结尾
  400f30:   48 8d 5c 24 04          lea    0x4(%rsp),%rbx                                       ；将rsp+4存到rbx
  400f35:   48 8d 6c 24 18          lea    0x18(%rsp),%rbp                                       ;将rsp +24 存到rbp
  400f3a:   eb db                   jmp    400f17 <phase_2+0x1b>                         ;跳转
  400f3c:   48 83 c4 28             add    $0x28,%rsp
  400f40:   5b                      pop    %rbx
  400f41:   5d                      pop    %rbp
  400f42:   c3                      retq   
```


### bomblab只做了两个练习掌握一下流程，后续有必要再深入研究汇编语言
> phase_3~phase_6:[参考](https://www.one-tab.com/page/bVPPA1wwRnekrX-2FiBmNQ)

