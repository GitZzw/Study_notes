gcc .\bit.c -o .\bit.exe

# 位操作

## 1.~ & 实现异或xor （排除00 11）

## 2.! ~ & ^ | + << >> 实现Tmin (移位)

## 3.! ~ & ^ | + 实现isTmax ()

## 4.! ~ & ^ | + << >> 实现判断奇数位是不是全为1 alloddbits (与0xAAAAAAAA再异或0xAAAAAAAA应该为0)

## 5. ! ~ & ^ | + << >> 实现取负数negate (取反+1)

## 6. ! ~ & ^ | + << >> 实现判断是不是数字isAsciiDigit (上下界 正负 移位31)

## 7. ! ~ & ^ | + << >> 实现条件函数conditional x ? y : z (通过两次非运算 x=0全变为0000....不为0全为1111....)    

## 8. ! ~ & ^ | + << >> 实现判断小于等于isLessOrEqual (~x + 1 + y的正负讨论)

## 9. ~ & ^ | + << >> 实现逻辑非 logicalNeg ！ (除了0之外其他数或自己相反数的符号位肯定为1)

## 10. ! ~ & ^ | + << >> 实现一个数用补码表示最少需要几位 howManyBits 

> (如果是一个正数，则需要找到它最高的一位（假设是n）是1的，再加上符号位，负数取反；判断高位是否有0  ———————> 判断是否为0：!!x ， 不断缩小范围)

## 11. Any integer/unsigned operations incl. ||, &&. also if, while 实现floatScale2 求2乘一个unsigned表示的浮点数 (浮点数的存储形式)

## 12. floatFloat2Int 实现将浮点数转换为整数

## 13. floatPower2 求2的x次方
