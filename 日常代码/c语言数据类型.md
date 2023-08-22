
#### 整型类型
----------------------------------------------------------------

| **类型** | **存储大小** | **值范围** |
| :------: | :------: | :------: |
| **char** | 1 byte | -128 ~ 127 or 0 ~ 255 |
| **unsigned char** | 1 byte | 0 ~ 255 |
| **signed char** | 1 byte | -128 ~ 127|
| **int** | 2 or 4 byte | -32,768 ~ 32,767 or <br> -2,147,483,648 ~ 2,147,483,647 |
| **unsigned int** | 2 or 4 byte | 0 ~ 65,535 or 0 ~ 4,294,967,295 |
| **short** | 2 byte | -32,768 ~ 32,767 |
| **unsigned short** | 2 byte | 0 ~ 65,535 |
| **long** | 4 byte | -2,147,483,648 ~ 2,147,483,647 |
| **unsigned long** | 4 byte | 0 ~ 4,294,967,295 |

使用 **sizeof(type)** 可以获取对象或类型的存储字节大小。
```c
#include <stdio.h>
#include <limits.h>

int main() {
    printf("sizeof(int) = %lu\n", sizeof(int));
    return 0;
}
```
**%lu** 为32位无符号整数，编译结果:
```c
sizeof(int) = 4
``` 
### 浮点类型
---

| **类型** | **存储大小** | **值范围** | **精度** |
| :------: | :------: | :------: | :------: |
| **float** | 4 byte | 1.2E-38 到 3.4E+38 | 6位 |
| **double** | 8 byte | 2.3E-308 到 1.7E+308 | 15位 |
| **long double** | 16 byte | 3.4E-4932 到 1.1E+4932 | 19位 |

```c
#include <stdio.h>
#include <float.h>

int main(){
    printf("float 存储最大字节数：%lu \n", sizeof(float));
    printf("float 最小值： %E \n", FLT_MIN);
    printf("float 最大值： %E \n", FLT_MAX);
    printf("精度值： %d \n", FLT_DIG);
}
```

```c
float 存储最大字节数 : 4 
float 最小值: 1.175494E-38
float 最大值: 3.402823E+38
精度值: 6
```

### 类型转换
---
```c
int i = 10;
float f = 3.14;
double d = i + f; // 隐式转换，向下兼容。int 转换为 double
int j = (int) d; // 显式转换，向上兼容。double 转换为 int
```

1、一种是需要建立存储空间的。例如：int a 在声明的时候就已经建立了存储空间。
2、另一种是不需要建立存储空间的，通过使用extern关键字声明变量名而不定义它。 例如：extern int a 其中变量 a 可以在别的文件中定义的。
```c
#include <stdio.h>
 
// 函数外定义变量 x 和 y
int x;
int y;
int addtwonum()
{
    // 函数内声明变量 x 和 y 为外部变量
    extern int x;
    extern int y;
    // 给外部变量（全局变量）x 和 y 赋值
    x = 1;
    y = 2;
    return x+y;
}
 
int main()
{
    int result;
    // 调用函数 addtwonum
    result = addtwonum();
    
    printf("result 为: %d",result);
    return 0;
}
```
```
result 为: 3
```
```c
char mychar = 'a';
int myAsciiValue = (int) mychar; // ASCII 值转化为整型值 97

```
1. auto 存储类是所有局部变量默认的存储类。定义在函数中的变量默认为 auto 存储类，这意味着它们在函数开始时被创建，在函数结束时被销毁。
2. register 存储类用于定义存储在寄存器中而不是 RAM 中的局部变量。这意味着变量的最大尺寸等于寄存器的大小（通常是一个字），且不能对它应用一元的 '&' 运算符（因为它没有内存位置）。register 存储类定义存储在寄存器，所以变量的访问速度更快，但是它不能直接取地址，因为它不是存储在 RAM 中的。在需要频繁访问的变量上使用 register 存储类可以提高程序的运行速度。
3. static 存储类指示编译器在程序的生命周期内保持局部变量的存在，而不需要在每次它进入和离开作用域时进行创建和销毁。因此，使用 static 修饰局部变量可以在函数调用之间保持局部变量的值。static 修饰符也可以应用于全局变量。当 static 修饰全局变量时，会使变量的作用域限制在声明它的文件内。全局声明的一个 static 变量或方法可以被任何函数或方法调用，只要这些方法出现在跟 static 变量或方法同一个文件中。静态变量在程序中只被初始化一次，即使函数被调用多次，该变量的值也不会重置。
4. extern 存储类用于定义在其他文件中声明的全局变量或函数。当使用 extern 关键字时，不会为变量分配任何存储空间，而只是指示编译器该变量在其他文件中定义。extern 存储类用于提供一个全局变量的引用，全局变量对所有的程序文件都是可见的。当您使用 extern 时，对于无法初始化的变量，会把变量名指向一个之前定义过的存储位置。当您有多个文件且定义了一个可以在其他文件中使用的全局变量或函数时，可以在其他文件中使用 extern 来得到已定义的变量或函数的引用。可以这么理解，extern 是用来在另一个文件中声明一个全局变量或函数。extern 修饰符通常用于当有两个或多个文件共享相同的全局变量或函数的时候。

| **运算符** | **描述** | **实例** |
| :------: | :------: | :------: |
| **sizeof** | 返回变量的大小 | sizeof(a) 将返回 4，其中 a 是整数。 |
| **&** | 返回变量的地址 | &a; 将给出变量的实际地址。 |
| \* | 指向一个变量 | *a; 将指向一个变量。 |
| **? :** | 条件表达式 | 如果条件为真 ? 则值为 X : 否则值为 Y |

C 语言把任何 **非零** 和 **非空** 的值假定为 **true**，把 **零** 或 **null** 假定为 **false**。

```c
#include <stdio.h>
 
/* 函数声明 */
int max(int num1, int num2);
 
int main ()
{
   /* 局部变量定义 */
   int a = 100;
   int b = 200;
   int ret;
 
   /* 调用函数来获取最大值 */
   ret = max(a, b);
 
   printf( "Max value is : %d\n", ret );
 
   return 0;
}
 
/* 函数返回两个数中较大的那个数 */
int max(int num1, int num2) 
{
   /* 局部变量声明 */
   int result;
 
   if (num1 > num2)
      result = num1;
   else
      result = num2;
 
   return result; 
}
```

**访问数组元素**
```c
#include <stdio.h>
 
int main ()
{
   int n[ 10 ]; /* n 是一个包含 10 个整数的数组 */
   int i,j;
 
   /* 初始化数组元素 */         
   for ( i = 0; i < 10; i++ )
   {
      n[ i ] = i + 100; /* 设置元素 i 为 i + 100 */
   }
   
   /* 输出数组中每个元素的值 */
   for (j = 0; j < 10; j++ )
   {
      printf("Element[%d] = %d\n", j, n[j] );
   }
 
   return 0;
}
```
```c
#include <stdio.h>
 
int main ()
{
   int n[ 10 ]; /* n 是一个包含 10 个整数的数组 */
   int i,j;
 
   /* 初始化数组元素 */         
   for ( i = 0; i < 10; i++ )
   {
      n[ i ] = i + 100; /* 设置元素 i 为 i + 100 */
   }
   
   /* 输出数组中每个元素的值 */
   for (j = 0; j < 10; j++ )
   {
      printf("Element[%d] = %d\n", j, n[j] );
   }
 
   return 0;
}
```
```
Element[0] = 100
Element[1] = 101
Element[2] = 102
Element[3] = 103
Element[4] = 104
Element[5] = 105
Element[6] = 106
Element[7] = 107
Element[8] = 108
Element[9] = 109
```
**枚举**
```
enum DAY
{
      MON=1, TUE, WED, THU, FRI, SAT, SUN
};
//第一个枚举成员的默认值为整型的 0，后续枚举成员的值在前一个成员上加 1。我们在这个实例中把第一个枚举成员的值定义为 1，第二个就为 2，以此类推。

enum season {spring, summer=3, autumn, winter};
// 没有指定值的枚举元素，其值为前一元素加 1。也就说 spring 的值为 0，summer 的值为 3，autumn 的值为 4，winter 的值为 5

// 介绍以下三种方式定义枚举，感觉有点像字典。
// 1
enum DAY
{
      MON=1, TUE, WED, THU, FRI, SAT, SUN
};
enum DAY day;
// 2
enum DAY
{
      MON=1, TUE, WED, THU, FRI, SAT, SUN
} day;
// 3
enum
{
      MON=1, TUE, WED, THU, FRI, SAT, SUN
} day;
```

**指针**
```c
#include <stdio.h>
 
int main ()
{
    int var_runoob = 10;
    int *p;              // 定义指针变量
    p = &var_runoob;
 
   printf("var_runoob 变量的地址： %p\n", p);
   return 0;
}
```
 ![Alt](https://www.runoob.com/wp-content/uploads/2014/09/c-pointer.png)
 
在这里，type 是指针的基类型，它必须是一个有效的 C 数据类型，var_name 是指针变量的名称。用来声明指针的星号 * 与乘法中使用的星号是相同的。但是，在这个语句中，星号是用来指定一个变量是指针。以下是有效的指针声明：
```
int    *ip;    /* 一个整型的指针 */
double *dp;    /* 一个 double 型的指针 */
float  *fp;    /* 一个浮点型的指针 */
char   *ch;    /* 一个字符型的指针 */
```
所有实际数据类型，不管是整型、浮点型、字符型，还是其他的数据类型，对应指针的值的类型都是一样的，都是一个代表内存地址的长的十六进制数。
不同数据类型的指针之间唯一的不同是，指针所指向的变量或常量的数据类型不同。

**如何使用指针**
```c
#include <stdio.h>
 
int main ()
{
   int  var = 20;   /* 实际变量的声明 */
   int  *ip;        /* 指针变量的声明 */
 
   ip = &var;  /* 在指针变量中存储 var 的地址 */
 
   printf("var 变量的地址: %p\n", &var  );
 
   /* 在指针变量中存储的地址 */
   printf("ip 变量存储的地址: %p\n", ip );
 
   /* 使用指针访问值 */
   printf("*ip 变量的值: %d\n", *ip );
 
   return 0;
}
```
```c
var 变量的地址: 0x7ffeeef168d8
ip 变量存储的地址: 0x7ffeeef168d8
*ip 变量的值: 20
```
```c
int *ptr = NULL;
printf("ptr 的地址是 %p\n", ptr  );
// ptr 的地址是 0x0

//在大多数的操作系统上，程序不允许访问地址为 0 的内存，因为该内存是操作系统保留的。然而，内存地址 0 有特别重要的意义，它表明该指针不指向一个可访问的内存位置。但按照惯例，如果指针包含空值（零值），则假定它不指向任何东西。如需检查一个空指针，您可以使用 if 语句，如下所示：
if(ptr)     /* 如果 p 非空，则完成 */
if(!ptr)    /* 如果 p 为空，则完成 */
```

**函数指针**
```c
#include <stdio.h>
 
int max(int x, int y)
{
    return x > y ? x : y;
}
 
int main(void)
{
    /* p 是函数指针 */
    int (* p)(int, int) = & max; // &可以省略
    int a, b, c, d;
 
    printf("请输入三个数字:");
    scanf("%d %d %d", & a, & b, & c);
 
    /* 与直接调用函数等价，d = max(max(a, b), c) */
    d = p(p(a, b), c); 
 
    printf("最大的数字是: %d\n", d);
 
    return 0;
}
```
**回调函数**
```c
#include <stdlib.h>  
#include <stdio.h>
 
void populate_array(int *array, size_t arraySize, int (*getNextValue)(void))
{
    for (size_t i=0; i<arraySize; i++)
        array[i] = getNextValue();
}
 
// 获取随机值
int getNextRandomValue(void)
{
    return rand();
}
 
int main(void)
{
    int myarray[10];
    /* getNextRandomValue 不能加括号，否则无法编译，因为加上括号之后相当于传入此参数时传入了 int , 而不是函数指针*/
    populate_array(myarray, 10, getNextRandomValue);
    for(int i = 0; i < 10; i++) {
        printf("%d ", myarray[i]);
    }
    printf("\n");
    return 0;
}
```
**字符串**
```c
#include <stdio.h>
 
int main ()
{
   char site[7] = {'R', 'U', 'N', 'O', 'O', 'B', '\0'};
   // char site = "RUNOOB";
   printf("菜鸟教程: %s\n", site );
   return 0;
}
```
1.	strcpy(s1, s2);
复制字符串 s2 到字符串 s1。
2.	strcat(s1, s2);
连接字符串 s2 到字符串 s1 的末尾。
3.	strlen(s1);
返回字符串 s1 的长度。
4.	strcmp(s1, s2);
如果 s1 和 s2 是相同的，则返回 0；如果 s1<s2 则返回小于 0；如果 s1>s2 则返回大于 0。
5.	strchr(s1, ch);
返回一个指针，指向字符串 s1 中字符 ch 的第一次出现的位置。
6.	strstr(s1, s2);
返回一个指针，指向字符串 s1 中字符串 s2 的第一次出现的位置。

应用：
```c
#include <stdio.h>
#include <string.h>
 
int main ()
{
   char str1[14] = "runoob";
   char str2[14] = "google";
   char str3[14];
   int  len ;
 
   /* 复制 str1 到 str3 */
   strcpy(str3, str1);
   printf("strcpy( str3, str1) :  %s\n", str3 );
 
   /* 连接 str1 和 str2 */
   strcat( str1, str2);
   printf("strcat( str1, str2):   %s\n", str1 );
 
   /* 连接后，str1 的总长度 */
   len = strlen(str1);
   printf("strlen(str1) :  %d\n", len );
 
   return 0;
}
```
```c
strcpy( str3, str1) :  runoob
strcat( str1, str2):   runoobgoogle
strlen(str1) :  12
```

**结构体参数**
```c
#include <stdio.h>
#include <string.h>
 
struct Books
{
   char  title[50];
   char  author[50];
   char  subject[100];
   int   book_id;
};
 
/* 函数声明 */
void printBook( struct Books book );
int main( )
{
   struct Books Book1;        /* 声明 Book1，类型为 Books */
   struct Books Book2;        /* 声明 Book2，类型为 Books */
 
   /* Book1 详述 */
   strcpy( Book1.title, "C Programming");
   strcpy( Book1.author, "Nuha Ali"); 
   strcpy( Book1.subject, "C Programming Tutorial");
   Book1.book_id = 6495407;
 
   /* Book2 详述 */
   strcpy( Book2.title, "Telecom Billing");
   strcpy( Book2.author, "Zara Ali");
   strcpy( Book2.subject, "Telecom Billing Tutorial");
   Book2.book_id = 6495700;
 
   /* 输出 Book1 信息 */
   printBook( Book1 );
 
   /* 输出 Book2 信息 */
   printBook( Book2 );
 
   return 0;
}
void printBook( struct Books book )
{
   printf( "Book title : %s\n", book.title);
   printf( "Book author : %s\n", book.author);
   printf( "Book subject : %s\n", book.subject);
   printf( "Book book_id : %d\n", book.book_id);
}
```

```c
Book title : C Programming
Book author : Nuha Ali
Book subject : C Programming Tutorial
Book book_id : 6495407
Book title : Telecom Billing
Book author : Zara Ali
Book subject : Telecom Billing Tutorial
Book book_id : 6495700
```

**指向结构体的指针**
```c

```