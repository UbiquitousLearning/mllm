# 代码规范

##  1. 文件命名

### **总述**

文件名的每个单词首字母均大写, 不可以包含下划线（ `_` ）或连字符（ `-` ）如：`FileName.cpp`。

### **说明**

文件命名示例: 
* `Tensor.cpp`
* `CPUAdd.cpp`

通常应尽量让文件名更加明确。 `NetParameter.hpp` 就比 `Parameter.hpp` 要好。\
定义类时文件名一般成对出现, 如 `CPUAdd.hpp` 和 `CPUAdd.cpp`, 对应于类 `CPUAdd`。\
内联函数定义必须放在 `.hpp` 文件中。 如果函数比较短, 就直接将实现也放在 `.hpp` 中。

通常实现类的`.cpp`和`.hpp`文件存放在`/src`及其子目录下。一些表示状态的结构体/枚举定义的`.hpp`文件存放在`/include`目录下。


## 2.类型命名

### **总述**

类型名称的每个单词首字母均大写, 不可以包含下划线（ `_` ）或连字符（ `-` ） 如：`ClassName`。

### **说明**

所有类型命名 —— **类, 结构体, 类型定义 (`typedef`), 枚举, 类型模板参数** —— 均使用相同约定, 即以大写字母开始, 每个单词首字母均大写, 不包含下划线. 例如:

```C++
// 类和结构体
class Tensor { ...}
struct NetParameter { ...}

// 类型定义
typedef map<string, int> OpParam;

// using 别名
using OpParam = map<string, int>;

// 枚举
enum OpType { ...}
```

## 3.变量命名

### **总述**

变量 (包括函数参数) 和数据成员名一律小写, 单词之间用下划线连接。 \
类的成员变量以下划线结尾, 如：`a_class_data_member_`。 \
结构体和普通变量不用以下划线结尾, 如: `a_local_variable`，`a_struct_data_member`。

### **说明**

#### **普通变量命名**

举例:
```C++
string op_name;  // 用下划线.
```
#### **类数据成员**
不管是静态的还是非静态的, 类数据成员都要小写并以下划线（ `_` ）结尾.

举例:
```C++
class Tensor {
    ...
 private:
    string name_; // 用下划线并以下划线结尾.
    void *host_ptr_;
    static int idx_;
};
```

#### **结构体变量**
不管是静态的还是非静态的, 结构体数据成员都可以和普通变量一样, 不用像类那样以下划线结尾

举例:
```C++
struct NetOp{
    ...
    vector<string> in_op; 
    string name;
    static int idx;
} ;
```

## 4.函数命名

### **总述**

常规函数和类/结构体的成员函数使用驼峰命名法, `myFuction()`。 

### **说明**
常规函数名和类的函数名除首单词外每个单词首字母大写（即“驼峰变量名”）。 没有下划线. 对于首字母缩写的单词, 更倾向于将它们视作一个单词进行首字母大写 (例如, 写作`startRpc()` 而非 `startRPC()`)。

函数的参数与普通变量命名规则相同。
```C++
//常规函数：驼峰命名法
void createNetParem(ETENSOR end_t, NetParameter &net_param);

class Tensor {
public:
    ...
    //类的成员函数：驼峰命名法
    bool reshape(const vector<int> &shape);
    ...
};
```



类的成员函数中取值和设值函数则要求与变量名匹:
```C++
class Tensor {
public:
    ...
    //赋值
    void setName(string name) {
        name_ = name;
    }
    //取值
    const string name() const {
        return name_;
    }
 private:
    string name_;
};
```

## 4.枚举/宏命名

### **总述**

全部字母为大写，并且单词之间用下划线（ `_` ）连接，如`MY_ENUM`，`MY_DEFINE`。

### **说明**

举例：
```C++
//枚举
enum OpType {
    INPUT = 0,
    ADD,
    ...，
    OP_NUM
};

//宏定义
#define PREDICT(x) !!(x)

//宏定义
#ifndef MLLM_CHECK_H
#define MLLM_CHECK_H
#endif
```

