# Github使用规范

##  1. git commit 规范

commit message格式
```
<type>: <subject>
```
### type(必须)

用于说明git commit的类别，只允许使用下面的标识。

feat：新功能（feature）。

fix：修复BUG。

style：格式（不影响代码运行的变动）。

refactor：重构（即不是新增功能，也不是修改bug的代码变动）。

perf：优化相关，比如提升性能、体验。

test：增加测试。

chore：构建过程或辅助工具的变动。

revert：回滚到上一个版本。

merge：代码合并。

sync：同步主线或分支的Bug。

## subject(必须)

subject是commit目的的简短描述，不超过50个字符。

建议使用中文（用中文描述问题能更清楚一些）或者简单的英文。

结尾不加句号或其他标点符号。


##  2. 分支命名规范
该仓库分支命名规则如下:

|分支:	|	命名:	|	说明:|
|:--------| :---------|:--------|
|主分支	 |	master	|	主分支，目前不使用|
|开发分支	|	develop 	|	开发分支，永远是功能最新最全的分支|
|成员开发分支	|	develop-XX	|成员开发分支，开发者XX使用的分支|
|发布版本	|	release-*|	发布定期要上线的功能, 目前不使用|
|修复分支	|	bug-*	|	修复线上代码的bug, 目前不使用|


<font color=red size=6>注意！！！</font>： 
**开发者小明需要新建一个分支`develop-xm`在此分支上继续开发,不要直接push到`develop`分支.**
