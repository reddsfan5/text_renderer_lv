# [python（八）：python使用lmdb数据库](https://www.cnblogs.com/zhangxianrong/p/14919706.html)

## 一、入门代码

LMDB的全称是Lightning Memory-Mapped Database(快如闪电的内存映射数据库)，它的文件结构简单，包含一个数据文件和一个锁文件：

![img](lmdb_tutorial.assets/201872112703222.jpg)

LMDB文件可以同时由多个进程打开，具有极高的数据存取速度，访问简单，不需要运行单独的数据库管理进程，只要在访问数据的代码里引用LMDB库，访问时给文件路径即可。

让系统访问大量小文件的开销很大，而LMDB使用内存映射的方式访问文件，使得文件内寻址的开销非常小，使用指针运算就能实现。数据库单文件还能减少数据集复制/传输过程的开销。

1. 生成一个空的lmdb数据库文件

```python
# -*- coding: utf-8 -*- 
import lmdb 
  
# 如果train文件夹下没有data.mbd或lock.mdb文件，则会生成一个空的，如果有，不会覆盖 
# map_size定义最大储存容量，单位是B，以下定义1TB容量 
env = lmdb.open("./train"，map_size=1*1024**4) 
env.close()
```

2. LMDB数据的添加、修改、删除

```python
# -*- coding: utf-8 -*- 
import lmdb 
  
# map_size定义最大储存容量，单位是kb，以下定义1TB容量 
env = lmdb.open("./train", map_size=1*1024**4) 
  
txn = env.begin(write=True) 
  
# 添加数据和键值 
txn.put(key = '1', value = 'aaa') 
txn.put(key = '2', value = 'bbb') 
txn.put(key = '3', value = 'ccc') 
  
# 通过键值删除数据 
txn.delete(key = '1') 
  
# 修改数据 
txn.put(key = '3', value = 'ddd') 
  
# 通过commit()函数提交更改 
txn.commit() 
env.close()
```

3. 查询lmdb数据库内容

```python
# -*- coding: utf-8 -*- 
import lmdb 
  
env = lmdb.open("./train") 
  
# 参数write设置为True才可以写入 
txn = env.begin(write=True) 
############################################添加、修改、删除数据 
  
# 添加数据和键值 
txn.put(key = '1', value = 'aaa') 
txn.put(key = '2', value = 'bbb') 
txn.put(key = '3', value = 'ccc') 
  
# 通过键值删除数据 
txn.delete(key = '1') 
  
# 修改数据 
txn.put(key = '3', value = 'ddd') 
  
# 通过commit()函数提交更改 
txn.commit() 
############################################查询lmdb数据 
txn = env.begin() 
  
# get函数通过键值查询数据 
print txn.get(str(2)) 
  
# 通过cursor()遍历所有数据和键值 
for key, value in txn.cursor(): 
  print (key, value) 
    
############################################ 
  
env.close()
```

4. 读取已有.mdb文件内容

```python
# -*- coding: utf-8 -*- 
import lmdb 
  
env_db = lmdb.Environment('trainC') 
# env_db = lmdb.open("./trainC") 
  
txn = env_db.begin() 
  
# get函数通过键值查询数据,如果要查询的键值没有对应数据，则输出None 
print txn.get(str(200)) 
  
for key, value in txn.cursor(): #遍历 
  print (key, value) 
  
env_db.close()
```

## 二、进阶

### LMDB 介绍

 

LMDB 全称为 Lightning Memory-Mapped Database，就是非常快的内存映射型数据库(键值存储数据库)，LMDB使用内存映射文件，可以提供更好的输入/输出性能，对于用于神经网络的大型数据集( 比如 [ImageNet](https://link.zhihu.com/?target=http%3A//image-net.org/) )，可以将其存储在 LMDB 中。

 

因为最开始 Caffe 就是使用的这个数据库，所以网上的大多数关于 LMDB 的教程都通过 Caffe 实现的，对于不了解 Caffe 的同学很不友好，所以本篇文章只讲解 LMDB。

 

LMDB属于key-value数据库，而不是关系型数据库( 比如 MySQL )，LMDB提供 key-value 存储，其中每个键值对都是我们数据集中的一个样本。LMDB的主要作用是提供数据管理，可以将各种各样的原始数据转换为统一的key-value存储。

 

LMDB效率高的一个关键原因是它是基于内存映射的，这意味着它返回指向键和值的内存地址的指针，而不需要像大多数其他数据库那样复制内存中的任何内容。

 

LMDB不仅可以用来存放训练和测试用的数据集，还可以存放神经网络提取出的特征数据。如果数据的结构很简单，就是大量的矩阵和向量，而且数据之间没有什么关联，数据内没有复杂的对象结构，那么就可以选择LMDB这个简单的数据库来存放数据。

 

LMDB的文件结构很简单，一个文件夹，里面是一个数据文件和一个锁文件。数据随意复制，随意传输。它的访问简单，不需要单独的数据管理进程。只要在访问代码里引用LMDB库，访问时给文件路径即可。

 

用LMDB数据库来存放图像数据，而不是直接读取原始图像数据的原因：

- 数据类型多种多样，比如：二进制文件、文本文件、编码后的图像文件jpeg、png等，不可能用一套代码实现所有类型的输入数据读取，因此通过LMDB数据库，转换为统一数据格式可以简化数据读取层的实现。
- lmdb具有极高的存取速度，大大减少了系统访问大量小文件时的磁盘IO的时间开销。LMDB将整个数据集都放在一个文件里，**避免了文件系统寻址的开销**，你的存储介质有多快，就能访问多快，不会因为文件多而导致时间长。LMDB使用了内存映射的方式访问文件，这使得文件内寻址的开销大幅度降低。

 

 **关键概念**：

1. `Environment`：所有的交互都在环境上下文中进行。环境是数据库的顶级抽象，可包含多个子数据库。
2. `Transaction`：对数据库的操作都在事务上下文中完成。有只读事务和读写事务两种。
3. `Database`：数据库在LMDB中是键值对的集合。每个环境可以包含多个数据库。
4. Cursor：是用于遍历数据库中的键值对的工具

> #### cursor是生成器么
>
> 在Python的LMDB库中，游标（Cursor）并不是生成器，而是一个可以用来迭代数据库中键值对的对象。虽然你可以像使用生成器一样使用游标（例如，你可以在游标上使用for循环），但是游标的工作方式与生成器有一些不同。
>
> 生成器是一种特殊类型的迭代器，它在每次迭代时运行一段代码来产生一个新的值。生成器只能被迭代一次，因为它们不会存储他们生成的值。
>
> 游标则不同，游标是一种可以多次迭代并且可以随时更改其位置的对象。游标指向数据库中的特定位置，并允许你在数据库中前后移动。你可以使用游标在数据库中插入，删除或者获取数据。
>
> 以下是一个例子，展示了如何在Python的LMDB库中使用游标：
>
> ```python
> import lmdb
> 
> env = lmdb.open('/tmp/test.lmdb')
> 
> with env.begin() as txn:
>     cursor = txn.cursor()
>     for key, value in cursor:
>         print(key, value)
> 
>     # Move cursor to a specific key
>     if cursor.set_key(b'my-key'):
>         print(cursor.key(), cursor.value())
>     else:
>         print('key not found')
> ```
>
> 在这个例子中，我们首先在一个事务中创建了一个游标，然后使用这个游标遍历了数据库中的所有键值对。然后，我们使用`set_key`方法将游标移动到了一个具有特定键的记录。如果找到了这个键，那么我们就打印出这个键和它的值。

上述概念主要是在键值存储数据库，特别是在LMDB中使用的。它们也确实与许多数据库相关的常见概念有关，但并非所有数据库系统都使用相同的术语或以相同的方式工作。

1. **Environment**：这个概念在LMDB中非常重要，但在其他数据库系统中，可能会有类似但不完全相同的概念。例如，在SQL数据库中，数据库环境可能涉及到服务器设置、连接池、权限等。
2. **Transaction**：事务的概念在许多数据库系统中都存在，它是一种机制，用于保证一组数据库操作的完整性和一致性。如果在事务执行过程中出现错误，那么所有在这个事务中的操作都会被回滚，以保证数据的完整性和一致性。SQL数据库和许多NoSQL数据库都支持事务。
3. **Cursor**：游标的概念在许多数据库系统中都存在，尤其是在SQL数据库中。游标允许你查询并遍历数据库中的记录。然而，游标在不同的数据库系统中可能会有所不同。例如，在SQL数据库中，游标可能会被用于复杂的数据处理和操作，而在键值存储数据库中，游标可能只是用于遍历键值对。

因此，虽然这些概念在很多数据库系统中都有应用，但它们的实现和用途可能会因数据库的类型和设计而异。在使用任何数据库系统时，都应阅读和理解该系统的文档和教程，以了解其特定的概念和工作方式。

### LMDB 的基本函数

- `env = lmdb.open()`：创建 lmdb 环境
- `txn = env.begin()`：建立事务
- `txn.put(key, value)`：进行插入和修改
- `txn.delete(key)`：进行删除
- `txn.get(key)`：进行查询
- `txn.cursor()`：进行遍历
- `txn.commit()`：提交更改

 

创建一个 lmdb 环境：

```python
# 安装：pip install lmdb
import lmdb

env = lmdb.open(lmdb_path, map_size=1*1024**4)
```

`lmdb_path` 指定存放生成的lmdb数据库的文件夹路径，如果没有该文件夹则自动创建。

 

`map_size` 指定创建的新数据库所需磁盘空间的最小值，`1*1024**4`＝１T。可以在这里进行 [存储单位换算](https://link.zhihu.com/?target=https%3A//cunchu.51240.com/)。

 

会在指定路径下创建 `data.mdb` 和 `lock.mdb` 两个文件，一是个数据文件，一个是锁文件。

 

修改数据库内容：

```python
txn = env.begin(write=True)

# insert/modify
txn.put(str(1).encode(), "Alice".encode())
txn.put(str(2).encode(), "Bob".encode())

# delete
txn.delete(str(1).encode())

txn.commit()
```

先创建一个事务(transaction) 对象 `txn`，所有的操作都必须经过这个事务对象。因为我们要对数据库进行写入操作，所以将 `write` 参数置为 `True`，默认其为 `False`。

 

使用 `.put(key, value)` 对数据库进行插入和修改操作，传入的参数为键值对。

 

值得注意的是，需要在键值字符串后加 `.encode()` 改变其编码格式，将 `str` 转换为 `bytes` 格式，否则会报该错误：`TypeError: Won't implicitly convert Unicode to bytes; use .encode()`。在后面使用 `.decode()` 对其进行解码得到原数据。

 

使用 `.delete(key)` 删除指定键值对。

 

对LMDB的读写操作在事务中执行，需要使用 `commit` 方法提交待处理的事务。

 

查询数据库内容：

```python
txn = env.begin()

print(txn.get(str(2).encode()))

for key, value in txn.cursor():
    print(key, value)

env.close()
```

每次 `commit()` 之后都要用 `env.begin()` 更新 txn（得到最新的lmdb数据库）。

 

使用 `.get(key)` 查询数据库中的单条记录。

 

使用 `.cursor()` 遍历数据库中的所有记录，其返回一个可迭代对象，相当于关系数据库中的游标，每读取一次，游标下移一位。

 

也可以想文件一样使用 `with` 语法：

```python
with env.begin() as txn:
    print(txn.get(str(2).encode()))

    for key, value in txn.cursor():
        print(key, value)
```

 

完整的demo如下：

```python
import lmdb
import os, sys

def initialize():
    env = lmdb.open("lmdb_dir")
    return env

def insert(env, sid, name):
    txn = env.begin(write=True)
    txn.put(str(sid).encode(), name.encode())
    txn.commit()

def delete(env, sid):
    txn = env.begin(write=True)
    txn.delete(str(sid).encode())
    txn.commit()

def update(env, sid, name):
    txn = env.begin(write=True)
    txn.put(str(sid).encode(), name.encode())
    txn.commit()

def search(env, sid):
    txn = env.begin()
    name = txn.get(str(sid).encode())
    return name

def display(env):
    txn = env.begin()
    cur = txn.cursor()
    for key, value in cur:
        print(key, value)


env = initialize()

print("Insert 3 records.")
insert(env, 1, "Alice")
insert(env, 2, "Bob")
insert(env, 3, "Peter")
display(env)

print("Delete the record where sid = 1.")
delete(env, 1)
display(env)

print("Update the record where sid = 3.")
update(env, 3, "Mark")
display(env)

print("Get the name of student whose sid = 3.")
name = search(env, 3)
print(name)

# 最后需要关闭关闭lmdb数据库
env.close()

# 执行系统命令
os.system("rm -r lmdb_dir")
```

 

### 图片数据示例

在图像深度学习训练中我们一般都会把大量原始数据集转化为lmdb格式以方便后续的网络训练。因此我们也需要对该数据集进行lmdb格式转化。

 

将图片和对应的文本标签存放到lmdb数据库：

```python
import lmdb

image_path = './cat.jpg'
label = 'cat'

env = lmdb.open('lmdb_dir')
cache = {}  # 存储键值对

with open(image_path, 'rb') as f:
    # 读取图像文件的二进制格式数据
    image_bin = f.read()

# 用两个键值对表示一个数据样本
cache['image_000'] = image_bin
cache['label_000'] = label

with env.begin(write=True) as txn:
    for k, v in cache.items():
        if isinstance(v, bytes):
            # 图片类型为bytes
            txn.put(k.encode(), v)
        else:
            # 标签类型为str, 转为bytes
            txn.put(k.encode(), v.encode())  # 编码

env.close()
```

这里需要获取图像文件的二进制格式数据，然后用两个键值对保存一个数据样本，即分开保存图片和其标签。

 

然后分别将图像和标签写入到lmdb数据库中，和上面例子一样都需要将键值转换为 `bytes` 格式。因为此处读取的图片格式本身就为 `bytes`，所以不需要转换，标签格式为 `str`，写入数据库之前需要先进行编码将其转换为 `bytes`。

 

从lmdb数据库中读取图片数据：

```python
import cv2
import lmdb
import numpy as np

env = lmdb.open('lmdb_dir')

with env.begin(write=False) as txn:
    # 获取图像数据
    image_bin = txn.get('image_000'.encode())
    label = txn.get('label_000'.encode()).decode()  # 解码

    # 将二进制文件转为十进制文件（一维数组）
    image_buf = np.frombuffer(image_bin, dtype=np.uint8)
    # 将数据转换(解码)成图像格式
    # cv2.IMREAD_GRAYSCALE为灰度图，cv2.IMREAD_COLOR为彩色图
    img = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
    cv2.imshow('image', img)
    cv2.waitKey(0)
```

先通过 `lmdb.open()` 获取之前创建的lmdb数据库。

 

这里通过键得到图片和其标签，因为写入数据库之前进行了编码，所以这里需要先解码。

- 标签通过 `.decode()` 进行解码重新得到字符串格式。
- 读取到的图片数据为二进制格式，所以先使用 `np.frombuffer()` 将其转换为十进制格式的文件，这是一维数组。然后可以使用 `cv2.imdecode()` 将其转换为灰度图（二维数组）或者彩色图（三维数组）。

## 三、原理

- Linux内存布局

- - 低址到高址分别是Text segment（代码段），数据段（已初始化的全局，静态变量），BSS段（未初始化的全局，静态变量），堆，内存映射区以及栈。

  - 内存分配的两种方式：

  - - brk

    - - brk是将数据段(.data)的最高地址指针_edata往高地址推。
      - malloc小于128k的内存，使用brk分配内存。
      - brk分配的内存需要等到高地址内存释放以后才能释放。

    - mmap

    - - mmap是在进程的虚拟地址空间中（堆和栈中间，称为文件映射区域的地方）找一块空闲的虚拟内存。
      - malloc大于128k的内存，使用mmap分配内存。
      - mmap分配的内存可以单独释放。

  - 这两种方式分配的都是虚拟内存，没有分配物理内存。在第一次访问已分配的虚拟地址空间的时候，发生缺页中断，操作系统负责分配物理内存，然后建立虚拟内存和物理内存之间的映射关系。

![img](lmdb_tutorial.assets/v2-2c88978907de3084fc56fb600b3feb30_1440w.jpg)

 

- - LMDB（Lightning Memory-Mapped Database）是一个小型数据库，具有一些强大的功能：

  - - 有序映射接口（键始终排序，支持范围查找）

    - 带有MVCC（多版本并发控制）的完全事务性，完全ACID（原子性，一致性，隔离性，耐久性）语义。

    - 读者/写者事务：读者不会阻止写者，写者也不会阻止读者。

    - - 写入程序已完全序列化，因此写入始终无死锁。

    - 读事务非常低开销，可以不使用malloc或任何其他阻塞调用来执行。

    - 支持多线程和多进程并发，环境可以由同一主机上的多个进程打开。

    - 可以创建多个子数据库，其中事务覆盖所有子数据库。

    - 内存映射，允许零拷贝查找和迭代。

    - 免维护，无需外部过程或后台清理/压缩。

    - 防崩溃，无需日志或崩溃恢复过程。

    - 没有应用程序级缓存。

    - - LMDB充分利用了操作系统的缓冲区高速缓存。

    - 32KB的目标代码和C的6KLOC。

 

- LMDB的基本结构

![img](lmdb_tutorial.assets/v2-31e05f14026db259c6722d92c9e83d3e_1440w.png)

 

- - 内存映射（Memory Map）

  - - 内存映射就是把物理内存映射到进程的地址空间之内，这些应用程序就可以直接使用输入输出的地址空间。

    - - 使用内存映射文件处理存储于磁盘上的文件时，将不需要由应用程序对文件执行I/O操作，这意味着在对文件进行处理时将不必再为文件申请并分配缓存，所有的文件缓存操作均由系统直接管理，由于取消了将文件数据加载到内存、数据从内存到文件的回写以及释放内存块等步骤，使得内存映射文件在处理大数据量的文件时能起到相当重要的作用。

 

- - 一般文件的io过程（两次数据拷贝）

1. 首先调用read()（系统调用）先将文件内容从硬盘拷贝到内核空间的一个缓冲区（一次数据拷贝）。
2. 然后再将这些数据拷贝到用户空间。（一次数据拷贝）

- 内存映射的文件io过程（一次数据拷贝）

1. 调用mmap()（系统调用）分配逻辑地址。
2. 逻辑地址转换成物理地址。
3. 进程第一次访问ptr指向的内存地址，发生缺页中断。由中断处理函数将数据拷贝到相应的内存地址上。（一次数据拷贝）
4. 虚拟地址置换。

- - lmdb使用mmap文件映射,不管这个文件存储实在内存上还是在持久存储上。

  - - lmdb的所有读取操作都是通过mmap将要访问的文件只读的映射到虚拟内存中，直接访问相应的地址。

    - 因为使用了read-only的mmap，同样避免了程序错误将存储结构写坏的风险。

    - 写操作，则是通过write系统调用进行的，这主要是为了利用操作系统的文件系统一致性，避免在被访问的地址上进行同步。

    - lmdb把整个虚拟存储组织成B+Tree存储,索引和值读存储在B+Tree的页面上（聚集索引）。

    - LMDB中使用append-only B+树，其更新操作最终都会转换为B+树的物理存储的append操作，文件不支持内部的修改，只支持append。

    - - append增加了存储开销，因为旧的数据依然存在。带来一个额外的好处就是，旧的链路依然存在，依然可以正常的访问，例如过去有个人持有了过去的root（root9）的指针，那么过去的整棵树都完整的存在。

![img](lmdb_tutorial.assets/v2-f202c1794ca61e233790d1398192e0fb_1440w.png)![img](lmdb_tutorial.assets/v2-f6db2c50a1092e9b7154ee56f86bae14_1440w.jpg)

- - COW(Copy-on-write)写时复制

  - - 如果有多个调用者（callers）同时要求相同资源（如内存或磁盘上的数据存储），他们会共同获取相同的指针指向相同的资源，直到某个调用者试图修改资源的内容时，系统才会真正复制一份专用副本（private copy）给该调用者，而其他调用者所见到的最初的资源仍然保持不变。

    - 因此多个调用者只是读取操作时可以共享同一份资源。

    - 优点

    - - 如果调用者没有修改该资源，就不会有副本（private copy）被创建。

 

- 当前读

- - 读取的是记录的最新版本，读取时还要保证其他并发事务不能修改当前记录，会对读取的记录进行加锁。

  - - 例如select lock in share mode(共享锁), select for update ; update, insert ,delete(排他锁)这些操作都是一种当前读

- 快照读

- - 快照读的实现是基于多版本并发控制，即MVCC，避免了加锁操作，降低了开销。

- - MVCC(Multiversion concurrency control )多版并发本控制

  - - 当MVCC数据库需要更新数据项时，它不会用新数据覆盖旧数据，而是将旧数据标记为已过时并在其他位置添加新版本。

    - - 因此存储了多个版本，但只有一个是最新版本。

    - MVCC通过保存数据的历史版本，根据比较版本号来处理数据的是否显示，从而达到读取数据的时候不需要加锁就可以保证事务隔离性的效果。

    - 数据库系统维护当前活跃的事务ID列表m_ids，其中最小值up_limit_id和最大值low_limit_id，被访问的事务ID：trx_id。

    - - 如果trx_id< up_limit_id，说明trx_id对应的事务在生成可读视图前已经被提交了，可以被当前事务访问
      - 如果trx_id> low_limit_id，说明事务trx_id生成可读视图后才生成的，所以不可以被当前事务访问到。
      - 如果up_limit_id <=trx_id<= low_limit_id，判断trx_id是否在m_ids中，若在，则说明trix_id在生成可读视图时还处于活跃，不可以被访问到；弱国不在m_ids中，说明在生成可读视图时该事务已经被提交了，故可以被访问到。

 

- MVCC/COW在LMDB中的实现

- - LMDB对MVCC加了一个限制，即只允许一个写线程存在，从根源上避免了写写冲突，当然代价就是写入的并发性能下降。

  - - 因为只有一个写线程，所以不会不需要wait日志、读写依赖队列、锁队列等一系列控制并发、事务回滚、数据恢复的基础工具。

  - MVCC的基础就是COW，对于不同的用户来说，若其在整个操作过程中不进行任何的数据改变，其就使用同一份数据即可。若需要进行改变，比如增加、删除、修改等，就需要在私有数据版本上进行，修改完成提交之后才给其他事务可见。

- - LMDB的事务实现

  - - Atom（A）原子性：LMDB中通过txn数据结构和cursor数据结构的控制，通过将脏页列表放入 dirtylist中，当txn进行提交时再一次性统一刷新到磁盘中或者abort时都不提交保证事务要不全成功、要不全失败。对于长事务，若页面spill到磁盘，因为COW技术，这些页面未与整棵B-Tree的rootpage产生关联，因此后续的事务还是不能访问到这些页面，同样保证了事务的原子性。

    - Consistency(C)一致性： 有如上的操作,保证其数据就是一致的，不存在因为多线程同时写数据导致数据产生错误的情况。

    - Isolation（I）隔离性：事务隔离通过锁控制（MUTEX），LMDB支持的锁互斥是进程级别/线程级别，支持的隔离方式为锁表支持，读读之间不锁，写等待读完成之后开始，读等待写完成后开始。

    - Duration（D）持久性：，没有使用WAL、undo/redo log等技术来保证系统崩溃时数据库的可用性，其保证数据持续可用的技术是COW技术和只有一线程写技术。

    - - 假如LMDB或者系统崩溃时，只有读操作，那么数据本来就没有发生变化，因此数据将不可能遭到破坏。假如崩溃时，有一个线程在进行写操作，则只需要判断最后的页面号与成功提交到数据库中的页面号是否一致，若不一致则说明写操作没有完成，则最后一个事务写失败，数据在最后一个成功的页面前的是正确的，后续的属于崩溃事务的，不能用，这样就保证了数据只要序列化到磁盘则一定可用，要不其就是还没有遵循ACI原则序列化到磁盘。

 

- 聚集索引和辅助索引

- - 聚集索引

  - - 指索引项的排序方式和表中数据记录排序方式一致的索引，每张表只能有一个聚集索引，聚集索引的叶子节点存储了整个行数据。

    - - 如果一个主键被定义了，那么这个主键就是作为聚集索引。
      - 如果没有主键被定义，那么该表的第一个唯一非空索引被作为聚集索引。
      - 如果没有主键也没有合适的唯一索引，那么innodb内部会生成一个隐藏的主键作为聚集索引，这个隐藏的主键是一个6个字节的列，该列的值会随着数据的插入自增。

    - 由于实际的数据页只能按照一颗B+树进行排序，因此每张表只能有一个聚集索引。

  - 辅助索引

  - - 一个表中的所有索引除了聚集索引，其他的都是二级（辅助）索引（secondary index）。
    - 辅助索引，其叶子节点并不包含行记录的全部数据，叶子结点除了包含键值以外，每个叶子结点中的索引行还包含了一个书签，该书签用来告诉存储引擎可以在哪找到相应的数据行，由于innodb引擎表是索引组织表，因此innodb存储引擎的辅助索引的书签就是相应行数据的聚集索引键，

- 示例代码

```python
import lmdb
# map_size定义最大储存容量，单位是kb，以下定义1TB容量
env = lmdb.open("./train", map_size=1099511627776)

# 参数write设置为True才可以写入
txn = env.begin(write=True) # 开启事务

# 通过cursor()遍历所有数据和键值
for key, value in txn.cursor():
        print (key, value)

# 添加数据和键值
txn.put(key = '1', value = 'aaa')
txn.put(key = '2', value = 'bbb')
txn.put(key = '3', value = 'ccc')

# 通过键值删除数据
txn.delete(key = '1')

# 修改数据
txn.put(key = '3', value = 'ddd')

# 通过commit()函数提交更改
txn.commit()                        # 提交事务

env.close()         # 关闭事务
```

------

### HDF5

HDF5 (Hierarchical Data Format) 是一种常见的跨平台数据储存文件，可以存储不同类型的图像和数码数据，并且可以在不同类型的机器上传输，同时还有统一处理这种文件格式的函数库。

- HDF5 文件结构中有 2 primary objects:

- - Groups

  - - 类似于文件夹，每个 HDF5 文件其实就是根目录 (root)为 group'/'

  - Datasets

  - - 类似于 NumPy 中的数组 array

    - 每个 dataset 可以分成两部分

    - - 原始数据 (raw) data values

      - 元数据 metadata

      - - 描述并提供有关其他数据的信息的数据

- 整个HDF5结构

```text
+-- /
|     +-- group_1
|     |     +-- dataset_1_1
|     |     |     +-- attribute_1_1_1
|     |     |     +-- attribute_1_1_2
|     |     |     +-- ...
|     |     |
|     |     +-- dataset_1_2
|     |     |     +-- attribute_1_2_1
|     |     |     +-- attribute_1_2_2
|     |     |     +-- ...
|     |     |
|     |     +-- ...
|     |
|     +-- group_2
|     |     +-- dataset_2_1
|     |     |     +-- attribute_2_1_1
|     |     |     +-- attribute_2_1_2
|     |     |     +-- ...
|     |     |
|     |     +-- dataset_2_2
|     |     |     +-- attribute_2_2_1
|     |     |     +-- attribute_2_2_2
|     |     |     +-- ...
|     |     |
|     |     +-- ...
|     |
|     +-- ...
|
```

- HDF5的特性

- - 自述性

  - - 对于一个HDF 文件里的每一个数据对象，有关于该数据的综合信息（元数据）

    - - 在没有任何外部信息的情况下，HDF 允许应用程序解释HDF文件的结构和内容。

  - 通用性

  - - 许多数据类型都可以被嵌入在一个HDF文件里

    - - 例如，通过使用合适的HDF 数据结构，符号、数字和图形数据可以同时存储在一个HDF 文件里。

  - 灵活性

  - - HDF允许用户把相关的数据对象组合在一起，放到一个分层结构中，向数据对象添加描述和标签。
    - 它还允许用户把科学数据放到多个HDF 文件里。

  - 扩展性

  - - HDF极易容纳将来新增加的数据模式，容易与其他标准格式兼容。

  - 跨平台性

  - - HDF 是一个与平台无关的文件格式。HDF 文件无需任何转换就可以在不同平台上使用。

## 四、pytorch使用

转：https://blog.csdn.net/P_LarT/article/details/103208405

https://www.yuque.com/lart/ugkv9f/hbnym1

文章https://blog.csdn.net/jyl1999xxxx/article/details/53942824中介绍了使用LMDB的原因：

Caffe使用LMDB来存放训练/测试用的数据集，以及使用网络提取出的feature（为了方便，以下还是统称数据集）。数据集的结构很简单，就是大量的矩阵/向量数据平铺开来。数据之间没有什么关联，数据内没有复杂的对象结构，就是向量和矩阵。既然数据并不复杂，Caffe就选择了LMDB这个简单的数据库来存放数据。

图像数据集归根究底从图像文件而来。引入数据库存放数据集，是为了减少IO开销。读取大量小文件的开销是非常大的，尤其是在机械硬盘上。LMDB的整个数据库放在一个文件里，避免了文件系统寻址的开销。LMDB使用内存映射的方式访问文件，使得文件内寻址的开销非常小，使用指针运算就能实现。数据库单文件还能减少数据集复制/传输过程的开销。一个几万，几十万文件的数据集，不管是直接复制，还是打包再解包，过程都无比漫长而痛苦。LMDB数据库只有一个文件，你的介质有多块，就能复制多快，不会因为文件多而慢如蜗牛。

在文章http://shuokay.com/2018/05/14/python-lmdb/中类似提到：

为什么要把图像数据转换成大的二进制文件？

简单来说，是**因为读写小文件的速度太慢**。那么，不禁要问，图像数据也是二进制文件，单个大的二进制文件例如 LMDB 文件也是二进制文件，为什么单个图像读写速度就慢了呢？这里分两种情况解释。

1. 机械硬盘的情况：机械硬盘的每次读写启动时间比较长，例如磁头的寻道时间占比很高，因此，如果单个小文件读写，尤其是随机读写单个小文件的时候，这个寻道时间占比就会很高，最后导致大量读写小文件的时候时间会很浪费；
2. NFS 的情况：在 NFS 的场景下，系统的一次读写首先要进行上百次的网络通讯，并且这个通讯次数和文件的大小无关。因此，如果是读写小文件，这个网络通讯时间占据了整个读写时间的大部分。

固态硬盘的情况下应该也会有一些类似的开销，目前没有研究过。

 

总而言之，使用LMDB可以为我们的数据读取进行加速。

 

## 具体操作

 

### LMDB主要类

 

#### [lmdb.Environment](https://lmdb.readthedocs.io/en/release/#environment-class)

 

```python
class lmdb.Environment(path, map_size=10485760, subdir=True, readonly=False, metasync=True, sync=True, map_async=False, mode=493, create=True, readahead=True, writemap=False, meminit=True, max_readers=126, max_dbs=0, max_spare_txns=1, lock=True)
```

Structure for a database environment. An environment may contain multiple databases, all residing in the same shared-memory map and underlying disk file.

 To write to the environment a [`Transaction`](https://lmdb.readthedocs.io/en/release/#lmdb.Transaction) must be created. One simultaneous write transaction is allowed, however there is no limit on the number of read transactions even when a write transaction exists.

**This class is aliased to lmdb.open.**

这是数据库环境的结构。 一个环境可能包含多个数据库，所有数据库都驻留在同一共享内存映射和基础磁盘文件中。

几个重要的实例方法：

-  begin(db=None, parent=None, write=False, buffers=False): 可以调用事务类 
-  `lmdb.Transaction` open_db(key=None, txn=None, reverse_key=False, dupsort=False, create=True, integerkey=False, integerdup=False, dupfixed=False): 打开一个数据库，返回一个不透明的句柄。重复`Environment.open_db()` 调用相同的名称将返回相同的句柄。作为一个特殊情况，主数据库总是开放的。命名数据库是通过在主数据库中存储一个特殊的描述符来实现的。环境中的所有数据库共享相同的文件。因为描述符存在于主数据库中，所以如果已经存在与数据库名称匹配的 `key` ，创建命名数据库的尝试将失败。此外，查找和枚举可以看到`key` 。如果主数据库`keyspace`与命名数据库使用的名称冲突，则将主数据库的内容移动到另一个命名数据库。

 

```
>>> env = lmdb.open('/tmp/test', max_dbs=2)
>>> with env.begin(write=True) as txn
...     txn.put('somename', 'somedata')
>>> # Error: database cannot share name of existing key!
>>> subdb = env.open_db('somename')
```

 

#### [lmdb.Transaction](https://lmdb.readthedocs.io/en/release/#transaction-class)

`class lmdb.Transaction(env, db=None, parent=None, write=False, buffers=False)` 。

 

所有操作都需要事务句柄，事务可以是只读或读写的。写事务可能不会跨越线程。事务对象实现了上下文管理器协议，因此即使面对未处理的异常，也可以可靠地释放事务：

 

```python
# Transaction aborts correctly:
with env.begin(write=True) as txn:
   crash()
# Transaction commits automatically:
with env.begin(write=True) as txn:
   txn.put('a', 'b')
```

 

这个类的实例包含着很多有用的操作方法。

 

- abort(): 中止挂起的事务。重复调用 `abort()` 在之前成功的 `commit()` 或 `abort()` 后或者在相关环境关闭后是没有效果的。
- 提交挂起的事务。

- Shortcut for `lmdb.Cursor(db, self)` 
- delete(key, value='', db=None): Delete a key from the database.

- The key to delete.
- value:如果数据库是以 `dupsort = True` 打开的，并且 `value` 不是空的 `bytestring` ，则删除仅与此 `(key, value)` 对匹配的元素，否则该 `key` 的所有值都将被删除。

- Returns `True` if at least one key was deleted.

- cursors
- pop(key, db=None): 使用临时cursor调用 `Cursor.pop()` 。

- 
- put(key, value, dupdata=True, overwrite=True, append=False, db=None): 存储一条记录(record)，如果记录被写入，则返回 `True` ，否则返回 `False` ，以指示key已经存在并且 `overwrite = False` 。成功后，cursor位于新记录上。

- key: Bytestring key to store.
- value: Bytestring value to store.

- dupdata: 如果 `True` ，并且数据库是用 `dupsort = True` 打开的，如果给定 `key` 已经存在，则添加键值对作为副本。否则覆盖任何现有匹配的 `key` 。
- overwrite: 
- append: 如果为 `True` ，则将对附加到数据库末尾，而不首先比较其顺序。附加不大于现有最高 `key` 的 `key` 将导致损坏。
- db: 要操作的命名数据库。如果未指定，默认为
- replace(key, value, db=None): 使用临时cursor调用 `Cursor.replace()` .
- Named database to operate on. If unspecified, defaults to the database given to the Transaction constructor.

- stat(db): Return statistics like `Environment.stat()` , except for a single DBI. `db` must be a database handle returned by `open_db()` .



#### [Imdb.Cursor](https://lmdb.readthedocs.io/en/release/#cursor-class)

```python

class lmdb.Cursor(db, txn)

	Structure for navigating a database.

	db:

	_Database to navigate.

	txn:

	Transaction to navigate.
```

As a convenience, `Transaction.cursor()` can be used to quickly return a cursor:

 

```
>>> env = lmdb.open('/tmp/foo')
>>> child_db = env.open_db('child_db')
>>> with env.begin() as txn:
...     cursor = txn.cursor()           # Cursor on main database.
...     cursor2 = txn.cursor(child_db)  # Cursor on child database.
```

 

游标以未定位的状态开始。如果在这种状态下使用 `iternext()` 或 `iterprev()` ，那么迭代将分别从开始处和结束处开始。迭代器直接使用游标定位，这意味着**在同一游标上存在多个迭代器时会产生奇怪的行为**。

 

从Python绑定的角度来看，一旦任何扫描或查找方法(例如 `next()` 、 `prev_nodup()` 、 `set_range()` )返回 `False` 或引发异常，游标将返回未定位状态。这主要是为了确保在面对任何错误条件时语义的安全性和一致性。

当游标返回到未定位的状态时，它的 `key()` 和 `value()` 返回空字符串，表示没有活动的位置，尽管在内部，LMDB游标可能仍然有一个有效的位置。

这可能会导致在迭代 `dupsort=True` 数据库的 `key` 时出现一些令人吃惊的行为，因为 `iternext_dup()` 等方法将导致游标显示为未定位，尽管它返回 `False` 只是为了表明当前键没有更多的值。在这种情况下，简单地调用 `next()` 将导致在下一个可用键处继续迭代。

This behaviour may change in future.

 

Iterator methods such as `iternext()` and `iterprev()` accept keys and values arguments. If both are `True` , then the value of `item()` is yielded on each iteration. If only keys is `True` , `key()` is yielded, otherwise only `value()` is yielded.

 

在迭代之前，游标可能定位在数据库中的任何位置

 

```
>>> with env.begin() as txn:
...     cursor = txn.cursor()
...     if not cursor.set_range('5'): # Position at first key >= '5'.
...         print('Not found!')
...     else:
...         for key, value in cursor: # Iterate from first key >= '5'.
...             print((key, value))
```

 

不需要迭代来导航，有时会导致丑陋或低效的代码。在迭代顺序不明显的情况下，或者与正在读取的数据相关的情况下，使用 `set_key()` 、 `set_range()` 、 `key()` 、 `value()` 和 `item()` 可能是更好的选择。

 

```
>>> # Record the path from a child to the root of a tree.
>>> path = ['child14123']
>>> while path[-1] != 'root':
...     assert cursor.set_key(path[-1]), \
...         'Tree is broken! Path: %s' % (path,)
...     path.append(cursor.value())
```

 

几个实例方法：

- set_key(key): Seek exactly to key, returning `True` on success or `False` if the exact key was not found. 对于 `set_key()` ，空字节串是错误的。对于使用 `dupsort=True` 打开的数据库，移动到键的第一个值(复制)。
- set_range(key): Seek to the first `key` greater than or equal to `key` , returning `True` on success, or `False` to indicate `key` was past end of database. Behaves like `first()` if key is the empty bytestring. 对于使用 `dupsort=True` 打开的数据库，移动到键的第一个值(复制)。

- get(key, default=None): Equivalent to `set_key()` , except `value()` is returned when key is found, otherwise default.

- item(): Return the current `(key, value)` pair.
- key(): Return the current key.

- value(): Return the current value.

 

### 操作流程

 

概况地讲，操作LMDB的流程是：

 

- 通过 `env = lmdb.open()` 打开环境
- 通过 `txn = env.begin()` 建立事务

- 通过 `txn.put(key, value)` 进行插入和修改
- 通过 `txn.delete(key)` 进行删除

- 通过 `txn.get(key)` 进行查询
- 通过 `txn.cursor()` 进行遍历

- 通过 `txn.commit()` 提交更改

 

这里要注意：

1. `put` 和 `delete` 后一定注意要 `commit` ，不然根本没有存进去
2. 每一次 `commit` 后，需要再定义一次 `txn=env.begin(write=True)` 

 

来自的代码：

 

```python
#!/usr/bin/env python

import lmdb
import os, sys

def initialize():
	env = lmdb.open("students");
	return env;

def insert(env, sid, name):
	txn = env.begin(write = True);
	txn.put(str(sid), name);
	txn.commit();

def delete(env, sid):
	txn = env.begin(write = True);
	txn.delete(str(sid));
	txn.commit();

def update(env, sid, name):
	txn = env.begin(write = True);
	txn.put(str(sid), name);
	txn.commit();

def search(env, sid):
	txn = env.begin();
	name = txn.get(str(sid));
	return name;

def display(env):
	txn = env.begin();
	cur = txn.cursor();
	for key, value in cur:
		print (key, value);

env = initialize();

print "Insert 3 records."
insert(env, 1, "Alice");
insert(env, 2, "Bob");
insert(env, 3, "Peter");
display(env);

print "Delete the record where sid = 1."
delete(env, 1);
display(env);

print "Update the record where sid = 3."
update(env, 3, "Mark");
display(env);

print "Get the name of student whose sid = 3."
name = search(env, 3);
print name;

env.close();

os.system("rm -r students");
```

 

### 创建图像数据集

 

这里主要借鉴自https://github.com/open-mmlab/mmsr/blob/master/codes/data_scripts/create_lmdb.py的代码。

 

改写为：

 

```python
import glob
import os
import pickle
import sys

import cv2
import lmdb
import numpy as np
from tqdm import tqdm


def main(mode):
    proj_root = '/home/lart/coding/TIFNet'
    datasets_root = '/home/lart/Datasets/'
    lmdb_path = os.path.join(proj_root, 'datasets/ECSSD.lmdb')
    data_path = os.path.join(datasets_root, 'RGBSaliency', 'ECSSD/Image')
    
    if mode == 'creating':
        opt = {
            'name': 'TrainSet',
            'img_folder': data_path,
            'lmdb_save_path': lmdb_path,
            'commit_interval': 100,  # After commit_interval images, lmdb commits
            'num_workers': 8,
        }
        general_image_folder(opt)
    elif mode == 'testing':
        test_lmdb(lmdb_path, index=1)


def general_image_folder(opt):
    """
    Create lmdb for general image folders
    If all the images have the same resolution, it will only store one copy of resolution info.
        Otherwise, it will store every resolution info.
    """
    img_folder = opt['img_folder']
    lmdb_save_path = opt['lmdb_save_path']
    meta_info = {'name': opt['name']}
    
    if not lmdb_save_path.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with 'lmdb'.")
    if os.path.exists(lmdb_save_path):
        print('Folder [{:s}] already exists. Exit...'.format(lmdb_save_path))
        sys.exit(1)
    
    # read all the image paths to a list
    
    print('Reading image path list ...')
    all_img_list = sorted(glob.glob(os.path.join(img_folder, '*')))
    # cache the filename, 这里的文件名必须是ascii字符
    keys = []
    for img_path in all_img_list:
        keys.append(os.path.basename(img_path))
    
    # create lmdb environment
    
    # 估算大概的映射空间大小
    data_size_per_img = cv2.imread(all_img_list[0], cv2.IMREAD_UNCHANGED).nbytes
    print('data size per image is: ', data_size_per_img)
    data_size = data_size_per_img * len(all_img_list)
    env = lmdb.open(lmdb_save_path, map_size=data_size * 10)
    # map_size：
    # Maximum size database may grow to; used to size the memory mapping. If database grows larger
    # than map_size, an exception will be raised and the user must close and reopen Environment.
    
    # write data to lmdb
    
    txn = env.begin(write=True)
    resolutions = []
    tqdm_iter = tqdm(enumerate(zip(all_img_list, keys)), total=len(all_img_list), leave=False)
    for idx, (path, key) in tqdm_iter:
        tqdm_iter.set_description('Write {}'.format(key))
        
        key_byte = key.encode('ascii')
        data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
        if data.ndim == 2:
            H, W = data.shape
            C = 1
        else:
            H, W, C = data.shape
        resolutions.append('{:d}_{:d}_{:d}'.format(C, H, W))
        
        txn.put(key_byte, data)
        if (idx + 1) % opt['commit_interval'] == 0:
            txn.commit()
            # commit 之后需要再次 begin
            txn = env.begin(write=True)
    txn.commit()
    env.close()
    print('Finish writing lmdb.')
    
    # create meta information
    
    # check whether all the images are the same size
    assert len(keys) == len(resolutions)
    if len(set(resolutions)) <= 1:
        meta_info['resolution'] = [resolutions[0]]
        meta_info['keys'] = keys
        print('All images have the same resolution. Simplify the meta info.')
    else:
        meta_info['resolution'] = resolutions
        meta_info['keys'] = keys
        print('Not all images have the same resolution. Save meta info for each image.')
    
    pickle.dump(meta_info, open(os.path.join(lmdb_save_path, 'meta_info.pkl'), "wb"))
    print('Finish creating lmdb meta info.')


def test_lmdb(dataroot, index=1):
    env = lmdb.open(dataroot, readonly=True, lock=False, readahead=False, meminit=False)
    meta_info = pickle.load(open(os.path.join(dataroot, 'meta_info.pkl'), "rb"))
    print('Name: ', meta_info['name'])
    print('Resolution: ', meta_info['resolution'])
    print('# keys: ', len(meta_info['keys']))
    
    # read one image
    key = meta_info['keys'][index]
    print('Reading {} for test.'.format(key))
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    
    C, H, W = [int(s) for s in meta_info['resolution'][index].split('_')]
    img = img_flat.reshape(H, W, C)
    
    cv2.namedWindow('Test')
    cv2.imshow('Test', img)
    cv2.waitKeyEx()


if __name__ == "__main__":
    # mode = creating or testing
    main(mode='creating')
```

 

### 配合DataLoader

 

这里仅对训练集进行LMDB处理，测试机依旧使用的原始的读取图片的方式。

 

```python
import os
import pickle

import lmdb
import numpy as np
from PIL import Image
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils import joint_transforms


def _get_paths_from_lmdb(dataroot):
    """get image path list from lmdb meta info"""
    meta_info = pickle.load(open(os.path.join(dataroot, 'meta_info.pkl'),
                                 'rb'))
    paths = meta_info['keys']
    sizes = meta_info['resolution']
    if len(sizes) == 1:
        sizes = sizes * len(paths)
    return paths, sizes


def _read_img_lmdb(env, key, size):
    """read image from lmdb with key (w/ and w/o fixed size)
    size: (C, H, W) tuple"""
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    C, H, W = size
    img = img_flat.reshape(H, W, C)
    return img


def _make_dataset(root, prefix=('.jpg', '.png')):
    img_path = os.path.join(root, 'Image')
    gt_path = os.path.join(root, 'Mask')
    img_list = [
        os.path.splitext(f)[0] for f in os.listdir(gt_path)
        if f.endswith(prefix[1])
    ]
    return [(os.path.join(img_path, img_name + prefix[0]),
             os.path.join(gt_path, img_name + prefix[1]))
            for img_name in img_list]


class TestImageFolder(Dataset):
    def __init__(self, root, in_size, prefix):
        self.imgs = _make_dataset(root, prefix=prefix)
        self.test_img_trainsform = transforms.Compose([
            # 输入的如果是一个tuple，则按照数据缩放，但是如果是一个数字，则按比例缩放到短边等于该值
            transforms.Resize((in_size, in_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        
        img = Image.open(img_path).convert('RGB')
        img_name = (img_path.split(os.sep)[-1]).split('.')[0]
        
        img = self.test_img_trainsform(img)
        return img, img_name
    
    def __len__(self):
        return len(self.imgs)


class TrainImageFolder(Dataset):
    def __init__(self, root, in_size, scale=1.5, use_bigt=False):
        self.use_bigt = use_bigt
        self.in_size = in_size
        self.root = root
        
        self.train_joint_transform = joint_transforms.Compose([
            joint_transforms.JointResize(in_size),
            joint_transforms.RandomHorizontallyFlip(),
            joint_transforms.RandomRotate(10)
        ])
        self.train_img_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])  # 处理的是Tensor
        ])
        # ToTensor 操作会将 PIL.Image 或形状为 H×W×D，数值范围为 [0, 255] 的 np.ndarray 转换为形状为 D×H×W，
        # 数值范围为 [0.0, 1.0] 的 torch.Tensor。
        self.train_target_transform = transforms.ToTensor()
        
        self.gt_root = '/home/lart/coding/TIFNet/datasets/DUTSTR/DUTSTR_GT.lmdb'
        self.img_root = '/home/lart/coding/TIFNet/datasets/DUTSTR/DUTSTR_IMG.lmdb'
        self.paths_gt, self.sizes_gt = _get_paths_from_lmdb(self.gt_root)
        self.paths_img, self.sizes_img = _get_paths_from_lmdb(self.img_root)
        self.gt_env = lmdb.open(self.gt_root, readonly=True, lock=False, readahead=False,
                                meminit=False)
        self.img_env = lmdb.open(self.img_root, readonly=True, lock=False, readahead=False,
                                 meminit=False)
    
    
    def __getitem__(self, index):
        gt_path = self.paths_gt[index]
        img_path = self.paths_img[index]
        
        gt_resolution = [int(s) for s in self.sizes_gt[index].split('_')]
        img_resolution = [int(s) for s in self.sizes_img[index].split('_')]
        img_gt = _read_img_lmdb(self.gt_env, gt_path, gt_resolution)
        img_img = _read_img_lmdb(self.img_env, img_path, img_resolution)
        if img_img.shape[-1] != 3:
            img_img = np.repeat(img_img, repeats=3, axis=-1)
        img_img = img_img[:, :, [2, 1, 0]]  # bgr => rgb
        img_gt = np.squeeze(img_gt, axis=2)
        gt = Image.fromarray(img_gt, mode='L')
        img = Image.fromarray(img_img, mode='RGB')
        
        img, gt = self.train_joint_transform(img, gt)
        gt = self.train_target_transform(gt)
        img = self.train_img_transform(img)
        
        if self.use_bigt:
            gt = gt.ge(0.5).float()  # 二值化
        
        img_name = self.paths_img[index]
        return img, gt, img_name
    
    def __len__(self):
        return len(self.paths_img)


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super(DataLoaderX, self).__iter__())
```

 

##  

 

- 文档：https://lmdb.readthedocs.io/en/release/
- http://shuokay.com/2018/05/14/python-lmdb/

- 关于LMDB的介绍：https://blog.csdn.net/jyl1999xxxx/article/details/53942824
- 代码示例：

- https://github.com/kophy/py4db
- https://www.jianshu.com/p/66496c8726a1

- https://blog.csdn.net/ayst123/article/details/44077903
- https://github.com/open-mmlab/mmsr

文章https://blog.csdn.net/jyl1999xxxx/article/details/53942824中介绍了使用LMDB的原因：

 

## 五、实战

```python
from pathlib import Path
import shutil
import lmdb
import numpy as np
from faiss_modules.faiss_index import FaissIndex

class DetectConfusion(object):
    def __init__(self, dataset_path=None, rebuild_cache=False, cache_path = ".similar_metrix"):
        if rebuild_cache or not Path(cache_path).exists():
            shutil.rmtree(cache_path, ignore_errors=True)
            if dataset_path is None:
                raise ValueError('创建缓存失败，数据集路径为空。')
            self.__build_cache(dataset_path, cache_path)
        self.env = lmdb.open(cache_path, create=False, lock=False, readonly=True)

    def __build_cache(self, dataset_path, cache_path):
        with lmdb.open(cache_path, map_size=int(1e10)) as env:
            with env.begin(write=True) as txn:
                score, line_index, avg = FaissIndex().detect()
                txn.put(b'score', score.tobytes())
                txn.put(b'line_index', line_index.tobytes())
                txn.put(b'avg', avg.tobytes())

    def main(self):
        with self.env.begin(write=False) as txn:
            score = np.frombuffer(txn.get(b'score'), dtype=np.uint32)
            line_index = np.frombuffer(txn.get(b'line_index'), dtype=np.uint32)
            avg = np.frombuffer(txn.get(b'avg'), dtype=np.uint32)
        print(score, line_index, avg)



if __name__ == '__main__':
   a = DetectConfusion("data/", True).main()
```

## python 有没有什么办法可以快速合并两个lmdb

* * *

**ChatGPT:**

在Python中，可以使用 `lmdb` 库来操作LMDB数据库。但是LMDB并未直接提供合并两个数据库的接口，因此需要自行实现合并的过程。

以下是一个简单的例子，展示了如何将两个LMDB数据库合并成一个新的数据库：

```python
import lmdb

def merge_lmdbs(lmdb1_path, lmdb2_path, merged_lmdb_path):
    env1 = lmdb.open(lmdb1_path, readonly=True, lock=False)
    env2 = lmdb.open(lmdb2_path, readonly=True, lock=False)
    merged_env = lmdb.open(merged_lmdb_path, map_size=env1.info()['map_size'] + env2.info()['map_size'])

    with env1.begin() as txn1:
        with env2.begin() as txn2:
            with merged_env.begin(write=True) as merged_txn:
                for key, value in txn1.cursor():
                    merged_txn.put(key, value)
                for key, value in txn2.cursor():
                    merged_txn.put(key, value)

    env1.close()
    env2.close()
    merged_env.close()

# 使用方式
merge_lmdbs('lmdb1_path', 'lmdb2_path', 'merged_lmdb_path')
```

请注意这个脚本假设两个数据库中不存在重复的键。如果有重复的键，那么需要根据你的具体需求来决定如何处理。在上述示例代码中，如果有重复键存在，后一个数据库的值会覆盖前一个数据库中的值。

另外，`map_size` 是LMDB数据库的最大大小，如果你知道你的数据需要更大的空间，你需要增大这个值。在上述代码中，我将 `map_size` 设为两个输入数据库 `map_size` 的总和，以确保有足够的空间来存储所有的数据。

最后，记得正确设置你的LMDB路径，替换 `'lmdb1_path'`, `'lmdb2_path'`, 和 `'merged_lmdb_path'` 为你的实际文件路径。

以上的示例代码简单直接，但在处理大型数据库时可能会有性能问题。如果你需要处理大型的LMDB数据库，你可能需要使用更复杂的方法来提高效率，例如使用多线程或者分批处理数据。

* * *



这样的合并过程需要对每个数据库进行遍历，可以不遍历么

* * *

**ChatGPT:**

由于LMDB数据库是基于键值对的存储，为了将两个数据库的内容合并，你必须遍历每个数据库的每个键值对。否则，你将无法访问和复制数据。

当然，如果你有其他关于数据库的信息，比如键的范围或结构，你可能可以设计一个更优化的方法。但是在通用情况下，你需要遍历整个数据库才能完成合并。

另外，如果两个数据库的大小非常大，导致遍历速度慢或者内存使用过大，你可以考虑将遍历和合并过程分批完成，每次只处理一部分数据。

如果你有对效率的要求，也可以考虑使用多线程或者并行计算的方法来加速遍历和合并过程。不过需要注意的是，由于LMDB的设计，同一时刻只能有一个写事务，所以并行写入可能并不会提高速度。

* * *

**You:**

也就是说，lmdb的合并不会太高效率

* * *

**ChatGPT:**

对的，LMDB 数据库的合并过程可能并不高效，这是因为它需要遍历两个数据库的所有键值对。虽然 LMDB 本身是非常高效的键值存储数据库，但由于其设计特性，合并数据库这一特定操作可能比较耗时。

以下是几个可能的解决方法：

1. **使用更大的事务**：LMDB 在处理大事务时可能更高效。在你的情况下，如果内存足够，你可以尝试将整个合并过程放在一个单独的事务中执行。

2. **使用更小的事务**：相反，如果你的数据库非常大，并且内存有限，你可以尝试将合并过程分解为多个小事务。这样可以减少内存使用，并可能提高效率。

3. **使用更快的硬件**：LMDB 非常依赖于存储和内存的性能。使用 SSD 和更大的内存可能会大幅提高合并的速度。

4. **避免合并**：如果可能，你可以尝试避免合并操作。例如，你可以在开始时创建一个足够大的数据库，然后直接在这个数据库中写入所有数据。

每个数据库的最佳实践可能会有所不同，所以你可能需要尝试不同的方法，看看哪种最适合你的情况。

* * *

**You:**

我可不可以，将A数据库扩容后，将B数据库的内容写入A

* * *

**ChatGPT:**

当然可以。这实际上是一个很好的方法，尤其在你想避免创建新的LMDB数据库的时候。

你需要做的是扩大第一个数据库的 `map_size`，然后把第二个数据库的数据写入第一个数据库。这里是一段示例代码：

```python
import lmdb

def extend_and_merge_lmdbs(lmdb1_path, lmdb2_path):
    env1 = lmdb.open(lmdb1_path, map_size=2*lmdb.open(lmdb1_path).info()['map_size'])
    env2 = lmdb.open(lmdb2_path, readonly=True, lock=False)

    with env2.begin() as txn2:
        with env1.begin(write=True) as txn1:
            for key, value in txn2.cursor():
                txn1.put(key, value)

    env1.close()
    env2.close()

# 使用方式
extend_and_merge_lmdbs('lmdb1_path', 'lmdb2_path')
```

请注意，这个代码片段假设你的第一个数据库有足够的空间来容纳第二个数据库的所有数据。我简单地将第一个数据库的 `map_size` 扩大了两倍，但实际上你可能需要更精确地计算所需的空间。

另外，这个代码片段还假设两个数据库中不存在重复的键。如果有重复的键，你需要决定如何处理。在这个示例代码中，如果有重复键存在，第二个数据库的值将会覆盖第一个数据库中的值。

* * *

**You:**

## lmdb扩容的时间复杂度是多少

* * *

**ChatGPT:**

LMDB扩容，即增加`map_size`（数据库的最大大小）是一个相对廉价的操作。改变`map_size`的时间复杂度接近常数时间（O(1)）。这是因为`map_size`仅仅是为LMDB环境设置的虚拟内存映射大小，增加它并不会导致所有数据都需要被复制或者移动。

在代码中设置一个新的`map_size`后，需要重新打开LMDB环境才能生效。重新打开操作的复杂度也是常数级别的。

注意，扩大`map_size`并不会立即消耗更多磁盘空间，只有当更多的数据被写入LMDB数据库时，才会真正使用更多的磁盘空间。这是因为LMDB使用的是稀疏文件，这些文件的大小会随着数据的增加而增加，直到达到`map_size`设置的最大值。

总的来说，增加LMDB的`map_size`是一个低成本操作，不会显着增加代码的执行时间。但是需要注意，确保你的系统中有足够的磁盘空间来容纳数据库的增长。

* * *

**You:**

也就是说，当我无法提前预估我的数据量和大小时，可以使用动态扩容的方式，逐渐写入数据库

* * *

**ChatGPT:**

对的，如果你无法预先知道你的数据量或大小，可以选择动态调整 `map_size` 以适应数据的增长。

在 LMDB 中，`map_size` 是一个虚拟内存映射大小，这个值并不会立即消耗你的磁盘空间。实际的磁盘使用量取决于你向 LMDB 写入的数据量。

当你需要写入的数据超过了当前 `map_size` 的时候，你可以关闭数据库，增大 `map_size`，然后重新打开数据库，再进行写入操作。

需要注意的是，频繁地更改 `map_size` 可能会带来一些性能问题，因为每次修改都需要重新打开数据库。如果可能的话，预先设定一个足够大的 `map_size` 是比较好的做法。如果你在初始化数据库的时候没有足够的信息来决定一个合适的 `map_size`，那么动态扩容就是一个合理的解决方案。

这里是一个例子，展示了如何在 LMDB `map_size` 不足时动态扩容：

```python
import lmdb

def dynamic_write(env_path, key, value, increment=2):
    while True:
        try:
            env = lmdb.open(env_path, map_size=lmdb.open(env_path).info()['map_size'])
            with env.begin(write=True) as txn:
                txn.put(key, value)
            break
        except lmdb.MapFullError:
            # 重新打开数据库并扩大 map_size
            env = lmdb.open(env_path, map_size=lmdb.open(env_path).info()['map_size']*increment)
        finally:
            env.close()
```

在这个函数中，`increment` 是每次增加 `map_size` 的倍数。如果 `lmdb.MapFullError` 异常被抛出，表示当前的 `map_size` 已经无法满足数据的写入，此时函数会捕获这个异常，然后将 `map_size` 扩大为原来的 `increment` 倍，然后再尝试写入。

这样，你就可以在写入数据的过程中动态地调整 `map_size` 了。

* * *

**You:**

既然map_size 是一个虚拟内存映射大小，这个值并不会立即消耗你的磁盘空间，那我是不是可以一开始设置这个值为无限大呢

* * *

**ChatGPT:**

理论上来说，你可以将 `map_size` 设置为一个非常大的值，比如硬盘的最大容量。但实际上，由于几个因素，这种做法并不推荐。

1. **系统限制**：虽然 `map_size` 并不直接消耗磁盘空间，但它是对系统的内存映射区域大小的预留。如果你设置一个非常大的 `map_size`，超过了系统允许的最大内存映射区域，可能会导致 LMDB 环境无法打开。

2. **可能的性能影响**：虽然目前我没有找到明确的证据表明设置过大的 `map_size` 会影响 LMDB 的性能，但这样做可能会影响其他基于内存映射的操作的性能，因为它们可能需要竞争有限的内存映射资源。

3. **稀疏文件限制**：虽然 LMDB 使用稀疏文件，这意味着文件的大小可以超过实际使用的磁盘空间，但并非所有的文件系统都支持或者良好地支持稀疏文件。在一些文件系统中，一个非常大的稀疏文件可能会导致问题。

4. **难以管理和预测**：如果你设置了一个非常大的 `map_size`，那么你可能很难预测和管理你的数据和磁盘使用情况。你可能会遇到突然的磁盘空间耗尽的问题。

所以，虽然你可以将 `map_size` 设置得非常大，但最好还是根据你的实际需要和你的系统的具体情况来确定一个合适的 `map_size`。如果你无法预先知道需要多大的空间，你可以选择逐渐增大 `map_size`，如我在之前的回答中所述。

* * *

**You:**

可是，我实际看到的是，我在打开一个数据库的时候，在写入数据之前，就会生成一个map_size大小的lmdb文件

* * *

**ChatGPT:**

对的，当你创建一个新的LMDB数据库并设置了`map_size`，系统会立即在磁盘上创建一个等于`map_size`大小的文件。这是因为LMDB使用了内存映射文件(memory-mapped files)的技术。虽然这个文件的大小等于`map_size`，但是实际上，它在物理磁盘上并不占用这么多空间，这是因为操作系统并没有为这个文件分配实际的磁盘空间。这就是所谓的"稀疏文件"（sparse file）。

然而，对于一些文件系统或者一些工具（如`du`），可能会误报这个文件的大小，显示它占用的空间等于`map_size`，尽管实际上并不是这样。你可以使用`ls -lh`命令来查看文件的"逻辑大小"，并使用`du -h`命令来查看文件实际占用的磁盘空间。你会发现，虽然`ls -lh`显示文件大小为`map_size`，但是`du -h`显示的实际占用空间要小得多。

需要注意的是，并非所有的文件系统都支持稀疏文件。在不支持稀疏文件的文件系统中，LMDB文件的大小可能真的会立即变为`map_size`。所以在这些文件系统上，你需要更小心地设置`map_size`，确保不会超过实际可用的磁盘空间。

在确定`map_size`时，一种好的策略是预估你的数据将占用多少空间，然后根据这个预估值设置`map_size`。如果你无法预估，或者数据的大小可能会有很大波动，那么可以选择动态扩容，如我之前所述。

* * *

**You:**

可是，当我在移动map_size为10000M的lmdb文件时，如果该lmdb实际只写入一个1M大小的数据，移动快不快

* * *

**ChatGPT:**

在进行文件移动操作时，实际上是文件系统如何处理该操作决定了操作的速度。

对于稀疏文件，如果你在同一文件系统内移动文件，通常操作系统只需要修改文件系统的索引信息，而不需要实际移动文件内容，因此这个操作的速度很快，几乎是立即的，与文件的“逻辑大小”（在这个例子中是 10000M）无关。在这种情况下，文件系统不需要理会文件的稀疏性，所以这个操作应该与实际写入的数据大小（在这个例子中是 1M）无关。

然而，如果你要将文件从一个文件系统移动到另一个文件系统，那么操作系统就需要复制文件的所有内容，包括那些稀疏的部分。在这种情况下，操作的速度将取决于实际写入的数据量，而不是文件的“逻辑大小”。因此，如果你的 LMDB 数据库实际只写入了 1M 的数据，那么这个操作应该相当快，尽管文件的“逻辑大小”为 10000M。

需要注意的是，这些行为可能会根据你的操作系统和文件系统的具体实现有所不同。有些工具或命令可能不完全理解稀疏文件的概念，可能会错误地处理这些文件，导致不必要的慢速操作。因此，如果你在实践中遇到任何问题，我建议你查阅你的操作系统和文件系统的文档，或者寻求专门的技术支持。

* * *

**You:**

如何并发的写入lmdb呢

* * *

**ChatGPT:**

LMDB 是一个事务性的数据库，它支持多个读者和一个写者（多读单写）的并发模型。因此，你不能在同一时间内有多个进程或线程写入同一个 LMDB 数据库。这是由于 LMDB 的底层设计，它用来保证数据的一致性和完整性。

然而，你可以使用多个进程或线程并发地读取同一个 LMDB 数据库。这是由于 LMDB 的多版本并发控制（MVCC）特性，可以让每个读事务有一个对数据库的一致性视图，而不需要锁定整个数据库。

如果你想要并发地写入数据，你可能需要调整你的设计。你可以将你的数据分成多个部分，然后在不同的 LMDB 数据库中并发地写入。然后，你可以在需要时将这些数据库合并在一起，或者你可以修改你的应用，让它可以同时从多个数据库读取数据。

或者，你也可以使用一个队列，让多个进程或线程将他们需要写入的数据放入队列，然后由一个单独的写入者进程或线程从队列中取出数据并写入 LMDB 数据库。

虽然这些方案可能会增加你的设计和实现的复杂性，但它们可以让你充分利用多核或多处理器的能力，从而提高你的数据写入速度。



* * *

**You:**

## lmdb 每次扩容都要重新打开 环境和事务么

* * *

**ChatGPT:**

是的，每次修改LMDB的`map_size`时，你都需要关闭并重新打开数据库。这是因为`map_size`参数决定了数据库的内存映射的大小，而内存映射是在打开数据库时创建的。一旦创建了内存映射，就不能修改其大小，所以如果你想要改变`map_size`，就必须重新打开数据库。

此外，所有未完成的事务都需要在关闭数据库之前完成（提交或放弃）。如果在关闭数据库时还有未完成的事务，你可能会收到一个错误，或者这些未完成的事务可能会丢失。

这是一个例子，展示了如何在`map_size`不足时增大`map_size`：

```python
import lmdb

def dynamic_write(env_path, key, value, increment=2):
    while True:
        try:
            env = lmdb.open(env_path, map_size=lmdb.open(env_path).info()['map_size'])
            with env.begin(write=True) as txn:
                txn.put(key, value)
            break
        except lmdb.MapFullError:
            # 重新打开数据库并扩大 map_size
            env = lmdb.open(env_path, map_size=lmdb.open(env_path).info()['map_size']*increment)
        finally:
            env.close()
```

在这个函数中，每次`lmdb.MapFullError`异常被抛出时，都会关闭数据库，增大`map_size`，然后重新打开数据库。

* * *

**You:**

事务是基于环境的，对么

* * *

**ChatGPT:**

是的，你的理解是正确的。在LMDB中，所有的读写操作都在事务（transaction）的上下文中执行。这些事务都是由一个LMDB环境（environment）对象创建的。

在LMDB中，环境可以被视为是数据库的一个容器。环境维护了数据库的状态，包括元数据、内存映射、锁和其他资源。同时，环境也提供了方法来管理事务。你可以使用环境的`begin`方法来开始一个新的事务，无论是读事务还是写事务。

事务是在环境的上下文中执行的。每个事务都有一个对环境（包括它的数据库）的一致性视图。在一个写事务中，你可以对数据库进行修改。这些修改只在该事务的上下文中可见，直到该事务提交。一旦事务提交，这些修改就对所有新的事务可见。在一个读事务中，你可以查询数据库，但不能进行修改。读事务总是看到它开始时数据库的状态，无论此后是否有其他写事务进行了修改。

请注意，虽然一个环境可以同时有多个读事务，但是在任何时候都只能有一个写事务。这是因为LMDB使用了写者优先（writer-preferred）的锁策略，以防止写者饥饿（writer starvation）。当一个写事务开始时，后续的读事务会被阻塞，直到写事务完成。但是已经开始的读事务不会被阻塞。

* * *

**You:**

我有一个lmdb文件，内容没有写满，如果，我用更小的尺寸打开这个lmdb，会不会导致数据丢失

* * *

**ChatGPT:**

如果你试图用一个小于实际数据库大小的`map_size`来打开LMDB数据库，LMDB将无法打开这个数据库，并可能抛出错误。这是因为`map_size`定义了LMDB可以使用的地址空间的大小，如果这个空间小于数据库的实际大小，那么LMDB将无法正常访问数据库的全部内容。

然而，即使你试图使用一个小的`map_size`打开数据库，也不会导致数据丢失。数据库文件本身并没有被修改，所以所有的数据都还在那里。你只需要再次用一个足够大的`map_size`来打开数据库就可以了。

需要注意的是，如果你的数据已经接近或超过了`map_size`，你可能会在试图写入新数据时遇到`lmdb.MapFullError`。在这种情况下，你需要增加`map_size`以便能够继续写入新的数据。

最后，你应该始终确保你的`map_size`大于或等于你的数据库可能会需要的最大空间。如果你不能提前知道这个值，那么你应该定期检查你的数据库的大小，并在需要时增大`map_size`。

* * *

**You:**

可事实上，我用小的map_size打开并没有报错，而是将lmdb文件缩小为数据实际占用尺寸。

* * *

**ChatGPT:**

我必须对此表示道歉。你的观察是正确的，我的先前回答可能导致了误解。

实际上，当你用一个小的 `map_size` 打开 LMDB 数据库时，LMDB 不会抛出错误。在 LMDB 中，`map_size` 参数定义了内存映射区域的大小，但并不限制实际数据库文件的大小。所以，即使实际的数据库文件大小超过了 `map_size`，你仍然可以打开这个数据库。

然而，你需要注意的是，如果你试图写入新的数据，并且这些数据会使数据库大小超过 `map_size`，你将会收到一个 `lmdb.MapFullError`。在这种情况下，你需要增大 `map_size` 以便能够继续写入新的数据。

如果你使用一个小的 `map_size` 打开了一个大的数据库，并进行了一些写操作（包括删除或更新数据），那么你可能会看到数据库文件的大小被减小。这是因为 LMDB 在执行写操作时可能会调整数据库文件的大小，以反映数据的实际占用空间。这种情况下，数据库文件的大小可能会接近于（但不一定等于）数据的实际占用空间。

我希望这个回答能解决你的问题，如果有其他问题，欢迎继续提问。

* * *

**You:**

## lmdb 如何存储numpy数组呢

* * *

**ChatGPT:**

要在LMDB数据库中存储Numpy数组，你首先需要将数组转换为可以被序列化的格式。通常我们使用Python的`pickle`库或Numpy的`tostring`或`tobytes`方法将数组转换为字节流，然后存储这个字节流。然后，在从数据库读取数据时，你可以使用相应的反序列化方法将字节流转换回Numpy数组。

这是一个例子，展示了如何将Numpy数组存储到LMDB数据库中：

```python
import lmdb
import numpy as np

# 创建一个Numpy数组
arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

# 使用tobytes将数组转换为字节流
arr_bytes = arr.tobytes()

env = lmdb.open('/path/to/mylmdb', map_size=10000000)
with env.begin(write=True) as txn:
    txn.put(b'mykey', arr_bytes)

# 从数据库读取数据并将其转换回Numpy数组
with env.begin() as txn:
    arr_bytes = txn.get(b'mykey')
    arr2 = np.frombuffer(arr_bytes, dtype=np.float32).reshape(arr.shape)

print(np.array_equal(arr, arr2))  # prints: True
```

这个例子首先创建了一个Numpy数组`arr`，然后使用`arr.tobytes()`将数组转换为字节流。然后，它打开一个LMDB环境并开始一个写事务，将字节流存储到数据库中。然后，它开始一个读事务，从数据库中读取字节流，并使用`np.frombuffer`将字节流转换回Numpy数组。最后，它检查原始数组和从数据库中读取的数组是否相等。

* * *

**You:**

## reshape(arr.shape) 这个操作是必须的么

* * *

**ChatGPT:**

是的，`reshape(arr.shape)`这个操作通常是必要的。因为当你将一个numpy数组转换为字节流时，其形状（shape）信息是不会被保存的。因此，当你从字节流重新构造数组时，你需要手动指定其形状。

假设我们有一个二维数组，例如上面例子中的 `arr`。当我们使用 `tobytes()` 方法将其转换为字节流时，这个数组会被“展平”为一维数组。因此，当我们使用 `np.frombuffer(arr_bytes, dtype=np.float32)` 将字节流转回为数组时，我们得到的会是一个一维数组。

所以，我们需要使用 `reshape` 方法将这个一维数组恢复为原来的形状。这就是为什么我们需要 `reshape(arr.shape)` 这个操作的原因。

这是一个例子，说明了为什么需要 `reshape`：

```python
import numpy as np

# 创建一个二维数组
arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

# 转换为字节流
arr_bytes = arr.tobytes()

# 从字节流恢复数组，但不使用reshape
arr2 = np.frombuffer(arr_bytes, dtype=np.float32)

print(arr2)  # prints: array([1., 2., 3., 4., 5., 6.], dtype=float32)

# 从字节流恢复数组，并使用reshape
arr3 = np.frombuffer(arr_bytes, dtype=np.float32).reshape(arr.shape)

print(arr3)  # prints: array([[1., 2., 3.],
             #                [4., 5., 6.]], dtype=float32)
```

你可以看到，如果我们不使用 `reshape`，我们得到的是一个一维数组。只有使用了 `reshape`，我们才能得到原来的二维数组。

* * *

**You:**

## 什么是可以被序列化的格式，numpy数组不可以被序列化么

* * *

**ChatGPT:**

序列化是指将复杂的数据结构（如列表、字典、对象、numpy数组等）转换为字节流（一串字节）的过程，这个字节流可以存储到文件或数据库中，或者通过网络发送到远程位置。反序列化则是将这个字节流恢复为原始的数据结构。

Python的pickle库提供了序列化和反序列化的功能。使用pickle，你可以将几乎任何Python对象序列化为字节流，包括numpy数组。

然而，numpy数组也提供了自己的序列化和反序列化方法，如 `tostring`、`tobytes` 和 `frombuffer`。这些方法通常比pickle更高效，因为它们直接操作数组的内存表示，而不需要创建一个新的Python对象。而且，使用numpy的方法序列化的数组可以直接在C或其他语言中使用，不需要经过Python的解析。

但是，numpy的方法只能序列化数组的数据部分，不能序列化数组的元数据，如形状和数据类型。所以，当你反序列化数组时，你需要手动指定这些元数据。而pickle则会自动保存和恢复这些元数据。

因此，如果你需要将数组存储到数据库或文件中，或者通过网络发送，并且你知道数组的形状和数据类型，那么你可以使用numpy的方法。如果你不知道数组的形状和数据类型，或者你需要将数组作为Python对象使用，那么你应该使用pickle。

* * *

**You:**

## lmdb使用pickle存储数组

* * *

**ChatGPT:**

当你选择使用pickle将numpy数组序列化存储到LMDB中时，可以参考以下代码：

```python
import lmdb
import numpy as np
import pickle

# 创建一个Numpy数组
arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

# 使用pickle将数组序列化为字节流
arr_bytes = pickle.dumps(arr)

env = lmdb.open('/path/to/mylmdb', map_size=10000000)
with env.begin(write=True) as txn:
    txn.put(b'mykey', arr_bytes)

# 从数据库读取数据并使用pickle将其反序列化回Numpy数组
with env.begin() as txn:
    arr_bytes = txn.get(b'mykey')
    arr2 = pickle.loads(arr_bytes)

print(np.array_equal(arr, arr2))  # prints: True
```

在这个例子中，我首先创建了一个Numpy数组`arr`，然后使用pickle的`dumps`函数将数组序列化为字节流。然后，我打开一个LMDB环境并开始一个写事务，将字节流存储到数据库中。然后，我开始一个读事务，从数据库中读取字节流，并使用pickle的`loads`函数将字节流反序列化回Numpy数组。最后，我检查原始数组和从数据库中读取的数组是否相等。

使用pickle的好处是，它会自动保存数组的形状和数据类型，所以你不需要手动指定这些信息。这使得存储和读取数据变得更加简单。然而，pickle的速度可能比numpy的`tostring`或`tobytes`方法慢，因为它需要创建和解析Python对象。

* * *

**You:**

## 通俗解释什么是字节流

* * *

**ChatGPT:**

字节流（Byte Stream）是一种数据处理方式，指的是数据（无论是文本、图片、音频、视频等）被转换为一系列的字节，然后按顺序传输或存储。

在计算机中，字节（Byte）是最基本的数据单位。1个字节等于8个比特（Bit），一个比特可以表示0或1，而一个字节可以表示256种不同的值（即0-255）。

我们可以将一份数据比喻成一本书，那么字节流就像是我们将这本书一页一页（一个字节一个字节）的阅读或者复制。这一系列的字节可以依次写入磁盘文件、发送到网络，或者存储到内存中。在接收方，可以依次读取这些字节，然后重建出原始的数据。

例如，在Python中，我们可以使用`open`函数以二进制模式（`'b'`）打开文件，然后使用`read`或`write`方法读取或写入字节流。这就是字节流的一个应用。

注意，因为字节流通常不包含数据的结构信息（比如列表、字典或者Numpy数组的形状、数据类型等），所以我们在写入或读取字节流时，通常需要有一种方法（比如序列化/反序列化）来保存或恢复这些信息。

* * *

**You:**

为什么python对象在写入lmdb时，要先进行序列化，我不能直接存储python对象么

* * *

**ChatGPT:**

LMDB，类似于其他键值存储数据库，其底层使用的是字节流来存储数据。这就需要我们将要存储的数据（无论是文本、数字、Python对象等）转换（序列化）成字节流。

Python对象是复杂的数据结构，它们在内存中的表示形式通常不是连续的，且包含许多Python特有的元数据。直接将这些对象的内存表示形式存储到数据库中通常是不可行的，因为：

1. Python对象的内存表示形式可能在不同版本或实现的Python中不同，甚至在同一版本的Python中也可能因操作系统或硬件的不同而不同。因此，直接存储Python对象的内存表示形式可能导致跨平台和跨版本的不兼容问题。

2. Python对象可能包含指向其他对象的引用，而这些引用在内存中的表示形式通常是依赖于特定Python进程的内存地址。如果我们直接存储Python对象的内存表示形式，那么在读取数据时，这些引用可能已经无效，导致错误或未定义的行为。

3. 直接存储Python对象的内存表示形式可能导致安全问题。因为Python对象可能包含敏感信息（如密码或密钥），而这些信息在内存中可能是以明文形式存在的。如果我们直接存储Python对象的内存表示形式，那么这些敏感信息可能会被泄露。

因此，我们通常使用序列化方法（如pickle）将Python对象转换为字节流，然后存储这个字节流。序列化方法会保留Python对象的数据和结构，而不保留其内存表示形式。这样，我们可以在任何平台和Python版本上读取并恢复这个对象，而不需要知道它在原始Python进程中的内存表示形式。

* * *

**You:**

tostring  与 tobytes有什么不一样

* * *

**ChatGPT:**

`numpy.ndarray.tostring()` 和 `numpy.ndarray.tobytes()` 两个方法在功能上是一致的。它们都是将数组中的数据转换成一个连续的字节流。

实际上，根据numpy的官方文档，`tostring()` 方法已经被废弃，建议使用 `tobytes()` 方法。

这是因为 `tostring()` 方法的名称可能会让人误解，让人以为这个方法会将数组转换为字符串。但实际上，这个方法是将数组转换为字节流，无论数组的数据类型是什么。而且，这个字节流是不能直接被解析为人类可读的文本的。因此，为了避免误解，numpy推荐使用 `tobytes()` 方法。

所以，如果你在代码中看到 `tostring()` 方法，你可以直接将其替换为 `tobytes()` 方法，因为这两个方法在功能上是一致的。

* * *

**You:**

b'string' 与 'string'.encode() 功能一样么

* * *

**ChatGPT:**

是的，`b'string'`和`'string'.encode()`在Python中都是用于创建字节对象的方式，它们的结果是等价的。

在Python中，字符串是由字符组成的，这些字符可以是ASCII字符，也可以是Unicode字符。而字节对象是由字节组成的，每个字节都是一个0-255的整数。

当你在一个字符串前面加上`b`，你会得到一个字节对象。这个字节对象中的每个字节都对应于原始字符串中的一个字符，但只有在该字符是ASCII字符时这个对应关系才成立。如果原始字符串中有非ASCII字符，那么这个操作会导致错误。

`str.encode()`方法也是将一个字符串转换为一个字节对象。默认情况下，这个方法会假设字符串是用UTF-8编码的，并将每个字符转换为其UTF-8编码的字节。如果字符串中有非ASCII字符，这个方法会正常工作，因为UTF-8可以表示所有的Unicode字符。

例如：

```python
s = 'Hello, world!'
b = b'Hello, world!'
print(s.encode() == b)  # prints: True
```

在这个例子中，`s.encode()`和`b`产生了相同的字节对象。

所以，如果你确定你的字符串只包含ASCII字符，那么你可以使用`b'string'`和`'string'.encode()`互换。如果你的字符串可能包含非ASCII字符，那么你应该使用`str.encode()`方法，以避免错误。