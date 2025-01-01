---
title: tranformer全流程零基础解析
date: 2024-07-21 10:57:29
tags: 深度学习
top_img: https://blogpicture-1310464487.cos.ap-nanjing.myqcloud.com/ typora/头图_7.png
cover: https://blogpicture-1310464487.cos.ap-nanjing.myqcloud.com/ typora/1a510a031ef340581f2b169fbf51597f490518248.jpg
---

## Transformer全流程零基础解析

> [!NOTE]
>
> 诈骗预警：虽然说是零基础，但是还是要有一般的深度学习基础，例如: 基本的深度网络概念、pytorch 基本使用、对权重的基本认识等。

本文将详细拆解Transformer的各个基本组件，并且配合基本的代码讲解，另外，本文的零基础为NLP领域零基础用户，需要有一定的深度学习认知，因此本文更偏向专业性，不会偏向通俗科普，但是读完本文可以秒杀市面上所有的Transformer介绍，下面开始正文。

原文链接：[[1706.03762\] Attention Is All You Need (arxiv.org)](https://arxiv.org/abs/1706.03762)

### 前言

本文的介绍思路将与市面上的大部分顺序不同，虽然Transformer提出的原文叫做**Attention Is All You Need**，但是本文并不会优先介绍 attention机制（毕竟还有一篇论文叫做**Attention Is Not All You Need Anymore**），*所以在详细介绍attention之前就请把他当作一个线性层罢*。

### Transformer总览

接下来这张图我们将会反复出现，这张图就是Transformer 的总体架构，其主要构成为Encoder-Decoder框架以及外围的结果输出部分和输入嵌入部分，见下图：

<img src="https://blogpicture-1310464487.cos.ap-nanjing.myqcloud.com/%20typora/image-20240725223605806.png" alt="image-20240725223605806" style="zoom: 67%;" />

嗯，看到这儿肯定是觉得一脸懵逼的，然后内心一阵暗骂：这都啥玩意？别急，下面我们将按照程序运行的基本顺序来逐个模块进行讲解，基本顺序为输入与嵌入、E-D架构（后面均简称为E-D架构）、输出，按照该顺序我们简单获得一个整体运行的认识。

### 输入与嵌入

相比写的已经烂大街的各种注意力机制的文章，大家最迷糊的一般都是从输入开始的，我们会疑惑的点大概有以下：

* 字符串怎么输入？输入后怎么计算？
* 嵌入有什么用？又该怎么嵌入？
* 位置编码如何发挥作用？又有什么用？

下面就一个一个的进行解决：

#### Token

我们首先来解决第一个问题，这要牵扯到最最最古老的NLP思想——Token。

> Token 是指文本中的一个基本单元，通常是词或短语。这个切分token的过程，成为分词（Tokenization）。 

我们先简单举个例子：

`"Never give up”`，考虑对这个句子进行切分，我们很容易想到可以将其切分为`-Never-give-up`，这是一种基于空格的切分方式，其中Never、give、up都是一个token。

有些叛逆的同学肯定觉得，我偏不，不就是切分句子吗，我要切分成

`-Nevergive-up`。甚至还有一些更叛逆的同学想：你说通常token是词或短语是不是也有不太通常的情况，我就要这样切`-N-e-v-e-r-g-i-v-e-u-p`。

虽然上面两个同学他都有一定的歪理，但是别忘了我们要解决的问题是什么。我们希望计算机能够理解自然语言，然而计算机并不能对字符串进行什么运算处理，因此我们应该力求通过分割句子的方式来让计算机理解句子的意义。

于是，词表（Vocabulary）出现了，词表是一个由token与数字组成的一个dict（hashmap）。计算机固然没办法理解一个字符形式的token，但理解一个与token对应的数字还是可以做到的。这让我想起一句话：语言不是什么玄而又玄的，而是十分符合统计学的。

然后我们来仔细聊聊分词罢。

#### Tokenization

我们刚才说两个同学都有一定歪理，是的，他俩甚至犟嘴的都有道理，甚至都有一定程度的应用。Tokenization 在 Transformer 中所处的位置如下：

![image-20240728093346164](https://blogpicture-1310464487.cos.ap-nanjing.myqcloud.com/%20typora/image-20240728093346164.png)

目前，分词任务有三个粒度的分词，分别是：

* 词粒度
* 字粒度
* Subword 粒度

再介绍这些分词的情况与手段前，我们先回顾并提出一点问题：

* 我们要解决的问题是有意义的分出token
* 需要兼顾词表的大小与推理速度
* 能不能保证模型能够认识从来没见过的词，即，OOV（Out of Vocabulary）问题

##### 词粒度

 词粒度基本是最直观的分词手段了，也是最符合我们平时认知的方式。英语的话，由于词与词之间存在空格，就非常方便的可以完成分词，中文的话相对比较麻烦。

优点：

* 非常的人类、能很好的保持语义信息

缺点

* 会有一张超级大的词表
* 难以避免OOV问题
  * 或许存在一些解决方法：使用未知标记（UNK）
* 一些同前缀、同后缀的单词难以获得相关性

##### 字粒度

字粒度比较狠，就直接分成一个个的字母，这种对于中文而言就比较亲民了，英文划分后几乎是没什么含义的

优点：

* 词表规模大大减小
* 很少出现未知词汇，基本都可以组合

缺点：

* 没有太多语义信息
* 句子切分得到的token 数量大大增加，提高运算量。

##### Subword（子词）粒度

当我们看到上面两种方式各有优劣，那我们能不能折中一下（折中！）拥有两个优点？

举个例子就可以简单理解了：

> 例句：he is likely to be unfriendly to me
>
> 分词：‘he' 'is' 'like' 'ly' 'to' 'be' 'un' 'friend' 'ly' 'to' 'me'

现在优点就很明显了：

* 词表尽可能小，因为可以用尽量少的子词来组成词汇
* 有更好的泛化能力，可以学习到词汇之间的变化与关系

划分子词的方法有常见的几种：

* Byte Pair Encoding（BPE）
* WordPiece
* SentencePiece

我们简单聊聊BPE方法罢。

###### BPE

BPE的构造流程和哈夫曼树非常像，其算法流程如下：

1. 规定subword词表的大小
1. 在每个单词后加上\</w>，此举的目的在于区分一些前缀和后缀 
1. 将语料库中的所有单词划分为单个字符，用所有的单个字符建立最初的词典，并统计每个字符的概率
1. 挑出频次最高的字符对，然后重复2，3直到到达规定的词表的大小

然后简单举个例子：

> a tidy tiger tied a tie tighter to tidy her tiny tail

1.给单词后边加上\</w>，并统计每个词出现的频率

> ​    'a </w>': 2,
> ​    't i d y </w>': 2,
> ​    't i g e r </w>': 1,
> ​    't i e d </w>': 1,
> ​    't i e </w>': 1,
> ​    't i g h t e r </w>': 1,
> ​    't o </w>': 1,
> ​    'h e r </w>': 1,
> ​    't i n y </w>': 1,
> ​    't a i l </w>': 1,

2. 拆成单个字符，并统计频率，构成初始的子词词典

>  '</w>': 12,
>     'a': 3,
>     't': 10,
>     'i': 8,
>     'd': 3,
>     'y': 3,
>     'g': 2,
>     'e': 5,
>     'r': 3,
>     'h': 2,
>     'o': 1,
>     'n': 1,
>     'l': 1,

3. 统计语料中相邻子词对的出现频率，选取频率最高的子词对合并成新的子词加入词表，并更新词典

> '</w>': 12,
>     'a': 3,
>     't': 3, # [修改]
>     'i': 1, # [修改]
>     'd': 3,
>     'y': 3,
>     'g': 2,
>     'e': 5,
>     'r': 3,
>     'h': 2,
>     'o': 1,
>     'n': 1,
>     'l': 1,
>     'ti': 7, # [增加]

以此类推，BPE 实现代码在[4]中有，感兴趣可以看一下

##### Tokenization 总结

盗张图先（），这是 RNN 接受token并处理的过程。

<img src="https://blogpicture-1310464487.cos.ap-nanjing.myqcloud.com/%20typora/rnn.gif" alt="rnn" style="zoom: 67%;" />

从上图我们可以看出来，Tokenization 的本质其实就是一个字符到数字的映射，**其维护的是一个字典，而不是权重**，也就是说每一个字符or词or短语都有一个唯一确定的数字与其对应，是不是有点熟悉，没错，one-hot就是这样的，但是明显one-hot太笨了，有没有更强一点的算法呢？

#### Embedding

首先，我们来看Input Embedding 和 Output Embedding，在这之前我们有必要了解一下什么是embedding以及为什么要embedding。

<img src="https://blogpicture-1310464487.cos.ap-nanjing.myqcloud.com/%20typora/image-20240728093023512.png" alt="image-20240728093023512" />

##### 为什么不直接用token？

当我们有一些实践经验后，我们可以清晰地看到Tokenization后的结果：

<img src="https://blogpicture-1310464487.cos.ap-nanjing.myqcloud.com/%20typora/image-20240728110916819.png" alt="image-20240728110916819" />

而其真正的输入方式正是用one-hot编码的形式，通过矩阵输入，也就是说token对应的正是one-hot编码中的`1`的index

所以现在就很清晰了，one-hot编码最大的问题就是：他是一个正交矩阵，这就意味着词与词之间不存在任何的相关性，因此有必要开发一种新方法。

##### word2vec

又是一个名词，word2vec，全称应该叫：word to vector ，这就很好理解了，是把词翻译为向量的模型。word2vec 包含两个模型：skip-gram 和 CBOW 。

<img src="https://blogpicture-1310464487.cos.ap-nanjing.myqcloud.com/%20typora/1_bBETsVNLyjnaFJgM9avkeQ.webp" />

###### skip-gram

skip-gram（跳元模型），其核心要义就是在给定中心词的情况下，来生成上下文词的条件概率。例如，给定一句话

> The man loves his son. 

设定中心词为`loves`，skip-gram模型考虑上下文生成词的条件概率：
$$
P('the','man','his','son'|'loves')=P('the'|'loves')P('man'|'loves')P(''his'|'loves')P('son'|'loves')
$$
了解这么多也差不多了（由于公式很怪），我们来简单粗暴的总结一下skip-gram的任务，skip-gram要在给定一个词的情况下，找出上下文概率最大的词，也就是一个多分类任务，那么下面该怎么做，我想大家应该心里都有数吧（）

决定还是继续写一下，简单写一下公式，小结的时候再理解一次本质。那么，当给定一个中心词$\omega_c$去预测上下文中的一个词$\omega_o$的概率那么就可以表示为
$$
P(\omega_o|\omega_c)=\frac{exp(u_o^Tv_c)}{\sum_{i\in V}exp(u_i^Tv_c)}
$$
其中，$u$ 和 $v$ 均为一个向量，分别表示上下文和中心词。

###### CBOW

CBOW（连续词袋模型），其实就是正好和skip-gram是反的，skip-gram给定中心词预测多个上下文，而CBOW是给定上下文进行选词填空，实际上，连续词袋模型依旧是一个分类任务，其模型大体上也和skip-gram相似。

在这里不再赘述公式（因为我觉得看了也看不太懂）

###### word2vec 小结

基本了解原理后，我们来从头来理解一下，word2vec的输入实际上是one-hot编码，而其架构实际上就是一个很简单的MLP，那么这个过程我们就可以很简单的表示为：
$$
u^{'}=Wu
$$
![使用独热向量进行矩阵乘法的影响](http://mccormickml.com/assets/word2vec/matrix_mult_w_one_hot.png)

所以，实际上我们可以将one-hot 编码理解为一种在MLP权重中查询知识的过程，而输出结果就是一个嵌入表示向量，而这个输出结果实际上是**可逆的**（就是一个矩阵方程而已）。

然而在实现的时候，比如skip-gram因为要查询整个词表，会有极大的运算负担，因此提出了负采样的优化方案，这里就要不继续介绍了。

#### 位置编码

位置编码是Transformer中很重要的一部分，再介绍位置编码之前，我们要先区分一下encoding和embedding。

embedding我们刚在前面介绍过，embedding是将token转换为一种稠密向量（与one-hot的稀疏向量对应）的手段，encoding（编码）同样也是一种生成词向量的方式，二者不同点主要有二：

* embedding主要指通过神经网络生成词向量的黑盒模式，而encoding是通过公式可以直观理解的白盒生成（如one-hot）
* embedding更侧重于嵌入的词向量结果，而encoding更侧重于编码的过程而不是词向量结果

接下来，我们要回答几个问题：

* 为什么要有位置编码？
* 位置编码原理如何？有何缺点？该作何改进？
* 位置编码是怎么作用于任务的？

###### 为什么要有位置编码

位置编码，顾名思义，就是通过一套规则对序列中元素的位置进行唯一的编码。那么为什么要有位置编码呢？

在旧时代的NLP中，我们一般使用CNN/RNN来建模文本，其中CNN可以编码一定的绝对位置信息（很大程度上来自zero-padding），而RNN的序列依赖特性更是天生适合序列问题或者位置信息的建模。因此，在旧时代的NLP，基本无须单独做位置编码。

Transformer和以前的应用于序列学习的框架并不同，其在计算时并不会参考位置，即使位置调换也完全没有关系（留个伏笔，等写到attention再回收）,因此需要添加位置编码作为位置信息。

###### 位置编码分类与介绍

位置编码一般被分为两种：**绝对位置编码**和**相对位置编码**。

*绝对位置编码*，也就是每个元素大家一人一个数字分别对应自己的index，还有一些模型采用了可学习的位置编码，例如bert

最常见的绝对位置编码也就是我们常用的数组index了，这具有一个很显著的缺点，按照这种编码方式，那么越是后面的元素其位置编码权重越大，因为越是后面的元素其位置编码值越大。（就是这么简单粗暴）

*相对位置编码*，即考虑词与词之间的相对位置，**而不是单纯的考虑其所在的位置。**

###### 正余弦位置编码

Transformer 中使用的经典正余弦位置编码属于绝对位置编码，其公式如下：
$$
PE(pos,2i)=sin(\frac{pos}{10000^{\frac{2i}{d_{model}}}}) \\
PE(pos,2i+1)=cos(\frac{pos}{10000^{\frac{2i}{d_{model}}}})
$$
其中$pos$表示其所在的位置，$2i$ 和 $2i+1$ 表示位置编码内的维度索引，而$d_{model}$表示词向量的维度。这看起来有点抽象，那让我们变形一下，让他变得直观一点：
$$
PE_{pos}=\begin{bmatrix}sin(\omega_1\cdot pos)\\cos(\omega_1\cdot pos)\\sin(\omega_2\cdot pos)\\cos(\omega_2\cdot pos)\\\vdots\\sin(\omega_{d/2}\cdot pos)\\cos(\omega_{d/2}\cdot pos)\end{bmatrix}
$$
其中，$\omega_i=\frac{1}{10000^{\frac{2i}{d_{model}}}}$，现在看着是不是舒服多了，那么这个编码到底是怎么来表示出位置关系的呢？

首先我们先回忆一下二进制表示十进制：

| 十进制 | 二进制 |
| ---- | ---- |
| 0    | 0000 |
| 1    | 0001 |
| 2    | 0010 |
| 3    | 0011 |
| 4   | 0100 |
| 5  | 0101 |
| 6 | 0110 |
| 7   | 0111 |
| 8   | 1000 |

二进制是如何表示十进制的？这里我们要引入位（bit）的概念，每一个位都由0,1两种可能组成，那么四个二进制位即可以表示16个数字，其通过01不同的交错排列而形成。而位置编码呢？原论文是这样描述的：

>位置编码的每个维度都对应于一个正弦曲线。波长形成一个从2π到(10000·2π)的几何轨迹。我们之所以选择这个函数，是因为我们假设它可以让模型很容易地通过相对位置进行学习，因为对于任何固定的偏移量k,$PE_{pos+k}$都可以表示为$PE_{pos}$的线性函数。

位置编码采用了不同频率的三角函数来表达，其意思是与01的位置是一致的，熟悉傅里叶级数的同学肯定能想到了。随着pos的增加，波长越来越长，这就很好的实现了低位变化快、高位变化慢的情况。

![image-20241109233523778](C:\Users\HP\AppData\Roaming\Typora\typora-user-images\image-20241109233523778.png)

另外我们还对一些东西感兴趣，比如相对位置关系：

在实际的Transformer的attention过程中，其利用位置编码是用来推断相对位置的：
$$ {split}
\begin{align}
PE_{pos} \cdot PE_{pos+k} & =\sum_{i=0}^{\frac{d}{2}-1}{sin(\omega_i\cdot pos) \cdot sin(\omega_i\cdot (pos+k))}\\
& =\sum_{i=0}^{\frac{d}{2}-1}cos(\omega _i(pos-(pos+k)))\\
& =\sum_{i=0}^{\frac{d}{2}-1}cos(\omega _i\cdot k)\
\end{align}
$$
我们可以看到，在这个判断相对位置的过程中是不会体现**相对距离k的正负的**，也就是说，transformer没办法判断两个位置的前后关系。

#### 为什么位置编码和嵌入相加而不是拼接？

我们之前看到了位置编码是越到高维度信息越是稀疏，也就是说位置信息是前半段比较有用，而嵌入是后半段比较有用，因此做相加是一个隐藏式的拼接，同时降低了维度减少了计算量。

#### 总结

>  **交趾之南有越裳国。周公居摄六年，制礼作乐，天下和平。越裳以三象重译而献白雉。**

理解输入与嵌入部分其实很简单，就是翻译，我们对自然语言做了层层翻译，从最原始的语言到tokenization再到简单的编码再到嵌入再加上位置编码，我们成功的将自然语言翻译成了机器能够理解的语言，就像经过“三象重译”，我们才和计算机建立了沟通。



<div style="text-align: center;">
  <iframe
    id="ppt"
    width="100%"
    src="https://onedrive.live.com/embed?resid=F0D7612D44C925FB%21550&authkey=!AJ96EnctAg9b05Q&em=2"
    frameborder="0"
    style="display: block; max-width: 960px; margin: auto;">
  </iframe>
</div>
### Encoder-Decoder





### 参考内容

[1] [What is Tokenization in NLP? Here’s All You Need To Know ](https://www.analyticsvidhya.com/blog/2020/05/what-is-tokenization-nlp/)

[2] [小米面试官：“Tokenization 是什么”。封面看着眼熟 (qq.com)](https://mp.weixin.qq.com/s/0ewnWvf8sQflmamcXpcUfQ)

[3] [NLP学习笔记(十) 分词(下)-CSDN博客](https://blog.csdn.net/wsmrzx/article/details/129296403)

[4] [BPE（Byte Pair Encoding）算法python实现 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/695414398)

[5] [14.1. 词嵌入（word2vec） — 动手学深度学习 2.0.0 documentation (d2l.ai)](https://zh.d2l.ai/chapter_natural-language-processing-pretraining/word2vec.html)

[6] [Word2Vec For Word Embeddings -A Beginner's Guide (analyticsvidhya.com)](https://www.analyticsvidhya.com/blog/2021/07/word2vec-for-word-embeddings-a-beginners-guide/)

[7] [The Illustrated Word2vec – Jay Alammar – Visualizing machine learning one concept at a time. (jalammar.github.io)](https://jalammar.github.io/illustrated-word2vec/)

[8] [word2vec 中的数学原理详解 - peghoty - 博客园 (cnblogs.com)](https://www.cnblogs.com/peghoty/p/3857839.html)

[9] [Transformer位置编码图解 - BimAnt](http://www.bimant.com/blog/transformer-positional-encoding-illustration/)

[10] [一文通透位置编码：从标准位置编码、旋转位置编码RoPE到ALiBi、LLaMA 2 Long(含NTK-aware简介)-CSDN博客](https://blog.csdn.net/v_JULY_v/article/details/134085503)

[11] [【OpenLLM 009】大模型基础组件之位置编码-万字长文全面解读LLM中的位置编码与长度外推性（上） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/626828066)

[12] [Transformer 架构：位置编码 - Amirhossein Kazemnejad 的博客](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)

[13] [一文搞懂Transformer的位置编码_transformer位置编码-CSDN博客](https://blog.csdn.net/xian0710830114/article/details/133377460)
