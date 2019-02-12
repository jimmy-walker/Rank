# Learning to Rank

Learning to Rank主要有3种算法：

1. Pointwise: Subset Ranking, McRank, Prank, OC SVM
2. Pairwise: Ranking SVM, RankBoost, RankNet, GBRank, IR SVM, Lambda Rank, LambdaMart
3. Listwise: ListNet, ListMLE, AdaRank, SVM MAP, Soft Rank

## 推荐文章
介绍系统层面：
https://www.jiqizhixin.com/articles/2018-12-21-11
讲到深度学习用于l2r**（J继续寻找该代码）**
https://www.jiqizhixin.com/articles/2019-01-18-20
专门介绍lambamart，其中mart就是gbdt：
https://liam.page/2016/07/10/a-not-so-simple-introduction-to-lambdamart/
图文讲解lambdamart
http://blog.jqian.net/post/lambdamart.html

讲到ftlr也是基于梯度下降的在线学习

https://www.zhihu.com/question/28025036/answer/107297334

里面提到用ranklib而非es的ltr插件进行排序更好，因为不依赖es的版本。

https://hackernoon.com/learning-to-rank-for-job-search-d1dc44d7070


## 推荐书籍



## 推荐代码

https://github.com/jattenberg/RankLib 基于java

https://github.com/codelibs/ranklib 基于java，下面项目的clone

大名鼎鼎的Ranklib库https://sourceforge.net/p/lemur/wiki/RankLib/
https://sourceforge.net/p/lemur/wiki/RankLib%20How%20to%20use/ 这是新页面
https://people.cs.umass.edu/~vdang/ranklib.html 这是老页面无法访问了

有讲到怎么下载数据集进行几种算法测试。

对ranklib用法的详解
https://github.com/cixuuz/LTR.project

基于tensorflow实现的l2r
https://github.com/ChenglongChen/tensorflow-LTR

## ctr

https://github.com/guoday/ctrNet-tool（实现了ctr的算法FM, FFM, NFFM ）

## 个性化排序实战

个性化排序实战