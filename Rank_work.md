## 排序工作

## ogeek

### 题目

####数据格式

| 字段             | 说明                                                         | 数据示例                                   |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------ |
| prefix           | 用户输入（query前缀）                                        | 刘德                                       |
| query_prediction | 根据当前前缀，预测的用户完整需求查询词，最多10条；预测的查询词可能是前缀本身，数字为统计概率 | {“刘德华”:  “0.5”, “刘德华的歌”: “0.3”, …} |
| title            | 文章标题                                                     | 刘德华                                     |
| tag              | 文章内容标签                                                 | 百科                                       |
| label            | 是否点击                                                     | 0或1                                       |

####数据格式解释
#####title，tag，query_prediction的关系

<https://tianchi.aliyun.com/forum/issueDetail?spm=5176.12586969.1002.285.651d75f0jBhqXn&postId=25393> 

对于给定的样例数据： 挂号

{"挂号信是什么": "0.023", "挂号网上预约": "0.029", "挂号网官网": "0.015", "挂号信": "0.082", "挂号": "0.066", "挂号信单号查询": "0.075", "挂号平台": "0.025", "挂号网": "0.225", "挂号信查询": "0.201", "挂号信查询中国邮政": "0.020", "挂号预约": "0.021"}

预约挂号网 应用

首先，“挂号”，是我在Activity中输入的prefix。也就是对应

![1.jpg](http://aliyuntianchipublic.cn-hangzhou.oss-pub.aliyun-inc.com/public/files/image/1095279354181/1538114245598_DSJ8Q2EQHt.jpg)

其次，“预约挂号网”，是Activity中的内容。也就是对应

![2.jpg](http://aliyuntianchipublic.cn-hangzhou.oss-pub.aliyun-inc.com/public/files/image/1095279354181/1538114310002_2oY9ZpVqwB.jpg)

再者，“应用”，是Activity中内容部分的标签。也就是对应

![3.jpg](http://aliyuntianchipublic.cn-hangzhou.oss-pub.aliyun-inc.com/public/files/image/1095279354181/1538114567562_DrbGQKgTdr.jpg)

然而，中括号中的内容仅仅是后台数据，是用来辅助理解prefix的，单单从数据本身来看对Activity中的内容部分并没有什么直接联系。

##### query_prediction总和

<https://tianchi.aliyun.com/forum/issueDetail?spm=5176.12586969.1002.264.651d75f0jBhqXn&postId=26359> 

注意中括号中存在给出少于10个prediction，但总和小于1的情况，会有低质过滤，简单理解就是说剩余那几个质量不高，被过滤掉了 。

#####样本的产生

<https://tianchi.aliyun.com/forum/issueDetail?spm=5176.12586969.1002.168.651d75f0jBhqXn&postId=27417> 

对应每一个prefix，浏览器弹出3个备选的title+tag。

用户可能点击其中一个，也可能一个都不点击。

官方这样的一次场景拆分成3条记录，放到数据集里面。

若果然如此，正样本率天然就低于1/3. 

当前训练集+验证集统计正样本率35%～40%之间，与上述推测略有差异



## 梳理

###文章浏览

#### [Learning to rank学习基础](http://kubicode.me/2016/02/15/Machine%20Learning/Learning-To-Rank-Base-Knowledge/#%E6%97%A5%E5%BF%97%E6%8A%BD%E5%8F%96 ) J关键是要赶快跑一下这些代码，知道到底有哪些特征

**难点主要在训练样本的构建(人工或者挖日志)参考ltr在淘宝中的应用那篇文章**

虽说Listwise效果最好，但是天下武功唯快不破。在工业界用的比较多的应该还是Pairwise，因为他构建训练样本相对方便，并且复杂度也还可以，所以Rank SVM就很火啊。



####[相关性特征在图片搜索中的实践](https://www.infoq.cn/article/myAAs8rIG5JVo*YgyyLi?utm_source=rss&utm_medium=article) J直接用点击数就进行二排，没有考虑到，本身的其他特征，因此需要用很多规则去限制。比如狮子王new等等，没考虑到热度。说穿了就是要融入很多特征进行ltr排序！ltr是一个概念，用机器学习进行排序，具体算法则是很多，甚至lr也可以用。

搜索最基础的两部分：召回 + 排序，

召回功能由索引完成；

排序就是对候选 doc 计算相关性分值，然后根据分值完成最终的检索结果排序。 

排序部分工作不是由一个模型完成的，**用一个模型计算 query 和 doc 的相关性分值就直接排序，这样太简单粗暴了**，也很难在工业上达到较好的效果。 

大多数设计模式是，**通过基础模型学习不同的特征维度来表示各个域的相关性**，如 query 和 doc 文本相关性、和图像相关性、站点质量、图片质量等特征，然后**使用模型将这些特征综合计算得到排序分值**。这里我们关注的重点是相关性特征的表示。

第二部分介绍的两类方法，大致就是计算 query 和 doc 相关性的两类思路，这些方法都是计算 query 和 doc 单一相关性，即 query 和 doc 文本、query 和 doc 图像等。得到这些基础的相关性特征后，然后再使用 ltr 模型 ( 如 lr\svmrnak ) 来计算最终的排序分值。 J**注意这里的意思是有些learning to rank的方法是可以采用lr算法来计算的，比如pointwise，也有部分[pairwise]()！**<https://github.com/isnowfy/ranklogistic> 

拿 lr 模型来说，**比如有 10 个基础相关性特征，经过训练之后，lr 模型就有 10 个固定的权重。稍加思考就知道，对于不同 query 权重应该是变化的**，比如“5 月伤感图片”、“老虎简笔画图片”这两个 query ，前者应该更倾向于语义特征，因为很难定义什么样的图像叫伤感图像，但后者应该更倾向于图像特征，至少该是个简笔画图片。 **后来看到有研究使用 Attention 机制来解决这个问题的**，感觉是个很好的思路。大体想法是，分别计算 query 和 <doc 文本，doc 图像 > 整体的相关性，然后根据 query 和 doc 的本身特征，学到两种相关性的权重。

####[Learning to rank在淘宝的应用](https://mp.weixin.qq.com/mp/appmsg/show?__biz=MjM5MTY3ODcxNA==&appmsgid=10000062&itemidx=1&sign=e00d2e1f1fd917b1457c6e47421a4336&scene=20&xtrack=1&uin=MTcwOTYyNDY2MA%3D%3D&key=c55f77d6ac845d334f02598df6f4ecf26c3b3997975c989a5166c9abc5af96486ceb76f84a66a8c9fb5e48a8a1eab064735d7b9624c0867dde754e1183951a6b093013d51738b09dac8c0f327d2eb516&ascene=1&devicetype=Windows+7&version=62060739&lang=zh_CN&pass_ticket=EngB48mcD8xDHpo2QLfAzMRWm10btoeqOyABAeVcCEyUGzDOQ8sWFJW5qwAUWfGm ) 生成pariwise的数据，里面有些成功与否与本身rank有意义，比如超过时长，多听次数，J但是这都拿不到，考虑用之前的点击模型方法计算是否最终听完超过30秒。再加上原本排序。找到PPT了，学习到了很强大！！！[曾安祥_阿里巴巴搜索事业部AI实验室负责人](http://topic.it168.com/factory/adc2013/doc/zenganxiang.pptx) [个人微博](https://weibo.com/zengxiang0217?is_all=1#_rnd1563183687522) 结合下面的那个总结，我认为分成两类：只针对热门进行研究，原排序超过20和20以内随机抽取；而利用点击模型计算得到吸引力指数生成所有浏览过的数据（其中差值需要有一定置信度，J还是设一个，但不是很大的那种，其实也可以不用，毕竟有了贝叶斯平滑了，但是如果只是对于用户点击而言，就用曝光，因为本身在前排的都差不多），然后利用本次点击，聆听，购买等操作进行去除。

选择pair的方法是通过用户的点击与购买反馈来生产表示商品好坏的pair对。 

**使用点击反馈来形成pair**

 统计同query下，每个商品的点击率，将点击率平滑后，将点击率相差一定以上的商品形成pair。

**使用无点击的数据来形成pair**

在样本中加入一部分通过原始排序来生成的pair，这样的目的是使排序的参数变化不至于太剧烈，形成一些意想不到的badcase。  

这部分的样本根据展示日志中原始商品的排序，第1个商品好于第20个商品，第2个商品好于第21个商品，以此类推。

**样本的混合与分层抽样**

![](picture/dataset.png)

![](picture/整体ctr.png)
![](picture/单次样本选择.png)
![](picture/无点击.png)



#### [Learning to Rank 的实践](https://blog.csdn.net/chikily_yongfeng/article/details/81396607 ) 输入形式和损失函数决定ltr分类。

**按照这些方法的输入形式（input presentation）和损失函数（loss function）将这些方法划分为：Pointwise，Pairwise，和 Listwise 方法**。

**Ranklib** 是一个开源项目，其中包含了绝大多数学术界中的 LTR 方法。 

<https://sourceforge.net/p/lemur/wiki/RankLib/> 

- MART (Multiple Additive Regression Trees, a.k.a. Gradient boosted regression tree) [6]
- RankNet [1]
- RankBoost [2]
- AdaRank [3]
- Coordinate Ascent [4]
- LambdaMART [5]
- ListNet [7]
- Random Forests [8]



#### [Learning to Rank(LTR)](https://blog.csdn.net/clheang/article/details/45674989 ) [Ranking SVM 简介](https://blog.csdn.net/clheang/article/details/45767103 ) 使用点击模型产生ltr的训练数据，类似上面的点击反馈。（就是后面的比前面的不但要点（只是点击），而且平滑后还高的比例，就类似点击率相差一定比例，等同于平滑后还高）另外就是原本排序，最后就是考虑听歌时长，时长大致相同情况下（去除短时间样本的），这就类似购买了，而评论数收藏数播放数点赞数发行时间距离搜索时间等等都可以作为特征！！！注意只从topquery中取得这些数据！！！

使用点击日志的偏多。比如，结果ABC分别位于123位，B比A位置低，但却得到了更多的点击，那么B的相关性可能好于A。点击数据隐式反映了同Query下搜索结果之间相关性的相对好坏。在搜索结果中，高位置的结果被点击的概率会大于低位置的结果，这叫做”点击偏见”（Click Bias）。但采取以上的方式，就绕过了这个问题。因为我们只记录发生了”点击倒置”的高低位结果，使用这样的”偏好对”作为训练数据。关于点击数据的使用，后续再单独开帖记录，这里不展开。 在实际应用中，除了点击数据，往往还会使用更多的数据。比如通过session日志，挖掘诸如页面停留时间等维度。 在实际场景中，搜索日志往往含有很多噪音。**且只有Top Query（被搜索次数较多的Query）才能产生足够数量能说明问题的搜索日志。**

![](picture/trainingdata.png)

使用Clickthrough数据作为Ranking SVM的训练数据，来源[Optimizing Search Engines using Clickthrough Data. Thorsten Joachims. SIGKDD,2002](https://www.cs.cornell.edu/people/tj/publications/joachims_02c.pdf )：

其中1, 3, 7三个结果被用户点击过, 其他的则没有。因为返回的结果本身是有序的, 用户更倾向于点击排在前面的结果, 所以用户的点击行为本身是有偏(Bias)的。为了从有偏的点击数据中获得文档的相关信息, 我们认为: 如果一个用户点击了a而没有点击b, 但是b在排序结果中的位置高于a, 则a>b。

所以上面的用户点击行为意味着: 3>2, 7>2, 7>4, 7>5, 7>6。J其实这个不合理，可见淘宝中的learning to rank应用。

希望改成下面这样的训练数据:

![](picture/feature.png)

####[learning to rank简介](https://daiwk.github.io/posts/nlp-ltr.html ) [百度ltr的github介绍](https://github.com/PaddlePaddle/models/tree/develop/legacy/ltr )提到了用户id的embedding及其稀疏的解决方式，从而为排序个性化指路，不过目前暂时不做这个。

常用方法是直接将用户ID经过Embedding后作为特征接入到模型中，但是最后上线的效果却不尽如人意。通过分析用户的行为数据，我们发现相当一部分用户ID的行为数据较为稀疏，导致用户ID的Embedding没有充分收敛，未能充分刻画用户的偏好信息。

Airbnb发表在KDD 2018上的文章[Real-time Personalization using Embeddings for Search Ranking at Airbnb](https://astro.temple.edu/~tua95067/kdd2018.pdf)为这种问题提供了一种解决思路——利用**用户基础画像和行为数据**对**用户ID进行聚类**。Airbnb的主要场景是为旅游用户提供民宿短租服务，一般**用户一年旅游的次数在1-2次之间**，因此Airbnb的用户行为数据相比点评搜索会更为稀疏一些。

百度的ltr介绍了大致的方法和代码，不过数据集是用的是开源的人工标注的。