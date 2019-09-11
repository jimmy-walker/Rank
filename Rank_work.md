# 排序工作

[TOC]



## 梳理

### 思索

- **~~先搞定贝叶斯平滑对ctr的修正，应用到relevance~~，~~关于intent的不行先用去除热门歌手的query~~，然后再用构造方法构造训练集**。在ubm论文索引中使用"smoothing "click model""
- **第一步：用lambdarank或lambdamart，使用lightgbm~~或tensorflow ranking~~实现**。
- 以CTR为基础，为什么不直接用CTR排序呢，这是因为一些新的项目或是一些新的query时，那就没法用CTR数据了，只能用这个模型进行预估。另外，有些是用`ctr*a+cvr`进行排序的。
- 为了构建出 <q, p, r-score> 的数值 pair 对。能否直接用前面位置（比如前10）的修正过的点击模型吸引力来进行构建呢，这就会避免贝叶斯平滑带来的问题了。即分别准备一个pairwise和一个pointwise数据集。
- 预估用户对每条资讯的偏好程度，一般以点击率衡，在很多场合下排序学习（ltr）就等同于点击率预估。J因为可能有时候业务需要用cvr来进行排序。所以CTR的方法都能用，但是难点在于数据集的构建。因为对于不同query下，不能用总体ctr，曝光也不一致。
- 可以考虑先用LR或者是树模型lightgbm或者lambdarank，有证据表明**xgb和wide&deep值得学习，所以考虑lightgbm和wide&deep**。。。。注意补充上交叉特征，并且可解释。关键先把数据构建完。。。避免稀疏。所有特征按类型划分为离散型、连续型、向量型三种类型。如item类别就是一个离散型特征、item ctr就是一个连续性特征、标题向量就是一个向量型特征。对于连续值我们部分参考youtube wide & deep论文中的等频归一化方法，简单来说加入ctr特征需要等屏成10个特征，即将ctr值按照分布取其10等分点，这10等分点就定义了10个区间，每个区间的样本数都占10%。 我们没有使用bagof word模型来表示标题，因为这非常稀疏，而是采用标题中关键词的word2vec向量组合生成标题表示，使用词向量来表示标题极大减少了特征规模，实现上比较方便。标题向量同时需要归一化成单位向量，单位向量的余弦相似度即两个向量的内积，这个优化显著提高了ltr在线模块的性能。**J可能不需要使用向量特征，毕竟需要重新训练而已，并且不像搜索引擎一样需要很多的召回，还是严格模式。**
- 部署：hive和reddit都有特征存储。使用一天的训练数据的情况下，整个特征空间规模约为30万维左右。我们使用N-2天前的日志做训练，N-1天前的日志做评估，需保证两部分日志的用户群体是一致的。我们采用1500万条样本做训练，300万条样本做评估。J在我这里还不需要利用时间上面进行训练，因为不进行个性化处理。
- 特征包括：文本特征（如何考察全包含，歌手名，歌曲名），音频质量得分（可以滤除4秒的音乐），收藏数，点赞数，发布时间，评论数，播放数，付费与否（决定于推广与否）等等。
- 构建样本时**J自己的想法**，对于点击样本，如果都前后两个都点击了，则不计算pair；对于未点击样本，并不是指未点击，而是不考虑点击而已。~~就算两个样本中有矛盾，这也不要紧，因为就是要把loss降低而已。所以要对热门的都进行计算~~**感觉还是不能矛盾，按照PPT的思路就是先用总体ctr生成pair（曝光的行为之前的，可以考虑不用贝叶斯平滑或者用贝叶斯平滑两者比较）（考虑相同名称的取最高值），~~然后对于单个行为时的具体行为，将不符合的去除，这样就避免矛盾了。最后再对同一query下进行set，去除重复的，这里到时候看看数据集要不要去除重复，因为合并的话其实回到单个行为意义就不打大了，毕竟肯定又是和原来的一样了。J我想到一点将各个pv中得到的A,B,1聚合起来，比如有矛盾的A,B,0,然后统计数量，毕竟不存在的不出现。最后如果有0，就把他改正过来变成1~~。** **<u>淘宝的数据利用是对每个pv的，因为每个用户不一样，所以用户id特征值不一样，J对于不同时间的可以合并起来，而我们这里似乎不用个性化了，就直接使用聚合即可，不用再返回到单独pv了，或者就回到单独pv，不过就是重复即可，等于是让其更重要</u>**
- 可解释功能待做，因为独立不影响。
- [From RankNet to LambdaRank to LambdaMART: An Overview](https://pdfs.semanticscholar.org/0df9/c70875783a73ce1e933079f328e8cf5e9ea2.pdf) 好好看看
- [A Novel Algorithm for Unbiased Learning to Rank ](https://arxiv.org/abs/1809.05818) 也看看点击模型在其中的用途
- [Unbiased Learning-to-Rank with Biased Feedback](https://www.ijcai.org/proceedings/2018/0738.pdf) 也看看电机模型在其中的用途
- 把算法详解中好好看看。**先搞清楚构建数据集，然后慢慢看算法原理，可以看代码理解原理。[比如这个](**<https://github.com/cnkuangshi/LightCTR> **)**
- **<u>用点击模型计算相关度时，要么使用改进过的点击模型浏览，要么使用直接用播放来代替或用大的贝叶斯平滑代替，对于歌手名来说，可以考虑用分布情况，分布很散，说明确实是浏览类型。</u>~~亦或者直接在ltr算法中传入关键字类型，从而让算法判断得到当这类算法时，就不用这么算了，不能用这个方法，标注还是不一样的~~**。**<u>可以考虑对于贝叶斯平滑低值附近的就不考虑了，因为太少了。</u>**
- "wide&deep"OR"wide deep"OR"wide and deep" "learning to rank"OR"ltr"没搜到，只说[可以这么干](https://zhuanlan.zhihu.com/p/53110408)
- 从论文中得知**ubm是使用em+最大似然估计，而我们可以考虑使用em+最大后验概率**，比如在下文的抛硬币的实验中假设只有一次和两次正面实验，$\theta, \theta^2$，那么利用最大似然概率，结果就是1，而利用最大后验概率，计算$100\theta^2-50\theta-1=0, 100\theta^2-50\theta-2=0$，得到结果为$0.519, 0.537$。目前经过研究得到了相关的最大后验概率基于ubm的解法，J其实就是类似之前的贝叶斯。。。不过着眼于概率选定和初始值，目前很棘手，没有通论，所以考虑使用聚类，分成两类，这样就剔除了底层的那些干扰项，保证了前述项不去进行后验概率，另外概率选定，可用计算公式。可以考虑用昨天的平均值减去1个或2个标准差。。。**并且尝试修改用全部曝光而非点击最后一位，这样等于是削弱了第一个和第二个之间的分母差距。**
- **另外看了BBM，其核心就是不用em算法，用类似贝叶斯的推断，计算后验概率，然后进行积分计算，好处是稳定，不容易被迭代的次数给左右，坏处是花时间不一定有效果，需要看积分公式**。
- 未来可以考虑在ltr基础上融合个人信息，[如此代码](https://github.com/QingyaoAi/Deep-Listwise-Context-Model-for-Ranking-Refinement )
- **clickmodels的方法是不对的，他们的intent是对图像等verticle更感兴趣**，而非浏览和寻找，还是看回UBM的论文和scala版的ubm。而intent这么操作：引入变异系数来做，结合android的曝光6个，就选择小于6个的一般目标较明确，采用所有的最小值，利用播放量前6位的来计算（对歌手名和歌曲名进行聚合，不包含版本）思路清晰了，就是利用这段代码去进行更改：

```scala
    def train(sessions: Seq[(Int, Seq[Int], Seq[Boolean])], maxIter: Int)= {
      val data = sessions.flatMap { case (q, url, click) =>
        val distance = click2distance(click)
        url.zip(distance).map{ case (u, (c, r, d)) =>
          (q, u, r, d, c) //flat and reorder
        }
      }.groupBy{identity}.mapValues{_.length} //key is (query, url, rank, distance, click), value is total number of the key
      for (i <- 0 until maxIter) {
        val updates = data.map { case ((q, u, r, d, c), cnt) =>
          val alpha_uq = alpha(u)(q)
          val gamma_rd = gamma(r)(d)
          val mu_q = mu(q)
          val mu_gamma = mu_q.zip(gamma_rd).map{ case (x, y) => x * y} //2 browsing model the length will become two
          val dot_prod_mu_gamma = mu_gamma.sum //即分母中的mu和gamma对不同m的和
          val Q_m_a1_e1_c1 = mu_gamma.map {
            _ / dot_prod_mu_gamma //就是得到各个意图下查看q中r的情况，that is c within essay
          }
          val Q_m_e1_c1 = Q_m_a1_e1_c1
          val Q_m_c1 = Q_m_a1_e1_c1
          val Q_a1_c1 = 1.0 //计算alpha时即点击时的S的系数
          val Q_a1_c0 = alpha_uq * (1 - dot_prod_mu_gamma) / (1 - alpha_uq * dot_prod_mu_gamma) //计算alpha时未点击时的S的系数
          val Q_m_e1_c0 = mu_gamma.map {
            _ * (1 - alpha_uq) / (1 - alpha_uq * dot_prod_mu_gamma) //计算gamma时A的未点击的S的系数
          }
          val Q_m_c0 = gamma_rd.map { gamma_rdm =>
            1 - alpha_uq * gamma_rdm
          }.zip(mu_q).map { //zip的用处可以这样平行连接，把两个向量中对应位置进行计算
            case (x, y) => x * y / (1 - alpha_uq * dot_prod_mu_gamma) //计算gamma时B的未点击的S的系数
          }
          //返回的是tuple
          val alpha_fraction = if (c) {
            (Q_a1_c1 * cnt, cnt) //点击时系数乘以S值，和该S值（为了之后计算分母）
          } else {
            (Q_a1_c0 * cnt, cnt) //未点击时系数乘以S值，和该S值（为了之后计算分母）
          }
          //这里返回的是seq，其中每一项是tuple
          val gamma_fraction = if (c) {
            Q_m_e1_c1.map{_ * cnt}.zip(Q_m_c1.map {_ * cnt}) //只是返回A,B的值，不像其他进行计算和累计S值（因为后面需要先累加再A/B，不需要计算分母）
          } else {
            Q_m_e1_c0.map {_ * cnt}.zip(Q_m_c0.map {_ * cnt}) //只是返回A,B的值，不像其他进行计算和累计S值（因为后面需要先累加再A/B，后面不需要计算分母）
          }
          //这里返回的是seq，其中每一项是tuple
          val mu_fraction = if (c) {
            Q_m_c1.map { q_m_c => ( q_m_c * cnt, cnt)} //等价于各个意图下查看q中r的情况乘以S值，和该S值（为了之后计算分母）
          } else {
            Q_m_c0.map { q_m_c => (q_m_c * cnt, cnt) } //等价于计算gamma时B的未点击的S的系数乘以S值，和该S值（为了之后计算分母）
          }
          ((q, u, r, d), (alpha_fraction, gamma_fraction, mu_fraction))
        }
      
        // update alpha
        updates.map { case ((q, u, r, d), (af, gf, mf)) =>
          ((u, q), af)
        }.groupBy {_._1}.mapValues { //以此为key，而与此key相同的元素组成value中的list，即返回是一个map：(u, q), List(((u, q) ,af) ...)
          _.map {_._2}.reduce[(Double, Int)] { case (x, y) => //Double, Int就是上文得到的系数乘以S值，和该S值
            (x._1 + y._1, x._2 + y._2) //表示将分子_1和分母_2分别累加
          }
        }.foreach{ case ((u, q), (num, den)) =>
          alpha(u)(q) = num / den //然后更新即可
        }
      
        // update gamma
        updates.map { case ((q, u, r, d), (af, gf, mf)) =>
          ((r, d), gf)
        }.groupBy{_._1}.mapValues { //以此为key，而与此key相同的元素组成value中的list，即返回是一个map：(r, d), List(((r, d) ,gf) ...)
          _.map {_._2}.reduce[Array[(Double, Double)]] { //Double, Double就是上文得到的A,B值，但是因为本身是seq，其中每一项是tuple
            case (xs, ys) => xs.zip(ys).map { //xs和ys代表seq中的各个tuple，重新组合成_1在一个tuple，_2在一个tuple中
              case (x, y) => (x._1 + y._1, x._2 + y._2) //表示将分子_1和分母_2分别累加
            }
          }
        }.foreach { case ((r, d), fs) => //其中fs是上面新生成的tuple：(x._1 + y._1, x._2 + y._2)，存储_1,_2的各自之和
          fs.zipWithIndex.foreach { case ((num, den), m) =>
            gamma(r)(d)(m) = num / den //然后更新即可
          }
        }
        // update mu
        updates.map { case ((q, u, r, d), (af, gf, mf)) =>
          (q, mf)
        }.groupBy{_._1}.mapValues {
          _.map{_._2}.reduce[Array[(Double, Int)]]{
            case (xs, ys) => xs zip ys map {
              case (x, y) => (x._1 + y._1, x._2 + y._2) //因为对于reduce，匿名函数定义了输出形式就如(x._1 + y._1, x._2 + y._2)
            }
          }
        }.foreach { case (q, fs) =>
          fs.zipWithIndex.foreach{ case ((num, den), m) =>
            mu(q)(m) = num / den
          }
        }
      }
        (alpha, gamma, mu)
    }
```


![](picture/ubm-intent.png)


###文章浏览

####基本概念

##### [Learning to rank学习基础](http://kubicode.me/2016/02/15/Machine%20Learning/Learning-To-Rank-Base-Knowledge/#%E6%97%A5%E5%BF%97%E6%8A%BD%E5%8F%96 ) 全部概述一遍，J关键是要赶快跑一下这些代码，知道到底有哪些特征

**难点主要在训练样本的构建(人工或者挖日志)参考ltr在淘宝中的应用那篇文章**

虽说Listwise效果最好，但是天下武功唯快不破。在工业界用的比较多的应该还是Pairwise，因为他构建训练样本相对方便，并且复杂度也还可以，所以Rank SVM就很火啊。

##### [Rank教程: 06-Learning to Rank 概述](http://www.bdpt.net/cn/2018/11/20/rank%E6%95%99%E7%A8%8B-06-learning-to-rank-%E6%A6%82%E8%BF%B0/ ) 讲述了ltr的分类，这里写错了是lambda rank

![](picture/category.png)

####训练数据

#####[Learning to rank在淘宝的应用](https://mp.weixin.qq.com/mp/appmsg/show?__biz=MjM5MTY3ODcxNA==&appmsgid=10000062&itemidx=1&sign=e00d2e1f1fd917b1457c6e47421a4336&scene=20&xtrack=1&uin=MTcwOTYyNDY2MA%3D%3D&key=c55f77d6ac845d334f02598df6f4ecf26c3b3997975c989a5166c9abc5af96486ceb76f84a66a8c9fb5e48a8a1eab064735d7b9624c0867dde754e1183951a6b093013d51738b09dac8c0f327d2eb516&ascene=1&devicetype=Windows+7&version=62060739&lang=zh_CN&pass_ticket=EngB48mcD8xDHpo2QLfAzMRWm10btoeqOyABAeVcCEyUGzDOQ8sWFJW5qwAUWfGm ) 生成pariwise的数据，里面有些成功与否与本身rank有意义，比如超过时长，多听次数，J但是这都拿不到，考虑用之前的点击模型方法计算是否最终听完超过30秒。再加上原本排序。找到PPT了，学习到了很强大！！！[曾安祥_阿里巴巴搜索事业部AI实验室负责人](http://topic.it168.com/factory/adc2013/doc/zenganxiang.pptx) [个人微博](https://weibo.com/zengxiang0217?is_all=1#_rnd1563183687522) 结合下面的那个总结，我认为分成两类：只针对热门进行研究，原排序超过20和20以内随机抽取；而利用点击模型计算得到吸引力指数生成所有浏览过的数据（其中差值需要有一定置信度，J还是设一个，但不是很大的那种，其实也可以不用，毕竟有了贝叶斯平滑了，但是如果只是对于用户点击而言，就用曝光，因为本身在前排的都差不多），然后利用本次点击，聆听，购买等操作进行去除。

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

##### [Learning to Rank(LTR)](https://blog.csdn.net/clheang/article/details/45674989 ) [Ranking SVM 简介](https://blog.csdn.net/clheang/article/details/45767103 ) 使用点击模型产生ltr的训练数据，类似上面的点击反馈。（就是后面的比前面的不但要点（只是点击），而且平滑后还高的比例，就类似点击率相差一定比例，等同于平滑后还高）另外就是原本排序，最后就是考虑听歌时长，时长大致相同情况下（去除短时间样本的），这就类似购买了，而评论数收藏数播放数点赞数发行时间距离搜索时间等等都可以作为特征！！！注意只从topquery中取得这些数据！！！

使用点击日志的偏多。比如，结果ABC分别位于123位，B比A位置低，但却得到了更多的点击，那么B的相关性可能好于A。点击数据隐式反映了同Query下搜索结果之间相关性的相对好坏。在搜索结果中，高位置的结果被点击的概率会大于低位置的结果，这叫做”点击偏见”（Click Bias）。但采取以上的方式，就绕过了这个问题。因为我们只记录发生了”点击倒置”的高低位结果，使用这样的”偏好对”作为训练数据。关于点击数据的使用，后续再单独开帖记录，这里不展开。 在实际应用中，除了点击数据，往往还会使用更多的数据。比如通过session日志，挖掘诸如页面停留时间等维度。 在实际场景中，搜索日志往往含有很多噪音。**且只有Top Query（被搜索次数较多的Query）才能产生足够数量能说明问题的搜索日志。**

![](picture/trainingdata.png)

使用Clickthrough数据作为Ranking SVM的训练数据，来源[Optimizing Search Engines using Clickthrough Data. Thorsten Joachims. SIGKDD,2002](https://www.cs.cornell.edu/people/tj/publications/joachims_02c.pdf )：

其中1, 3, 7三个结果被用户点击过, 其他的则没有。因为返回的结果本身是有序的, 用户更倾向于点击排在前面的结果, 所以用户的点击行为本身是有偏(Bias)的。为了从有偏的点击数据中获得文档的相关信息, 我们认为: 如果一个用户点击了a而没有点击b, 但是b在排序结果中的位置高于a, 则a>b。

所以上面的用户点击行为意味着: 3>2, 7>2, 7>4, 7>5, 7>6。J其实这个不合理，可见淘宝中的learning to rank应用。

希望改成下面这样的训练数据:

![](picture/feature.png)

#####[1 号店 11.11：机器排序学习在电商搜索中的实战](https://www.infoq.cn/article/yhd-11-11-mlr-ec-search/ ) 采用了5级分值

在得到相关度数据和特征数据后，就可以根据 LambdaMART 训练数据的格式（如下所示），构建完整的训练数据集。每一行代表一个训练数据，项与项之间用空格分隔，其中 \<target> 是相关性得分，\<qid> 是每个 query 的序号，\<feature> 是特征项的编号，\<value> 是对应特征项归一化后的数值，\<info> 是注释项，不影响训练结果。

```
<target> qid:<qid> <feature>:<value> <feature>:<value> ... <feature>:<value> # <info>
```

图 3 表示项目中使用的实际训练数据（这里选取了其中 10 个特征作为示例，# 后面可以增加 Query 和商品名称，方便分析时的查看）：

![img](picture/sample.png)



#####[机器学习在高德搜索建议中的应用优化实践](https://yq.aliyun.com/articles/708264?utm_content=g_1000065961 ) 数据规模是百万query，这个是有用的信息，然后将一个session内的搜索进行融合，我这里可以改造下，就是后面点击的歌曲有前面的query的包含，则算作前面未点击的情况。但是感觉好难哦。。。。
样本构造难题，我们解决方案是：

•考量用户在出行场景的行为session，不光看在suggest的某次点击行为，更重要的是，考察用户在出行场景下的行为序列。比如suggest给出搜索建议后，继续搜索的是什么词，出行的地点是去哪里，等等。

•不是统计某个query下的点击， 而是把session看作一个整体，**用户在session最后的点击行为，会泛化到session中的所有query上**。

详细方案
第一步，融合服务端多张日志表，包括搜索建议、搜索、导航等。接着，进行session的切分和清洗。
最后，通过把输入session中，末尾query的点击计算到session中所有query上，以此满足实现用户输入session最短的优化目标。
最终，**抽取线上点击日志超过百万条的随机query**，每条query召回前N条候选POI。利用上述样本构造方案，最终生成千万级别的有效样本作为gbrank的训练样本。


####特征工程

#####[相关性特征在图片搜索中的实践](https://www.infoq.cn/article/myAAs8rIG5JVo*YgyyLi?utm_source=rss&utm_medium=article) J直接用点击数就进行二排，没有考虑到，本身的其他特征，因此需要用很多规则去限制。比如狮子王new等等，没考虑到热度。说穿了就是要融入很多特征进行ltr排序！ltr是一个概念，用机器学习进行排序，具体算法则是很多，甚至lr也可以用。

搜索最基础的两部分：召回 + 排序，

召回功能由索引完成；

排序就是对候选 doc 计算相关性分值，然后根据分值完成最终的检索结果排序。 

排序部分工作不是由一个模型完成的，**用一个模型计算 query 和 doc 的相关性分值就直接排序，这样太简单粗暴了**，也很难在工业上达到较好的效果。 

大多数设计模式是，**通过基础模型学习不同的特征维度来表示各个域的相关性**，如 query 和 doc 文本相关性、和图像相关性、站点质量、图片质量等特征，然后**使用模型将这些特征综合计算得到排序分值**。这里我们关注的重点是相关性特征的表示。

第二部分介绍的两类方法，大致就是计算 query 和 doc 相关性的两类思路，这些方法都是计算 query 和 doc 单一相关性，即 query 和 doc 文本、query 和 doc 图像等。得到这些基础的相关性特征后，然后再使用 ltr 模型 ( 如 lr\svmrnak ) 来计算最终的排序分值。 J**注意这里的意思是有些learning to rank的方法是可以采用lr算法来计算的，比如pointwise，也有部分[pairwise]()！**<https://github.com/isnowfy/ranklogistic> 

拿 lr 模型来说，**比如有 10 个基础相关性特征，经过训练之后，lr 模型就有 10 个固定的权重。稍加思考就知道，对于不同 query 权重应该是变化的**，比如“5 月伤感图片”、“老虎简笔画图片”这两个 query ，前者应该更倾向于语义特征，因为很难定义什么样的图像叫伤感图像，但后者应该更倾向于图像特征，至少该是个简笔画图片。 **后来看到有研究使用 Attention 机制来解决这个问题的**，感觉是个很好的思路。大体想法是，分别计算 query 和 <doc 文本，doc 图像 > 整体的相关性，然后根据 query 和 doc 的本身特征，学到两种相关性的权重。

#####[learning to rank简介](https://daiwk.github.io/posts/nlp-ltr.html ) [百度ltr的github介绍](https://github.com/PaddlePaddle/models/tree/develop/legacy/ltr )提到了用户id的embedding及其稀疏的解决方式，从而为排序个性化指路，不过目前暂时不做这个。

常用方法是直接将用户ID经过Embedding后作为特征接入到模型中，但是最后上线的效果却不尽如人意。通过分析用户的行为数据，我们发现相当一部分用户ID的行为数据较为稀疏，导致用户ID的Embedding没有充分收敛，未能充分刻画用户的偏好信息。

Airbnb发表在KDD 2018上的文章[Real-time Personalization using Embeddings for Search Ranking at Airbnb](https://astro.temple.edu/~tua95067/kdd2018.pdf)为这种问题提供了一种解决思路——利用**用户基础画像和行为数据**对**用户ID进行聚类**。Airbnb的主要场景是为旅游用户提供民宿短租服务，一般**用户一年旅游的次数在1-2次之间**，因此Airbnb的用户行为数据相比点评搜索会更为稀疏一些。

百度的ltr介绍了大致的方法和代码，不过数据集是用的是开源的人工标注的。



##### [1 号店 11.11：机器排序学习在电商搜索中的实战](https://www.infoq.cn/article/yhd-11-11-mlr-ec-search/ ) 特征工程这几类，主要是查询相关性，热度，new指标，付费与否，

![](picture/11feature.png)

#####[逻辑回归和learning to rank算法以及相关实践](https://luminarytian.github.io/2016/%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92%E5%92%8Clearning-to-rank%E7%AE%97%E6%B3%95%E4%BB%A5%E5%8F%8A%E7%9B%B8%E5%85%B3%E5%AE%9E%E8%B7%B5/ ) 特征在初期不要加入过多：那初期就暂时不要添加新的特征了，就用人工排序公式的特征进行学习
上图是一个搜索引擎返回的排序结果，加粗的是用户点击过的，简单的可以认为3、7是正例，2、4、5、6是负例，即排在后面有正例，前面的才能作为负例。

当然上述的组织方式在使用learning to rank的Pointwise Approach学习的过程中会将大部分权重都学成负(why？因为我们已经有一个人工的权重排序公式了，这样的训练集就是打人工公式的脸，人工排在前面的，都是负例，人工排在后面点击的才是正例)，相信大部分搞算法的都碰到过这样的问题。那这部分训练集到底代表了什么样的信息呢？**这部分训练集代表了用户在有偏的条件下，还极端的选择了后面的样例，可以肯定后面的样例一定有吸引用户的特征，只不过全部用这类样例来训练模型的话就会过拟了，so我们只需要对原始的排序模型进行采样并合并所有的样例，参见Learning to rank在淘宝的应用，再实施学习算法，基本就能simulate线上的情况，以及学习到的参数你就可解释了，而不是一脸懵逼**，能对学习到的参数的变化进行解释非常重要，这是分析调参和分析数据集以及证明你的想法不是瞎BB的基础。 

计算广告完成的目的是什么？有很多广告，又有很多用户，我不知道该怎么分发这些广告，而且也没有分发的依据，但我有很多广告和用户的特征，以及用户点击了广告的行为，所以我只要有了以前的日志，就可以学习到用户点广告的规律，这些规律用特征学习到的参数呈现。所以逻辑回归算法是完成一个从无到有的规律探索过程，而且实施了之后一定比目前线上的结果好。而对于搜索引擎来说，一般都会有一个几经锤炼的排序公式了，再用逻辑回归去学习参数，虽然理论上是能提高的，但实际效果提高可能不明显，并且如果再添加一些无用的特征(噪音)来干扰排序，那往往都是适得其反，所以搜索重排序可以看做是一个90到100的过程，**那初期就暂时不要添加新的特征了，就用人工排序公式的特征进行学习**，那这种算法就是经典的learning to rank(排序学习)了。

#####[一个可供参考的搜索引擎排序架构实践案例](https://www.infoq.cn/article/a-search-engine-scheduling-architecture-for-reference/ ) 特征包括：文本特征（如何考察全包含，歌手名，歌曲名），音频质量得分（可以滤除4秒的音乐），收藏数，点赞数，发布时间，评论数，播放数等等。

1） 文档本身的静态特征，包括文档的文本特征，如带权重的词向量，文档不同域（主标题、段落标题、描述内容、锚文本、URL 链接等）的 TF、IDF、BM25 和其他语言模型得分，也包括文档的质量分、网页文档的 PageRank 等重要性得分。关于文档的质量分，达观搜索根据不同的业务场景有不同的计算指标，比如电商相关的商品的质量分计算除了要考虑商品本身的文本与图片丰富度，更多的还要考虑商品的各种业务指标如销量变化、收藏、价格、库存、类别、上架时间、评论、商家信誉等级、是否作弊等，而媒体相关的文章的则需要考虑阅读数、转发数、赞数、收藏、评论、发文时间、主题类型等。 

2） 文档和 query 关联的特征，比如 query 对应文档的 TD-IDF score， BM25 score 等。 

3） query 本身的特征，比如文本特征，带权重的词向量，query 长度，query 所述的分类或主题，query 的 BM25 的 sum/avg/min/max/median 分数，query 上个月的热度等。

#####[大众点评搜索基于知识图谱的深度学习排序实践](https://www.jiqizhixin.com/articles/2019-01-18-20 )但是对于CTR预估和排序学习的领域，目前深度学习尚未在自动特征挖掘上对人工特征工程形成碾压之势，因此人工特征工程依然很重要。 特征离散化。下面embedding理解为对抽象query进行的必要的特征工作，比如粤菜，肯定不能完全匹配，而是要风格，这就可以进行embedding，使得某些商户embedding共现所以相似，然后利用点击行为使得query和商户又有embedding相似。

但是对于CTR预估和排序学习的领域，目前深度学习尚未在自动特征挖掘上对人工特征工程形成碾压之势，因此人工特征工程依然很重要。 

**特征离散化**：工业界一般很少直接使用连续值作为特征，而是将特征离散化后再输入到模型中。一方面因为离散化特征对于异常值具有更好的鲁棒性，其次可以为特征引入非线性的能力。并且，离散化可以更好的进行Embedding，我们主要使用如下两种离散化方法：

- 等频分桶：按样本频率进行等频切分，缺失值可以选择给一个默认桶值或者单独设置分桶。
- 树模型分桶：等频离散化的方式在特征分布特别不均匀的时候效果往往不好。此时可以利用单特征结合Label训练树模型，以树的分叉点做为切分值，相应的叶子节点作为桶号。

**加速Embedding特征的收敛**

在我们的深度学习排序模型中，除了Embedding特征，也存在大量Query、Shop和用户维度的强记忆特征，能够很快收敛。而Embedding特征是更为稀疏的弱特征，收敛速度较慢，为了加速Embedding特征的收敛，我们尝试了如下几种方案：

- **低频过滤**：针对出现频率较低的特征进行过滤，可以很大程度上减少参数量，避免过拟合。
- **预训练**：利用多类模型对稀疏Embedding特征进行预训练，然后进入模型进行微调：
  - 通过无监督模型如Word2vec、Fasttext对用户-商户点击关系建模，生成共现关系下的商户Embedding。
  - 利用DSSM等监督模型对Query-商户点击行为建模得到Query和商户的Embedding。
- **Multi-Task**：针对稀疏的Embedding特征，单独设置一个子损失函数，如下图所示。此时Embedding特征的更新依赖两个损失函数的梯度，而子损失函数脱离了对强特征的依赖，可以加快Embedding特征的收敛。



####开源的代码

#####[走马观花Google TF-Ranking的源代码](https://zhuanlan.zhihu.com/p/52447211) [用NumPy手工打造 Wide & Deep](https://zhuanlan.zhihu.com/p/53110408) [tf ranking](https://github.com/tensorflow/ranking) google官方tf ranking并且有介绍文章，只是看不大懂。。。


##### [Tensorflow implementations of various Learning to Rank (LTR) algorithms](https://github.com/ChenglongChen/tensorflow-LTR ) 基于tensorflow实现ltr算法，非官方

- pointwise
    - classification
        - DNN
        - LR
- pairwise
    - RankNet
    - LambdaRank
- listwise
    - ListNet (TODO)

##### [Learning to Rank 的实践](https://blog.csdn.net/chikily_yongfeng/article/details/81396607 ) 输入形式和损失函数决定ltr分类。

**按照这些方法的输入形式（input presentation）和损失函数（loss function）将这些方法划分为：Pointwise，Pairwise，和 Listwise 方法**。

**Ranklib** 是一个开源项目，其中包含了绝大多数学术界中的 LTR 方法。 

<https://sourceforge.net/p/lemur/wiki/RankLib/> 

<https://sourceforge.net/p/lemur/wiki/RankLib%20How%20to%20use/> 介绍怎么用

- MART (Multiple Additive Regression Trees, a.k.a. Gradient boosted regression tree) [6]
- RankNet [1]
- RankBoost [2]
- AdaRank [3]
- Coordinate Ascent [4]
- LambdaMART [5]
- ListNet [7]
- Random Forests [8]

#####[Learning to rank (software, datasets)](http://arogozhnikov.github.io/2015/06/26/learning-to-rank-software-datasets.html ) 介绍了数据集和代码，[排序学习研究进展与展望](http://html.rhhz.net/ZDHXBZWB/html/2018-8-1345.htm)中文的介绍数据集和代码

![](picture/code.png)

#####[DeepCTR：易用可扩展的深度学习点击率预测算法库](https://www.infoq.cn/article/iDt28zXQuKqBfG*wCWfc?utm_source=rss&utm_medium=article ) 先搞清楚构建数据集，然后慢慢看算法原理，可以看代码理解原理。

#####[Learning To Rank 研究与应用](https://zhuanlan.zhihu.com/p/26221188 ) 简单的实现例子，建议看代码时先看这个

####算法详解

#####[排序学习调研](http://xtf615.com/2018/12/25/learning-to-rank/) 强烈推荐，里面提到点击模型与ltr的结合，即无偏见的排序学习(Unbiased Learning-to-Rank)！J我认为相对来说音乐领域的无偏较好些，毕竟针对性很强。所以如果看不懂，可以暂时不看先。我感觉这里的第二个论文其实就是我想用的思路，将得到的吸引数值作为训练的分值！只不过这里还融入了预先值不同不是0.5。tf-ranking可用于工业界场景。
由于排序学习主要应用于大数据的场景，因此急需一个扩展性强的工具支持排序学习。到2018年12月之前，都不存在一个开源的工作能够大规模的进行排序学习，工业界一般都会自己搭建分布式系统来进行排序学习。幸运的是，Google于2018年12月初新开源了既支持开箱即用、又可扩展的TensorFlow库TF-Ranking，可用于排序学习。

我感觉这里的第二个论文其实就是我想用的思路，将得到的吸引数值作为训练的分值！只不过这里还融入了预先值不同不是0.5。

第二个方法：http://xtf615.com/2018/12/25/learning-to-rank/#Unbiased-Learning-to-Rank




#####[机器学习算法-L2R进一步了解](https://jiayi797.github.io/2017/09/25/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%AE%97%E6%B3%95-L2R%E8%BF%9B%E4%B8%80%E6%AD%A5%E4%BA%86%E8%A7%A3/)就pairwise中的RankSVM、GBRank和LambdaRank做简要介绍。

#####[前深度学习时代 CTR 预估模型的演化之路：从 LR 到 FFM](https://www.infoq.cn/article/wEdZsdQZ2pIKjs_x3JGJ?utm_source=rss&utm_medium=article ) 讲了很多算法线性模型，交叉模型 J我认为就是所谓的fm，树模型j我认为lambdamart也算，深度模型

#####[大众点评搜索基于知识图谱的深度学习排序实践](https://www.jiqizhixin.com/articles/2019-01-18-20 )

LR可以视作单层单节点的线性网络结构。模型优点是可解释性强。通常而言，良好的解释性是工业界应用实践比较注重的一个指标，它意味着更好的可控性，同时也能指导工程师去分析问题优化模型。但是LR需要依赖大量的人工特征挖掘投入，有限的特征组合自然无法提供较强的表达能力。

FM可以看做是在LR的基础上增加了一部分二阶交叉项。引入自动的交叉特征有助于减少人工挖掘的投入，同时增加模型的非线性，捕捉更多信息。FM能够自动学习两两特征间的关系，但更高量级的特征交叉仍然无法满足。

GBDT是一个Boosting的模型，通过组合多个弱模型逐步拟合残差得到一个强模型。树模型具有天然的优势，能够很好的挖掘组合高阶统计特征，兼具较优的可解释性。GBDT的主要缺陷是依赖连续型的统计特征，对于高维度**稀疏特征（feature index is much larger than the actual number of non zero features）**、时间序列特征不能很好的处理。

#####稀疏特征和离散特征，文章链接见下面url

将每个样本表示成一个10000维的向量 , 
下面是某个样本的向量表示:

0 123:0.122519644924 2097:0.0707110935549

0代表类别标签 , 123和2097就是feature index , 后面的数就是特征表达也就是说这个向量在123和2097维上有数据其余都是0 假如每个维度上都有值 , 本来是在10000个维度上来表达这个样本, 而现在只有两个维度上有效地表达了这个样本, 你说, 特征是不是超级稀疏啊。

**LR,FM,FFM适合处理离散化特征：其中之一就是更稳定。**

特征离散化后，模型会更稳定，比如如果对用户年龄离散化，20-30作为一个区间，不会因为一个用户年龄长了一岁就变成一个完全不同的人。当然处于区间相邻处的样本会刚好相反，所以怎么划分区间是门学问； 

<https://www.zhihu.com/question/31989952> 

**而GBDT适合处理连续化特征：**

**优点：**我们可以把树的生成过程理解成**自动进行多维度的特征组合**的过程，从根结点到叶子节点上的整个路径(多个特征值判断)，才能最终决定一棵树的预测值。另外，对于**连续型特征**的处理，GBDT 可以拆分出一个临界阈值，比如大于 0.027 走左子树，小于等于 0.027（或者 default 值）走右子树，这样很好的规避了人工离散化的问题。

**缺点：**对于**海量的 id 类特征**，GBDT 由于树的深度和棵树限制（防止过拟合），不能有效的存储；另外海量特征在也会存在性能瓶颈，经笔者测试，当 GBDT 的 one hot 特征大于 10 万维时，就必须做分布式的训练才能保证不爆内存。所以 GBDT 通常配合少量的反馈 CTR 特征来表达，这样虽然具有一定的范化能力，但是同时会有**信息损失**，对于头部资源不能有效的表达。

<https://cloud.tencent.com/developer/article/1005416> 

#####[LambdaMART 不太简短之介绍](https://liam.page/2016/07/10/a-not-so-simple-introduction-to-lambdamart/) 讲得不错，好好看看



####业界经验

##### [一个可供参考的搜索引擎排序架构实践案例](https://www.infoq.cn/article/a-search-engine-scheduling-architecture-for-reference/ ) 达观搜索采用listwise算法，position-aware ListMLE(p-ListMLE) ，然后再用点击模型去调整得到的结果。没说怎么调整的。难道是相乘？也没说训练数据怎么生成的。

我们在排序实践中还发现 MLR 无法充分利用用户对搜索结果的点击反馈。俗话说群众的眼睛是雪亮的，用户对不同位置的搜索结果的点击行为直接反应了搜索结果的好坏。**我们根据用户的历史点击记录生成了点击模型，通过点击模型对 MLR 的结果再进行一次调整**。 

 主要通过方式二生成训练数据，自动分析搜索点击日志，分析用户在同一个搜索 session 内对 query 的各种变换、对搜索结果中不同位置的文档的点击行为以及后继的筛选、翻页等行为，综合计算出一个可以标记训练数据的搜索满意度得分。 我们的搜索实践表明，**通过分析搜索点击日志可以实现模型训练数据的自动生成和实时更新**，同时也可以达到比较满意的搜索效果。

#####[1 号店 11.11：机器排序学习在电商搜索中的实战](https://www.infoq.cn/article/yhd-11-11-mlr-ec-search/ ) 为了构建出 <q, p, r-score> 的数值 pair 对，以CTR为基础，为什么不直接用CTR排序呢，这是因为一些新的项目或是一些新的query时，那就没法用CTR数据了，只能用这个模型进行预估。扣除阈值小的就忽略。然后利用点击数去除position bias效应。 
离线训练目标数据的获取有 2 种方法，人工标注和点击日志。**两者都是为了构建出 <q, p, r-score> 的数值 pair 对**，作为机器学习的训练数据集。这里 q 表示用户的查询 query，p 表示通过 q 召回的商品 product，r-score 表示商品 p 在查询 q 条件下的相关性得分。

其中人工标注一般步骤为，根据给定的 query 商品对, 判断商品和 query 是否相关，以及相关强度。该强度值可以用数值序列来表示，常见的是用 5 档评分，如 1- 差，2- 一般，3- 好，4- 优秀，5- 完美。人工标注一方面是标注者的主观判断，会受标注者背景、爱好等因素的影响，另一方面，实际查询的 query 和相关商品数量比较多，所以全部靠人工标注工作量大，一般很少采用。

因此，在我们的实践探索中，寻找获取方便且具有代表性的相关性的度量指标则成为重中之重。经过初期的探索，**我们确定以 CTR 为基础，实现了低成本的 query-product 相关性标注（虽然不完美，但在实际工程中切实可行）。**具体步骤如下：从用户真实的搜索和点击日志中，挖掘出同一个 query 下，商品的排序位置以及在这个位置上的点击数，如 query_1 有 3 个排好序的商品 a, b 和 c，但是 b 得到了更多的点击，那么 b 的相关性可能好于 a。点击数据隐式地反映了相同 query 下搜索结果相关性的好坏。

在搭建相关性数据的过程中，需要避免“展示位置偏见”position bias 现象，即在搜索结果中，排序靠前的结果被点击的概率会大于排序靠后的结果；在我们的训练模型中，如图 2 所示，第 6 位开始的商品点击数相比前 5 位有明显的下降，所以我们会直接去除搜索结果前 5 个商品的点击数 [5]。**J额感觉没啥用啊**

同时在实际场景中，搜索日志也通常含有噪音，只有足够多的点击次数才能体现商品相关性的大小。因此为了提升训练效率和训练效果，我们也针对 Click 数少于某个阈值的情况（即少于某个阈值的点击，我们就直接忽略）进行了测试，分别为 2,3,4,5,6,7,8。经过离线分析，在阈值为 4 的情况下，NDCG 分别有 20%-30% 的提升（这是 NDCG@60 on validation data 和当前线上的 NDCG 相比），效果提升比较明显。

#####[美团搜索的深度学习实践](https://awps-assets.meituan.net/mit-x/slide-bundle-2018-2/48/3-%E7%BE%8E%E5%9B%A2%E6%90%9C%E7%B4%A2%E4%B8%AD%E7%9A%84%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5.pdf) 路线从pairwise到深度学习

![](picture/meituanmodel.png)

#####[深度学习在搜索业务中的探索与实践](https://tech.meituan.com/2019/01/10/deep-learning-in-meituan-hotel-search-engine.html )似乎我的特征数很少。但是看tf[官网](https://vimsky.com/article/3659.html)里子也有特征数比较少用wide，deep的。提示我可以将离散特征如下图一样embedding再传入，这样参数量就大了！！！！但还是特征数少而已。

实际线上使用的连续特征接近400种。

**XGB适合连续特征**，至今为止，**XGB都是数据量较小场景下的主模型，但是树模型优化到后期遇到了瓶颈，比如特征工程收益变小、增大数据量没有额外收益等，此外树模型不适合做在线学习的问题愈发严重。** 

MLP(Multiple-Layer Perception)多层感知机，也就是全连接神经网络。**在很长一段时间内，在特征相同的情况下，MLP效果不如XGB，所以有段时间线上使用的是XGB和MLP的融合模型。后来经过大量的网络结构调整和参数调整，调参经验越来越丰富，MLP才逐步超越XGB**。 

深度排序模型对离散特征的处理有两大类方法，**一类是对离散特征做Embedding，这样离散特征就可以表示成连续的向量放到神经网络中去，另一类是Wide&Deep，把离散特征直接加到Wide侧** 。J我认为就是所谓的预训练和同时训练。

![](picture/embedding.png)

![](picture/meituanmodel2.png)

#####[Multi Task Learning在工业界如何更胜一筹](https://cloud.tencent.com/info/8f4c84190bd753bf2338e84308328935.html) ltr形式上与CTR预估类似，但不同的是输入信息中需要包含具体的商品排序特征（Ranking Features）

任务一是CTR预估任务，这在广告和排序推荐中较为常用，例如预估用户对某电影视频或者音乐的点击率。淘宝会使用CTR来预估用户对某些商品的点击率。公式中主要运用似然函数来表示。输入包括256位的用户表达和商品的embedding，此处的embedding即为用户行为序列中的embedding，两处共享。这些输入信息经过三层网络便可以得到预估结果。

![](picture/ctrali.jpg)

![](picture/ctrfali.jpg)

任务二是L2R（Learning to Rank，也可做LTR）任务，**形式上与CTR预估类似，但不同的是输入信息中需要包含具体的商品排序特征（Ranking Features）**。在CTR预估中将用户表达与embedding做全连接操作，而L2R任务中是将用户表达经过两层网络后和商品特征进行线性相乘。它的优势是最上一层网络容易理解，并且便于查错。而公式中与CTR不同的是加入了权重信息，来表示注重或者忽略哪些行为。比如用户点击商品、购买商品的行为需要较大的权重，而浏览商品后无任何行为可以予以忽视。

![](picture/ltrali.jpg)

![](picture/ltrfali.jpg)

#### 部署问题

##### [当推荐算法开源包多如牛毛，为什么我们还要专门的推荐算法工程师](https://www.cnblogs.com/flclain/p/4211685.html) 介绍了一些场景，包括部署，我感觉意思是离线选择完所有排序，然后线上再根据实时进行微调。目前我就先做线下把，之后再上实时。离线评测的分布式也缓一缓。

算法一般分好几个部分，一般有线下部分和线上部分，**比如很多做法是线下选离线商品池子，线上CTR预估。**这样就涉及到大量的离线评测和线上算法架构。

一般的公司项目的数据量都是单节点运算支撑不起的。百万级有行为用户和几十万的商品都是很常见的，小规模也有几万，**所以要离线评测一般需要并行和分布式**。这些也是很难自动化的。

线上的话架构等模型定下来做LR这种，**有实时的特征（最近30个点击这种）**，**也有离线算好的，一般会对用户大体离线算个离线池子，这样线上压力小**，storm都是不错的选择，反正不需要实时训练，数据增量记录好一天甚至一周一迭代就好。 

#####[用好学习排序 (LTR) ,资讯信息流推荐效果翻倍](https://www.jiqizhixin.com/articles/2019-03-05-13) 部署情况：hive和reddit都有特征存储。使用一天的训练数据的情况下，整个特征空间规模约为30万维左右。我们使用N-2天前的日志做训练，N-1天前的日志做评估，需保证两部分日志的用户群体是一致的。我们采用1500万条样本做训练，300万条样本做评估。J在我这里还不需要利用时间上面进行训练，因为不进行个性化处理。

由于离线和在线都会需要User和item端特征，我们在hive数仓和ssdb集群总中都存储一份，离线负责join hive表，在线负责读取ssdb。 

训练数据按照libsvm格式进行打印。**使用一天的训练数据的情况下，整个特征空间规模约为30万维左右** 。

简单起见，**我们使用N-2天前的日志做训练，N-1天前的日志做评估，需保证两部分日志的用户群体是一致的**。

实际过程中，**我们采用1500万条样本做训练，300万条样本做评估，**训练完成后离线auc为0.79-0.8区间内，在线auc为0.75-0.76区间内，存在一定差距。关于auc可以自行参考技术文章，简单来说auc就是衡量模型将正样本排在负样本前面的概率，即排序能力。 

#####[大众点评搜索基于知识图谱的深度学习排序实践](https://www.jiqizhixin.com/articles/2019-01-18-20 ) 在tensorflow上的开发细节。

我们利用TensorFlow分布式框架训练LambdaDNN模型。如前文所述，Lambda梯度需要对同Query下的样本进行计算，但是正常情况下所有的样本是随机Shuffle到各个Worker的。因此我们需要对样本进行预处理：

- 通过QueryId进行Shuffle，将同一个Query的样本聚合在一起，同一个Query的样本打包进一个TFRecord。
- 由于每次请求Query召回的Doc数不一样，对于可变Size的Query样本在拉取数据进行训练时需要注意，TF会自动补齐Mini-Batch内每个样本大小一致，导致输入数据中存在大量无意义的默认值样本。这里我们提供两点处理方式：
- MR过程中对Key进行处理，使得多个Query的样本聚合在一起，然后在训练的时候进行动态切分。
- 读取到补齐的样本，根据设定的补齐标记获取索引位，去除补齐数据。

为了提升训练效率，我们与基础研发平台数据平台中心紧密协同，一起探索并验证了多项优化操作：

- 将ID类特征的映射等操作一并在预处理中完成，减少多轮Training过程中的重复计算。
- 将样本转TfRecord，利用RecordDataSet方式读取数据并计算处理，Worker的计算性能大概提升了10倍。
- Concat多个Categorical特征，组合成Multi-Hot的Tensor进行一次Embedding_Lookup操作，减少Map操作的同时有助于参数做分片存储计算。
- 稀疏Tensor在计算梯度以及正则化处理时保留索引值，仅对有数值的部分进行更新操作。
- 多个PS服务器间进行分片存储大规模Tensor变量，减少Worker同步更新的通讯压力，减少更新阻塞，达到更平滑的梯度更新效果。

整体下来，对于30亿左右的样本量、上亿级别的特征维度，一轮迭代大概在半小时内完成。适当的增加并行计算的资源，可以达到分钟级的训练任务。

####推荐系统

#####[机器如何「猜你喜欢」？深度学习模型在1688的应用实践](https://www.jiqizhixin.com/articles/2019-01-02-2) 推荐中训练数据没有位置概念，直接曝光点击label为1，曝光未点击则label为0。采用wide&deep。

训练数据来自目标日前七天内用户在1688首页猜你喜欢模块行为数据，曝光点击label为1，曝光未点击则label为0。

![](picture/1688feature.png)
图中滑窗期指目标前1/3/5/7/15/30天的行为窗口。

#####[用好学习排序 (LTR) ,资讯信息流推荐效果翻倍](https://www.jiqizhixin.com/articles/2019-03-05-13) 很多场合，排序学习就是点击率预估，但是问题在于样本的构造，此外有些场合可能更看重cvr！！考虑使用lr作为baseline，并且采用交叉特征，避免稀疏。所有特征按类型划分为离散型、连续型、向量型三种类型。如item类别就是一个离散型特征、item ctr就是一个连续性特征、标题向量就是一个向量型特征
推荐整个流程可以分为召回、排序、重排序这三个阶段，通俗来说，召回就是找到用户可能喜欢的几百条资讯，排序就是对这几百条资讯利用机器学习的方法预估用户对每条资讯的偏好程度，**一般以点击率衡量，所以学习排序在很多情况下等同于点击率预估**，都是将用户最可能点击的资讯优先推给用户；重排序更多考虑业务逻辑，如在推荐结果的多样性、时效性、新颖性等方面进行控制。

**交叉特征**：用户对item类别的一二级类别点击率、用户对item标签的偏好、用户对item素材类型的曝光、点击次数和点击率、最近16个点击item与预测item标题向量的余弦相似度、相似度最大值等。 

**我们没有使用bagof word模型来表示标题，因为这非常稀疏，而是采用标题中关键词的word2vec向量组合生成标题表示**，使用词向量来表示标题极大减少了特征规模，实现上比较方便。标题向量同时需要归一化成单位向量，单位向量的余弦相似度即两个向量的内积，这个优化显著提高了ltr在线模块的性能。

我们将**所有特征按类型划分为离散型、连续型、向量型三种类型。如item类别就是一个离散型特征、item ctr就是一个连续性特征、标题向量就是一个向量型特征**。对于每种特征，其处理方式都会不太一样，对于离散型一般直接根据离散值做feature name，对于连续值我们部分参考youtube wide & deep论文中的等频归一化方法，简单来说加入ctr特征需要等屏成10个特征，即将ctr值按照分布取其10等分点，这10等分点就定义了10个区间，每个区间的样本数都占10%。**需要注意的是，ltr在线部分需要hardcode写死这10个区间来提高特征离散化的效率。** 

#####[深度学习在 Airbnb 中的探索与应用](https://www.infoq.cn/article/ZhYeLX*ylFawD26KpDRo?utm_source=rss&utm_medium=article ) 采用多个模型，有论文记录
JavaServer 处理 query
Spark 记录 logs
Tensorflow 进行模型训练
JavaNNLibrary 线上低延迟预测

#### 其他资源

- [A curated list of awesome information retrieval resources](https://github.com/harpribot/awesome-information-retrieval)
- [A collection of resources for Recommender Systems (RecSys)](https://github.com/chihming/competitive-recsys) 里面有l2r资源


####强化学习

#####[强化学习在携程酒店推荐排序中的应用探索](https://www.infoq.cn/article/uqgproEH7TGBlBCf*cJ7?utm_source=rss&utm_medium=article) J应该未来趋势是用强化学习，其可以逐渐解决position bias。不过目前还是用传统的ltr。

####bert
#####[Bert时代的创新（应用篇）：Bert在NLP各领域的应用进展](https://zhuanlan.zhihu.com/p/68446772)就是把query和document输入bert判断相似性，然后进行排序，但是目前暂时不用这个方法，因为是纯文本，没考虑到用户行为，比如热度等等。
就是把Query和Document输入Bert，利用Bert的深层语言处理能力。作出两者是否相关的判断。如果是Passage级别的短文档检索，其实流程基本和QA是一样的；而如果是长文档检索，则需要增加一个如何处理长文档的技术方案，然后再走Bert去做相关性判断。
短文档效果显著：
![](picture/bert.jpg)
长文档可以看这篇文章：https://arxiv.org/pdf/1903.10972.pdf
大致思路是：在重排序阶段，很难把整个文档输入到Bert中，所以这个工作采取了一个简单的方法：把文档分割成句子，利用Bert判断每个句子和查询Q的相关性，然后累加得分最高的Top N句子（结论是取得分最高的3个句子就够了，再多性能会下降），获得文档和查询Q的相关性得分，这样就将长文档问题转化成了文档部分句子的得分累加的模式。实验表明相对强基准BM25+RM3，使用Bert会有大约10%的效果提升。
归纳一下的话：在搜索领域应用Bert，如果是Passage这种短文档搜索，往往效果有非常巨大的提升；而目前长文档的搜索，使用Bert也能有一定幅度的提升，但是效果不如短文档那么明显，很可能原因在于搜索的长文档处理方式有它自己的特点，还需要继续摸索更合理的更能体现搜索中长文档特性的方法，以进一步发挥Bert的效果。

#### 个性化

#####[Click-Through Rate Prediction with the User Memory Network](https://github.com/rener1199/deep_memory ) 

一个很好的思路，将用户作为两个memory输入，避免像rnn这么复杂，可以用来增强wide&deep模型。可以看做个性化的方向。

#####[Practice on Long Sequential User Behavior Modeling for Click-Through Rate Prediction](https://github.com/UIC-Paper/MIMN)

从模型和服务端解决长用户行为序列的讨论，已经部署在阿里广告系统中。

#####[Lifelong Sequential Modeling with Personalized Memorization for User Response Prediction](https://paperswithcode.com/paper/lifelong-sequential-modeling-with )
还是淘宝解决长时间序列的工作。

#### 抽象搜索

##### [大众点评搜索基于知识图谱的深度学习排序实践](https://www.jiqizhixin.com/articles/2019-01-18-20 ) J这理解为抽象搜索，无需人工运营了！下面embedding理解为对抽象query进行的必要的特征工作，比如粤菜，肯定不能完全匹配，而是要风格，这就可以进行embedding，使得某些商户embedding共现所以相似，然后利用点击行为使得query和商户又有embedding相似。

**加速Embedding特征的收敛**

在我们的深度学习排序模型中，除了Embedding特征，也存在大量Query、Shop和用户维度的强记忆特征，能够很快收敛。而Embedding特征是更为稀疏的弱特征，收敛速度较慢，为了加速Embedding特征的收敛，我们尝试了如下几种方案：

- **低频过滤**：针对出现频率较低的特征进行过滤，可以很大程度上减少参数量，避免过拟合。
- **预训练**：利用多类模型对稀疏Embedding特征进行预训练，然后进入模型进行微调：
  - 通过无监督模型如Word2vec、Fasttext对用户-商户点击关系建模，生成共现关系下的商户Embedding。
  - 利用DSSM等监督模型对Query-商户点击行为建模得到Query和商户的Embedding。
- **Multi-Task**：针对稀疏的Embedding特征，单独设置一个子损失函数，如下图所示。此时Embedding特征的更新依赖两个损失函数的梯度，而子损失函数脱离了对强特征的依赖，可以加快Embedding特征的收敛。



#### 可解释
##### [大众点评搜索基于知识图谱的深度学习排序实践](https://www.jiqizhixin.com/articles/2019-01-18-20 ) 原理就是扰动样本之后进行排序，得出原因，这个可以待做。

关于机器学习模型的可解释性研究，业界已经有了一些探索。Lime(Local Interpretable Model-Agnostic Explanations)是其中的一种，如下图所示：通过对单个样本的特征生成扰动产生近邻样本，观察模型的预测行为。根据这些扰动的数据点距离原始数据的距离分配权重，基于它们学习得到一个可解释的模型和预测结果[5]。举个例子，如果需要解释一个情感分类模型是如何预测“我讨厌这部电影”为负面情感的，我们通过丢掉部分词或者乱序构造一些样本预测情感，最终会发现，决定“我讨厌这部电影”为负面情感的是因为“讨厌”这个词。 

![](picture/interpret.png)



#### IRGAN

#####[IRGAN ：生成对抗网络在搜狗图片搜索排序中的应用](https://zhuanlan.zhihu.com/p/31373052 ) 没看懂，未来研究方向

#### 转化率（CTR）的贝叶斯平滑

#####二项分布的贝叶斯平滑

Beta分布中参数α和β的本质含义，即：α表示点击数，β表示曝光数。因为贝叶斯平滑的具体公式就是：

$$SmoothCTR = \frac{(α + CurrentC - 1)}{( α + β + CurrentI -2)}$$

公式由来：
- 一般来说，点击还是不点击，这是服从伯努利二项分布的。**$r$表示点击率这个参数。**
- 而二项分布的共轭分布就是Beta分布，也就是说，点击率服从Beta分布
- 我们可以从历史数据当中学到历史数据的Beta分布的具体参数，**也就是先验分布$\pi(r)$ （不加任何条件限制的分布），表示点击率这个参数的分布。**
- 共轭先验有一个特性：如果找到一个$\pi(r)$，它是$\pi(x|r)$的共轭先验，那么r的后验分布$\pi(r|x)$和先验分布$\pi(r)$会有一样的形式。
- 这个特性告诉我们：先验分布$\pi(r)$ （也就是历史数据）的分布与后验分布$\pi(r|x)$ （也就是x条件下点击率r的分布）是一个形式的
- 既然我们知道了先验分布$\pi(r)$ （也就是历史数据）的分布是beta分布，那么我们就知道了后验分布$\pi(r|x)$ （也就是x条件下点击率r的分布）分布也是beta分布
- 也就是说 ：先验分布$\pi(r)$ （也就是历史数据） + 后验知识 —-&gt; 后验分布$\pi(r|x)$ （也就是x条件下点击率r的分布）
- 那么接下来我们就需要求解后验分布$\pi(r|x)$的beta分布参数了
- 根据相关的证明（代入贝叶斯公式中推到而成），后验分布的参数$\alpha = \alpha+ C, \beta = \beta + I  - C$
- 其中I是展示数量，C是点击数量，$\alpha和 \beta$ 是历史数据的beta分布参数
- 那么后验分布$\pi(r|x)$ 就是 $beta( \alpha+ C,  \beta + I -  C)$
- **阶段总结**：知道了$\pi(r)$的分布（**因为我们自己定义成了beta分布**），然后知道了二项分布（这是由于点击行为的特点），因而我们知道后验概率也是beta分布了（相关的推导证明）。
- 设连续性随机变量X的概率密度函数为f(x)，若积分绝对收敛，则称积分的值为随机变量的数学期望：即下面那个积分式。
- 如果我们要让点击率误差最小，那么取后验分布的均值（均值即期望，针对square error，$L(\hat{r}, r) = (\hat{r} - r)^2$，估计出来的参数通常用$\hat{r}$表示，其是通过样本计算出来的，参数的真值用$r$表示。 根据相关证明，太复杂就没列出，那么当$\hat{r}$等于$r$在$\pi(r|x)$上的期望时，贝叶斯风险最小。 如果知道$\pi(r|x)$的形式，那么只需求$\int r\pi(r|x)\ dr$就可以了，**实际上发现这就是期望的定义**），就是最好的点击率了！！！！也就是（直接根据beta分布的期望公式求解）：$$mean = \frac{\alpha}{\alpha + \beta} =  \frac{\alpha` + C}{\alpha + \beta + I}$$
- **阶段总结**：针对参数，需要选一个值，使得损失函数最小，即对某广告的点击率而言，需要预估一个点击率，使得对于发生的样本而言，损失函数最小。**发现用后验分布的期望**即可，就是上面那个公式，有C和I加入。
- **阶段总结矩估计**：现在问题聚焦到怎么预估$\alpha$和$\beta$，然后就能根据点击情况和曝光情况计算出贝叶斯平滑后的值了。方法是将所有广告的点击率进行计算后（看做很多独立的样本），取均值和方差，然后让总体的原点矩与样本的原点矩相等，因为对于点击率参数其总体的Beta分布中，可以计算得到，E(x) = α / (α+β)，D(x) = αβ / (α+β)2(α+β+1)，解出参数$\alpha$和$\beta$。**J我可以考虑对于同一query下所有的歌曲或者所有query，进行计算。**

​```python
        self.alpha = (mean+0.000001) * ((mean+0.000001) * (1.000001 - mean) / (var+0.000001) - 1)
        #self.beta = (1-mean)*(mean*(1-mean)/(var+0.000001)-1)
        self.beta = (1.000001 - mean) * ((mean+0.000001) * (1.000001 - mean) / (var+0.000001) - 1)
```



​```python

#!/usr/bin/python
# coding=utf-8
import numpy
import random
import scipy.special as special
import pandas as pd
import time
import math
from math import log
class HyperParam(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
    def sample_from_beta(self, alpha, beta, num, imp_upperbound):
        sample = numpy.random.beta(alpha, beta, num)
        I = []
        C = []
        for click_ratio in sample:
            imp = random.random() * imp_upperbound
            #imp = imp_upperbound
            click = imp * click_ratio
            I.append(imp)
            C.append(click)
#             I.append(int(imp))
#             C.append(int(click))
        return I, C
    # 平滑方式1
    def update_from_data_by_FPI(self, tries, success, iter_num, epsilon):
        '''estimate alpha, beta using fixed point iteration'''
        '''tries ： 展示次数
           success : 点击次数
           iter_num : 迭代次数
           epsilon : 精度
        '''
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(tries, success, self.alpha, self.beta)
            # 当迭代稳定时，停止迭代
            if abs(new_alpha-self.alpha)<epsilon and abs(new_beta-self.beta)<epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta
    def __fixed_point_iteration(self, tries, success, alpha, beta):
        '''fixed point iteration'''
        sumfenzialpha = 0.0
        sumfenzibeta = 0.0
        sumfenmu = 0.0
        # digamma 指伽马函数，是阶乘在实数与复数域的扩展
        sumfenzialpha = special.digamma(success+alpha) - special.digamma(alpha)
        print (sumfenzialpha)
        # for i in range(len(tries)):
        #     sumfenzialpha += (special.digamma(success[i]+alpha) - special.digamma(alpha))
        #     sumfenzibeta += (special.digamma(tries[i]-success[i]+beta) - special.digamma(beta))
        #     sumfenmu += (special.digamma(tries[i]+alpha+beta) - special.digamma(alpha+beta))
        return alpha*(sumfenzialpha/sumfenmu), beta*(sumfenzibeta/sumfenmu)
      
    # 平滑方式2
    def update_from_data_by_moment(self, tries, success):
        '''estimate alpha, beta using moment estimation'''
        # 求均值和方差
        mean, var = self.__compute_moment(tries, success)
        #print 'mean and variance: ', mean, var
        #self.alpha = mean*(mean*(1-mean)/(var+0.000001)-1)
        self.alpha = (mean+0.000001) * ((mean+0.000001) * (1.000001 - mean) / (var+0.000001) - 1)
        #self.beta = (1-mean)*(mean*(1-mean)/(var+0.000001)-1)
        self.beta = (1.000001 - mean) * ((mean+0.000001) * (1.000001 - mean) / (var+0.000001) - 1)
    
    def __compute_moment(self, tries, success):
        # 求均值和方差
        '''moment estimation'''
        ctr_list = []
        # var = 0.0
        mean = (success / tries).mean()
#         mean = [(x*1.0)/y for x, y in zip(success, tries)].mean()
        if len(tries) == 1:
            var = 0
        else:
            var = (success / tries).var()
#             var = [(x*1.0)/y for x, y in zip(success, tries)].var()
        # for i in range(len(tries)):
        #     ctr_list.append(float(success[i])/tries[i])
        # mean = sum(ctr_list)/len(ctr_list)
        # for ctr in ctr_list:
        #     var += pow(ctr-mean, 2)
        return mean, var

def test():
    #设定初始值
    hyper = HyperParam(1, 1)
    #--------sample training data--------
    I, C = hyper.sample_from_beta(10, 1000, 10000, 1000)
    print (I, C)
    #--------estimate parameter using moment estimation--------
    hyper.update_from_data_by_moment(numpy.asarray(I, dtype=numpy.float32), numpy.asarray(C, dtype=numpy.float32))
    print (hyper.alpha, hyper.beta)
    

```

[公式讲解1](https://blog.csdn.net/jinping_shi/article/details/78334362)

[公式讲解2](https://blog.csdn.net/jinping_shi/article/details/53444100)

[新的估计方法](https://www.cnblogs.com/bentuwuying/p/6498370.html ) 此方法反馈是在较小曝光下并不准，更依赖于初始的alpha和beta，因此索性就直接使用矩估计吧

[新的估计方法2](https://blog.csdn.net/mytestmy/article/details/19088519 )

[代码演示](https://jiayi797.github.io/2017/07/09/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%AE%9E%E8%B7%B5-%E8%BD%AC%E5%8C%96%E7%8E%87%E9%A2%84%E4%BC%B0%E4%B9%8B%E8%B4%9D%E5%8F%B6%E6%96%AF%E5%B9%B3%E6%BB%91/ )

##### <u>发现对于本身这个情况，也可以使用贝叶斯平滑，因为对于点击模型而言，也是二项分布，虽然不点击时不为0，但是概率密度函数，还是下面的表达式。两级思考：先用平滑补出alpha和beta，得到点击率，然后再用点击模型从点击率中计算出attractiveness，对于平滑补出的beta-alpha就不进行计算了，毕竟起点都一样了，而且没有实际点击位置意义！</u>

因而可以证明，关于点击模型的点击率的后验概率也是beta分布。

**这么理解，本来得到多少点击多少曝光，然后在进入点击模型之前，我们估算出平滑后有多少先验的点击和曝光，到这里为止，是贝叶斯平滑的范畴没有问题。然后分别针对实际的点击曝光和贝叶斯的点击曝光$\alpha$和$\beta$，进行点击模型计算，这样就能得出平滑后的attractiveness了。**

**唯一的问题是贝叶斯的点击曝光怎么计算attractiveness。J我考虑不进行计算，反正对于同一query下的song都是同样的起点（分子）。此外也符合点击模型定义，毕竟这些贝叶斯的点击曝光是没有具体位置的。**

$$\begin{align}P(\theta|X) &= \frac{P(X|\theta)P(\theta)}{P(X)}=\frac{P(X|\theta)P(\theta)}{\int_0^1P(X|\theta)P(\theta)d\theta}\\&=\frac{C_2^5\theta^3(1-\theta)^2\frac{1}{\mathbf{B}(a,b)}\theta^{a-1}(1-\theta)^{b-1}}{\int_0^1C_2^5\theta^3(1-\theta)^2\frac{1}{\mathbf{B}(a,b)}\theta^{a-1}(1-\theta)^{b-1} d\theta}\\&=\frac{\theta^{(a+3-1)}(1-\theta)^{(b+2-1)}}{\int_0^1 \theta^{(a+3-1)}(1-\theta)^{(b+2-1)} d\theta}\\&=\frac{\theta^{(a+3-1)}(1-\theta)^{(b+2-1)}}{\mathbf{B}(a+3,b+2)}\\&=Beta(\theta | a+3, b+2)\end{align}$$



### 点击模型

#### UBM

- 就是利用这段代码去进行更改：

```scala
    def train(sessions: Seq[(Int, Seq[Int], Seq[Boolean])], maxIter: Int)= {
      val data = sessions.flatMap { case (q, url, click) =>
        val distance = click2distance(click)
        url.zip(distance).map{ case (u, (c, r, d)) =>
          (q, u, r, d, c) //flat and reorder
        }
      }.groupBy{identity}.mapValues{_.length} //key is (query, url, rank, distance, click), value is total number of the key
      for (i <- 0 until maxIter) {
        val updates = data.map { case ((q, u, r, d, c), cnt) =>
          val alpha_uq = alpha(u)(q)
          val gamma_rd = gamma(r)(d)
          val mu_q = mu(q)
          val mu_gamma = mu_q.zip(gamma_rd).map{ case (x, y) => x * y} //2 browsing model the length will become two
          val dot_prod_mu_gamma = mu_gamma.sum //即分母中的mu和gamma对不同m的和
          val Q_m_a1_e1_c1 = mu_gamma.map {
            _ / dot_prod_mu_gamma //就是得到各个意图下查看q中r的情况，that is c within essay
          }
          val Q_m_e1_c1 = Q_m_a1_e1_c1
          val Q_m_c1 = Q_m_a1_e1_c1
          val Q_a1_c1 = 1.0 //计算alpha时即点击时的S的系数
          val Q_a1_c0 = alpha_uq * (1 - dot_prod_mu_gamma) / (1 - alpha_uq * dot_prod_mu_gamma) //计算alpha时未点击时的S的系数
          val Q_m_e1_c0 = mu_gamma.map {
            _ * (1 - alpha_uq) / (1 - alpha_uq * dot_prod_mu_gamma) //计算gamma时A的未点击的S的系数
          }
          val Q_m_c0 = gamma_rd.map { gamma_rdm =>
            1 - alpha_uq * gamma_rdm
          }.zip(mu_q).map { //zip的用处可以这样平行连接，把两个向量中对应位置进行计算
            case (x, y) => x * y / (1 - alpha_uq * dot_prod_mu_gamma) //计算gamma时B的未点击的S的系数
          }
          //返回的是tuple
          val alpha_fraction = if (c) {
            (Q_a1_c1 * cnt, cnt) //点击时系数乘以S值，和该S值（为了之后计算分母）
          } else {
            (Q_a1_c0 * cnt, cnt) //未点击时系数乘以S值，和该S值（为了之后计算分母）
          }
          //这里返回的是seq，其中每一项是tuple
          val gamma_fraction = if (c) {
            Q_m_e1_c1.map{_ * cnt}.zip(Q_m_c1.map {_ * cnt}) //只是返回A,B的值，不像其他进行计算和累计S值（因为后面需要先累加再A/B，不需要计算分母）
          } else {
            Q_m_e1_c0.map {_ * cnt}.zip(Q_m_c0.map {_ * cnt}) //只是返回A,B的值，不像其他进行计算和累计S值（因为后面需要先累加再A/B，后面不需要计算分母）
          }
          //这里返回的是seq，其中每一项是tuple
          val mu_fraction = if (c) {
            Q_m_c1.map { q_m_c => ( q_m_c * cnt, cnt)} //等价于各个意图下查看q中r的情况乘以S值，和该S值（为了之后计算分母）
          } else {
            Q_m_c0.map { q_m_c => (q_m_c * cnt, cnt) } //等价于计算gamma时B的未点击的S的系数乘以S值，和该S值（为了之后计算分母）
          }
          ((q, u, r, d), (alpha_fraction, gamma_fraction, mu_fraction))
        }
      
        // update alpha
        updates.map { case ((q, u, r, d), (af, gf, mf)) =>
          ((u, q), af)
        }.groupBy {_._1}.mapValues { //以此为key，而与此key相同的元素组成value中的list，即返回是一个map：(u, q), List(((u, q) ,af) ...)
          _.map {_._2}.reduce[(Double, Int)] { case (x, y) => //Double, Int就是上文得到的系数乘以S值，和该S值
            (x._1 + y._1, x._2 + y._2) //表示将分子_1和分母_2分别累加
          }
        }.foreach{ case ((u, q), (num, den)) =>
          alpha(u)(q) = num / den //然后更新即可
        }
      
        // update gamma
        updates.map { case ((q, u, r, d), (af, gf, mf)) =>
          ((r, d), gf)
        }.groupBy{_._1}.mapValues { //以此为key，而与此key相同的元素组成value中的list，即返回是一个map：(r, d), List(((r, d) ,gf) ...)
          _.map {_._2}.reduce[Array[(Double, Double)]] { //Double, Double就是上文得到的A,B值，但是因为本身是seq，其中每一项是tuple
            case (xs, ys) => xs.zip(ys).map { //xs和ys代表seq中的各个tuple，重新组合成_1在一个tuple，_2在一个tuple中
              case (x, y) => (x._1 + y._1, x._2 + y._2) //表示将分子_1和分母_2分别累加
            }
          }
        }.foreach { case ((r, d), fs) => //其中fs是上面新生成的tuple：(x._1 + y._1, x._2 + y._2)，存储_1,_2的各自之和
          fs.zipWithIndex.foreach { case ((num, den), m) =>
            gamma(r)(d)(m) = num / den //然后更新即可
          }
        }
        // update mu
        updates.map { case ((q, u, r, d), (af, gf, mf)) =>
          (q, mf)
        }.groupBy{_._1}.mapValues {
          _.map{_._2}.reduce[Array[(Double, Int)]]{
            case (xs, ys) => xs zip ys map {
              case (x, y) => (x._1 + y._1, x._2 + y._2) //因为对于reduce，匿名函数定义了输出形式就如(x._1 + y._1, x._2 + y._2)
            }
          }
        }.foreach { case (q, fs) =>
          fs.zipWithIndex.foreach{ case ((num, den), m) =>
            mu(q)(m) = num / den
          }
        }
      }
        (alpha, gamma, mu)
    }
```



#### 引入intent-确实可以缓解对于浏览形惩罚过于严重的现象，最后的$\alpha$与intent无关

比如0.4,0,8和0.8\*0.4+0.2\*0.8，针对$\alpha$来研究。关键初始值要给其进行计算，**对于intent的预估不能是一样的**。


![](picture/ubm.png)

##### 引入变异系数来做，结合android的曝光6个，就选择小于6个的一般目标较明确，采用所有的最小值，利用播放量前6位的来计算（对歌手名和歌曲名进行聚合，不包含版本）。

变异系数：$c_{v} = \frac {\sigma }{\mu } $
其中$\sigma$表示该组数据的标准差，$\mu$表示该组数据的均值。
该变异系数的取值范围为$[ 0,\sqrt{n-1} ]$，其中$n$为数组中的数据个数。



### [贝叶斯推断](http://noahsnail.com/2018/05/17/2018-05-17-%E8%B4%9D%E5%8F%B6%E6%96%AF%E4%BC%B0%E8%AE%A1%E3%80%81%E6%9C%80%E5%A4%A7%E4%BC%BC%E7%84%B6%E4%BC%B0%E8%AE%A1%E3%80%81%E6%9C%80%E5%A4%A7%E5%90%8E%E9%AA%8C%E6%A6%82%E7%8E%87%E4%BC%B0%E8%AE%A1/ )

好文，讲得很好，已经备份到了rank的picture中。

![](picture/贝叶斯估计、最大似然估计、最大后验概率估计.png)



## ogeek

### 题目

#### 数据格式

| 字段             | 说明                                                         | 数据示例                                   |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------ |
| prefix           | 用户输入（query前缀）                                        | 刘德                                       |
| query_prediction | 根据当前前缀，预测的用户完整需求查询词，最多10条；预测的查询词可能是前缀本身，数字为统计概率 | {“刘德华”:  “0.5”, “刘德华的歌”: “0.3”, …} |
| title            | 文章标题                                                     | 刘德华                                     |
| tag              | 文章内容标签                                                 | 百科                                       |
| label            | 是否点击                                                     | 0或1                                       |

#### 数据格式解释

##### title，tag，query_prediction的关系

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

##### 样本的产生

<https://tianchi.aliyun.com/forum/issueDetail?spm=5176.12586969.1002.168.651d75f0jBhqXn&postId=27417> 

对应每一个prefix，浏览器弹出3个备选的title+tag。

用户可能点击其中一个，也可能一个都不点击。

官方这样的一次场景拆分成3条记录，放到数据集里面。

若果然如此，正样本率天然就低于1/3. 

当前训练集+验证集统计正样本率35%～40%之间，与上述推测略有差异

