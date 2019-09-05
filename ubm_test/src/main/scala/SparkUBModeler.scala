import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
/**
  *
  * author: jomei
  * date: 2018/12/18 16:57
  */
case class Conf(
                 max_queries: Long,
                 max_url_per_query: Int,
                 browsingModes: Int,
                 maxIter: Int,
                 minDelta: Double,
                 numPartitions: Int
               )

class UBMModel(
                val max_queries: Long,
                val max_url_per_query: Int,
                val browsingModes: Int = 1) {

  var gamma: Option[Map[(Int, Int), Array[Double]]] = None
  var alpha: Option[RDD[((Long, Long), Double)]] = None
  var mu: Option[RDD[(Long, Array[Double])]] = None
}

object SparkUBModeler{

  //rdd为两列，一列是tuple(q, u, r, d)，另一列是seq，其中每一项是tuple,具体为(c, cnt)表示点击与否和数量
  def train(train_data: RDD[((Long, Long, Int, Int), Seq[(Boolean, Int)])], //(Long, Long, Int, Int)表示(q, u, r, d)
            conf: Conf) = {
    println("train data: " + train_data.count())

    val Conf(max_queries, max_url_per_query, browsingModes, maxIter, minDelta, numPartitions) = conf
    val sc = train_data.context //rdd.Context() returns the SparkContext that this RDD was created on.
    val gamma = train_data.map{
      case ((q,u,r,d), _) => (r,d)
    }.distinct().collect().map { case (r, d) =>
      ((r, d), Array.fill(browsingModes)(0.5)) //将(r, d)进行聚合，初始值0.5
    }.toMap //因为数量较少，不用rdd保存，用map即可
    // 形式为Map，key为(r,d)，value为0.5
    var gamma_br = sc.broadcast(gamma) //广播
    var alpha = train_data.map { case ((q, u, r, d), _) =>
      (q, u)
    }.distinct(numPartitions).map { //numPartitions表示结果RDD的分区数量
      (_, 0.5)
    }.cache() //初始化alpha，将该rdd保存，数量大，用rdd保存，不能用map，暂时看不到结果，因为是lazy
    //形式为rdd，整个列为tuple((q,u), 0.5)
    var mu = train_data.map {
      _._1._1
    }.distinct().map { q =>
      (q, Array.tabulate(browsingModes) { _ => 1.0 / browsingModes})
    }.cache() //初始化mu，将该rdd保存，数量大，用rdd保存，不能用map，暂时看不到结果，因为是lazy
    //形式为rdd，整个列为tuple(q, Array(0.5))

    var delta = Double.PositiveInfinity
    var joined_data = train_data.map { case ((q, u, r, d), cnts) =>
      ((q, u), (r, d, cnts)) //为了用下面的join操作，先变成(k,v)的pairrdd
    }.join(alpha).map { case ((q, u), ((r, d, cnts), alpha_qu)) => //对两个pairrdd进行操作，join会返回(k, (v1,v2))
      (q, (u, r, d, cnts, alpha_qu)) //map的用处对于pairrdd再进一步交换位置
    }.join(mu) //返回形式(q, ((u, r, d, cnts, alpha_qu), mu_q)

    for (i <- 0 until maxIter if delta > minDelta) {
      val updates = joined_data.flatMap { case (q, ((u, r, d, cnts, alpha_qu), mu_q)) => //为了打破复杂的嵌套，就用flatMap
        val gamma_rd = gamma_br.value(r,d) //value直接取出了对象时Map，可以用((r,d))，也可以用(r,d)索引

        val mu_gamma = mu_q zip gamma_rd map { case (x, y) => x * y} //2 browsing model the length will become two
        val dot_prod_mu_gamma = mu_gamma.sum //即分母中的mu和gamma对不同m的和
        val Q_m_a1_e1_c1 = mu_gamma.map {
          _ / dot_prod_mu_gamma //就是得到各个意图下查看q中r的情况
        }
        val Q_m_e1_c1 = Q_m_a1_e1_c1
        val Q_m_c1 = Q_m_a1_e1_c1
        val Q_a1_c1 = 1.0 //计算alpha时即点击时的S的系数
        val Q_a1_c0 = alpha_qu * (1 - dot_prod_mu_gamma) / (1 - alpha_qu * dot_prod_mu_gamma) //计算alpha时未点击时的S的系数
        val Q_m_e1_c0 = mu_gamma.map {
          _ * (1 - alpha_qu) / (1 - alpha_qu * dot_prod_mu_gamma) //计算gamma时A的未点击的S的系数
        }
        val Q_m_c0 = gamma_rd.map { gamma_rdm =>
          1 - alpha_qu * gamma_rdm
        }.zip(mu_q).map { //zip的用处可以这样平行连接，把两个向量中对应位置进行计算
          case (x, y) => x * y / (1 - alpha_qu * dot_prod_mu_gamma) //计算gamma时B的未点击的S的系数
        }

        val fractions = cnts.map { case (c, cnt) =>
          //这里返回的是rdd，每一行是tuple
          val alpha_fraction = if (c) {
            (Q_a1_c1 * cnt, cnt) //点击时系数乘以S值，和该S值（为了之后计算分母）
          } else {
            (Q_a1_c0 * cnt, cnt) //未点击时系数乘以S值，和该S值（为了之后计算分母）
          }
          //这里返回的是rdd，每一行是seq，其中每一项是tuple
          val gamma_fraction = if (c) { //注意这里tuple是(Am1,Bm1)，另一个tuple是(Am2,Bm2)
            Q_m_e1_c1.map {_ * cnt}.zip(Q_m_c1.map {_ * cnt}) //只是返回A,B的值，不像其他进行计算和累计S值（因为后面需要先累加再A/B，不需要计算分母）
          } else {
            Q_m_e1_c0.map {_ * cnt}.zip(Q_m_c0.map {_ * cnt}) //只是返回A,B的值，不像其他进行计算和累计S值（因为后面需要先累加再A/B，后面不需要计算分母）
          }
          //这里返回的是rdd，每一行是seq，其中每一项是tuple
          val mu_fraction = if (c) { //同上，每一个tuple对应一个m
            Q_m_c1.map { q_m_c => (q_m_c * cnt, cnt)} //等价于各个意图下查看q中r的情况乘以S值，和该S值（为了之后计算分母）
          } else {
            Q_m_c0.map { q_m_c => (q_m_c * cnt, cnt)} //等价于计算gamma时B的未点击的S的系数乘以S值，和该S值（为了之后计算分母）
          }

          (alpha_fraction, gamma_fraction, mu_fraction)
        }
        fractions.map{ case fs => ((q,u,r,d), fs)} //重新包装成一个rdd
      }.cache()

      // update alpha
      val new_alpha = updates.map { case ((q, u, r, d), fractions) =>
        ((q, u), fractions._1)
      }.reduceByKey { case (lhs, rhs) => //rdd有直接分组汇总的方法reducebykey， 定义一个函数，对相同key的相邻之间进行操作(lhs, rhs),因为是pairrdd的方法，因此肯定是()tuple类型操作
        (lhs._1 + rhs._1, lhs._2 + rhs._2) //表示将分子_1和分母_2分别累加，从原来的(tuple,tuple)变成一个tuple，看里面是数不是tuple
      }.mapValues { case (num, den) =>
        num / den //pairrdd的操作mapvalues只修改该value值，得到要更新的值
      }.cache()

      val delta_alpha = alpha.join(new_alpha).values.map{ //把新的alpha值和上一次的alpha值进行join，返回(k, (v1,v2))
        case (x, y) => math.abs(x - y)
      }.max() //取出与原来之间最大的变化量

      // update mu
      val new_mu = updates.map { case ((q, u, r, d), fractions) =>
        (q, fractions._3)
      }.reduceByKey { case (x, y) => //对fraction._3中每行seq进行遍历[(分子m1，分母m1),(分子m2，分母m2)]；[(分子m1，分母m1),(分子m2，分母m2)]
        x zip y map { case (lhs, rhs) => //将两个array组合zip在一起，[((分子m1，分母m1),(分子m1，分母m1)),((分子m2，分母m2),(分子m2，分母m2))]
          (lhs._1 + rhs._1, lhs._2 + rhs._2) //在这里形式为 [(分子m1和，分母m1和),(分子m2和，分母m2和)]
        }
      }.mapValues {
        _.map { case (num, den) => //这里对_中的每一个tuple都进行分子/分母
          num / den
        }//在这里rdd形式为(q,Array(m个数))
      }.cache()

      val delta_mu = mu.join(new_mu).values.map{ //join后的形式为 (q, (Array(0.5),分值)), 只取value则为(Array(0.5),分值)
        case (lhs, rhs) => lhs.zip(rhs).map{ //对每一行进行zip,则(0.5，新分值), (0.5，新分值)成为这样一个seq
          case (x, y) => math.abs(x - y) //然后再进行计算差值
        }.max //取出该行最大的差值
      }.max() //取出全部rdd中最大的差值

      delta = math.max(delta_alpha, delta_mu) //取得两者最大的变化量

      // update gamma
      updates.map { case ((q, u, r, d), fractions) =>
        ((r, d), fractions._2)
      }.reduceByKey { case (x, y) =>
        x zip y map { case (lhs, rhs) =>
          (lhs._1 + rhs._1, lhs._2 + rhs._2)
        }
      }.mapValues {
        _.map { case (num, den) =>
          num / den
        } //将gamma值collect取出作为array形式，每一项是tuple((r,d),Array(m个数)))
      }.collect().foreach { case ((r, d), gamma_rd) =>
        gamma_rd.zipWithIndex.foreach { //对Array(m个数)进行遍历，其序号就代表模型编号
          case (gamma_rdm, m) =>
            delta = math.max(delta, math.abs(gamma(r,d)(m) - gamma_rdm)) //将保存最大的delta
            gamma(r,d)(m) = gamma_rdm //更新
        }
      }
      gamma_br = sc.broadcast(gamma) //重新将其广播，替换掉新值
      //释放rdd空间
      updates.unpersist()
      alpha.unpersist()
      mu.unpersist()
      joined_data.unpersist()
      //更新数值
      alpha = new_alpha
      mu = new_mu
      //利用新的数据生成下一轮需要更新的数据
      joined_data = train_data.map { case ((q, u, r, d), cnts) =>
        ((q, u), (r, d, cnts))
      }.join(alpha).map { case ((q, u), ((r, d, cnts), alpha_qu)) =>
        (q, (u, r, d, cnts, alpha_qu))
      }.join(mu).cache()

      val perplexity = joined_data.flatMap{
        case (q, ((u, r, d, cnts, alpha_qu), mu_q)) =>
          val gamma_rd = gamma_br.value(r, d)

          cnts.map{ case (c, cnt) =>
            val p_c1 = alpha_qu * gamma_rd.zip(mu_q).map{ case (x, y) => x * y}.sum //把Array()和Array()对应相乘为一个Array，然后合计，再乘以alpha
            (if (c) - cnt * log2(p_c1) else - cnt * log2(1-p_c1), cnt) //(分子，分母)
          }
      }.reduce{
        (x, y) => (x._1 + y._1, x._2 + y._2) //(分子之和，分母之和)
      }
      //      logInfo(f"iteration $i: delta = $delta%.6f")
      println(f"iteration $i: delta = $delta%.6f, " +
        f"perplexity = ${math.pow(2, perplexity._1 / perplexity._2)}%.6f")
    }

    val model = new UBMModel(max_queries, max_url_per_query, browsingModes)
    model.gamma = Some(gamma)
    model.alpha = Some(alpha)
    model.mu = Some(mu)

    model
  }

  def log2(x: Double): Double = math.log(x) / math.log(2)

//  def lab(train_data: RDD[((Long, Long, Int, Int), Seq[(Boolean, Int)])]) = {
//    println("world")
//  }
//  def lab2(train_data: Seq[((Long, Long, Int, Int), Seq[(Boolean, Int)])]) = {
//    println("world")
//  }

  def click2distance(cs: Seq[Boolean]): Seq[(Boolean, Int, Int)] = { //返回(点击结果， 该rank， 与前一点击rank间的差值，0表示前一个，一直未点则为本身rank)
    var pre_click = -1
    cs.zipWithIndex.map { case (c, r) =>
      val d = r - pre_click - 1
      if (c) pre_click = r
      (c, r, d) //则返回的也是array数组，即利用zipWithIndex，将序号考虑进来进行操作
    }
  }

  def main(args: Array[String]):Unit = {
    System.setProperty("hadoop.home.dir", "E:\\hadoop-2.6.4")
    val conf = new SparkConf().setAppName("Simple Application")
    val sc = new SparkContext(conf)
//    val data2 = List(
//      ((51.toLong ,12.toLong ,1,0), List((false,1), (true,1))),
//      ((51.toLong ,18.toLong ,7,5), List((false,1)))
//    )
//    val data = List(
//      ((51.toLong ,12.toLong ,1,0), Seq((false,1), (true,1))),
//      ((51.toLong ,18.toLong ,7,5), Seq((false,1)))
//    )
//    val test = sc.parallelize(data)
//    lab2(data) //it's ok //so I think the problem is related to the rdd
//    lab(test)
//    train(test, Conf(24705,10,1,50,0.001,1))
    val rdd = sc.textFile("E:\\sparkdata.txt")
    val sessions = rdd.map(line => line.split(" ")).map(array =>
      (
        array(0).toLong,
        List(array(1).toLong, array(2).toLong, array(3).toLong, array(4).toLong, array(5).toLong, array(6).toLong, array(7).toLong, array(8).toLong, array(9).toLong, array(10).toLong),
        List(array(11).toBoolean, array(12).toBoolean, array(13).toBoolean, array(14).toBoolean, array(15).toBoolean, array(16).toBoolean, array(17).toBoolean, array(18).toBoolean, array(19).toBoolean, array(20).toBoolean)
      )
    )
    val train_data = sessions.flatMap{ case (q, url, click) => //这里输入的是tuple
      val distance = click2distance(click)
      url.zip(distance).map{ case (u, (c, r, d)) => //这里输出的其实是List[tuple]，而使用flatMap后,会把此flat成各个tuple。
        (q, u, r, d, c) //flat and reorder
      }
    }.groupBy{identity}.mapValues{_.size}.map{ case ((q, u, r, d, c), cnt) => //because it's not pairrdd, cannot use gourpbykey
      ((q, u, r, d) ,(c,cnt))
    }.groupByKey.mapValues(_.toSeq)

//    val resultmodel = train(train_data, Conf(24705,10,1,1,0.001,1))
    val resultmodel = train(train_data, Conf(24705,10,1,50,0.001,1))

    val result_alpha = resultmodel.alpha.get.collect
    val result_gamma = resultmodel.gamma.get
    val result_mu = resultmodel.mu.get.collect

    val test_query = 317
    val test_order = result_alpha.map{ case ((q,u), score) =>
      (q, u, score)
    }.filter{case (q, _, _) => q == test_query}.sortBy{case (_, _, score) => score}.reverse

    println ("end")
  }
}
