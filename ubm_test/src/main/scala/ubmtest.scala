/**
  *
  * author: jomei
  * date: 2018/12/14 17:02
  */
class LocalModeler {
  var browsingModes = 2 //猜测是离散变量m，这里其实默认是一种意图而已
  var max_urls = 30
  var max_queries = 2
//  var max_urls = 200000 //202586
//  var max_queries = 20000 //24705
  var max_url_per_query = 10 //自行定义
  val alpha = Array.tabulate(max_urls, max_queries) { (u, q) => 0.5 } // q -> u -> alpha
  val gamma = Array.tabulate(max_url_per_query, max_url_per_query, browsingModes) {
    (r, d, m) => if (d <= r) 0.5 else 0.0
  } // r -> d -> m -> gamma
//  val mu = Array.tabulate(max_queries, browsingModes) { (q, m) => 1.0 / browsingModes } // q -> m -> mu
  val mu = Array( Array(0.4, 0.6), Array(0.6, 0.4) )

  def click2distance(cs: Seq[Boolean]): Seq[(Boolean, Int, Int)] = { //返回(点击结果， 该rank， 与前一点击rank间的差值，0表示前一个，一直未点则为本身rank)
    var pre_click = -1
    cs.zipWithIndex.map { case (c, r) =>
      val d = r - pre_click - 1
      if (c) pre_click = r
      (c, r, d)
    }
  }

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
}

object ubmtest {
  def main(args: Array[String]): Unit = {
    val maxIter = 50;
    val ubm = new LocalModeler;
    val sessions = List(
      (0, List(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), List(false, false, false, false, true, true, false, false, false, false)),
      (1, List(11, 12, 13, 14, 15, 16, 17, 18, 19, 20), List(true, false, false, false, false, false, false, false, false, false)),
      (1, List(11, 12, 13, 14, 15, 16, 17, 18, 19, 20), List(true, true, false, false, false, false, false, false, false, false))
    )

    //test
//    val data = sessions.flatMap{ case (q, url, click) =>
//      val distance = ubm.click2distance(click)
//      url.zip(distance).map { case (u, (c, r, d)) =>
//        (q, u, r, d, c) //flat and reorder
//      }
//    }.groupBy{identity}.mapValues{_.length} //key is (query, url, rank, distance, click), value is total number of the key
//    for (i <- 0 until maxIter) {
//      val updates = data.map { case ((q, u, r, d, c), cnt) =>
//        val alpha_uq = ubm.alpha(u)(q)
//        val gamma_rd = ubm.gamma(r)(d)
//        val mu_q = ubm.mu(q)
//        val mu_gamma = mu_q.zip(gamma_rd).map{ case (x, y) => x * y} //2 browsing model the length will become two
//      val dot_prod_mu_gamma = mu_gamma.sum //即分母中的mu和gamma对不同m的和
//      val Q_m_a1_e1_c1 = mu_gamma.map {
//        _ / dot_prod_mu_gamma //就是得到各个意图下查看q中r的情况
//      }
//        val Q_m_e1_c1 = Q_m_a1_e1_c1
//        val Q_m_c1 = Q_m_a1_e1_c1
//        val Q_a1_c1 = 1.0 //计算alpha时即点击时的S的系数
//      val Q_a1_c0 = alpha_uq * (1 - dot_prod_mu_gamma) / (1 - alpha_uq * dot_prod_mu_gamma) //计算alpha时未点击时的S的系数
//      val Q_m_e1_c0 = mu_gamma.map {
//        _ * (1 - alpha_uq) / (1 - alpha_uq * dot_prod_mu_gamma) //计算gamma时A的未点击的S的系数
//      }
//        val Q_m_c0 = gamma_rd.map { gamma_rdm =>
//          1 - alpha_uq * gamma_rdm
//        }.zip(mu_q).map { //zip的用处可以这样平行连接，把两个向量中对应位置进行计算
//          case (x, y) => x * y / (1 - alpha_uq * dot_prod_mu_gamma) //计算gamma时B的未点击的S的系数
//        }
//        //返回的是tuple
//        val alpha_fraction = if (c) {
//          (Q_a1_c1 * cnt, cnt) //点击时系数乘以S值，和该S值（为了之后计算分母）
//        } else {
//          (Q_a1_c0 * cnt, cnt) //未点击时系数乘以S值，和该S值（为了之后计算分母）
//        }
//        //这里返回的是seq，其中每一项是tuple
//        val gamma_fraction = if (c) {
//          Q_m_e1_c1.map{_ * cnt}.zip(Q_m_c1.map {_ * cnt}) //只是返回A,B的值，不像其他进行计算和累计S值（因为后面需要先累加再A/B，不需要计算分母）
//        } else {
//          Q_m_e1_c0.map {
//            _ * cnt
//          }.zip(Q_m_c0.map { //只是返回A,B的值，不像其他进行计算和累计S值（因为后面需要先累加再A/B，后面不需要计算分母）
//            _ * cnt
//          })
//        }
//        //这里返回的是seq，其中每一项是tuple
//        val mu_fraction = if (c) {
//          Q_m_c1.map { q_m_c => ( q_m_c * cnt, cnt)} //等价于各个意图下查看q中r的情况乘以S值，和该S值（为了之后计算分母）
//        } else {
//          Q_m_c0.map { q_m_c => (q_m_c * cnt, cnt) } //等价于计算gamma时B的未点击的S的系数乘以S值，和该S值（为了之后计算分母）
//        }
//
//        ((q, u, r, d), (alpha_fraction, gamma_fraction, mu_fraction))
//      }
//
//      // update alpha
//      updates.map { case ((q, u, r, d), (af, gf, mf)) =>
//        ((u, q), af)
//      }.groupBy {_._1}.mapValues { //以此为key，而与此key相同的元素组成value中的list，即返回是一个map：(u, q), List(((u, q) ,af) ...)
//        _.map {_._2}.reduce[(Double, Int)] { case (x, y) => //Double, Int就是上文得到的系数乘以S值，和该S值
//          (x._1 + y._1, x._2 + y._2) //表示将分子_1和分母_2分别累加
//        }
//      }.foreach{ case ((u, q), (num, den)) =>
//        ubm.alpha(u)(q) = num / den //然后更新即可
//      }
//
//      // update gamma
//      updates.map { case ((q, u, r, d), (af, gf, mf)) =>
//        ((r, d), gf)
//      }.groupBy{_._1}.mapValues { //以此为key，而与此key相同的元素组成value中的list，即返回是一个map：(r, d), List(((r, d) ,gf) ...)
//        _.map {_._2}.reduce[Array[(Double, Double)]] { //Double, Double就是上文得到的A,B值，但是因为本身是seq，其中每一项是tuple
//          case (xs, ys) => xs.zip(ys).map { //xs和ys代表seq中的各个tuple，重新组合成_1在一个tuple，_2在一个tuple中
//            case (x, y) => (x._1 + y._1, x._2 + y._2) //表示将分子_1和分母_2分别累加
//          }
//        }
//      }.foreach { case ((r, d), fs) => //其中fs是上面新生成的tuple：(x._1 + y._1, x._2 + y._2)，存储_1,_2的各自之和
//        fs.zipWithIndex.foreach { case ((num, den), m) =>
//          ubm.gamma(r)(d)(m) = num / den //然后更新即可
//        }
//      }
//      // update mu
//      updates.map { case ((q, u, r, d), (af, gf, mf)) =>
//        (q, mf)
//      }.groupBy{_._1}.mapValues {
//        _.map{_._2}.reduce[Array[(Double, Int)]]{
//          case (xs, ys) => xs zip ys map {
//            case (x, y) => (x._1 + y._1, x._2 + y._2) //因为对于reduce，匿名函数定义了输出形式就如(x._1 + y._1, x._2 + y._2)
//          }
//        }
//      }.foreach { case (q, fs) =>
//        fs.zipWithIndex.foreach{ case ((num, den), m) =>
//          ubm.mu(q)(m) = num / den
//        }
//      }
////      (ubm.alpha, ubm.gamma, ubm.mu)
//    }
    val list1 = scala.io.Source.fromFile("E:\\sparkdata.txt").getLines.map(line => line.split(" ")).map( array =>
      (
        array(0).toInt,
        List(array(1).toInt, array(2).toInt, array(3).toInt, array(4).toInt, array(5).toInt, array(6).toInt, array(7).toInt, array(8).toInt, array(9).toInt, array(10).toInt),
        List(array(11).toBoolean, array(12).toBoolean, array(13).toBoolean, array(14).toBoolean, array(15).toBoolean, array(16).toBoolean, array(17).toBoolean, array(18).toBoolean, array(19).toBoolean, array(20).toBoolean)
      )
    ).toList
//    val t = ubm.train(list1, 1)
    val t = ubm.train(sessions, maxIter+10)
    println("hello")
  }

}