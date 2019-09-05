import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
/**
  *
  * author: jomei
  * date: 2018/12/19 19:57
  */
object testcreate {
  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "E:\\hadoop-2.6.4")
    val conf = new SparkConf().setAppName("Simple Application")
    val sc = new SparkContext(conf)
    val rdd = sc.textFile("E:\\sparkdata.txt")
//    for (line <- lines.collect) {
//      System.out.println(line)
//    }
    val train_data = rdd.map(line => line.split(" ")).map(array =>
      (
        array(0),
        Seq(array(1), array(2), array(3), array(4), array(5), array(6), array(7), array(8), array(9), array(10)),
        Seq(array(11), array(12), array(13), array(14), array(15), array(16), array(17), array(18), array(19), array(20))
      )
    )

    val train_loc = train_data.collect

//    val pairs = lines.map(x => (x.split(" ")(0), x.split(" ")(1)))
    println("hello")
  }
}
