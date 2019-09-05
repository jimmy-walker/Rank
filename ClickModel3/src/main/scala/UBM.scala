import java.io.File

import org.apache.spark.SparkContext
import org.apache.spark.sql.expressions.{Window, WindowSpec}
import org.apache.spark.sql.{Column, Row, SparkSession}

import scala.reflect.runtime.universe._
import org.apache.spark.sql.functions._
/**
  *
  * author: jomei
  * date: 2019/2/14 18:02
  */
object UBM {
  def main(args: Array[String]): Unit = {
    //====================================================================
    //1)connect to cluster
    val warehouseLocation = new File("spark-warehouse").getAbsolutePath
    val spark = SparkSession
      .builder()
      .appName("Click Model")
      .config("spark.sql.warehouse.dir", warehouseLocation)
      .enableHiveSupport()
      .getOrCreate()
    import spark.implicits._
    val sc: SparkContext = spark.sparkContext
    spark.sparkContext.setLogLevel("ERROR")

    //====================================================================
    //2)initial variable
    val date_start = args(0)
    val date_end = args(1)
    val bayes = args(2).toInt
    val datatable = args(3)
    val thisdatatable = args(4)
    val space = args(5).toDouble
    val rank_click = args(6).toInt
    val prior_position =args(7).toDouble
    val round_bit = args(8).toInt
    val gamma_variable = args(9)
    val percent_rank_value = args(10).toDouble
    val long_rank = args(11).toInt

    println ("parameter:")
    println ("date_start:" + date_start)
    println ("date_end:" + date_end)
    println ("bayes:" + bayes)
    println ("datatable:" + datatable)
    println ("thisdatatable:" + thisdatatable)
    println ("space:" + space)
    println ("rank_click:" + rank_click)
    println ("prior_position:" + prior_position)
    println ("round_bit:" + round_bit)
    println ("gamma_variable:" + gamma_variable)
    println ("percent_rank:" + percent_rank_value)
    println ("long_rank:" + long_rank)

    //-------------------------origin------------------------------------//
    //====================================================================
    //3)acquire the data between date_start and date_end of datatable_song
    //====================================================================
    val sql_sessions_new_combine_read= s"select q, u, r, d, c, s, cnt, choric_singer, songname from "+s"$datatable"+s"_sessions where cdt between "+s"'$date_start' and '$date_end'"
    val df_sessions_new_combine_read = spark.sql(sql_sessions_new_combine_read).
      groupBy("q", "u", "r", "d", "c", "s", "choric_singer", "songname").
      agg(sum("cnt").alias("cnt"))
    df_sessions_new_combine_read.persist()

    val df_alpha = df_sessions_new_combine_read.
      select("q","u").
      distinct().
      withColumn("qu", struct(col("q"), col("u"))).withColumn("alpha", lit(0.5)) //just like tuple
    //update the position of three days combine
    val sql_position_new_read= s"select r, d, numerator, denominator, gamma_origin, gamma_max, gamma_avg, gamma_sum from "+s"$thisdatatable"+s"_position_new where cdt = '$date_end'"
    val df_position_read = spark.sql(sql_position_new_read)

    val gamma_new = df_position_read.
      withColumn("rd", struct(col("r"), col("d"))).
      withColumn("gamma_new", round(col(s"$gamma_variable"), round_bit)).
      select(col("rd"), col("gamma_new")).as[((Int, Int), Double)].collect.toMap
    var gamma_br_new = sc.broadcast(gamma_new)
    val update_new = udf{(r: Int, d: Int, c: Boolean, s: Int, cnt: Long, alpha_uq: Double) =>
      var gamma_rd = 0.0
      try{
        gamma_rd = gamma_br_new.value(r,d)
      } catch {
        case e: NoSuchElementException => gamma_rd = 0.5
      }
      if (s == 1){
        if (!c)
          ((alpha_uq * (1 - gamma_rd) / (1 - alpha_uq * gamma_rd)) * cnt,
            1.0 * cnt,
            (gamma_rd * (1 - alpha_uq) / (1 - alpha_uq * gamma_rd)) * cnt,
            1.0 * cnt)
        else
          (1.0 * cnt,
            1.0 * cnt,
            1.0 * cnt,
            1.0 * cnt) //if all cnt it will be Long, add 1.0, it will declare Double
      }
      else{
        if (!c)
          ((alpha_uq * (1 - gamma_rd) / (1 - alpha_uq * gamma_rd)) * cnt,
            1.0 * cnt,
            0.0,
            0.0)
        else
          (1.0 * cnt,
            1.0 * cnt,
            0.0,
            0.0) //if all cnt it will be Long, add 1.0, it will declare Double
      }
    }
    //====================================================================
    //4)create alpah and gamma
    //====================================================================
    //update the position of three days combine
    val df_song_new_combine = df_sessions_new_combine_read.as("d1").
      join(df_alpha.as("d2"), ($"d1.q" === $"d2.q") && ($"d1.u" === $"d2.u")).
      select($"d1.*", $"d2.alpha").
      withColumn("update", update_new($"r", $"d", $"c", $"s", $"cnt", $"alpha")).
      withColumn("alpha_numerator", $"update._1").
      withColumn("alpha_denominator", $"update._2").
      withColumn("gamma_numerator", $"update._3").
      withColumn("gamma_denominator", $"update._4").
      select($"q", $"u", $"r", $"d", $"c", $"s", $"cnt", $"alpha_numerator", $"alpha_denominator", $"gamma_numerator", $"gamma_denominator", $"choric_singer", $"songname")
    df_song_new_combine.persist()

    //====================================================================
    //5)bayes average
    //====================================================================
    val min_alpha_new_combine = df_song_new_combine.
      groupBy("q", "u", "choric_singer", "songname").
      agg(sum("alpha_numerator").alias("numerator"),
        sum("alpha_denominator").alias("denominator")).
      withColumn("alpha", $"numerator"/$"denominator").
      agg(min("alpha")).first.getDouble(0)

    println("min_alpha_new_combine:" + min_alpha_new_combine)

    //calculate the local impact to calculate the true alpha
    //calculate each q's top 5 alpha value's min alpha
    val w_average_alpha_new_combine = Window.
      partitionBy("q").
      orderBy(desc("denominator"))
    val df_average_alpha_new_combine = df_song_new_combine.
      groupBy("q", "u", "choric_singer", "songname").
      agg(sum("alpha_numerator").alias("numerator"),
        sum("alpha_denominator").alias("denominator"),
        sum(when($"s" === 0, $"alpha_denominator")).as("local")).
      na.fill(0, Seq("local")).
      withColumn("alpha", ($"numerator" - $"local" )/($"denominator" - $"local")).
      na.fill(min_alpha_new_combine, Seq("alpha")).
      withColumn("rank_alpha", rank().over(w_average_alpha_new_combine)).
      filter($"rank_alpha" <=bayes).
      groupBy("q").
      agg(min("alpha").alias("mean_alpha"))
    //use avg instead of min, will produce higher mean_alpha to make yiluzhixia bottom
    //we adjust local to search, cause local is smaller than search
    //calculate each q's top 5 alpha value's avg value nums
    val w_average_nums_new_combine = Window.partitionBy("q").orderBy(desc("click"))
    val df_average_nums_new_combine = df_song_new_combine.
      groupBy("q", "u", "choric_singer", "songname").
      agg(sum("alpha_numerator").alias("numerator"),
        sum("alpha_denominator").alias("denominator"),
        sum(when($"s" =!= 0 && $"c" === true, $"alpha_denominator")).as("click")).
      na.fill(0, Seq("click")).
      withColumn("rank_click", rank().over(w_average_nums_new_combine)).
      filter($"rank_click" <=bayes).
      groupBy("q").
      agg(avg("click").alias("mean_nums"))

    //new mean_num and mean_alpha
    val df_average_alpha_new_combine2 = df_song_new_combine.
      groupBy("q", "u", "choric_singer", "songname").
      agg(sum("alpha_numerator").alias("numerator"),
        sum("alpha_denominator").alias("denominator"),
        sum(when($"s" === 0, $"alpha_denominator")).as("local")).
      na.fill(0, Seq("local")).
      withColumn("alpha", ($"numerator" - $"local" )/($"denominator" - $"local")).
      na.fill(min_alpha_new_combine, Seq("alpha")).
      withColumn("rank_alpha", rank().over(w_average_alpha_new_combine)).
      withColumn("percent_rank_alpha", percent_rank().over(w_average_alpha_new_combine)).
      filter($"rank_alpha" <=bayes || $"percent_rank_alpha" < percent_rank_value).
      filter($"rank_alpha" <= long_rank).
      groupBy("q").
      agg(min("alpha").alias("mean_alpha_2"))

    val w_average_nums_new_combine3 = Window.partitionBy("q").orderBy(desc("denominator"))
    val df_average_nums_new_combine3 = df_song_new_combine.
      groupBy("q", "u", "choric_singer", "songname").
      agg(sum("alpha_numerator").alias("numerator"),
        sum("alpha_denominator").alias("denominator")).
      withColumn("rank_click", rank().over(w_average_nums_new_combine3)).
      filter($"rank_click" <=bayes).
      groupBy("q").
      agg(avg("denominator").alias("mean_nums_3"))

    val df_average_alpha_new_combine3 = df_song_new_combine.
      groupBy("q", "u", "choric_singer", "songname").
      agg(sum("alpha_numerator").alias("numerator"),
        sum("alpha_denominator").alias("denominator")).
      withColumn("alpha", ($"numerator")/($"denominator")).
      na.fill(min_alpha_new_combine, Seq("alpha")).
      withColumn("rank_alpha", rank().over(w_average_alpha_new_combine)).
      withColumn("percent_rank_alpha", percent_rank().over(w_average_alpha_new_combine)).
      filter($"rank_alpha" <=bayes || $"percent_rank_alpha" < percent_rank_value).
      filter($"rank_alpha" <= long_rank).
      groupBy("q").
      agg(min("alpha").alias("mean_alpha_3"))

    //combine two value calculated above to be joined below
    val df_average_new_combine = df_average_alpha_new_combine.
      as("d1").
      join(df_average_nums_new_combine.as("d2"),  $"d1.q"===$"d2.q", "left").
      select($"d1.*", $"d2.mean_nums").
      as("d3").
      join(df_average_alpha_new_combine2.as("d4"),  $"d3.q"===$"d4.q", "left").
      select($"d3.*", $"d4.mean_alpha_2").
      as("d5").
      join(df_average_nums_new_combine3.as("d6"),  $"d5.q"===$"d6.q", "left").
      select($"d5.*", $"d6.mean_nums_3").
      as("d7").
      join(df_average_alpha_new_combine3.as("d8"),  $"d7.q"===$"d8.q", "left").
      select($"d7.*", $"d8.mean_alpha_3")

    //val df_result = df_song_read.groupBy("q", "u", "choric_singer", "songname").agg(sum("alpha_numerator").alias("numerator"), sum("alpha_denominator").alias("denominator"), sum(when($"s" === 0, $"alpha_denominator")).as("local")).na.fill(0, Seq("local")).as("d1").join(df_average.as("d2"),  $"d1.q"===$"d2.q", "left").select($"d1.*", $"d2.mean_alpha", $"d2.mean_nums").withColumn("prior_alpha", when($"mean_alpha"-space>min_alpha, $"mean_alpha"-space).otherwise(min_alpha)).withColumn("alpha", ($"mean_nums"*$"prior_alpha" + $"numerator" - $"local" )/($"mean_nums" + $"denominator" - $"local")).withColumn("alpha_search", ($"numerator" - $"local" )/($"denominator" - $"local")).na.fill(min_alpha, Seq("alpha_search")).withColumn("alpha_direct", ($"numerator")/($"denominator")).select("q", "u", "choric_singer", "songname", "numerator", "denominator", "local", "mean_nums", "mean_alpha", "alpha", "alpha_search", "alpha_direct")
    val df_result_new_combine = df_song_new_combine.
      groupBy("q", "u", "choric_singer", "songname").
      agg(sum("alpha_numerator").alias("numerator"),
        sum("alpha_denominator").alias("denominator"),
        sum(when($"s" =!= 0 && $"c" === true, $"alpha_denominator")).as("click"),
        sum(when($"s" === 0, $"alpha_denominator")).as("local")).
      na.fill(0, Seq("click")).
      na.fill(0, Seq("local")).as("d1").
      join(df_average_new_combine.as("d2"),  $"d1.q"===$"d2.q", "left").
      select($"d1.*", $"d2.mean_alpha", $"d2.mean_nums", $"d2.mean_alpha_2", $"d2.mean_nums_3", $"d2.mean_alpha_3").
      withColumn("prior_alpha", when($"mean_alpha"-space>min_alpha_new_combine, $"mean_alpha"-space).
        otherwise(min_alpha_new_combine)).
      withColumn("prior_alpha_2", when($"mean_alpha_2"-space>min_alpha_new_combine, $"mean_alpha_2"-space).
        otherwise(min_alpha_new_combine)).
      withColumn("prior_alpha_3", when($"mean_alpha_3"-space>min_alpha_new_combine, $"mean_alpha_3"-space).
        otherwise(min_alpha_new_combine)).
      withColumn("alpha", ($"mean_nums"*$"prior_alpha" + $"numerator" - $"local" )/($"mean_nums" + $"denominator" - $"local")).
      withColumn("alpha_t", ($"mean_nums"*$"prior_alpha_2" + $"numerator" - $"local" )/($"mean_nums" + $"denominator" - $"local")).
      withColumn("alpha_c", ($"mean_nums"*$"prior_alpha_3" + $"numerator")/($"mean_nums" + $"denominator")).
      withColumn("alpha_search", ($"numerator" - $"local" )/($"denominator" - $"local")).
      na.fill(min_alpha_new_combine, Seq("alpha_search")).
      withColumn("alpha_v", when(($"denominator"- $"local") > $"mean_nums", $"alpha_search").
        otherwise(($"mean_nums"*$"prior_alpha" + $"numerator" - $"local" )/($"mean_nums" + $"denominator" - $"local"))).
      withColumn("alpha_direct", ($"numerator")/($"denominator")).
      select("q", "u", "choric_singer", "songname",
        "numerator", "denominator", "local", "click",
        "mean_nums", "mean_alpha", "mean_nums_3", "mean_alpha_2", "mean_alpha_3",
        "alpha", "alpha_t", "alpha_c", "alpha_search", "alpha_direct", "alpha_v")

    df_result_new_combine.createOrReplaceTempView("result_new_combine_data")

    val sql_result_new_combine_create= """
create table if not exists """+s"""$thisdatatable"""+"""_result_new_combine_v_2
(
q string,
u string,
choric_singer string,
songname string,
numerator double,
denominator double,
local double,
click double,
mean_nums double,
mean_alpha double,
mean_nums_3 double,
mean_alpha_2 double,
mean_alpha_3 double,
alpha double,
alpha_t double,
alpha_c double,
alpha_search double,
alpha_direct double,
alpha_v double
)
partitioned by (cdt string)
row format delimited fields terminated by '|' lines terminated by '\n' stored as textfile
"""

    spark.sql(sql_result_new_combine_create)

    val sql_result_new_combine_save= s"""
INSERT OVERWRITE TABLE """+s"""$thisdatatable"""+s"""_result_new_combine_v_2 PARTITION(cdt='$date_end') select
                               q, u, choric_singer, songname, numerator, denominator, local, click,
                               mean_nums, mean_alpha, mean_nums_3, mean_alpha_2, mean_alpha_3,
                               alpha, alpha_t, alpha_c, alpha_search, alpha_direct, alpha_v from result_new_combine_data
"""

    spark.sql(sql_result_new_combine_save)

    df_sessions_new_combine_read.unpersist()
    df_song_new_combine.unpersist()

    //-------------------------click------------------------------------//
    //====================================================================
    //3)acquire the data between date_start and date_end of datatable_song
    //====================================================================
    val sql_sessions_new_combine_click_read= s"select q, u, r, d, c, s, cnt, choric_singer, songname from "+s"$datatable"+s"_sessions_click where cdt between "+s"'$date_start' and '$date_end'"
    val df_sessions_new_combine_click_read = spark.sql(sql_sessions_new_combine_click_read).
      groupBy("q", "u", "r", "d", "c", "s", "choric_singer", "songname").
      agg(sum("cnt").alias("cnt")) //cause different date, we recount the sum
    df_sessions_new_combine_click_read.persist()

    val df_alpha_click = df_sessions_new_combine_click_read.
      select("q","u").
      distinct().
      withColumn("qu", struct(col("q"), col("u"))).withColumn("alpha", lit(0.5)) //just like tuple
    //update the position of three days combine
    val sql_position_new_click_read= s"select r, d, numerator, denominator, gamma_origin, gamma_max, gamma_avg, gamma_sum from "+s"$thisdatatable"+s"_position_new_click where cdt = '$date_end'"
    val df_position_click_read = spark.sql(sql_position_new_click_read)

    val gamma_new_click = df_position_click_read.
      withColumn("rd", struct(col("r"), col("d"))).
      withColumn("gamma_new", round(col(s"$gamma_variable"), round_bit)).
      select(col("rd"), col("gamma_new")).as[((Int, Int), Double)].collect.toMap
    var gamma_br_new_click = sc.broadcast(gamma_new_click)
    val update_new_click = udf{(r: Int, d: Int, c: Boolean, s: Int, cnt: Long, alpha_uq: Double) =>
      var gamma_rd = 0.0
      try{
        gamma_rd = gamma_br_new_click.value(r,d)
      } catch {
        case e: NoSuchElementException => gamma_rd = 0.5
      }
      if (s == 1){
        if (!c)
          ((alpha_uq * (1 - gamma_rd) / (1 - alpha_uq * gamma_rd)) * cnt,
            1.0 * cnt,
            (gamma_rd * (1 - alpha_uq) / (1 - alpha_uq * gamma_rd)) * cnt,
            1.0 * cnt)
        else
          (1.0 * cnt,
            1.0 * cnt,
            1.0 * cnt,
            1.0 * cnt) //if all cnt it will be Long, add 1.0, it will declare Double
      }
      else{
        if (!c)
          ((alpha_uq * (1 - gamma_rd) / (1 - alpha_uq * gamma_rd)) * cnt,
            1.0 * cnt,
            0.0,
            0.0)
        else
          (1.0 * cnt,
            1.0 * cnt,
            0.0,
            0.0) //if all cnt it will be Long, add 1.0, it will declare Double
      }
    }
    //====================================================================
    //4)create alpah and gamma
    //====================================================================
    //update the position of three days combine
    val df_song_new_combine_click = df_sessions_new_combine_click_read.as("d1").
      join(df_alpha_click.as("d2"), ($"d1.q" === $"d2.q") && ($"d1.u" === $"d2.u")).
      select($"d1.*", $"d2.alpha").
      withColumn("update", update_new_click($"r", $"d", $"c", $"s", $"cnt", $"alpha")).
      withColumn("alpha_numerator", $"update._1").
      withColumn("alpha_denominator", $"update._2").
      withColumn("gamma_numerator", $"update._3").
      withColumn("gamma_denominator", $"update._4").
      select($"q", $"u", $"r", $"d", $"c", $"s", $"cnt",
        $"alpha_numerator", $"alpha_denominator", $"gamma_numerator", $"gamma_denominator",
        $"choric_singer", $"songname")
    df_song_new_combine_click.persist()
    df_song_new_combine_click.createOrReplaceTempView("song_combine_click_data")

    //====================================================================
    //5)bayes average
    //====================================================================
    val min_alpha_new_combine_click = df_song_new_combine_click.
      groupBy("q", "u", "choric_singer", "songname").
      agg(sum("alpha_numerator").alias("numerator"),
        sum("alpha_denominator").alias("denominator")).
      withColumn("alpha", $"numerator"/$"denominator").
      agg(min("alpha")).first.getDouble(0)

    println("min_alpha_new_combine_click:" + min_alpha_new_combine_click)

    //calculate the local impact to calculate the true alpha
    //calculate each q's top 5 alpha value's min alpha
    val w_average_alpha_new_combine_click = Window.
      partitionBy("q").
      orderBy(desc("denominator"))
    val df_average_alpha_new_combine_click = df_song_new_combine_click.
      groupBy("q", "u", "choric_singer", "songname").
      agg(sum("alpha_numerator").alias("numerator"),
        sum("alpha_denominator").alias("denominator")).
      withColumn("alpha", ($"numerator")/($"denominator")).
      na.fill(min_alpha_new_combine_click, Seq("alpha")).
      withColumn("rank_alpha", rank().over(w_average_alpha_new_combine_click)).
      filter($"rank_alpha" <=bayes).
      groupBy("q").
      agg(min("alpha").alias("mean_alpha"))
    //use avg instead of min, will produce higher mean_alpha to make yiluzhixia bottom
    //we adjust local to search, cause local is smaller than search
    //calculate each q's top 5 alpha value's avg value nums
    val w_average_nums_new_combine_click = Window.partitionBy("q").orderBy(desc("click"))
    val df_average_nums_new_combine_click = df_song_new_combine_click.
      groupBy("q", "u", "choric_singer", "songname").
      agg(sum("alpha_numerator").alias("numerator"),
        sum("alpha_denominator").alias("denominator"),
        sum(when($"s" =!= 0 && $"c" === true, $"alpha_denominator")).as("click")). //for click, c will always be true, but it doesn't matter
      na.fill(0, Seq("click")).
      withColumn("rank_click", rank().over(w_average_nums_new_combine_click)).
      filter($"rank_click" <=bayes).
      groupBy("q").
      agg(avg("click").alias("mean_nums"))

    //new mean_num and mean_alpha
    val df_average_alpha_new_combine_click_2 = df_song_new_combine_click.
      groupBy("q", "u", "choric_singer", "songname").
      agg(sum("alpha_numerator").alias("numerator"),
        sum("alpha_denominator").alias("denominator")).
      withColumn("alpha", ($"numerator")/($"denominator")).
      na.fill(min_alpha_new_combine_click, Seq("alpha")).
      withColumn("rank_alpha", rank().over(w_average_alpha_new_combine_click)).
      withColumn("percent_rank_alpha", percent_rank().over(w_average_alpha_new_combine_click)).
      filter($"rank_alpha" <=bayes || $"percent_rank_alpha" < percent_rank_value).
      filter($"rank_alpha" <= long_rank).
      groupBy("q").
      agg(min("alpha").alias("mean_alpha_2"))

    val df_average_nums_new_combine_click_2 = df_song_new_combine_click.
      groupBy("q", "u", "choric_singer", "songname").
      agg(sum("alpha_numerator").alias("numerator"),
        sum("alpha_denominator").alias("denominator"),
        sum(when($"s" =!= 0 && $"c" === true, $"alpha_denominator")).as("click")). //for click, c will always be true, but it doesn't matter
      na.fill(0, Seq("click")).
      withColumn("rank_click", rank().over(w_average_alpha_new_combine_click)).
      filter($"rank_click" <=bayes).
      groupBy("q").
      agg(avg("click").alias("mean_nums_2"))

    //combine two value calculated above to be joined below
    val df_average_new_combine_click = df_average_alpha_new_combine_click.as("d1").
      join(df_average_nums_new_combine_click.as("d2"),  $"d1.q"===$"d2.q", "left").
      select($"d1.*", $"d2.mean_nums").
      as("d3").
      join(df_average_alpha_new_combine_click_2.as("d4"),  $"d3.q"===$"d4.q", "left").
      select($"d3.*", $"d4.mean_alpha_2").
      as("d5").
      join(df_average_nums_new_combine_click_2.as("d6"),  $"d5.q"===$"d6.q", "left").
      select($"d5.*", $"d6.mean_nums_2")

    //get mixsongid's complete play statistic
    val sql_song_playstatus_click= """
select
    a.*,
    b.play_count,
    b.play_count_30,
    b.play_count_60,
    b.play_count_all
from song_combine_click_data a
left join (
    select
            mixsongid,
            sum(play_count) as play_count,
            sum(case when regexp_extract(spttag,'[0-9]+$',0)>30 then play_count else 0 end) as play_count_30,
            sum(case when regexp_extract(spttag,'[0-9]+$',0)>60 then play_count else 0 end) as play_count_60,
            sum(case when status='完整播放' then play_count else 0 end) as play_count_all
    from dsl.restruct_dwm_list_all_play_d
    where dt = """ + s"""'$date_end'""" + """
            and pt='android'
            and sty='音频'
            and status<>'播放错误'
            and (fo rlike '搜索/' or fo rlike '^(/)?搜索$')
    group by
            mixsongid
) b
on a.u = b.mixsongid
"""

    val df_song_new_combine_click_playcount = spark.sql(sql_song_playstatus_click)

    val df_result_new_combine_click = df_song_new_combine_click_playcount.
      groupBy("q", "u", "choric_singer", "songname", "play_count", "play_count_30", "play_count_60", "play_count_all").
      agg(sum("alpha_numerator").alias("numerator"),
        sum("alpha_denominator").alias("denominator"),
        sum(when($"s" =!= 0 && $"c" === true, $"alpha_denominator")).as("click")).//for click, c will always be true, but it doesn't matter
      na.fill(0, Seq("click")).
      as("d1").
      join(df_average_new_combine_click.as("d2"),  $"d1.q"===$"d2.q", "left").
      select($"d1.*", $"d2.mean_alpha", $"d2.mean_nums", $"d2.mean_alpha_2", $"d2.mean_nums_2").
      withColumn("prior_alpha", when($"mean_alpha"-space>min_alpha_new_combine_click, $"mean_alpha"-space).
        otherwise(min_alpha_new_combine_click)).
      withColumn("prior_alpha_2", when($"mean_alpha_2"-space>min_alpha_new_combine_click, $"mean_alpha_2"-space).
        otherwise(min_alpha_new_combine_click)).
      withColumn("alpha", ($"mean_nums"*$"prior_alpha" + $"numerator")/($"mean_nums" + $"denominator")).
      withColumn("alpha_t", ($"mean_nums_2"*$"prior_alpha_2" + $"numerator")/($"mean_nums_2" + $"denominator")).
      withColumn("alpha_search", ($"numerator")/($"denominator")).
      na.fill(min_alpha_new_combine_click, Seq("alpha_search")).
      withColumn("alpha_v", when(($"denominator") > $"mean_nums", $"alpha_search"). //add threshold to make the threshold bigger
        otherwise($"alpha")).
      withColumn("alpha_r", $"alpha"*$"play_count_30"/$"play_count").
      select("q", "u", "choric_singer", "songname", "numerator", "denominator",
        "click",
        "mean_nums", "mean_alpha", "mean_nums_2", "mean_alpha_2",
        "play_count", "play_count_30", "play_count_60", "play_count_all",
        "alpha", "alpha_t", "alpha_search", "alpha_v", "alpha_r")

    df_result_new_combine_click.createOrReplaceTempView("result_new_combine_click_data")

    val sql_result_new_combine_click_create= """
create table if not exists """+s"""$thisdatatable"""+"""_result_new_combine_v_2_click
(
q string,
u string,
choric_singer string,
songname string,
numerator double,
denominator double,
click double,
mean_nums double,
mean_alpha double,
mean_nums_2 double,
mean_alpha_2 double,
alpha double,
alpha_t double,
alpha_search double,
alpha_v double,
alpha_r double,
play_count bigint,
play_count_30 bigint,
play_count_60 bigint,
play_count_all bigint
)
partitioned by (cdt string)
row format delimited fields terminated by '|' lines terminated by '\n' stored as textfile
"""

    spark.sql(sql_result_new_combine_click_create)

    val sql_result_new_combine_click_save= s"""
INSERT OVERWRITE TABLE """+s"""$thisdatatable"""+s"""_result_new_combine_v_2_click PARTITION(cdt='$date_end') select
                               q, u, choric_singer, songname, numerator, denominator, click,
                               mean_nums, mean_alpha, mean_nums_2, mean_alpha_2,
                               alpha, alpha_t, alpha_search, alpha_v, alpha_r,
                               play_count, play_count_30, play_count_60, play_count_all from result_new_combine_click_data
"""

    spark.sql(sql_result_new_combine_click_save)

    df_sessions_new_combine_click_read.unpersist()
    df_song_new_combine_click.unpersist()

    //10)end
    spark.stop() //to avoid ERROR LiveListenerBus: SparkListenerBus has already stopped! Dropping event SparkListenerExecutorMetricsUpdate


  }
}
