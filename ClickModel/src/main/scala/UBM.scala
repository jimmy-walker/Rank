import java.io.File

import org.apache.spark.SparkContext
import org.apache.spark.sql.expressions.{Window, WindowSpec}
import org.apache.spark.sql.{Column, Row, SparkSession}

import scala.reflect.runtime.universe._
import org.apache.spark.sql.functions._
/**
  *
  * author: jomei
  * date: 2019/2/14 9:13
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
    val edition = args(2)
    val lvt1 = args(3)
    val lvt2 = args(4)
    val threshold = args(5).toInt
    val maxrank = args(6).toInt
    val bayes = args(7).toInt
    val datatable = args(8)
    val thisdatatable = args(9)
    val date_before = args(10)
    val space = args(11).toDouble
    val prior_position = args(12).toDouble
    val round_bit = args(13).toInt
    val gamma_variable = args(14)
    val restart_switch = args(15)

    println ("parameter:")
    println ("date_start:" + date_start)
    println ("date_end:" + date_end)
    println ("edition:" + edition)
    println ("lvt1:" + lvt1)
    println ("lvt2:" + lvt2)
    println ("threshold:" + threshold)
    println ("maxrank:" + maxrank)
    println ("bayes:" + bayes)
    println ("datatable:" + datatable)
    println ("thisdatatable:" + thisdatatable)
    println ("data_before:" + date_before)
    println ("space:" + space)
    println ("prior_position:" + prior_position)
    println ("round_bit:" + round_bit)
    println ("gamma_variable:" + gamma_variable)
    println ("restart_switch:" + restart_switch)

    //====================================================================
    //====================================================================
    //3)raw data
    val sql_raw = """
select
    a,
    scid_albumid,
    coalesce(cast(ivar2 as bigint),0) ivar2,
    case when cast(tv1 as int) is not null then tv1
        else coalesce(cast(tv as int),'unknown') end as tv,
    coalesce(trim(fo),'unknown') fo,
    lower(kw) as kw,
    mid,
    i,
    lvt,
    svar2,
    case when trim(sty)='音频' then '音频'
        when trim(sty)='视频' then '视频'
        else 'unknown' end sty,
    case when trim(fs)='完整播放' then '完整播放'
        when trim(fs)='播放错误' then '播放错误'
        when trim(fs) in ('被终止','播放中退出','暂停时退出','被?止') then '未完整播放'
        else '未知播放状态' end status,
    case when abs(coalesce(cast(st as bigint),0))>=10000000 then abs(coalesce(cast(st as bigint),0))/1000
        else abs(coalesce(cast(st as bigint),0)) end st,
    case when coalesce(cast(spt as bigint),0)>=10000000 then coalesce(cast(spt as bigint),0)/1000
        else coalesce(cast(spt as bigint),0) end spt
from ddl.dt_list_ard_d
where
(dt between """+s"""'$date_start'"""+""" and """+s"""'$date_end'"""+"""
    and lvt between """+s"""'$lvt1'"""+""" and """+s"""'$lvt2'"""+"""
    and a='3'
    and action='search'
    and fs='有搜索结果'
    and sct='歌曲'
    and coalesce(CAST(tv1 AS INT),CAST(tv AS INT))>="""+s"""'$edition'"""+""")
OR
(dt between """+s"""'$date_start'"""+""" and """+s"""'$date_end'"""+"""
    and lvt between """+s"""'$lvt1'"""+""" and """+s"""'$lvt2'"""+"""
    and a in ('9697', '10650', '10654')
    and scid_albumid IS NOT NULL
    and CAST(ivar2 AS BIGINT) > 0
    and action='search'
    and coalesce(CAST(tv1 AS INT),CAST(tv AS INT))>="""+s"""'$edition'"""+""")
OR
(dt between """+s"""'$date_start'"""+""" and """+s"""'$date_end'"""+"""
    and lvt between """+s"""'$lvt1'"""+""" and """+s"""'$lvt2'"""+"""
    and a ='14124'
    and scid_albumid IS NOT NULL
    and action='exposure'
    and coalesce(CAST(tv1 AS INT),CAST(tv AS INT))>="""+s"""'$edition'"""+"""
    and fo regexp '/搜索/[^/]+/(?:单曲)')
OR
(dt between """+s"""'$date_start'"""+""" and """+s"""'$date_end'"""+"""
    and lvt between """+s"""'$lvt1'"""+""" and """+s"""'$lvt2'"""+"""
    and a='4'
    and scid_albumid IS NOT NULL
    and action='play'
    and trim(fs)<>'播放错误'
    and trim(ivar10)='主动播放'
    and (trim(reason)<>'1' or reason is null)
    and coalesce(CAST(tv1 AS INT),CAST(tv AS INT))>="""+s"""'$edition'"""+"""
    and ((trim(sty)='音频' and fo regexp '/搜索/[^/]+/(?:单曲|歌曲)')
            or (trim(sty)='视频' and fo regexp '/搜索/[^/]+$')))
"""

    val df_raw = spark.sql(sql_raw).distinct //to avoid duplicate like 24's44615593561466220233863321800162079931
    df_raw.createOrReplaceTempView("raw_data")

    val sql_raw_create= """
create table if not exists """+s"""$datatable"""+"""_raw
(
a string,
scid_albumid string,
ivar2 bigint,
tv string,
fo string,
kw string,
mid string,
i string,
lvt string,
svar2 string,
sty string,
status string,
st double,
spt double
)
partitioned by (cdt string)
row format delimited fields terminated by '|' lines terminated by '\n' stored as textfile
"""
    //"unknown" and int so become string; double and bigint so become double

    spark.sql(sql_raw_create)

    val sql_raw_save= s"""
INSERT OVERWRITE TABLE """+s"""$datatable"""+s"""_raw PARTITION(cdt='$date_end') select * from raw_data
"""

    spark.sql(sql_raw_save)
    //====================================================================
    //4)Modify data by status
    val sql_raw_read= s"select a, scid_albumid, ivar2, tv, fo, kw, mid, i, lvt, svar2, sty, status, st, spt from "+s"$datatable"+s"_raw where cdt = '$date_end'"
    val df_raw_read = spark.sql(sql_raw_read)
    //two: modify data by status
    val count_time = udf{(tv: String, spt: Double, ivar2: Long, st: Double) =>
      val t = (tv.toDouble, spt.toDouble, ivar2.toDouble) match {
        case (a, b, c) if a < 7900 && b < 0 => 0
        case (a, b, c) if a < 7900 && b >= 0 => b
        case (a, b, c) if a >= 7900 && c < 0 => 0
        case (a, b, c) if a >= 7900 && c >= 0 => c
        case _ => 0
      }
      if (t > 50*(st.toDouble)) ((st.toDouble)/1000).formatted("%.6f") else (t/1000).formatted("%.6f")
    }
    //a=3中kw有搜索原始词（未强纠词）
    //a="9697", "10650", "10654"中，fo是强纠词/搜索实际对应的词，svar2是原词（未强纠词）
    //a=14124，fo是搜索原始词
    //a=4中fo是强纠词/搜索实际对应的词
    //从所有的字段的fo中提取出一个query。需要注意对于点击的a需要考虑强纠是否进行替换.有强纠的，那么query就是原词（为了匹配下面的a=4寻找），新建一个列表保存真实词
    //最后得到query（原词等）表示用户想搜的包括强纠前的词以及用户输入正确的词，改过埋点4后不需要了，直接用keyword中间变量废弃
    //a="9697", "10650", "10654"中correct表示有真实强纠词或""，为空则表示未发生强纠，改过埋点4变成都是强纠后的词后而非原词，不需要有原词，中间变量废弃
    //keyword表示该条埋点的关键词，对于a="9697", "10650", "10654"和a=4来说都是搜索结果的真实词
    //origin表示强纠前的原词或null，中间变量废弃
    //ivar表示点击的位置，提前置于0，是为了避免其他记录有播放时长，时间太大。
    val df_edit = df_raw_read.
      withColumn("keyword", when($"kw".isNull,
        when($"a" === "14124",
          regexp_extract($"fo","/搜索/([^/]+)/(?:单曲)",1)).
          when($"sty" === "音频",
            regexp_extract($"fo","/搜索/([^/]+)/(?:单曲|歌曲)",1)).
          when($"sty" === "视频",
            regexp_extract($"fo","/搜索/([^/]+)$",1)).
          otherwise(regexp_extract($"fo","/搜索/([^/]+)$",1))).
        otherwise($"kw")).
      withColumn("spt_cnt", when($"a" === "4", count_time($"tv", $"spt", $"ivar2", $"st")).otherwise(null)).
      withColumn("valid", when($"a" === "4",
        when($"status" === "完整播放" || $"spt_cnt" >= 30, 1).
          otherwise(0)).
        otherwise(null)).
      withColumn("ivar", when($"a".isin("9697", "10650", "10654"), $"ivar2").otherwise(0)).
      select("a", "mid", "i", "scid_albumid", "lvt", "keyword", "valid", "ivar")

    df_edit.createOrReplaceTempView("edit_data")

    val sql_edit_create= """
create table if not exists """+s"""$datatable"""+"""_edit
(
a string,
mid string,
i string,
scid_albumid string,
lvt string,
keyword string,
valid int,
ivar bigint
)
partitioned by (cdt string)
row format delimited fields terminated by '|' lines terminated by '\n' stored as textfile
"""
    //"unknown" and int so become string; double and bigint so become double

    spark.sql(sql_edit_create)

    val sql_edit_save= s"""
INSERT OVERWRITE TABLE """+s"""$datatable"""+s"""_edit PARTITION(cdt='$date_end') select * from edit_data
"""

    spark.sql(sql_edit_save)
    //====================================================================
    //5)attach play records to search record
    val sql_edit_read= s"select a, mid, i, scid_albumid, lvt, keyword, valid, ivar from "+s"$datatable"+s"_edit where cdt = '$date_end'"
    val df_edit_read = spark.sql(sql_edit_read)
    //three: create play session
    //1.对于a=3标注顺序tag从1开始,row_number.over从1开始
    val window_tag = Window.
      partitionBy("a", "mid", "i").
      orderBy(asc("lvt")).
      rowsBetween(Long.MinValue, 0) //without "a", it will consider other "a" by 1...4...13
    val df_tag = df_edit_read.
      withColumn("tag", when($"a" === "3", row_number.over(window_tag)).otherwise(0))
    //2.对a=3下的点击进行tag标签从而得到哪个归哪个，不按照keyword来分，只是按照前后来分。如果某个点击前面没有3，那么就会显示为0，也很合理，说明其产生的a=4操作只能当做本地播放来看
    //val a = Seq("39612569::0:2728:0:1,32100650::0:2730:0:2,32218352::0:2730:0:3,32042828::0:2730:0:4,105894131::0:2730:0:5,32144418::0:2728:0:6,40288371::0:2728:0:7", "32029511::0:2730:0:8,32218351::0:2728:0:9", "32029500::0:2730:0:8,32029511::0:2728:0:8")
    val convert2Map = udf {(scid: Seq[String] ) =>
      if (scid != null){
        scid.map(_.split(",")).
          flatten.
          map(_.split(":")).
          filter(_.size == 6). //to avoid Caused by: java.lang.ArrayIndexOutOfBoundsException
          map((i:Array[String]) => (i(5), i(0))).
          groupBy(_._1).
          values.
          map((i:Seq[(String, String)]) => i.groupBy(identity).mapValues{_.length}.maxBy(_._2)._1).
          map((i:(String, String))  => Map(i._1.toInt -> i._2)).
          reduce(_ ++ _)
      }
      else{
        null
      }
    }

    val window_click_position = Window.
      partitionBy("mid", "i").
      orderBy(asc("lvt")).
      rowsBetween(Long.MinValue, 0) //Long.MinValue means "UNBOUNDED PRECEDING"
    val df_click_position = df_tag.withColumn("search",
      when($"a" === "14124",
        max($"tag").over(window_click_position)).
        otherwise(0)).
      filter($"search" =!= 0 && $"a" === "14124").
      groupBy("mid", "i", "search", "keyword").
      agg(collect_list($"scid_albumid") as "info").
      withColumn("result", convert2Map($"info")).
      select("mid", "i", "search", "keyword", "result")

    df_click_position.createOrReplaceTempView("click_position_data")

    val sql_click_position_create= """
create table if not exists """+s"""$datatable"""+"""_click_position
(
mid string,
i string,
search int,
keyword string,
result map<int, string>
)
partitioned by (cdt string)
row format delimited fields terminated by '|' lines terminated by '\n' stored as textfile
"""

    spark.sql(sql_click_position_create)

    val sql_click_position_save= s"""
INSERT OVERWRITE TABLE """+s"""$datatable"""+s"""_click_position PARTITION(cdt='$date_end') select * from click_position_data
"""

    spark.sql(sql_click_position_save)

    //3.对a=3下的点击进行tag标签从而得到哪个归哪个，不按照keyword来分，只是按照前后来分。如果某个点击前面没有3，那么就会显示为0，也很合理，说明其产生的a=4操作只能当做本地播放来看
    val window_click_tag = Window.
      partitionBy("mid", "i").
      orderBy(asc("lvt")).
      rowsBetween(Long.MinValue, 0) //Long.MinValue means "UNBOUNDED PRECEDING"
    val df_click_tag = df_tag.filter($"a" =!= "14124").
      withColumn("session", when($"a".isin("9697", "10650", "10654"),
        max($"tag").over(window_click_tag)).
        otherwise(0))
    //4.根据mid，i, scid_albumid,keyword进行对a=4溯源。返回parent，position
    val window_previous_tag = Window.
      partitionBy("mid", "i", "scid_albumid", "keyword").
      orderBy(asc("lvt")).
      rowsBetween(Long.MinValue, 0)

    val find_recent_click = udf{(asession: Seq[Row]) =>
      if (asession != null){
        var plays = 0
        var withinops = 0 //点击行为操作内的播放个数
        var cops: List[Int] = List()
        var ops: List[Int] = List()
        var clocation: List[Long] = List()
        var olocation: List[Long] = List()
        for (Row(a:String, s:Int, p:Long) <- asession.reverse){
          a match {
            case "4" => plays += 1
            case "10650" => {
              withinops = plays
              cops = cops:+s
              clocation = clocation:+p
            }
            case "10654" | "9697" => {
              withinops = plays
              ops = ops:+s
              olocation = olocation:+p
            }
            case _ =>
          }
        }
        if (withinops > cops.size+ops.size || withinops == 0){
          (0, 1) //it means not found, we give it first place
        }
        else{
          if (withinops > cops.size){
            (ops(withinops-cops.size-1), olocation(withinops-cops.size-1).toInt) //it means find insert and next in forward sequence with 0 start
          }
          else{
            (cops(withinops-1), clocation(withinops-1).toInt) //it means find click in forward sequence with 0 start
          }
        }
      }
      else{
        (-1, -1) //it means null, we do not consider it
      }
    }
    //we filter the groupby mid, i, other than mid,i,scid_albumid,query
    //cause it means all has problems.
    val df_click_tag_filter =  df_click_tag.
      groupBy("mid", "i").
      count().
      filter($"count" < threshold)
    //df_click_tag_filter.sort($"count".desc).show(100,false)

    val df_click_tag_new = df_click_tag_filter.as("d1").
      join(df_click_tag.as("d2"),
        ($"d1.mid"===$"d2.mid") && ($"d1.i"===$"d2.i"),
        "left").
      select($"d2.*")

    val df_parent = df_click_tag_new.withColumn("combine",
      when($"a" =!= "3",
        struct("a", "session", "ivar")).
        otherwise(null)).
      withColumn("click",
        when($"a" === "4",
          collect_list("combine").over(window_previous_tag)).
          otherwise(null)).
      withColumn("pair",
        when($"a" === "4",
          find_recent_click($"click")).
          otherwise(null)).
      withColumn("parent", $"pair._1").
      withColumn("position", $"pair._2").
      select("a", "mid", "i", "scid_albumid", "lvt", "parent", "position", "keyword", "valid", "session", "ivar")

    df_parent.createOrReplaceTempView("parent_data")

    val sql_parent_create= """
create table if not exists """+s"""$datatable"""+"""_parent
(
a string,
mid string,
i string,
scid_albumid string,
lvt string,
parent int,
position int,
keyword string,
valid int,
session int,
ivar bigint
)
partitioned by (cdt string)
row format delimited fields terminated by '|' lines terminated by '\n' stored as textfile
"""

    spark.sql(sql_parent_create)

    val sql_parent_save= s"""
INSERT OVERWRITE TABLE """+s"""$datatable"""+s"""_parent PARTITION(cdt='$date_end') select * from parent_data
"""

    spark.sql(sql_parent_save)
    //6)create play items
    val sql_parent_read= s"select a, mid, i, scid_albumid, lvt, parent, position, keyword, valid, session, ivar  from "+s"$datatable"+s"_parent where cdt = '$date_end'"
    val df_parent_read = spark.sql(sql_parent_read)
    //find mid to research
    //df_parent_read.filter($"a" === "4" && $"parent" === 0 && $"valid" === 1).groupBy("mid").count().filter($"count" === 10).sort($"count".desc).show(100,false)
    //5.不用join，试着使用mid，parent下word列最多的项，来提取word
    val count_most = udf{(items: Seq[String]) => items.groupBy(identity).maxBy(_._2.size)._1}
    val df_search = df_parent_read.
      filter($"a" === "4" && $"parent".isNotNull && $"parent" =!= 0 && $"parent" =!= -1).
      groupBy("mid", "i", "parent").
      agg(collect_list("scid_albumid") as "scid_albumid", collect_list("valid") as "valid",
        collect_list("position") as "position", collect_list("keyword") as "keyword").
      withColumn("kw", count_most($"keyword")).
      drop("keyword")

    val df_local = df_parent_read.
      filter($"a" === "4" && $"parent" === 0 && $"valid" === 1).
      select($"mid", $"i", $"parent", array($"scid_albumid") as "scid_albumid",
        array($"valid") as "valid", array($"position") as "position", $"keyword" as "kw")

    val df_play = df_search.union(df_local)

    df_play.createOrReplaceTempView("play_data")

    val sql_play_create= """
create table if not exists """+s"""$datatable"""+"""_play
(
mid string,
i string,
parent int,
scid_albumid array<string>,
valid array<int>,
position array<int>,
kw string
)
partitioned by (cdt string)
row format delimited fields terminated by '|' lines terminated by '\n' stored as textfile
"""

    spark.sql(sql_play_create)

    val sql_play_save= s"""
INSERT OVERWRITE TABLE """+s"""$datatable"""+s"""_play PARTITION(cdt='$date_end') select * from play_data
"""

    spark.sql(sql_play_save)
    //====================================================================
    //7)create sessions of play
    val sql_play_read= s"select mid, i, parent, scid_albumid, valid, position, kw from "+s"$datatable"+s"_play where cdt = '$date_end'"
    val df_play_read = spark.sql(sql_play_read)

    val sql_click_position_read= s"select mid, i, search, keyword, result from "+s"$datatable"+s"_click_position where cdt = '$date_end'"
    val df_click_position_read = spark.sql(sql_click_position_read)

    //four: create all session
    val sortCount = udf {(arr: Seq[Row]) =>
      arr.map{ case Row(s: String, c: Long) => (s, c) }.sortBy(- _._2)
    }

    val df_query = df_play_read.as("d1").
      join(df_click_position_read.as("d2"), $"d1.mid"===$"d2.mid" && $"d1.i"===$"d2.i" && $"d1.parent"===$"d2.search" && $"d1.kw"===$"d2.keyword", "left").
      select($"d1.*", $"d2.result")

    val create_session = udf{(song: Seq[String], valid: Seq[Int], position: Seq[Int], result: Map[Int, String]) =>
      val length = position.max
      //use distinct to eliminate the duplicate click
      //groupBy(i => (i._1, i._2)) to choose the bigger valid
      //notice if existed different song within one position, it will all be saved! like: Map(1 -> List((1,55,0), (1,77,1)), 3 -> List((3,66,1)))
      var click = (position, song, valid).zipped.toList.distinct.groupBy(i => (i._1, i._2)).values.map(_.maxBy(_._3)).toList.groupBy(_._1)
      //var with immutable, can be added by +=
      if (result != null){
        for (i <- 1 to length){
          //use contains check map's key
          if (!click.contains(i) && result.contains(i)){
            click += (i -> List((i, result(i), 0)))
          }
        }
        click
      }
      else{
        click
      }
    }

    // val df_look = df_test.withColumn("session", create_session($"scid_albumid", $"valid", $"position", $"result"))

    val df_session = df_query.withColumn("session", create_session($"scid_albumid", $"valid", $"position", $"result"))

    df_session.createOrReplaceTempView("session_data")

    val sql_session_create= """
create table if not exists """+s"""$datatable"""+"""_session
(
mid string,
i string,
parent int,
scid_albumid array<string>,
valid array<int>,
position array<int>,
kw string,
session map<int,array<struct<p:int, s:string, c:int>>>
)
partitioned by (cdt string)
row format delimited fields terminated by '|' lines terminated by '\n' stored as textfile
"""

    spark.sql(sql_session_create)

    val sql_session_save= s"""
INSERT OVERWRITE TABLE """+s"""$datatable"""+s"""_session PARTITION(cdt='$date_end') select mid, i, parent, scid_albumid, valid, position, kw, session from session_data
"""

    spark.sql(sql_session_save)
    //====================================================================
    //8)create song dataframe and tuple of q,u,r,d
    val sql_session_read= s"select mid, i, parent, scid_albumid, position, kw, session from "+s"$datatable"+s"_session where cdt = '$date_end'"
    val df_session_read = spark.sql(sql_session_read)
    //val df_test = df_session.filter($"mid" === "203558088414556161490737452342408042744")

    val click2distance = udf{(position: Seq[Int], session: Map[Int, Seq[Row]]) =>
      val length = position.max
      var pre_click = -1
      //var actions = List()
      //only List() will raise error scala.MatchError: Nothing (of class scala.reflect.internal.Types$ClassNoArgsTypeRef)
      var actions = List[(String, Int, Int, Boolean, Int)]()
      for (i <- 1 to length){
        if(session.contains(i)){
          var times = 0
          var new_pre_click = pre_click
          val nums = session(i).size
          var search = 2
          var valid_nums = 0
          actions = actions ++ session(i).map{ // we consider the current session(i)'s different item (caused by fold)
            case Row(p:Int, scid: String, c: Int) => {
              val d = i - pre_click - 2
              if (c == 1){
                new_pre_click = i - 1
                valid_nums += 1
              }
              times += 1 //calculate the current session(i)'s times
              if (nums==1){ //it means the current session(i) doesnot have fold item
                search =1 //it means that we update both position and attractiveness
              }
              else{
                if (valid_nums == 1){ //it means although the current session(i) have fold item, but we consider the first valid one as normal one
                  search = 1 ////it means that we update both position and attractiveness
                }
                if (times == nums && valid_nums == 0){ //it means although the current session(i) have fold item, but if all invalid, we consider the las invalid one as normal one
                  search = 1 //it means within fold nothing click, we should consider the position and attract
                  //other we only condsider attract to avoid duplicate update position!
                }
              }
              (scid, i-1, d, if (c == 1) true else false, search)
            }
          }
          //it means the current session(i) has been finished, so we can update pre_click
          pre_click = new_pre_click
        }
      }
      actions
    }
    //we have checked with df_session.filter($"parent" === 0).filter(size($"position") =!= 1).show()
    //only save the scid_albumid to see it as first place
    val local2distance = udf{(scid: Seq[String]) =>
      var actions = List[(String, Int, Int, Boolean, Int)]()
      actions = actions ++ Seq((scid(0), 0, 0, true, 0)) //scid_albumid, rank, previous_click_rank, click, whether_search
      actions
    }

    val df_sessions_distance = df_session_read.withColumn("click2distance", when($"parent" === 0, local2distance($"scid_albumid")).
      otherwise(click2distance($"position", $"session")))

    val reorder = udf{(kw: String, uc: Row) =>
      (kw, uc.getString(0), uc.getInt(1), uc.getInt(2), uc.getBoolean(3), uc.getInt(4))
    }

    val df_sessions_pre = df_sessions_distance.
      select($"kw", explode($"click2distance").alias("group")).
      withColumn("reorder", reorder($"kw", $"group")).
      groupBy("reorder").
      agg(count("reorder").alias("cnt")).
      withColumn("q", $"reorder._1").
      withColumn("u", $"reorder._2").
      withColumn("r", $"reorder._3").
      withColumn("d", $"reorder._4").
      withColumn("c", $"reorder._5").
      withColumn("s", $"reorder._6").
      select("q", "u", "r", "d", "c", "s", "cnt")

    df_sessions_pre.createOrReplaceTempView("sessions_pre_data")

    val sql_song_retrieve= s"""
select
    a.*,
    b.choric_singer,
    b.songname
from sessions_pre_data a
left join (
    select
            mixsongid,
            choric_singer,
            songname
    from common.st_k_mixsong_part
    where dt = '$date_end'
    group by
             mixsongid,
             choric_singer,
             songname
) b
on a.u = b.mixsongid
"""

    val df_sessions = spark.sql(sql_song_retrieve)

    df_sessions.createOrReplaceTempView("sessions_data")

    val sql_sessions_create= """
create table if not exists """+s"""$datatable"""+"""_sessions
(
q string,
u string,
r int,
d int,
c boolean,
s int,
cnt bigint,
choric_singer string,
songname string
)
partitioned by (cdt string)
row format delimited fields terminated by '|' lines terminated by '\n' stored as textfile
"""

    spark.sql(sql_sessions_create)

    val sql_sessions_save= s"""
INSERT OVERWRITE TABLE """+s"""$datatable"""+s"""_sessions PARTITION(cdt='$date_end') select q, u, r, d, c, s, cnt, choric_singer, songname from sessions_data
"""

    spark.sql(sql_sessions_save)

    //====================================================================
    //9)create alpah and gamma
    val sql_sessions_read= s"select q, u, r, d, c, s, cnt, choric_singer, songname from "+s"$datatable"+s"_sessions where cdt = '$date_end'"
    val df_sessions_read = spark.sql(sql_sessions_read)
    df_sessions_read.persist()

    //begin train by default 0.5 of position
    //initial the parameters
    //for save by iteration
    val df_gamma = df_sessions_read.
      select("r","d").
      distinct().
      withColumn("rd", struct(col("r"), col("d"))).
      withColumn("gamma", lit(0.5)) //just like tuple.withColumn("gamma", lit(0.5))

    val gamma = df_gamma.select(col("rd"), col("gamma")).as[((Int, Int), Double)].collect.toMap
    var gamma_br = sc.broadcast(gamma)

    val df_alpha = df_sessions_read.
      select("q","u").
      distinct().
      withColumn("qu", struct(col("q"), col("u"))).
      withColumn("alpha", lit(0.5)) //just like tuple

    //val alpha = df_alpha.select(col("qu"), col("alpha")).as[((String, String), Double)].collect.toMap
    //var alpha_br = sc.broadcast(alpha)

    //define udf function
    val update = udf{(r: Int, d: Int, c: Boolean, s: Int, cnt: Long, alpha_uq: Double) =>
      val gamma_rd = gamma_br.value(r,d)
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

    val df_song = df_sessions_read.as("d1").
      join(df_alpha.as("d2"), ($"d1.q" === $"d2.q") && ($"d1.u" === $"d2.u")).
      select($"d1.*", $"d2.alpha").
      withColumn("update", update($"r", $"d", $"c", $"s", $"cnt", $"alpha")).
      withColumn("alpha_numerator", $"update._1").
      withColumn("alpha_denominator", $"update._2").
      withColumn("gamma_numerator", $"update._3").
      withColumn("gamma_denominator", $"update._4").
      select($"q", $"u", $"r", $"d", $"c", $"s", $"cnt",
        $"alpha_numerator", $"alpha_denominator", $"gamma_numerator", $"gamma_denominator",
        $"choric_singer", $"songname")

    df_song.createOrReplaceTempView("song_data")

    val sql_song_create= """
    create table if not exists """+s"""$datatable"""+"""_song
    (
    q string,
    u string,
    r int,
    d int,
    c boolean,
    s int,
    cnt bigint,
    alpha_numerator double,
    alpha_denominator double,
    gamma_numerator double,
    gamma_denominator double,
    choric_singer string,
    songname string
    )
    partitioned by (cdt string)
    row format delimited fields terminated by '|' lines terminated by '\n' stored as textfile
    """

    spark.sql(sql_song_create)

    val sql_song_save= s"""
    INSERT OVERWRITE TABLE """+s"""$datatable"""+s"""_song PARTITION(cdt='$date_end') select q, u, r, d, c, s, cnt, alpha_numerator, alpha_denominator, gamma_numerator, gamma_denominator, choric_singer, songname from song_data
    """

    spark.sql(sql_song_save)
    //====================================================================
    //10)only calculate the position
    val sql_song_read= "select q, u, r, d, c, s, cnt, alpha_numerator, alpha_denominator, gamma_numerator, gamma_denominator, choric_singer, songname from "+s"$datatable"+"_song where cdt = "+s"'$date_end'"
    val df_song_read = spark.sql(sql_song_read)
    df_song_read.persist()

    //position donot deal with s === 1, cause s =!= 1 we return 0
    //for the position, we use the overall to count mean
    //val sumNum = df_song_read.agg(sum("gamma_numerator")).first.getDouble(0)
    val sumDen_new = df_song_read.
      agg(sum("gamma_denominator")).first.getDouble(0)
    val avgDen_new = df_song_read.
      groupBy("r", "d").
      agg(sum("gamma_denominator").alias("denominator")).
      agg(avg("denominator")).first.getDouble(0)
    val maxDen_new = df_song_read.
      groupBy("r", "d").
      agg(sum("gamma_denominator").alias("denominator")).
      agg(max("denominator")).first.getDouble(0)
    //val df_position = df_song_read.groupBy("r", "d").agg(sum("gamma_numerator").alias("numerator"), sum("gamma_denominator").alias("denominator")).withColumn("gamma", (lit(avgDen*sumNum/sumDen) + $"numerator")/(lit(avgDen) + $"denominator"))
    //use 0.5 as prior
    //just keep four decimal places
    val df_position_new = df_song_read.
      groupBy("r", "d").
      agg(sum("gamma_numerator").alias("numerator"),
        sum("gamma_denominator").alias("denominator")).
      withColumn("gamma_origin", $"numerator"/$"denominator").
      withColumn("gamma_max", (lit(maxDen_new*prior_position) + $"numerator")/(lit(maxDen_new) + $"denominator")).
      withColumn("gamma_avg", (lit(avgDen_new*prior_position) + $"numerator")/(lit(avgDen_new) + $"denominator")).
      withColumn("gamma_sum", (lit(sumDen_new*prior_position) + $"numerator")/(lit(sumDen_new) + $"denominator"))

    df_position_new.createOrReplaceTempView("position_new_data")

    val sql_position_new_create= """
create table if not exists """+s"""$thisdatatable"""+"""_position_new
(
r int,
d int,
numerator double,
denominator double,
gamma_origin double,
gamma_max double,
gamma_avg double,
gamma_sum double
)
partitioned by (cdt string)
row format delimited fields terminated by '|' lines terminated by '\n' stored as textfile
"""

    spark.sql(sql_position_new_create)

    val sql_position_new_save= s"""
INSERT OVERWRITE TABLE """+s"""$thisdatatable"""+s"""_position_new PARTITION(cdt='$date_end') select r, d, numerator, denominator, gamma_origin, gamma_max, gamma_avg, gamma_sum from position_new_data
"""

    spark.sql(sql_position_new_save)
    df_song_read.unpersist()

    //    val sql_play_read= s"select mid, i, parent, scid_albumid, valid, position, kw from "+s"$datatable"+s"_play where cdt = '$date_end'"
    //    val df_play_read = spark.sql(sql_play_read)
    //
    //    val sql_click_position_read= s"select mid, i, search, keyword, result from "+s"$datatable"+s"_click_position where cdt = '$date_end'"
    //    val df_click_position_read = spark.sql(sql_click_position_read)
    //
    //    //four: create all session
    //    val sortCount = udf {(arr: Seq[Row]) =>
    //      arr.map{ case Row(s: String, c: Long) => (s, c) }.sortBy(- _._2)
    //    }
    //
    //    val df_query = df_play_read.as("d1").
    //      join(df_click_position_read.as("d2"), $"d1.mid"===$"d2.mid" && $"d1.i"===$"d2.i" && $"d1.parent"===$"d2.search" && $"d1.kw"===$"d2.keyword", "left").
    //      select($"d1.*", $"d2.result")


    //=============================click===============================
    //6)create play items
    //the following three things is same with above!!!!!!!!!!!!!!!!!!!!
//    val sql_parent_read= s"select a, mid, i, scid_albumid, lvt, parent, position, keyword, valid, session, ivar from "+s"$datatable"+s"_parent where cdt = '$date_end'"
//    val df_parent_read = spark.sql(sql_parent_read)

    //5.不用join，试着使用mid，parent下word列最多的项，来提取word
//    val count_most = udf{(items: Seq[String]) => items.groupBy(identity).maxBy(_._2.size)._1}

    //begin new function
    val df_play_click = df_parent_read.
      filter($"a".isin("9697", "10650", "10654") && $"session".isNotNull && $"session" =!= 0 ).
      groupBy("mid", "i", "session").
      agg(collect_list("scid_albumid") as "scid_albumid", collect_list("ivar") as "ivar", collect_list("keyword") as "keyword").
      withColumn("kw", count_most($"keyword")).
      drop("keyword")

    df_play_click.createOrReplaceTempView("play_click_data")

    val sql_play_click_create= """
create table if not exists """+s"""$datatable"""+"""_play_click
(
mid string,
i string,
session int,
scid_albumid array<string>,
ivar array<int>,
kw string
)
partitioned by (cdt string)
row format delimited fields terminated by '|' lines terminated by '\n' stored as textfile
"""

    spark.sql(sql_play_click_create)

    val sql_play_click_save= s"""
INSERT OVERWRITE TABLE """+s"""$datatable"""+s"""_play_click PARTITION(cdt='$date_end') select * from play_click_data
"""

    spark.sql(sql_play_click_save)
    //====================================================================
    //7)create sessions of play: c means click not valid, v means valid !!!!!!!!!!!!!!!!!!
    val sql_play_click_read= s"select mid, i, session, scid_albumid, ivar, kw from "+s"$datatable"+s"_play_click where cdt = '$date_end'"
    val df_play_click_read = spark.sql(sql_play_click_read)

    val sql_click_position_click_read= s"select mid, i, search, keyword, result from "+s"$datatable"+s"_click_position where cdt = '$date_end'"
    val df_click_position_click_read = spark.sql(sql_click_position_click_read)

    //four: create all session
    val df_query_click = df_play_click_read.as("d1").
      join(df_click_position_click_read.as("d2"), $"d1.mid"===$"d2.mid" && $"d1.i"===$"d2.i" && $"d1.session"===$"d2.search" && $"d1.kw"===$"d2.keyword", "left").
      select($"d1.*", $"d2.result")

    val create_session_click = udf{(song: Seq[String], ivar: Seq[Int], result: Map[Int, String]) =>
      val length = ivar.max
      val valid = Seq.fill(ivar.size)(1) //add t tag to 1 show it's clicked
    var click = (ivar, song, valid).zipped.toList.distinct.groupBy(i => (i._1, i._2)).values.map(_.maxBy(_._3)).toList.groupBy(_._1)
      //var with immutable, can be added by +=
      if (result != null){
        for (i <- 1 to length){
          //use contains check map's key
          if (!click.contains(i) && result.contains(i)){
            click += (i -> List((i, result(i), 0)))
          }
        }
        click
      }
      else{
        click
      }
    }

    // val df_look = df_test.withColumn("session", create_session($"scid_albumid", $"valid", $"position", $"result"))

    val df_session_click = df_query_click.withColumn("session_total", create_session_click($"scid_albumid", $"ivar", $"result"))

    df_session_click.createOrReplaceTempView("session_click_data")

    val sql_session_click_create= """
create table if not exists """+s"""$datatable"""+"""_session_click
(
mid string,
i string,
session int,
scid_albumid array<string>,
ivar array<int>,
kw string,
session_total map<int,array<struct<p:int, s:string, c:int>>>
)
partitioned by (cdt string)
row format delimited fields terminated by '|' lines terminated by '\n' stored as textfile
"""

    spark.sql(sql_session_click_create)

    val sql_session_click_save= s"""
INSERT OVERWRITE TABLE """+s"""$datatable"""+s"""_session_click PARTITION(cdt='$date_end') select mid, i, session, scid_albumid, ivar, kw, session_total from session_click_data
"""

    spark.sql(sql_session_click_save)
    //====================================================================
    //8)create song dataframe and tuple of q,u,r,d
    val sql_session_click_read= s"select mid, i, scid_albumid, ivar, kw, session_total from "+s"$datatable"+s"_session_click where cdt = '$date_end'"
    val df_session_click_read = spark.sql(sql_session_click_read)
    //val df_test = df_session.filter($"mid" === "203558088414556161490737452342408042744")

    val click2distance_click = udf{(ivar: Seq[Int], session: Map[Int, Seq[Row]]) =>
      val length = ivar.max
      var pre_click = -1
      //var actions = List()
      //only List() will raise error scala.MatchError: Nothing (of class scala.reflect.internal.Types$ClassNoArgsTypeRef)
      var actions = List[(String, Int, Int, Boolean, Int)]()
      for (i <- 1 to length){
        if(session.contains(i)){
          var times = 0
          var new_pre_click = pre_click
          val nums = session(i).size
          var search = 2
          var valid_nums = 0
          actions = actions ++ session(i).map{ // we consider the current session(i)'s different item (caused by fold)
            case Row(p:Int, scid: String, c: Int) => {
              val d = i - pre_click - 2
              if (c == 1){
                new_pre_click = i - 1
                valid_nums += 1
              }
              times += 1 //calculate the current session(i)'s times
              if (nums==1){ //it means the current session(i) doesnot have fold item
                search =1 //it means that we update both position and attractiveness
              }
              else{
                if (valid_nums == 1){ //it means although the current session(i) have fold item, but we consider the first valid one as normal one
                  search = 1 ////it means that we update both position and attractiveness
                }
                if (times == nums && valid_nums == 0){ //it means although the current session(i) have fold item, but if all invalid, we consider the las invalid one as normal one
                  search = 1 //it means within fold nothing click, we should consider the position and attract
                  //other we only condsider attract to avoid duplicate update position!
                }
              }
              (scid, i-1, d, if (c == 1) true else false, search)
            }
          }
          //it means the current session(i) has been finished, so we can update pre_click
          pre_click = new_pre_click
        }
      }
      actions
    }

    val df_sessions_distance_click = df_session_click_read.
      withColumn("click2distance", click2distance_click($"ivar", $"session_total"))

    val reorder_click = udf{(kw: String, uc: Row) =>
      (kw, uc.getString(0), uc.getInt(1), uc.getInt(2), uc.getBoolean(3), uc.getInt(4))
    }

    val df_sessions_pre_click = df_sessions_distance_click.
      select($"kw", explode($"click2distance").alias("group")).
      withColumn("reorder", reorder_click($"kw", $"group")).
      groupBy("reorder").
      agg(count("reorder").alias("cnt")).
      withColumn("q", $"reorder._1").
      withColumn("u", $"reorder._2").
      withColumn("r", $"reorder._3").
      withColumn("d", $"reorder._4").
      withColumn("c", $"reorder._5").
      withColumn("s", $"reorder._6").
      select("q", "u", "r", "d", "c", "s", "cnt") //now "s" doesnot mean anything, cause all is search

    df_sessions_pre_click.createOrReplaceTempView("sessions_pre_click_data")

    val sql_song_retrieve_click= s"""
select
    a.*,
    b.choric_singer,
    b.songname
from sessions_pre_click_data a
left join (
    select
            mixsongid,
            choric_singer,
            songname
    from common.st_k_mixsong_part
    where dt = '$date_end'
    group by
             mixsongid,
             choric_singer,
             songname
) b
on a.u = b.mixsongid
"""

    val df_sessions_click = spark.sql(sql_song_retrieve_click)

    df_sessions_click.createOrReplaceTempView("sessions_click_data")

    val sql_sessions_click_create= """
create table if not exists """+s"""$datatable"""+"""_sessions_click
(
q string,
u string,
r int,
d int,
c boolean,
s int,
cnt bigint,
choric_singer string,
songname string
)
partitioned by (cdt string)
row format delimited fields terminated by '|' lines terminated by '\n' stored as textfile
"""

    spark.sql(sql_sessions_click_create)

    val sql_sessions_click_save= s"""
INSERT OVERWRITE TABLE """+s"""$datatable"""+s"""_sessions_click PARTITION(cdt='$date_end') select q, u, r, d, c, s, cnt, choric_singer, songname from sessions_click_data
"""

    spark.sql(sql_sessions_click_save)

    //====================================================================
    //9)create alpah and gamma
    val sql_sessions_click_read= s"select q, u, r, d, c, s, cnt, choric_singer, songname from "+s"$datatable"+s"_sessions_click where cdt = '$date_end'"
    val df_sessions_click_read = spark.sql(sql_sessions_click_read)
    df_sessions_click_read.persist()

    //begin train by default 0.5 of position
    //initial the parameters
    //for save by iteration
    val df_gamma_click = df_sessions_click_read.
      select("r","d").
      distinct().
      withColumn("rd", struct(col("r"), col("d"))).
      withColumn("gamma", lit(0.5)) //just like tuple.withColumn("gamma", lit(0.5))

    val gamma_click = df_gamma_click.
      select(col("rd"), col("gamma")).as[((Int, Int), Double)].
      collect.
      toMap
    var gamma_br_click = sc.broadcast(gamma_click)

    val df_alpha_click = df_sessions_click_read.
      select("q","u").
      distinct().
      withColumn("qu", struct(col("q"), col("u"))).
      withColumn("alpha", lit(0.5)) //just like tuple

    //val alpha = df_alpha.select(col("qu"), col("alpha")).as[((String, String), Double)].collect.toMap
    //var alpha_br = sc.broadcast(alpha)

    //define udf function
    val update_click = udf{(r: Int, d: Int, c: Boolean, s: Int, cnt: Long, alpha_uq: Double) =>
      val gamma_rd = gamma_br_click.value(r,d)
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

    val df_song_click = df_sessions_click_read.as("d1").
      join(df_alpha_click.as("d2"), ($"d1.q" === $"d2.q") && ($"d1.u" === $"d2.u")).
      select($"d1.*", $"d2.alpha").
      withColumn("update", update_click($"r", $"d", $"c", $"s", $"cnt", $"alpha")).
      withColumn("alpha_numerator", $"update._1").
      withColumn("alpha_denominator", $"update._2").
      withColumn("gamma_numerator", $"update._3").
      withColumn("gamma_denominator", $"update._4").
      select($"q", $"u", $"r", $"d", $"c", $"s", $"cnt",
        $"alpha_numerator", $"alpha_denominator", $"gamma_numerator", $"gamma_denominator",
        $"choric_singer", $"songname")

    df_song_click.createOrReplaceTempView("song_click_data")

    val sql_song_click_create= """
    create table if not exists """+s"""$datatable"""+"""_song_click
    (
    q string,
    u string,
    r int,
    d int,
    c boolean,
    s int,
    cnt bigint,
    alpha_numerator double,
    alpha_denominator double,
    gamma_numerator double,
    gamma_denominator double,
    choric_singer string,
    songname string
    )
    partitioned by (cdt string)
    row format delimited fields terminated by '|' lines terminated by '\n' stored as textfile
    """

    spark.sql(sql_song_click_create)

    val sql_song_click_save= s"""
    INSERT OVERWRITE TABLE """+s"""$datatable"""+s"""_song_click PARTITION(cdt='$date_end') select q, u, r, d, c, s, cnt, alpha_numerator, alpha_denominator, gamma_numerator, gamma_denominator, choric_singer, songname from song_click_data
    """

    spark.sql(sql_song_click_save)
    //====================================================================
    //10)only calculate the position
    val sql_song_click_read= "select q, u, r, d, c, s, cnt, alpha_numerator, alpha_denominator, gamma_numerator, gamma_denominator, choric_singer, songname from "+s"$datatable"+"_song_click where cdt = "+s"'$date_end'"
    val df_song_click_read = spark.sql(sql_song_click_read)
    df_song_click_read.persist()

    //position donot deal with s === 1, cause s =!= 1 we return 0
    //for the position, we use the overall to count mean
    //val sumNum = df_song_read.agg(sum("gamma_numerator")).first.getDouble(0)
    val sumDen_new_click = df_song_click_read.
      agg(sum("gamma_denominator")).first.getDouble(0)
    val avgDen_new_click = df_song_click_read.
      groupBy("r", "d").
      agg(sum("gamma_denominator").alias("denominator")).
      agg(avg("denominator")).first.getDouble(0)
    val maxDen_new_click = df_song_click_read.
      groupBy("r", "d").
      agg(sum("gamma_denominator").alias("denominator")).
      agg(max("denominator")).first.getDouble(0)
    //val df_position = df_song_read.groupBy("r", "d").agg(sum("gamma_numerator").alias("numerator"), sum("gamma_denominator").alias("denominator")).withColumn("gamma", (lit(avgDen*sumNum/sumDen) + $"numerator")/(lit(avgDen) + $"denominator"))
    //use 0.5 as prior
    //just keep four decimal places
    val df_position_new_click = df_song_click_read.
      groupBy("r", "d").
      agg(sum("gamma_numerator").alias("numerator"),
        sum("gamma_denominator").alias("denominator")).
      withColumn("gamma_origin", $"numerator"/$"denominator").
      withColumn("gamma_max", (lit(maxDen_new_click*prior_position) + $"numerator")/(lit(maxDen_new_click) + $"denominator")).
      withColumn("gamma_avg", (lit(avgDen_new_click*prior_position) + $"numerator")/(lit(avgDen_new_click) + $"denominator")).
      withColumn("gamma_sum", (lit(sumDen_new_click*prior_position) + $"numerator")/(lit(sumDen_new_click) + $"denominator"))

    df_position_new_click.createOrReplaceTempView("position_new_click_data")

    val sql_position_new_click_create= """
create table if not exists """+s"""$thisdatatable"""+"""_position_new_click
(
r int,
d int,
numerator double,
denominator double,
gamma_origin double,
gamma_max double,
gamma_avg double,
gamma_sum double
)
partitioned by (cdt string)
row format delimited fields terminated by '|' lines terminated by '\n' stored as textfile
"""

    spark.sql(sql_position_new_click_create)

    val sql_position_new_click_save= s"""
INSERT OVERWRITE TABLE """+s"""$thisdatatable"""+s"""_position_new_click PARTITION(cdt='$date_end') select r, d, numerator, denominator, gamma_origin, gamma_max, gamma_avg, gamma_sum from position_new_click_data
"""

    spark.sql(sql_position_new_click_save)
    df_song_click_read.unpersist()
    //====================================================================
    //11)end
    spark.stop() //to avoid ERROR LiveListenerBus: SparkListenerBus has already stopped! Dropping event SparkListenerExecutorMetricsUpdate

  }
}
