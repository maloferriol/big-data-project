import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.explode
import org.apache.spark.sql.functions.udf
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.feature.Normalizer
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.SQLTransformer
import org.apache.spark.ml.feature.MinMaxScaler



object SimpleApp {

    /* SimpleApp.scala */
    case class Fixture(team: String,date: String,ftr: Int, Goal_Scored: Int, Goal_Conceded: Int, Shot: Int, Shot_conceded: Int, Shot_On_Target: Int, Shot_conceded_On_Target: Int)

    
    def main(args: Array[String]) {
        
        val spark = SparkSession.builder.appName("Simple Application").getOrCreate()

        import spark.implicits._ 

        val t1 = System.nanoTime

        // input data in DataFrame
        var df = spark.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("/Users/maloferriol/Downloads/season09-10.csv","/Users/maloferriol/Downloads/season10-11.csv","/Users/maloferriol/Downloads/season11-12.csv","/Users/maloferriol/Downloads/season12-13.csv","/Users/maloferriol/Downloads/season13-14.csv","/Users/maloferriol/Downloads/season14-15.csv","/Users/maloferriol/Downloads/season15-16.csv","/Users/maloferriol/Downloads/season16-17.csv","/Users/maloferriol/Downloads/season17-18.csv","/Users/maloferriol/Downloads/season18-19.csv")


        def assignHomeResult(ftr: String):Int = ftr match {
            case "A" => 0
            case "D" => 1
            case "H" => 3
        }

        def assignAwayResult(ftr: String):Int = ftr match {
            case "A" => 3
            case "D" => 1
            case "H" => 0
        }

        // create class fixture 
        // It summurize the statistics of a game for one team
       
        // UDF => transform input data from a game (here a row) into a Seq list of fixture object for each team playing that game
        val fixtureList = udf( (Date:String,HomeTeam:String,AwayTeam:String,FTHG:Int,FTAG:Int,FTR:String,HS:Int,AS:Int,HST:Int,AST:Int) => Seq( Fixture(HomeTeam, Date, assignHomeResult(FTR), FTHG, FTAG, HS, AS, HST, AST), Fixture(AwayTeam, Date, assignAwayResult(FTR), FTAG, FTHG, AS, HS, AST, HST)) )

        // extract feature 
        val dataset = df.withColumn("fixture", fixtureList($"Date", $"HomeTeam", $"AwayTeam", $"FTHG", $"FTAG", $"FTR", $"HS", $"AS", $"HST", $"AST")).select(explode($"fixture"))


        val sqlTrans = new SQLTransformer().setStatement("SELECT col.team AS team, COUNT(col.date) AS date, SUM(col.ftr) AS ftr, SUM(col.Goal_Scored) AS Goal_Scored, SUM(col.Goal_Conceded) AS Goal_Conceded, SUM(col.Shot) AS Shot, SUM(col.Shot_conceded) AS Shot_conceded, SUM(col.Shot_On_Target) AS Shot_On_Target, SUM(col.Shot_conceded_On_Target) AS Shot_conceded_On_Target FROM __THIS__ GROUP BY col.team")

        // define assembler object to extract feature 
        val assembler = new VectorAssembler().setInputCols(Array( "date", "ftr","Goal_Scored","Goal_Conceded","Shot","Shot_conceded","Shot_On_Target","Shot_conceded_On_Target")).setOutputCol("featuresCol")

        // scale the feature value
        val scaler = new MinMaxScaler().setInputCol("featuresCol").setOutputCol("features")

        // Trains a k-means model.
        val kmeans = new KMeans().setK(3).setSeed(1L)

        val pipeline = new Pipeline().setStages(Array(sqlTrans, assembler, scaler, kmeans))

        // Fit the pipeline to training documents.
        val model = pipeline.fit(dataset)

        // Make predictions
        val predictions = model.transform(dataset).select("team", "features", "prediction")

        predictions.show(40)

        // Evaluate clustering by computing Silhouette score
        val evaluator = new ClusteringEvaluator()

        val silhouette = evaluator.evaluate(predictions)
        println(s"Silhouette with squared euclidean distance = $silhouette")

        val duration2 = (System.nanoTime - t1) / 1e9d        
        println(s"Duration = $duration2")

        spark.stop()
    }
}
