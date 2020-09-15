/* SimpleApp.scala */
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.SQLContext 
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.feature.Normalizer
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.SQLTransformer


val sqlContext = new SQLContext(sc) 

// input data in DataFrame
var df = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("season09-10.csv","season10-11.csv","season11-12.csv","season12-13.csv","season13-14.csv","season14-15.csv","season15-16.csv","season16-17.csv","season17-18.csv","season18-19.csv")

val t1 = System.nanoTime

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
case class Fixture(team: String,date: String,ftr: Int, Goal_Scored: Int, Goal_Conceded: Int, Shot: Int, Shot_conceded: Int, Shot_On_Target: Int, Shot_conceded_On_Target: Int)

// UDF => transform input data from a game (here a row) into a Seq list of fixture object for each team playing that game
val fixtureList = udf( (Date:String,HomeTeam:String,AwayTeam:String,FTHG:Int,FTAG:Int,FTR:String,HS:Int,AS:Int,HST:Int,AST:Int) => Seq( Fixture(HomeTeam, Date, assignHomeResult(FTR), FTHG, FTAG, HS, AS, HST, AST), Fixture(AwayTeam, Date, assignAwayResult(FTR), FTAG, FTHG, AS, HS, AST, HST)) )

// extract feature 
val dataset = df.withColumn("fixture", fixtureList($"Date", $"HomeTeam", $"AwayTeam", $"FTHG", $"FTAG", $"FTR", $"HS", $"AS", $"HST", $"AST")).select(explode($"fixture"))


val sqlTrans = new SQLTransformer().setStatement("SELECT col.team AS team, COUNT(col.date) AS date, SUM(col.ftr) AS ftr, SUM(col.Goal_Scored) AS Goal_Scored, SUM(col.Goal_Conceded) AS Goal_Conceded, SUM(col.Shot) AS Shot, SUM(col.Shot_conceded) AS Shot_conceded, SUM(col.Shot_On_Target) AS Shot_On_Target, SUM(col.Shot_conceded_On_Target) AS Shot_conceded_On_Target FROM __THIS__ GROUP BY col.team")

// define assembler object to extract feature 
val assembler = new VectorAssembler().setInputCols(Array( "date", "ftr","Goal_Scored","Goal_Conceded","Shot","Shot_conceded","Shot_On_Target","Shot_conceded_On_Target")).setOutputCol("featuresCol")

// scale the feature value
//val scaler = new StandardScaler().setInputCol("featuresCol").setOutputCol("features").setWithStd(true).setWithMean(false)

val normalizer = new Normalizer().setInputCol("featuresCol").setOutputCol("features").setP(1.0)

// Trains a k-means model.
val kmeans = new KMeans().setK(3).setSeed(1L)

val pipeline = new Pipeline().setStages(Array(sqlTrans, assembler, normalizer, kmeans))

// Fit the pipeline to training documents.
val model = pipeline.fit(dataset)

// Make predictions
val predictions = model.transform(dataset).select("team", "features", "prediction")

predictions.show(40)

// Evaluate clustering by computing Silhouette score
val evaluator = new ClusteringEvaluator()

val silhouette = evaluator.evaluate(predictions)
println(s"Silhouette with squared euclidean distance = $silhouette")

val duration = (System.nanoTime - t1) / 1e9d
