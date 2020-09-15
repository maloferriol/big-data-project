/* SimpleApp.scala */
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.SQLContext 
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.feature.Normalizer
import org.apache.spark.ml.feature.StandardScaler

val sqlContext = new SQLContext(sc) 

val t1 = System.nanoTime

// input data in DataFrame
var df = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("season09-10.csv","season10-11.csv","season11-12.csv","season12-13.csv","season13-14.csv","season14-15.csv","season15-16.csv","season16-17.csv","season17-18.csv","season18-19.csv")


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
case class Fixture(team: String,
				   date: String,
				   ftr: Int, 
				   Goal_Scored: Int, 
				   Goal_Conceded: Int, 
				   Shot: Int, 
				   Shot_conceded: Int, 
				   Shot_On_Target: Int, 
				   Shot_conceded_On_Target: Int)

// UDF => transform input data from a game (here a row) into a Seq list of fixture object for each team playing that game
val fixtureList = udf( (Date:String,HomeTeam:String,AwayTeam:String,FTHG:Int,FTAG:Int,FTR:String,HS:Int,AS:Int,HST:Int,AST:Int) => 
	Seq( 
		Fixture(HomeTeam, Date, assignHomeResult(FTR), FTHG, FTAG, HS, AS, HST, AST), 
		Fixture(AwayTeam, Date, assignAwayResult(FTR), FTAG, FTHG, AS, HS, AST, HST)
		) 
	)

// extract feature 
val df2 = df.withColumn("fixture", fixtureList($"Date", $"HomeTeam", $"AwayTeam", $"FTHG", $"FTAG", $"FTR", $"HS", $"AS", $"HST", $"AST")).select(explode($"fixture")).select( "col.*", "*")

// group by team 
val df4 = df2.groupBy("team").agg(
  "date" -> "count",
  "ftr" -> "sum",
  "Goal_Scored" -> "sum",
  "Goal_Conceded" -> "sum",
  "Shot" -> "sum",
  "Shot_conceded" -> "sum",
  "Shot_On_Target" -> "sum",
  "Shot_conceded_On_Target" -> "sum"
)

// df4.show(40)

// define assembler object to extract feature 
val assembler = new VectorAssembler().setInputCols(Array( "count(date)", "sum(ftr)","sum(Goal_Scored)","sum(Goal_Conceded)","sum(Shot)","sum(Shot_conceded)","sum(Shot_On_Target)","sum(Shot_conceded_On_Target)")).setOutputCol("features")

// perform feature extraction
val df5 = assembler.transform(df4)

// select the column features and label (here the name of the team)
val df6 = df5.select($"team".as("label"), $"features")

// normalize the feature value
val normalizer = new Normalizer().setInputCol("features").setOutputCol("normFeatures").setP(1.0)

// compute normalization of features values with norm set to 1 
val dataset = normalizer.transform(dataframe).select($"normFeatures".as("features"), $"label")

// scale the feature value
//val scaler = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures").setWithStd(true).setWithMean(false)

// Compute summary statistics by fitting the StandardScaler.
//val scalerModel = scaler.fit(df6)

// Normalize each feature to have unit standard deviation.
//val dataset = scalerModel.transform(df6).select($"scaledFeatures".as("features"), $"label")

val duration1 = (System.nanoTime - t1) / 1e9d

// dataset.show(40)

// Trains a k-means model.
val kmeans = new KMeans().setK(3).setSeed(1L)//.setFeaturesCol("normFeatures")
val model = kmeans.fit(dataset)

// Make predictions
val predictions = model.transform(dataset)

// Evaluate clustering by computing Silhouette score
val evaluator = new ClusteringEvaluator()

val silhouette = evaluator.evaluate(predictions)
println(s"Silhouette with squared euclidean distance = $silhouette")

// Shows the result.
println("Cluster Centers: ")
model.clusterCenters.foreach(println)

model.summary.predictions.show(40)

val duration2 = (System.nanoTime - t1) / 1e9d
