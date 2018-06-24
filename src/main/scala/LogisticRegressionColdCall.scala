import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object LogisticRegressionColdCall extends Serializable {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("LogisticRegressionColdCall")
      .master("local")
      .config("spark.ui.enabled", "false")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    val vertexDetail = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(getClass.getResource("").getPath + "/carinsurance/carInsurance_train.csv")
      .select(col("CarInsurance").alias("label"), col("Marital"), col("Education"), col("Age"), col("Balance"))

    vertexDetail.printSchema()
    vertexDetail.show()

    val maritalIndexer = new StringIndexer()
      .setInputCol("Marital")
      .setOutputCol("MaritalIndexer")

    val educationIndexer = new StringIndexer()
      .setInputCol("Education")
      .setOutputCol("EducationIndexer")

    val encoder = new OneHotEncoderEstimator()
      .setInputCols(Array("MaritalIndexer", "EducationIndexer"))
      .setOutputCols(Array("MaritalVec", "EducationVec"))

    val assembler = new VectorAssembler()
      .setInputCols(Array("Age", "Balance", "MaritalVec", "EducationVec"))
      .setOutputCol("features")

    val Array(training, test) = vertexDetail.randomSplit(Array(0.7, 0.3), 12345)

    val lr = new LogisticRegression()

    val pipeline = new Pipeline()
      .setStages(Array(maritalIndexer, educationIndexer, encoder, assembler, lr))

    val model = pipeline.fit(training)

    val predictions = model.transform(test)

    predictions.select("prediction", "label", "Age", "Balance", "Marital", "Education", "features").show(10, false)

    val evaluator = new BinaryClassificationEvaluator()
      .setMetricName("areaUnderROC")

    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error = ${(1.0 - accuracy)}")


    predictions
      .select("prediction", "label", "Age", "Balance", "Marital", "Education")
      .withColumn("prediction", col("prediction").cast("Int"))
      .withColumn("predictionResult", lit(col("prediction") === col("label")))
      .write
      .mode("overwrite")
      .option("header", "true")
      .option("delimiter","\t")
      .csv(getClass.getResource("").getPath + "/car-insurance-predication-result/")

    spark.stop()
  }
}
