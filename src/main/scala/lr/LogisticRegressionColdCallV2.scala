package lr

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

object LogisticRegressionColdCallV2 extends Serializable {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("LogisticRegressionColdCall")
      .master("local")
      .config("spark.ui.enabled", "false")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    val sample: DataFrame = loadSample(spark, "carInsurance_train.csv")
    sample.show()

    val Array(training, test) = sample.randomSplit(Array(0.7, 0.3), 12345)

    val pipeline: Pipeline = definePipeline

    val model = pipeline.fit(training)

    val predictions = model.transform(test)

    val evaluator = new BinaryClassificationEvaluator()
      .setMetricName("areaUnderROC")

    val accuracy = evaluator.evaluate(predictions)
    //Test Error = 0.1303923548018049
    println(s"Test Error = ${1.0 - accuracy}")

    /*predictions
      .select("prediction", "label", "Age", "Balance", "Marital", "Education")
      .withColumn("prediction", col("prediction").cast("Int"))
      .withColumn("predictionResult", lit(col("prediction") === col("label")))
      .write
      .mode("overwrite")
      .option("header", "true")
      .option("delimiter", "\t")
      .csv(getClass.getResource("").getPath + "/../car-insurance-predication-result/")*/

    spark.stop()
  }

  private def definePipeline = {
    val indexers = Array("Marital", "Education", "Job", "Communication").map(col => {
      new StringIndexer().setInputCol(col).setOutputCol(s"${col}Indexer")
    })

    val encoder = new OneHotEncoderEstimator()
      .setInputCols(Array("MaritalIndexer", "EducationIndexer", "JobIndexer", "CommunicationIndexer"))
      .setOutputCols(Array("MaritalVec", "EducationVec", "JobVec", "CommunicationVec"))

    val assembler = new VectorAssembler()
      .setInputCols(Array("Age", "Balance", "MaritalVec", "EducationVec", "JobVec", "CommunicationVec", "CarLoan", "LastContactDay", "CallDuration"))
      .setOutputCol("features")

    val lr = new LogisticRegression()

    val pipeline = new Pipeline()
      .setStages(indexers ++ Array(encoder, assembler, lr))
    pipeline
  }

  private def loadSample(spark: SparkSession, fileName:String) = {
    val sample = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(getClass.getResource("").getPath + s"/../carinsurance/$fileName")
      .select(col("CarInsurance").alias("label"), col("Marital"),
        col("Education"), col("Age"), col("Balance"),
        col("Job"), col("Communication"), col("CarLoan"), col("LastContactDay"),
        unix_timestamp(col("CallStart"), "HH:mm:ss").alias("CallStart"),
        unix_timestamp(col("CallEnd"), "HH:mm:ss").alias("CallEnd"))
      .withColumn("CallDuration", (col("CallEnd") - col("CallStart")) / 100)
    sample
  }
}
