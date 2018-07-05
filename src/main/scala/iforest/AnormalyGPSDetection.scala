package iforest

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.iforest.IForest
import org.apache.spark.sql.SparkSession

object AnormalyGPSDetection {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .master("local")
      .config("spark.ui.enabled", "false")
      .appName(getClass.getName)
      .getOrCreate()

    val dataset = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(getClass.getResource("").getPath + "/../gps/gps.csv")

    dataset.show(10)

    val assembler = new VectorAssembler()
    assembler.setInputCols(Array("latitude", "longitude"))
    assembler.setOutputCol("features")

    val iForest = new IForest()
        .setNumTrees(100)
        .setMaxSamples(256)
        .setContamination(0.15)
        .setBootstrap(false)
        .setMaxDepth(100)
        .setSeed(123456L)

    val pipeline = new Pipeline().setStages(Array(assembler, iForest))
    val model = pipeline.fit(dataset)
    val predictions = model.transform(dataset)

    predictions.select("latitude", "longitude", "prediction").show()

    spark.stop()
  }
}
