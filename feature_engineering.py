import os
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window

BASE = os.path.dirname(os.path.abspath(__file__))

conf = pyspark.SparkConf().setAll([
    ('spark.master', 'local[6]'),
    ('spark.app.name', 'Feature Engineering'),
    ('spark.driver.memory', '6g'),
    ('spark.executor.memory', '6g'),
    ('spark.sql.shuffle.partitions', '12'),
])
spark = SparkSession.builder.config(conf=conf).getOrCreate()
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")

schema = StructType([
    StructField("subject_id",     IntegerType(), True),
    StructField("activity_label", StringType(),  True),
    StructField("timestamp",      LongType(),    True),
    StructField("x",              FloatType(),   True),
    StructField("y",              FloatType(),   True),
    StructField("z",              StringType(),  True),
])

watch_accelDF = spark.read.csv(f"{BASE}/data/wisdm-dataset/raw/watch/accel/*.txt", schema=schema)
watch_gyroDF  = spark.read.csv(f"{BASE}/data/wisdm-dataset/raw/watch/gyro/*.txt",  schema=schema)
phone_accelDF = spark.read.csv(f"{BASE}/data/wisdm-dataset/raw/phone/accel/*.txt", schema=schema)
phone_gyroDF  = spark.read.csv(f"{BASE}/data/wisdm-dataset/raw/phone/gyro/*.txt",  schema=schema)

def clean_z(df):
    return df.withColumn("z", trim(regexp_replace(col("z"), ";", "")).cast("float"))

watch_accelDF = clean_z(watch_accelDF)
watch_gyroDF  = clean_z(watch_gyroDF)
phone_accelDF = clean_z(phone_accelDF)
phone_gyroDF  = clean_z(phone_gyroDF)

watch_accelDF = watch_accelDF.withColumn("timestamp", (round(col("timestamp") / 50) * 50).cast("long"))
watch_gyroDF  = watch_gyroDF.withColumn("timestamp",  (round(col("timestamp") / 50) * 50).cast("long"))
phone_accelDF = phone_accelDF.withColumn("timestamp", (round(col("timestamp") / 50) * 50).cast("long"))
phone_gyroDF  = phone_gyroDF.withColumn("timestamp",  (round(col("timestamp") / 50) * 50).cast("long"))

watch = watch_accelDF.select(
    "subject_id", "activity_label", "timestamp",
    col("x").alias("wa_x"), col("y").alias("wa_y"), col("z").alias("wa_z")
).join(
    watch_gyroDF.select(
        "subject_id", "timestamp",
        col("x").alias("wg_x"), col("y").alias("wg_y"), col("z").alias("wg_z")
    ),
    on=["subject_id", "timestamp"], how="inner"
)

phone = phone_accelDF.select(
    "subject_id", "activity_label", "timestamp",
    col("x").alias("pa_x"), col("y").alias("pa_y"), col("z").alias("pa_z")
).join(
    phone_gyroDF.select(
        "subject_id", "timestamp",
        col("x").alias("pg_x"), col("y").alias("pg_y"), col("z").alias("pg_z")
    ),
    on=["subject_id", "timestamp"], how="inner"
)

def engineer_features(df, prefix_pairs):
    w = Window.partitionBy("subject_id", "activity_label").orderBy("timestamp")

    df = df.withColumn("seq_id", concat_ws("-",
        col("subject_id").cast("string"),
        col("activity_label"),
        row_number().over(w).cast("string")
    ))

    for px, py, pz in prefix_pairs:
        # delta time between consecutive readings (not cumulative)
        delta_t = (col("timestamp") - lag("timestamp", 1).over(w)) / 1e9

        df = df \
            .withColumn(f"{px}_dx", coalesce(col(px) - lag(px, 1).over(w), lit(0.0))) \
            .withColumn(f"{py}_dy", coalesce(col(py) - lag(py, 1).over(w), lit(0.0))) \
            .withColumn(f"{pz}_dz", coalesce(col(pz) - lag(pz, 1).over(w), lit(0.0))) \
            .withColumn(f"{px}_ax", when(delta_t.isNull() | (delta_t == 0), 0.0).otherwise(col(f"{px}_dx") / delta_t)) \
            .withColumn(f"{py}_ay", when(delta_t.isNull() | (delta_t == 0), 0.0).otherwise(col(f"{py}_dy") / delta_t)) \
            .withColumn(f"{pz}_az", when(delta_t.isNull() | (delta_t == 0), 0.0).otherwise(col(f"{pz}_dz") / delta_t)) \
            .withColumn(f"{px}_magnitude", sqrt(col(px)**2 + col(py)**2 + col(pz)**2)) \
            .withColumn(f"{px}_{py}",      col(px) * col(py)) \
            .withColumn(f"{px}_{pz}",      col(px) * col(pz)) \
            .withColumn(f"{py}_{pz}",      col(py) * col(pz)) \
            .withColumn(f"{px}_jerk",      sqrt(col(f"{px}_ax")**2 + col(f"{py}_ay")**2 + col(f"{pz}_az")**2))

    return df  # keep timestamp for windowing in model.py

watch = engineer_features(watch, [("wa_x", "wa_y", "wa_z"), ("wg_x", "wg_y", "wg_z")])
phone = engineer_features(phone, [("pa_x", "pa_y", "pa_z"), ("pg_x", "pg_y", "pg_z")])

watch.cache()
phone.cache()

print("watch rows:", watch.count())
print("phone rows:", phone.count())
print("watch columns:", watch.columns)

watch.write.parquet(f"{BASE}/data/watch_features.parquet", mode="overwrite")
phone.write.parquet(f"{BASE}/data/phone_features.parquet", mode="overwrite")

print("Done. Parquet files written.")
