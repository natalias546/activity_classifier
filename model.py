import os
import numpy as np
import pandas as pd
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

BASE = os.path.dirname(os.path.abspath(__file__))

conf = pyspark.SparkConf().setAll([
    ('spark.master', 'local[6]'),
    ('spark.app.name', 'Activity Classifier'),
    ('spark.driver.memory', '6g'),
    ('spark.executor.memory', '6g'),
    ('spark.sql.shuffle.partitions', '12'),
])
spark = SparkSession.builder.config(conf=conf).getOrCreate()
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")

# ── Load ──────────────────────────────────────────────────────────────────────
watch = spark.read.parquet(f"{BASE}/data/watch_features.parquet")
phone = spark.read.parquet(f"{BASE}/data/phone_features.parquet")

# ── Windowing ─────────────────────────────────────────────────────────────────
WINDOW_SIZE = 200   # 10s at ~20Hz
STEP_SIZE   = 100   # 50% overlap

META_COLS  = {"subject_id", "activity_label", "seq_id", "timestamp"}
watch_feat = [c for c in watch.columns if c not in META_COLS]
phone_feat = [c for c in phone.columns if c not in META_COLS]

def make_window_schema(feat_cols):
    fields = [
        StructField("subject_id",    IntegerType(), False),
        StructField("activity_label", StringType(), False),
        StructField("window_index",  IntegerType(), False),
    ]
    for c in feat_cols:
        fields.append(StructField(f"{c}_mean", FloatType(), True))
        fields.append(StructField(f"{c}_std",  FloatType(), True))
    return StructType(fields)

def make_windower(feat_cols):
    def windower(pdf):
        pdf = pdf.sort_values("timestamp").reset_index(drop=True)
        subject_id     = int(pdf["subject_id"].iloc[0])
        activity_label = str(pdf["activity_label"].iloc[0])
        signals = pdf[feat_cols].values.astype(np.float32)
        rows = []
        for i, start in enumerate(range(0, len(signals) - WINDOW_SIZE + 1, STEP_SIZE)):
            window = signals[start:start + WINDOW_SIZE]
            row = {"subject_id": subject_id, "activity_label": activity_label, "window_index": i}
            for j, c in enumerate(feat_cols):
                row[f"{c}_mean"] = float(np.mean(window[:, j]))
                row[f"{c}_std"]  = float(np.std(window[:, j]))
            rows.append(row)
        if not rows:
            cols = ["subject_id", "activity_label", "window_index"] + \
                   [f"{c}_{s}" for c in feat_cols for s in ["mean", "std"]]
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(rows)
    return windower

watch_schema = make_window_schema(watch_feat)
phone_schema = make_window_schema(phone_feat)

print("Creating watch windows..i .")
watch_win = watch.groupBy("subject_id", "activity_label") \
                 .applyInPandas(make_windower(watch_feat), schema=watch_schema)

print("Creating phone windows...")
phone_win = phone.groupBy("subject_id", "activity_label") \
                 .applyInPandas(make_windower(phone_feat), schema=phone_schema)

watch_win.cache()
phone_win.cache()
print("watch windows:", watch_win.count())
print("phone windows:", phone_win.count())

# ── Split by subject_id (60/20/20) ────────────────────────────────────────────
subjects = [r.subject_id for r in watch_win.select("subject_id").distinct().collect()]
subjects.sort()
n = len(subjects)
train_subs = subjects[:int(n * 0.6)]
val_subs   = subjects[int(n * 0.6):int(n * 0.8)]
test_subs  = subjects[int(n * 0.8):]

print(f"Subjects — train: {len(train_subs)}, val: {len(val_subs)}, test: {len(test_subs)}")

watch_train = watch_win.filter(col("subject_id").isin(train_subs))
watch_val   = watch_win.filter(col("subject_id").isin(val_subs))
watch_test  = watch_win.filter(col("subject_id").isin(test_subs))

phone_train = phone_win.filter(col("subject_id").isin(train_subs))
phone_val   = phone_win.filter(col("subject_id").isin(val_subs))
phone_test  = phone_win.filter(col("subject_id").isin(test_subs))

# ── Build pipeline ────────────────────────────────────────────────────────────
def build_pipeline(feat_cols, prefix):
    indexer   = StringIndexer(inputCol="activity_label", outputCol="label", handleInvalid="keep")
    assembler = VectorAssembler(inputCols=feat_cols, outputCol=f"{prefix}_raw", handleInvalid="skip")
    scaler    = StandardScaler(inputCol=f"{prefix}_raw", outputCol=f"{prefix}_scaled",
                               withMean=True, withStd=True)
    rf        = RandomForestClassifier(
        featuresCol=f"{prefix}_scaled", labelCol="label",
        numTrees=100, seed=42
    )
    return Pipeline(stages=[indexer, assembler, scaler, rf])

watch_win_feat = [c for c in watch_win.columns if c not in {"subject_id", "activity_label", "window_index"}]
phone_win_feat = [c for c in phone_win.columns if c not in {"subject_id", "activity_label", "window_index"}]

watch_pipeline = build_pipeline(watch_win_feat, "watch")
phone_pipeline = build_pipeline(phone_win_feat, "phone")

# ── Train base models ─────────────────────────────────────────────────────────
print("Training watch RF...")
watch_model = watch_pipeline.fit(watch_train)

print("Training phone RF...")
phone_model = phone_pipeline.fit(phone_train)

# ── Val set accuracy of base models ──────────────────────────────────────────
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                               metricName="accuracy")
watch_val_preds = watch_model.transform(watch_val)
phone_val_preds = phone_model.transform(phone_val)

print(f"Watch RF val accuracy: {evaluator.evaluate(watch_val_preds):.4f}")
print(f"Phone RF val accuracy: {evaluator.evaluate(phone_val_preds):.4f}")

activity_key = {"A":"Walking","B":"Jogging","C":"Stairs","D":"Sitting","E":"Standing",
                "F":"Typing","G":"Teeth","H":"Soup","I":"Chips","J":"Pasta",
                "K":"Drinking","L":"Sandwich","M":"Kicking","O":"Catch",
                "P":"Dribbling","Q":"Writing","R":"Clapping","S":"Folding"}

def per_activity_accuracy(preds, name):
    print(f"\n{name} per-activity accuracy:")
    rows = (
        preds
        .withColumn("correct", (col("prediction") == col("label")).cast("int"))
        .groupBy("activity_label")
        .agg(avg("correct").alias("accuracy"), count("*").alias("windows"))
        .orderBy(col("accuracy").desc())
        .collect()
    )
    print(f"{'Activity':<20} {'Accuracy':>10} {'Windows':>10}")
    print("-" * 42)
    for r in rows:
        name_str = activity_key.get(r.activity_label, r.activity_label)
        print(f"{r.activity_label} — {name_str:<15} {r.accuracy:>9.1%} {r.windows:>10}")

per_activity_accuracy(watch_val_preds, "Watch RF")
per_activity_accuracy(phone_val_preds, "Phone RF")

def print_feature_importance(model, feat_cols, name, top_n=15):
    importances = model.stages[3].featureImportances
    pairs = sorted(zip(feat_cols, importances.toArray()), key=lambda x: x[1], reverse=True)
    print(f"\n{name} top {top_n} feature importances:")
    print(f"{'Feature':<30} {'Importance':>12}")
    print("-" * 44)
    for feat, imp in pairs[:top_n]:
        print(f"{feat:<30} {imp:>12.4f}")

print_feature_importance(watch_model, watch_win_feat, "Watch RF")
print_feature_importance(phone_model, phone_win_feat, "Phone RF")

# ── Stack: join val predictions on subject_id + activity_label + window_index ─
join_keys = ["subject_id", "activity_label", "window_index"]
watch_val_out = watch_val_preds.select(*join_keys, "label", col("probability").alias("watch_prob"))
phone_val_out = phone_val_preds.select(*join_keys, col("probability").alias("phone_prob"))

val_combined = watch_val_out.join(phone_val_out, on=join_keys, how="inner")
print("Val combined rows:", val_combined.count())

meta_assembler = VectorAssembler(inputCols=["watch_prob", "phone_prob"],
                                  outputCol="meta_features")
val_meta = meta_assembler.transform(val_combined)

print("Training meta Decision Tree...")
meta_dt    = DecisionTreeClassifier(featuresCol="meta_features", labelCol="label", seed=42)
meta_model = meta_dt.fit(val_meta)

# ── Test evaluation ───────────────────────────────────────────────────────────
watch_test_out = watch_model.transform(watch_test).select(*join_keys, "label", col("probability").alias("watch_prob"))
phone_test_out = phone_model.transform(phone_test).select(*join_keys, col("probability").alias("phone_prob"))

test_combined = watch_test_out.join(phone_test_out, on=join_keys, how="inner")
test_final    = meta_model.transform(meta_assembler.transform(test_combined))

evaluator.setPredictionCol("prediction")
evaluator.setMetricName("accuracy")
print(f"\nStacked model test accuracy: {evaluator.evaluate(test_final):.4f}")

# ── Per-activity accuracy ─────────────────────────────────────────────────────
# map numeric label back to activity name using StringIndexer labels
labels = watch_model.stages[0].labels  # index -> activity letter
label_map = {float(i): l for i, l in enumerate(labels)}
map_expr = create_map([lit(x) for pair in label_map.items() for x in pair])

per_activity = (
    test_final
    .withColumn("activity", map_expr[col("label")])
    .withColumn("correct", (col("prediction") == col("label")).cast("int"))
    .groupBy("activity")
    .agg(
        avg("correct").alias("accuracy"),
        count("*").alias("windows")
    )
    .orderBy(col("accuracy").desc())
)

print("\nPer-activity accuracy (test set):")
per_activity.show(truncate=False)
