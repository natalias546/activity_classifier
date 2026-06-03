import os
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS = os.path.join(BASE, "plots")
os.makedirs(PLOTS, exist_ok=True)

conf = pyspark.SparkConf().setAll([
    ('spark.master', 'local[6]'),
    ('spark.app.name', 'EDA'),
    ('spark.driver.memory', '6g'),
    ('spark.executor.memory', '6g'),
    ('spark.sql.shuffle.partitions', '12'),
])
spark = SparkSession.builder.config(conf=conf).getOrCreate()

activity_key = {
    "A": "Walking",   "B": "Jogging",    "C": "Stairs",   "D": "Sitting",
    "E": "Standing",  "F": "Typing",     "G": "Teeth",    "H": "Soup",
    "I": "Chips",     "J": "Pasta",      "K": "Drinking", "L": "Sandwich",
    "M": "Kicking",   "O": "Catch",      "P": "Dribbling","Q": "Writing",
    "R": "Clapping",  "S": "Folding"
}

# Load 
watch = spark.read.parquet(f"{BASE}/data/watch_features.parquet")
phone = spark.read.parquet(f"{BASE}/data/phone_features.parquet")

meta_cols  = {"subject_id", "activity_label", "seq_id", "timestamp"}
watch_feat = [c for c in watch.columns if c not in meta_cols]
phone_feat = [c for c in phone.columns if c not in meta_cols]

# Dataset Overview 
print("Dataset")
print(f"Watch rows:     {watch.count():,}")
print(f"Phone rows:     {phone.count():,}")
print(f"Watch features: {len(watch_feat)}")
print(f"Phone features: {len(watch_feat)}")
print(f"Watch subjects: {watch.select('subject_id').distinct().count()}")
print(f"Phone subjects: {phone.select('subject_id').distinct().count()}")
print(f"Activities:     {watch.select('activity_label').distinct().count()}")

print("Activity dist")

watch_counts = watch.groupBy("activity_label").count().orderBy("activity_label").toPandas()
phone_counts = phone.groupBy("activity_label").count().orderBy("activity_label").toPandas()
watch_counts["name"] = watch_counts["activity_label"].map(activity_key)
phone_counts["name"] = phone_counts["activity_label"].map(activity_key)

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
for ax, df, title in [(axes[0], watch_counts, "Watch"), (axes[1], phone_counts, "Phone")]:
    bars = ax.bar(df["name"], df["count"], color=sns.color_palette("husl", len(df)))
    ax.set_title(f"{title} — Activity Distribution", fontsize=13)
    ax.set_xlabel("Activity")
    ax.set_ylabel("Row Count")
    ax.tick_params(axis="x", rotation=45)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=7)
plt.tight_layout()
plt.savefig(f"{PLOTS}/eda_activity_distribution.png", dpi=150)
plt.show()

print("Rows per subject")

watch_subj = watch.groupBy("subject_id").count().orderBy("subject_id").toPandas()
phone_subj = phone.groupBy("subject_id").count().orderBy("subject_id").toPandas()

fig, axes = plt.subplots(1, 2, figsize=(18, 5))
for ax, df, title in [(axes[0], watch_subj, "Watch"), (axes[1], phone_subj, "Phone")]:
    ax.bar(df["subject_id"].astype(str), df["count"])
    ax.set_title(f"{title} — Rows per Subject", fontsize=13)
    ax.set_xlabel("Subject ID")
    ax.set_ylabel("Row Count")
    ax.tick_params(axis="x", rotation=90)
    ax.axhline(df["count"].mean(), color="red", linestyle="--", label=f"Mean: {int(df['count'].mean()):,}")
    ax.legend()
plt.tight_layout()
plt.savefig(f"{PLOTS}/eda_rows_per_subject.png", dpi=150)
plt.show()

# Stats
print("Stats — WATCH (raw sensor)")
watch.select("wa_x", "wa_y", "wa_z", "wg_x", "wg_y", "wg_z").describe().show()

print("DESCRIPTIVE STATISTICS — PHONE (raw sensor)")
phone.select("pa_x", "pa_y", "pa_z", "pg_x", "pg_y", "pg_z").describe().show()

#  Sample to pandas for plotting 
watchpd = watch.sample(fraction=0.05, seed=42).toPandas()
phonepd = phone.sample(fraction=0.05, seed=42).toPandas()
watchpd["activity_name"] = watchpd["activity_label"].map(activity_key)
phonepd["activity_name"] = phonepd["activity_label"].map(activity_key)

# 6. Raw sensor boxplots by activity 
for source, df, raw_cols, title in [
    ("watch", watchpd, ["wa_x","wa_y","wa_z","wg_x","wg_y","wg_z"], "Watch"),
    ("phone", phonepd, ["pa_x","pa_y","pa_z","pg_x","pg_y","pg_z"], "Phone"),
]:
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    for ax, c in zip(axes.flatten(), raw_cols):
        df.boxplot(column=c, by="activity_name", ax=ax)
        ax.set_title(c, fontsize=11)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
    plt.suptitle(f"{title} — Raw Sensor Values by Activity", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/eda_{source}_boxplots.png", dpi=150)
    plt.show()

#7. Magnitude distribution by activity 
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
for ax, df, col_, title in [
    (axes[0], watchpd, "wa_x_magnitude", "Watch Accel Magnitude"),
    (axes[1], phonepd, "pa_x_magnitude", "Phone Accel Magnitude"),
]:
    order = sorted(df["activity_name"].unique())
    sns.boxplot(data=df, x="activity_name", y=col_, order=order, ax=ax,
                palette="husl")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Activity")
    ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(f"{PLOTS}/eda_magnitude_by_activity.png", dpi=150)
plt.show()

# Feature correlation heatmap 
raw_watch = ["wa_x","wa_y","wa_z","wg_x","wg_y","wg_z",
             "wa_x_magnitude","wg_x_magnitude","wa_x_jerk","wg_x_jerk"]
raw_phone = ["pa_x","pa_y","pa_z","pg_x","pg_y","pg_z",
             "pa_x_magnitude","pg_x_magnitude","pa_x_jerk","pg_x_jerk"]

for df, cols, title, fname in [
    (watchpd, raw_watch, "Watch", "watch"),
    (phonepd, raw_phone, "Phone", "phone"),
]:
    corr = df[cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                xticklabels=cols, yticklabels=cols)
    plt.title(f"{title} — Key Feature Correlation", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{PLOTS}/eda_{fname}_correlation.png", dpi=150)
    plt.show()



print("EDA complete. Plots saved to project directory.")
