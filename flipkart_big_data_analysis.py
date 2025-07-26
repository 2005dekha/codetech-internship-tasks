
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, desc

# Step 1: Start Spark Session
spark = SparkSession.builder \
    .appName("Flipkart E-Commerce Big Data Analysis") \
    .getOrCreate()

# Step 2: Load Dataset
df = spark.read.csv("flipkart_sample.csv", header=True, inferSchema=True)

# Step 3: Preview Data
df.show(5)
df.printSchema()

# Step 4: Clean Data - Drop nulls
df_clean = df.dropna()

# Step 5: Top Categories by Product Count
top_categories = df_clean.groupBy("category").agg(count("*").alias("product_count")) \
    .orderBy(desc("product_count"))
top_categories.show(10)

# Step 6: Average Product Rating per Category
avg_ratings = df_clean.groupBy("category").agg(avg("rating").alias("avg_rating")) \
    .orderBy(desc("avg_rating"))
avg_ratings.show(10)

# Step 7: Most Expensive Products
top_expensive = df_clean.select("product_name", "price", "category") \
    .orderBy(desc("price")) \
    .limit(10)
top_expensive.show()

# Step 8: Save insights to CSV (optional)
top_categories.write.csv("output/top_categories", header=True, mode="overwrite")
avg_ratings.write.csv("output/avg_ratings", header=True, mode="overwrite")
top_expensive.write.csv("output/top_expensive", header=True, mode="overwrite")

# Step 9: Stop Spark Session
spark.stop()
