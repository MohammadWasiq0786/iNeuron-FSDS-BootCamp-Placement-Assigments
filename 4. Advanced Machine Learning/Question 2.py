"""
# Submitted by: Mohammad Wasiq

## Email: gl0427@myamu.ac.in

# Placement Advanced Machine Learning Assignment 2
"""

'''
Q2. A chemist had two chemical flasks labeled 0 and 1 which consist of two
different chemicals. He extracted 3 features from these chemicals in order to
distinguish between them, you provided the results derived by the chemicals and
your task is to create a model that will label chemical 0 or 1 given its three features
and built-in docker and use some library to display that in frontend.
Note : Use only pyspark

Dataset:- https://www.kaggle.com/datasets/uciml/indian-liver-patient-records
'''

# Ans:


# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.sql.functions import col

# Create a Spark session
spark = SparkSession.builder.appName("ChemicalClassification").getOrCreate()

# Load the dataset
data = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("path_to_dataset.csv")

# Print the schema of the dataset
data.printSchema()

# Select relevant columns for training
selected_columns = ['feature1', 'feature2', 'feature3', 'label']
data = data.select(*selected_columns)

# Convert the label column to numeric type
data = data.withColumn("label", col("label").cast("double"))

# Split the dataset into training and testing sets
(training_data, testing_data) = data.randomSplit([0.8, 0.2], seed=42)

# Create a vector assembler to combine the features into a single vector column
assembler = VectorAssembler(inputCols=['feature1', 'feature2', 'feature3'], outputCol="features")

# Create a logistic regression classifier
lr = LogisticRegression(labelCol="label", featuresCol="features")

# Create a pipeline to assemble the features and train the model
pipeline = Pipeline(stages=[assembler, lr])

# Train the model
model = pipeline.fit(training_data)

# Make predictions on the testing data
predictions = model.transform(testing_data)

# Display the predictions
predictions.select("features", "label", "prediction").show()

# Stop the Spark session
spark.stop()

