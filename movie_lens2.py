from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.types import Row, DateType
from pyspark.sql.functions import col , column, when, desc
import pyspark.sql.functions as f
import matplotlib.pyplot as plt
import pandas
import numpy
import re
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder

conf = SparkConf().setMaster('yarn').setAppName('test app')
sc = SparkContext(conf = conf)
sqlContext = SQLContext(sc)


input1 = sc.textFile('file:///home/ubuntu1/husain/ml-1m/users.dat').map(lambda x:x.split('::'))
input2 = sc.textFile('file:///home/ubuntu1/husain/ml-1m/movies.dat').map(lambda x:x.split('::'))
input3 = sc.textFile('file:///home/ubuntu1/husain/ml-1m/ratings.dat').map(lambda x:x.split('::'))

spark = SparkSession(sc)
spark.sparkContext.setLogLevel('WARN')
hasattr(input1, "toDF")
hasattr(input2, "toDF")
hasattr(input3, "toDF")

users = input1.toDF(['user_id','gender','age','occupation','zip_code'])
movies = input2.toDF(['movie_id','title','genre'])
ratings = input3.toDF(['user_id','movie_id','rating','timestamp'])

users_ratings = users.join(ratings,['user_id'], "inner")
dataset = users_ratings.join(movies,['movie_id'], "inner")

dataset = dataset.withColumn("age", col("age").cast("integer"))
dataset = dataset.withColumn("user_id", col("user_id").cast("integer"))
dataset = dataset.withColumn("movie_id", col("movie_id").cast("integer"))
dataset = dataset.withColumn("rating", col("rating").cast("integer"))
dataset = dataset.select('*', (f.from_unixtime('timestamp').cast(DateType())).alias('date'))
dataset = dataset.select('*',f.year(dataset.date).alias('year'))
dataset = dataset.withColumn("timestamp", f.from_unixtime("timestamp", "dd/MM/yyyy HH:MM:SS"))

dataset = dataset.withColumn("occupation", when(dataset["occupation"] == 0, "other").otherwise(when(dataset.occupation == 1, "academic/educator").otherwise(when(dataset.occupation == 2, "artist").otherwise(when(dataset.occupation == 3, "clerical/admin").otherwise(when(dataset.occupation == 4, "college/grad student").otherwise(when(dataset.occupation == 5, "customer service").otherwise(when(dataset.occupation == 6, "doctor/health care").otherwise(when(dataset.occupation == 7, "executive/managerial").otherwise(when(dataset.occupation == 8, "farmer").otherwise(when(dataset.occupation == 9, "homemaker").otherwise(when(dataset.occupation == 10, "K-12 student").otherwise(when(dataset.occupation == 11, "lawyer").otherwise(when(dataset.occupation == 12, "programmer").otherwise(when(dataset.occupation == 13, "retired").otherwise(when(dataset.occupation == 14, "sales/marketting").otherwise(when(dataset.occupation == 15, "scientist").otherwise(when(dataset.occupation == 16, "self-employed").otherwise(when(dataset.occupation == 17, "technician/engineer").otherwise(when(dataset.occupation == 18, "tradesman/craftsman").otherwise(when(dataset.occupation == 19, "unemployed").otherwise(when(dataset.occupation == 20, "writer").otherwise(dataset['occupation']))))))))))))))))))))))

dataset = dataset.select('movie_id','user_id','gender','age','zip_code','rating','timestamp','title','genre','date','year', f.regexp_replace(col("occupation"), "/", "|").alias('occupation'))

#dataset.printSchema()

df = dataset.select([c for c in dataset.columns if c in ['user_id', 'movie_id', 'rating']])
#print(df)
#df.show()

(training, test) = df.randomSplit([0.8, 0.2])

als = ALS(userCol="user_id", itemCol="movie_id", ratingCol="rating", coldStartStrategy="drop", nonnegative=True)

param_grid = ParamGridBuilder().addGrid(als.rank, [12,13,14]).addGrid(als.maxIter, [18,19,20]).addGrid(als.regParam, [.17,.18,.19]).build()

evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

tvs = TrainValidationSplit(estimator = als, estimatorParamMaps = param_grid, evaluator=evaluator)

model = tvs.fit(training)

best_model = model.bestModel

predictions = best_model.transform(test)

rmse = evaluator.evaluate(predictions)

print("RMSE" + str(rmse))
print("***** Best Model *********")
print("Rank "), best_model.rank
print("MaxIter"), best_model._java_obj.parent().getMaxIter()
print("RegParam"), best_model._java_obj.parent().getRegParam()

users_recs = best_model.recommendForAllUsers(10)
l = users_recs.where(users_recs.user_id == 1)
x = l.withColumn("recommendations", explode(l.recommendations))
y = x.select('user_id', 'recommendations.*')
final = y.join(movies, ["movie_id"], "inner")


movies_recs = best_model.recommendForAllItems(10)
m = movies_recs.where(movies_recs.movie_id == 1871)
n = m.withColumn("recommendations", explode(m.recommendations))
o = n.select('movie_id', 'recommendations.*')
final2 = o.join(users, ["user_id"], "inner")

print(m)
type(m)
#movies_recs.show(truncate = False)
#print(movies_recs)
#users_recs.toPandas().to_csv("users_recs.csv")

#get_recs_for_user(1).show()
#get_recs_for_user(2).show()

#predictions.orderBy("user_id", "rating").show()
