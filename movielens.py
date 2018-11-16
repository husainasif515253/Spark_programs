####################################################import all the necessary modules#########################################################

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.types import Row
from pyspark.sql.functions import col , column, when, desc
import pyspark.sql.functions as f


######################################################create spark configurations############################################################

conf = SparkConf().setMaster('local').setAppName('test app')
sc = SparkContext(conf = conf)


#########################################################load the datasets in RDDs###########################################################

input1 = sc.textFile('/user/ubuntu1/husain/ml-1m/users.dat').map(lambda x:x.split('::'))
input2 = sc.textFile('/user/ubuntu1/husain/ml-1m/movies.dat').map(lambda x:x.split('::'))
input3 = sc.textFile('/user/ubuntu1/husain/ml-1m/ratings.dat').map(lambda x:x.split('::'))


###########################################################Create spark session##############################################################

spark = SparkSession(sc)
spark.sparkContext.setLogLevel('WARN')
hasattr(input1, "toDF")
hasattr(input2, "toDF")
hasattr(input3, "toDF")


##########################################################Convert RDDs to dataframes#########################################################

users = input1.toDF(['user_id','gender','age','occupation','zip_code'])
movies = input2.toDF(['movie_id','title','genre'])
ratings = input3.toDF(['user_id','movie_id','rating','timestamp'])


############################################Print the counts of rows and columns of each dataframes##########################################

#print("Number of rows in users dataset is: ", users.count(), " number of columns in users dataset is: ", len(users.columns))
#print("Number of rows in movies dataset is: ", movies.count(), " number of columns in movies dataset is: ", len(movies.columns))
#print("Number of rows in ratings dataset is: ", ratings.count(), " number of columns in ratings dataset is: ", len(ratings.columns))


###########################################################Print all the 3 datasets##########################################################

#users.show()
#movies.show()
#ratings.show()


#####################################################Print the schemas of all the datasets###################################################

#users.printSchema()
#movies.printSchema()
#ratings.printSchema()


##################################################Join the datasets to obtain the final dataset##############################################

users_ratings = users.join(ratings,['user_id'], "inner")
#print("Number of rows in users_ratings dataset is: ", users_ratings.count(), " number of columns in users_ratings dataset is: ", len(users_ratings.columns))
#users_ratings.show()
dataset = users_ratings.join(movies,['movie_id'], "inner")
#print("Number of rows in 'dataset' dataset is: ", dataset.count(), " number of columns in 'dataset' dataset is: ", len(dataset.columns))
#dataset.show()
#dataset.printSchema()


###########################################Convert the datatypes of the columns in respective formats########################################

dataset = dataset.withColumn("age", col("age").cast("integer"))
dataset = dataset.withColumn("rating", col("rating").cast("integer"))
dataset = dataset.withColumn("timestamp", f.from_unixtime("timestamp", "dd/MM/yyyy HH:MM:SS"))
#dataset.show()


#############################################Replacing values of occupation column with respective names#####################################

dataset = dataset.withColumn("occupation", when(dataset["occupation"] == 0, "other").otherwise(when(dataset.occupation == 1, "academic/educator").otherwise(when(dataset.occupation == 2, "artist").otherwise(when(dataset.occupation == 3, "clerical/admin").otherwise(when(dataset.occupation == 4, "college/grad student").otherwise(when(dataset.occupation == 5, "customer service").otherwise(when(dataset.occupation == 6, "doctor/health care").otherwise(when(dataset.occupation == 7, "executive/managerial").otherwise(when(dataset.occupation == 8, "farmer").otherwise(when(dataset.occupation == 9, "homemaker").otherwise(when(dataset.occupation == 10, "K-12 student").otherwise(when(dataset.occupation == 11, "lawyer").otherwise(when(dataset.occupation == 12, "programmer").otherwise(when(dataset.occupation == 13, "retired").otherwise(when(dataset.occupation == 14, "sales/marketting").otherwise(when(dataset.occupation == 15, "scientist").otherwise(when(dataset.occupation == 16, "self-employed").otherwise(when(dataset.occupation == 17, "technician/engineer").otherwise(when(dataset.occupation == 18, "tradesman/craftsman").otherwise(when(dataset.occupation == 19, "unemployed").otherwise(when(dataset.occupation == 20, "writer").otherwise(dataset['occupation']))))))))))))))))))))))

dataset.show()

#dataset.printSchema()
#dataset.describe().show()

##################################################Split the dataset to training and testing###################################################

#training = dataset.sample(False, 0.8, 42)
#testing = dataset.sample(False, 0.2, 43)
#print("Number of rows in training dataset is: ", training.count(), " number of columns in training dataset is: ", len(training.columns))
#print("Number of rows in testing dataset is: ", testing.count(), " number of columns in testing dataset is: ", len(testing.columns))


######################################################Analysis on 'dataset' dataset###########################################################

#---------- find the average ratings of movies ordered in descending manner ------------ 
#movie_avg_rating = dataset.groupby('title').agg({'rating':'mean'}).select([f.col("title").alias("title"), f.col("avg(rating)").alias("avg_rating")])
#movie_avg_rating.show()
#movie_avg_rating.orderBy(movie_avg_rating.avg_rating.desc()).show()

#---------- Finding the generes of movies that each gender prefer ----------------------

#f_genre = dataset.select('gender', 'genre')
#f_genre = f_genre.filter(f_genre.gender == 'F')
#print(f_genre.count())
#f_genre = f_genre.groupby('genre').count()
#f_genre.show(truncate = False)

#m_genre = dataset.select('gender', 'genre')
#m_genre = m_genre.filter(m_genre.gender == 'M')
#print(m_genre.count())
#m_genre = m_genre.groupby('genre').count()
#m_genre.show(truncate = False)

#-------------Finding what kind of occupations watch more movies------------------------

#occupation_count = dataset.select('occupation').groupby('occupation').count()
#type(occupation_count)
#occupation_count.orderBy(occupation_count.count.desc()).show()

#------------------Finding the genre of movies based on age------------------------------

#age_group = dataset.select('age').groupby('age').count()
#age_group.show()
#age_list = age_group.select('age')
#age_array = [int(i.age) for i in age_list.collect()]
#print(age_array)



