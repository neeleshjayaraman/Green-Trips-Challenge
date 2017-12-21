
#Importing standard libraries to aid in performing the data analysis

import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
sns.set_style('whitegrid')
from datetime import datetime


# # Question 1
# 
# 1. Programmatically download and load into your favorite analytical tool the trip data for September 2015.
# 
# 2. Report how many rows and columns of data you have loaded.



#Code to load the data into the iPython Notebook

trip_df= pd.read_csv('green_tripdata_2015-09.csv')
trip_df.head()



#Shows the columns in the dataset
trip_df.info()




#Displays the total number of rows and columns in the dataset

trip_df.shape


# # Question 2
# 
# 1. Plot a histogram of the number of the trip distance ("Trip Distance").
# 
# 2. Report any structure you find and any hypotheses you have about that structure.



trip_df['Trip_distance'].hist(bins=100)

trip_df['Trip_distance'].value_counts()

plt.hist(trip_df['Trip_distance'][trip_df['Trip_distance']<20],bins = 100)
plt.title('Histogram of trip distances')
plt.xlabel('Trip Distance')
plt.ylabel('No of occurences')


# ## This above Histogram shows us that the occurences were higher for short trip distances and the occurences reduced as the distance increased. The trips are not random. If they were random, we would have a symmetric Gaussian distribution.

# # Question 3
# 1. Report mean and median trip distance grouped by hour of day.
# 
# 2. We'd like to get a rough sense of identifying trips that originate or terminate at one of the NYC area airports. Can you provide a count of how many transactions fit this criteria, the average fair, and any other interesting characteristics of these trips.

trip_df['lpep_pickup_datetime'].head()

trip_df['pickup'] = trip_df['lpep_pickup_datetime'].apply(lambda x: 
                                                       datetime.strptime(x, '%Y-%m-%d %H:%M:%S'

#collecting hourly statistics in order to give the mean and median by the hour of the day
trip_df['pickup_hour'] = trip_df['pickup'].apply(lambda x: x.hour)

trip_df['dropoff'] = trip_df['Lpep_dropoff_datetime'].apply(lambda x: 
                                                       datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
trip_df['dropoff_hour'] = trip_df['dropoff'].apply(lambda x: x.hour)

#Plotting the mean and median trip distance grouped by the hour of the day

trip_df[['Trip_distance','pickup_hour']].groupby('pickup_hour').mean().plot.bar()
plt.title('Mean trip distance')
plt.ylabel('Miles')
plt.show()

trip_df[['Trip_distance','pickup_hour']].groupby('pickup_hour').median().plot.bar()
plt.title('Median trip distance')
plt.ylabel('Miles')
plt.show()


# Looking at the dataset we know that JFK and Newark are the two airports in the NYC area having RateCode IDs 2 and 3. So, we use them for our analysis.

#Average trips to/from NYC area airports
#Creating airport_trips to help in our analysis

airports_trips = trip_df[(trip_df.RateCodeID==2) | (trip_df.RateCodeID==3)]

print("Number of trips to/from NYC airports: ", airports_trips.shape[0])
print("Average fare of trips to/from NYC airports: $", airports_trips.Fare_amount.mean(),"per trip")
print("Average total charged amount of trips to/from NYC airports: $", airports_trips.Total_amount.mean(),"per trip")


# # Question 5
# Choose only one of these options to answer for Question 5. There is no preference as to which one you choose. Please select the question that you feel best suits your particular skills and/or expertise. If you answer more than one, only the first will be scored.
# Option A: Distributions
# o    Build a derived variable representing the average speed over the course of a trip.
# 
# o    Can you perform a test to determine if the average trip speeds are materially the same in all weeks of September? If you decide they are not the same, can you form a hypothesis regarding why they differ?
# 
# o    Can you build up a hypothesis of average trip speed as a function of time of day?


#import statistics libraries for visualization

import scipy.stats as stats
import statistics

ans_t = (trip_df['dropoff'] - trip_df['pickup']).apply(lambda x: x.total_seconds())


#Omitting entries with less than a minute travel time

print('Percentage of entries with travel time less than a minute: ',100 * trip_df[ans_t < 60].shape[0]/trip_df.shape[0],'%')

#Code for Finding the average speed
trip_df['travel_time'] = (trip_df['dropoff'] - trip_df['pickup']).apply(lambda x: x.total_seconds()) 
trip_df = trip_df[trip_df['travel_time'] > 60]
trip_df['average_speed'] = 3600*(trip_df['Trip_distance']/trip_df['travel_time'])

trip_df['average_speed'].plot.hist(bins=300)


# We remove the entries with over 100 miles per hour of average speed as it is unreasonable and a result of erroneous data.



print('No of entries with average speed over 100 miles per hour: ',(trip_df['average_speed']>200).value_counts()[1])
trip_df = trip_df[trip_df['average_speed']<200]



trip_df['week'] = trip_df['dropoff'].apply(lambda x: x.week)




week_1 = trip_df['average_speed'][trip_df['week']==36].as_matrix()
week_2 = trip_df['average_speed'][trip_df['week']==37].as_matrix()
week_3 = trip_df['average_speed'][trip_df['week']==38].as_matrix()
week_4 = trip_df['average_speed'][trip_df['week']==39].as_matrix()
week_5 = trip_df['average_speed'][trip_df['week']==40].as_matrix()




stats.f_oneway(week_1,week_2, week_3,week_4, week_5)


# ## The test indicates a large f-value and a small p-value, therefore we reject the null hypothesis and we conclude that the differences between the groups are statistically significant which implies that the week of the month does seem to be related to the average speed. 



print(week_1.mean(),week_2.mean(),week_3.mean(),week_4.mean(),week_5.mean())




print(statistics.median(week_1),statistics.median(week_2),statistics.median(week_3),statistics.median(week_4),
      statistics.median(week_5))




#WeeklyStatistics 
plt.rcParams["figure.figsize"] = [20,12]
plt.subplot(3,2,1)
plt.hist(week_1,bins = 50,label = 'week 1')
plt.legend()
plt.subplot(3,2,2)
plt.hist(week_2,bins = 50,label = 'week 2')
plt.legend()
plt.subplot(3,2,3)
plt.hist(week_3,bins = 50,label = 'week 3')
plt.legend()
plt.subplot(3,2,4)
plt.hist(week_4,bins = 50,label = 'week 4')
plt.legend()
plt.subplot(3,2,5)
plt.hist(week_5,bins = 50,label = 'week 5')
plt.legend()
plt.legend()
plt.savefig('task5')
plt.show()




grouped = trip_df.groupby('pickup_hour')
samples = []

for name,group in grouped:
    samples.append(group['average_speed'])



sample = samples
stats.f_oneway(sample[0],sample[1],sample[2],sample[3], sample[4],sample[5],sample[6],sample[7],sample[8],sample[9],
              sample[10],sample[11],sample[12],sample[13],sample[14],sample[15],sample[16],sample[17],sample[18],
               sample[19],
              sample[20],sample[21],sample[22],sample[23])


# 
# ## The test for sets partitioned as per the hour of the journey also gives a high f-value and p-value of 0, implying that there are statistifically significant differences in the data sets considered.



#Computing Mean and Median Speeds every hour
means = []
medians = []
for hour in range(24):
    means.append(statistics.mean(sample[hour]))
    print('Mean:',statistics.mean(sample[hour]))
    medians.append(statistics.median(sample[hour]))
    print('Median:',statistics.median(sample[hour]))




#Plotting Mean Speed at the hour
plt.rcParams["figure.figsize"] = [5,3]
index = np.arange(24)
bar_width = 0.35
opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = plt.bar(index, means, bar_width,
                 alpha=opacity,
                 color='b',
                 error_kw=error_config,
                 label='Average speed of a ride at the hour')
plt.legend()
plt.savefig('task_5_b_1')

plt.show()




#Plotting Median Speed at the hour

plt.rcParams["figure.figsize"] = [5,3]
index = np.arange(24)
bar_width = 0.35
opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = plt.bar(index, medians, bar_width,
                 alpha=opacity,
                 color='b',
                 error_kw=error_config,
                 label='Median speed of a ride at the hour')
plt.legend()
plt.savefig('task_5_b_2')

plt.show()


# # Question 4
# 1. Build a derived variable for tip as a percentage of the total fare.
# 
# 2. Build a predictive model for tip as a percentage of the total fare. Use as much of the data as you like (or all of it). We will validate a sample.


trip_df['tip_percent'] = trip_df['Tip_amount']/trip_df['Fare_amount']
trip_df['tip_percent'].head()



from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics



#Cleaning the data

def clean_dataset(trip_df):
    assert isinstance(trip_df, pd.DataFrame), "trip_df needs to be a pd.DataFrame"
    trip_df.dropna(inplace=True)
    indices_to_keep = ~trip_df.isin([np.nan, np.inf, -np.inf]).any(1)
    return trip_df[indices_to_keep].astype(np.float64)



trip_df.dropna(subset = ['Trip_type '],inplace = True)


# To deal with categorical values we would need to use the one-hot encoding process from the sklearn library which works only on numpy arrays, hence we convert the relevant columns to numpy arrays in further analysis and are encoded to one-hot vectors.


#One-Hot encoding 
X_pre_encode = trip_df['VendorID'].as_matrix()
le = preprocessing.LabelEncoder()
le.fit(X_pre_encode.reshape(X_pre_encode.shape[0],1))
ans = le.transform(X_pre_encode)

enc = OneHotEncoder()
enc.fit(ans.reshape(X_pre_encode.shape[0],1))
ans1 = enc.transform(ans.reshape(X_pre_encode.shape[0],1)).toarray()

X_pre_encode = trip_df['Store_and_fwd_flag'].as_matrix()
le = preprocessing.LabelEncoder()
le.fit(X_pre_encode.reshape(X_pre_encode.shape[0],1))
ans = le.transform(X_pre_encode)

enc = OneHotEncoder()
enc.fit(ans.reshape(X_pre_encode.shape[0],1))
ans2 = enc.transform(ans.reshape(X_pre_encode.shape[0],1)).toarray()

X = np.c_[ans1,ans2]
print(X.shape)


X = np.c_[X , trip_df['Passenger_count'].as_matrix(),trip_df['Trip_distance'].as_matrix(),trip_df['Fare_amount'].as_matrix(),
         trip_df['Extra'].as_matrix(),trip_df['MTA_tax'].as_matrix(),trip_df['Tolls_amount'].as_matrix(),
         trip_df['improvement_surcharge'].as_matrix()]


print(X.shape)


for col in ['Payment_type','Trip_type ','pickup_hour','dropoff_hour']:

    X_pre_encode = trip_df[col].as_matrix()
    le = preprocessing.LabelEncoder()
    le.fit(X_pre_encode.reshape(X_pre_encode.shape[0],1))
    ans = le.transform(X_pre_encode)

    enc = OneHotEncoder()
    enc.fit(ans.reshape(X_pre_encode.shape[0],1))
    ans2 = enc.transform(ans.reshape(X_pre_encode.shape[0],1)).toarray()
    X = np.c_[X,ans2]


# We choose 13 columns as features for regression.

X.shape



y = trip_df['tip_percent'].as_matrix()



np.isnan(X).any(), np.isnan(y).any()



from sklearn.preprocessing import Imputer
imputer= Imputer(missing_values='NaN', strategy= 'mean', axis =1) 
imputer.fit(y)



np.isnan(X).any(), np.isnan(y).any()



import sklearn
from sklearn.linear_model import LinearRegression



#Building the Linear Regression Model

lreg= LinearRegression()



# Train the model using the training sets and performing regression
lreg.fit(X, y)
# The coefficients
print('Coefficients: \n', lreg.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((lreg.predict(X) - y) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % lreg.score(X, y))



np.abs(lreg.coef_)


