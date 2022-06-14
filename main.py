import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tf = pd.read_csv('tensorflow_data.csv')

#dimensions
initial=tf.shape
print(initial)

#let's drop the rows with no authors or no assignees
tf = tf.drop(tf[tf['assignees.login'].isna()].index)
tf = tf.drop(tf[tf['author.login'].isna()].index)

#clean1=tf.shape
#print(clean1)
#drop empty columns
tf.dropna(axis=1, how='all', inplace=True)

#only comments
commentsOnly = tf.loc[(tf['createdAt'].isna()) & (tf['number'].isna()) & (tf['labels'].isna()) & (tf['state'].isna()) & (
    tf['title'].isna()) & (tf['body'].isna()) & (tf['updatedAt'].isna()) & (tf['closedAt'].isna())]

#convert createdAt to datetime
#tf['createdAt'] = pd.to_datetime(tf['createdAt'], utc=True)
#print(tf['createdAt'])

#add formatted date column to the dataframe
tf['date'] = pd.to_datetime(tf['createdAt'], utc=True, format="%d/%m/%Y")

# Separate date further into month and year columns
tf['month'] = tf['date'].dt.month
tf['year'] = tf['date'].dt.year
#aggregate amount of repository changes by year and month to see how active the project has been recently
counts = tf.groupby(['month', 'year'])['number'].sum().plot(kind='bar')
plt.ylabel("Number of changes")
plt.xlabel("Month and Year of changes")
plt.show()