import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots()
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
counts = tf.groupby(['month', 'year'])['number'].sum().reset_index()
#avg = df.groupby("b", sort=False)["a"].mean()\.reset_index(name="mean")
#counts.rename(columns = {'month':'month', 'year':'year', '':'sum'}, inplace = True)

print(counts)
ax = sns.barplot(x='month', y='number',data=counts, order= counts.sort_values(['year', 'month'], ascending=[True, True]).month)
#plt.title("Recent Contributions to the TensorFlow Repository")
#fig.tight_layout()
plt.show()

#show distribution of users and contributions
user_dist = tf.groupby('author.login')["number"].count().nlargest(20)
sns.histplot(user_dist)
plt.title("Distribution of users working on the repository")
plt.xlabel("Number of users")
plt.ylabel("Number of changes")
plt.show()

# contributions by top 20 users
author_counts = tf.groupby("author.login")["number"].count().nlargest(20).reset_index()
print(author_counts)
author=sns.barplot(x=author_counts['author.login'], y=author_counts['number'], data=author_counts)
#ax = author_counts.plot.bar(title="Contribution of top 20 users", xlabel="Usernames", ylabel="Number of Changes")
plt.subplots_adjust(bottom=0.4)
plt.xticks(rotation=90)
plt.show()


