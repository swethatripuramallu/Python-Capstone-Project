import pandas
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
'''
 The following is the starting code for path1 for data reading to make your first step easier.
 'dataset_1' is the clean data for path1.
'''

with open('behavior-performance.txt','r') as f:
    raw_data = [x.strip().split('\t') for x in f.readlines()]
df = pandas.DataFrame.from_records(raw_data[1:],columns=raw_data[0])
df['VidID']       = pandas.to_numeric(df['VidID'])
df['fracSpent']   = pandas.to_numeric(df['fracSpent'])
df['fracComp']    = pandas.to_numeric(df['fracComp'])
df['fracPlayed']  = pandas.to_numeric(df['fracPlayed'])
df['fracPaused']  = pandas.to_numeric(df['fracPaused'])
df['numPauses']   = pandas.to_numeric(df['numPauses'])
df['avgPBR']      = pandas.to_numeric(df['avgPBR'])
df['stdPBR']      = pandas.to_numeric(df['stdPBR'])
df['numRWs']      = pandas.to_numeric(df['numRWs'])
df['numFFs']      = pandas.to_numeric(df['numFFs'])
df['s']           = pandas.to_numeric(df['s'])
dataset_1 = df

# print(dataset_1[15620:25350].to_string()) #This line will print out the first 35 rows of your data

'''
    Question 1:
    For this analysis, we will use KMeans clustering algorithm to group or cluster students based on their video-watching behavior. 
    We will use the features fracSpent, fracComp, fracPaused, numPauses, avgPBR, numRWs, and numFFs for this analysis. We will only use 
    students that completed at least five videos. We will start with a small number of clusters and increase the number of clusters until
    we find the optimal number of clusters based on the within-cluster sum of squares (WCSS). We will use the elbow method to identify 
    the optimal number of clusters.
'''

print(
     "\nQuestion 1: How well can the students be naturally grouped or clustered by their video-watching behavior " + 
     "\n(fracSpent, fracComp, fracPaused, numPauses, avgPBR, numRWs, and numFFs)? You should use all students that" +  
     "\ncomplete at least five of the videos in your analysis. Hints: KMeans or distribution parameters " + 
     "\n(mean and standard deviation) of Gaussians. \n")

#Calculate the number of videos completed for each student
data = dataset_1[dataset_1['VidID'] == 0]
userId = data['userID']
videos_completed = dataset_1[dataset_1['fracComp'] >= 0.9]
num_videos = [sum(videos_completed['userID'] == user) for user in userId]   

#Filtering out the data
userId_filtered, num_videos_filtered = zip(*[(u, n) for u, n in zip(userId, num_videos) if n >= 5])
X = dataset_1[dataset_1['userID'].isin(userId_filtered)]

#Selecting the features that will be used in clustering
features = ['fracSpent', 'fracComp', 'fracPaused', 'numPauses', 'avgPBR', 'numRWs', 'numFFs']
X_features = X[features]

#Standardizing the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

#Determining the optimal number of clusters using the elbow method
wcss = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 20), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#From the plot, we can see that the optimal number of clusters is 7. Performing KMeans clustering with 7 clusters
kmeans = KMeans(n_clusters=7, init='k-means++', random_state=42, n_init = 10)
kmeans.fit(X_scaled)
cluster_labels = kmeans.labels_

#Adding the cluster labels to the dataframe
X.loc[:, 'Cluster'] = cluster_labels
X.loc[:, 'UserID'] = X.loc[:, 'userID'].tolist()  # use .loc instead of chained indexing


#Printing the size of the clusters
cluster_sizes = X['Cluster'].value_counts()
print(cluster_sizes)

#Printing the mean value of the features in the clusters: 
cluster_means = X.groupby('Cluster').mean()
print(cluster_means)

print(
    "\n" + "Can student's video-watching behavior be used to predict a student's performance" + 
    "(i.e., average score s across all quizzes)?" + "\n")

#Select the features to be used in the logistic regression
x = X['fracComp']
y = X['s'].apply(lambda x: 1 if x == 1 else 0)

#Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#Fit a logistic regression model to the training data
lr = LogisticRegression(random_state=42)

# Convert x_train to a 2D numpy array
x_train_2d = np.array(x_train).reshape(-1, 1)

# Fit the model on the training data
lr.fit(x_train_2d, y_train)

# Convert x_train to a 2D numpy array
x_test_2d = np.array(x_test).reshape(-1, 1)

#Predict the labels for the test data
y_pred = lr.predict(x_test_2d)

#Compute the accuracy of the predictions
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f%%' % (accuracy * 100))
print('\nLooking at the accuracy we can see there is a relationship' + 
      ' between fraction completed of the video and quiz score')


print('\nQuestion 3: Taking this a step further, how well can you predict a ' +
      'students performance on a particular in-video quiz question' + 
      '\n(i.e., whether they will be correct or incorrect) based on their video-watching behaviors' + 
      ' while watching the corresponding video? You should use all student-video pairs in your analysis.\n')


# Select the features to be used in the logistic regression
features2 = ['fracSpent', 'fracComp', 'fracPlayed', 'fracPaused', 'numPauses', 'avgPBR', 'stdPBR', 'numRWs', 'numFFs']
x2 = X[features2]
y2 = X['s'].apply(lambda x: 1 if x == 1 else 0)

# Split the data into training and testing sets
x_train2, x_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size=0.3, random_state=42)

# Fit a logistic regression model to the training data
lr = LogisticRegression(random_state=42)

# Fit the model on the training data
lr.fit(x_train2, y_train)

# Predict the labels for the test data
y_pred2 = lr.predict(x_test2)

# Compute the accuracy of the predictions
accuracy = accuracy_score(y_test2, y_pred2)
print('\nAccuracy: %.2f%%' % (accuracy * 100))