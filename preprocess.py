from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
import pandas as pd
import numpy

def run(votes, test_data):
    votes = make_unique(votes)
    votes_training, votes_test_dev = splitData(votes, test_data)
    x_train = set_x(votes_training)
    x_test = set_x(votes_test_dev)
    y_train = set_y(votes_training)
    y_test = set_y(votes_test_dev)

    x_train, x_test = standardize(x_train, x_test)

    return x_train, x_test, y_train, y_test

def make_unique(votes): # Set the average vote of each unique video
    votes_on_same_video = 1
    unique_votes = []
    for i in range(len(votes)):

        if i != len(votes)-1:
            if votes[i][0] == votes[i+1][0]:
                votes_on_same_video += 1
                continue

        if votes_on_same_video > 1:
            for j in range(votes_on_same_video-1, -1, -1):
                rating += votes[i-j][1]

            rating = rating / votes_on_same_video

            unique_vote = (votes[i][0], int(round(rating)))
            unique_votes.append(unique_vote)
            votes_on_same_video = 1
        else:
            unique_votes.append(votes[i])

        rating = 0

    return unique_votes

def set_x(votes):  # Get features (AU & Gaze) based on videoName
    dataFrames = []
    for vote in votes:
        # Select relevant rows to extract based on timestamp
        time = int(vote[0][-3:])
        rows = list(range(time * 150 + 1, (time * 150) + 150 + 1, 1))
        rows.append(0) # Headers

        # Get data from csv-file
        dataFrame = pd.read_csv("./../CSV-filer/" + vote[0][:-4] + ".csv", skiprows=lambda x: x not in rows)
        #dataFrame.drop(dataFrame.columns[0:5], axis=1, inplace=True) # Too drop columns instead/if use all features
        gaze_y = dataFrame[dataFrame.columns[12:13]]
        dataFrame = dataFrame[dataFrame.columns[-35:-18]]
        dataFrame["Eyegaze_y"] = gaze_y
        dataFrames.append(dataFrame)

    # Extract all x-vectors from each Dataframe
    x = []
    for dataFrame in dataFrames:
        for row in dataFrame.iterrows():
            x.append(row[1])

    return x

def standardize(x_train, x_test): # Make all features scaled the same way
    sc = StandardScaler()

    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    return x_train, x_test

def set_y(votes): # Get all labels
    y = []
    for vote in votes:
        for i in range(150):  # Every sequence consist of 150 frames/rows
            y.append(vote[1])

    return y

def splitData(votes, test_data):
    votes_training = []
    votes_test = []

    for vote in votes:
        if vote[0][0:-4] in test_data:
            votes_test.append(vote)
        else:
            votes_training.append(vote)

    return votes_training, votes_test

def best_features(x, y):
    best_features = SelectKBest(k=10)
    fit = best_features.fit(x, y)

    numpy.set_printoptions(precision=3)
    print(fit.scores_)

    x_new = best_features.fit_transform(x)

    return x_new, y
