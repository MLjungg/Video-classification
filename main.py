import dataBase
import classify
import preprocess


def ignore_very(y):
    for i in range(len(y)):
        if y[i] == 1:
            y[i] = 2
        elif y[i] == 5:
            y[i] = 4
    return y

# Connect to database
db = dataBase.connect()

# Get votes from database. Holds in tuple: (videoName, vote)
votes = dataBase.getVotes(db)

# Clips in data
clips = [
        'sc53.sc52.personal.sc52',
        'sc.56.sc57.personal.sc56',
        'sc.56.sc57.personal.sc57',
        'sc58.sc61.personal.sc61',
        'sc59.sc60.personal.sc59',
        'sc59.sc60.personal.sc60',
        'sc62.sc63.personal.sc63',
        'sc66.sc67.personal.sc66',
        'sc66.sc67.personal.sc67',
        'sc74.sc75.personal.sc74',
        'sc74.sc75.personal.sc75',
        'sc76.sc77.personal.sc76',
        'sc76.sc77.personal.sc77',
        'sc82.sc83.personal.sc83',
        'sc84.sc65.personal.sc65']

number_of_clips = 0
accuracy_svm = 0
accuracy_dummy = 0
for test_clip in clips:
    # Extract x,y-test/train vectors from data and plot dist. Linear.
    x_train, x_test, y_train, y_test = preprocess.run(votes, test_clip)

    # Classify using SVM with different kernels
    y_pred_linear = classify.SVM_linear(x_train, x_test, y_train)

    y_pred_rbf = classify.SVM_rbf(x_train, x_test, y_train)

    # Combine classifiers
    y_final = classify.combine(y_pred_linear, y_pred_rbf)

    # Make very engaged -> engaged.
    y_final = ignore_very(y_final)
    y_test = ignore_very(y_test)

    # Evaluate model
    accuracy_svm += classify.evaluate(y_final, y_test, "SVM", test_clip) * len(y_test)

    # Classify using dummy to get baseline
    y_dummy = classify.dummy(x_train, x_test, y_train)

    # Evaluate dummy
    accuracy_dummy += classify.evaluate(y_dummy, y_test, "Dummy", test_clip) * len(y_test)

    # Count number of clips
    number_of_clips += len(y_test)

    # matrix
    classify.matrix(y_test, y_final)

print("Final results:")
print("SVM:")
print(accuracy_svm/number_of_clips)
print("Dummy:")
print(accuracy_dummy/number_of_clips)
