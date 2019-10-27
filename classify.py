from sklearn import svm
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import pyplot
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from sklearn import metrics

def SVM_linear(x_train, x_test, y_train):
    # Initialize models
    clf_linear = svm.SVC(kernel='linear', C=1)

    # Train models
    clf_linear.fit(x_train, y_train)

    # Predict test-values
    y_pred_linear = clf_linear.predict(x_test)

    return y_pred_linear

def SVM_rbf(x_train, x_test, y_train):
    # Initialize models
    clf_rbf = svm.SVC(kernel='rbf', C=1)

    # Train models
    clf_rbf.fit(x_train, y_train)

    # Predict test-values
    y_pred_rbf = clf_rbf.predict(x_test)

    return y_pred_rbf

def combine(y_pred_linear, y_pred_rbf):
    y_final = []
    for i in range(0,len(y_pred_linear)):
        if y_pred_linear[i] == 3:
            if y_pred_rbf[i] == 2 or y_pred_rbf[i] == 4:
                y_final.append(y_pred_rbf[i])
            else:
                y_final.append(y_pred_linear[i])
        else:
            y_final.append(y_pred_linear[i])

    return y_final

def dummy(x_train, x_test, y_train):
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(x_train, y_train)
    y_pred = dummy.predict(x_test)
    # Evaluate model
    return y_pred

def matrix(y_test, y_pred, combined_matrix, test_clip):
    #Saves all confusion matrix to one combined

    # Confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred, labels = [1,2,3,4,5])

    #conbined_matrix
    for row in range(5):
        for elem in range(5):
            combined_matrix[row][elem] += conf_mat[row][elem]

    print(conf_mat)
    print('combined_matrix', combined_matrix)
    fig, ax = pyplot.subplots(figsize=(5, 5))
    labels = ['Väldigt oengagerad', 'Oengagerad', 'Neutral', 'Engagerad', 'Väldigt engagerad']
    sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=labels, yticklabels=labels,
                cmap="YlGnBu_r")
    pyplot.ylabel('Actual')
    pyplot.xlabel('Predicted')
    ax.set_title(test_clip)
    pyplot.show()
    fig, ax = pyplot.subplots(figsize=(5, 5))
    labels = ['Väldigt oengagerad', 'Oengagerad', 'Neutral', 'Engagerad', 'Väldigt engagerad']
    sns.heatmap(combined_matrix, annot=True, fmt='d', xticklabels=labels, yticklabels=labels,
                cmap="YlGnBu_r")
    pyplot.ylabel('Actual')
    pyplot.xlabel('Predicted')
    ax.set_title('Combined Matrix')
    pyplot.show()
    return combined_matrix

def evaluate(y_pred, y_test, classifier, test_data):
    print("Accuracy of " + classifier + " with test_data: " + test_data +" = ", metrics.accuracy_score(y_test, y_pred))
    #print('Precision of ' + classifier + ' = ', metrics.average_precision_score(y_test, y_pred))


    if classifier == "Dummy":
        print('This batch had:' + str(len(y_test)) + " frames, and " + str(len(y_test)/150) + " clips in the test_set.")

    print('\n')


<<<<<<< HEAD
    return metrics.accuracy_score(y_test, y_pred)

=======
    # Confusion matrix
    #comb_matrix = matrix(y_test, y_pred)

    # Data
    print(classification_report(y_test, y_pred))
    return metrics.accuracy_score(y_test, y_pred)
>>>>>>> 46c68dfe02f1047556af33bf35ee4efe6e0a3977
