from sklearn import svm
def TrainClassifier(train_features, train_labels):
    clf =  svm.SVC(gamma="auto")
    clf.fit(train_features, train_labels)
    return clf