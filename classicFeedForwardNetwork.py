from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import genLib as GL
from sklearn.metrics import roc_auc_score, accuracy_score,f1_score,precision_recall_curve,auc,precision_score,recall_score
from sklearn import svm
from sklearn.preprocessing import normalize
import time

equivClassifier=MLPClassifier(hidden_layer_sizes=(4,4), max_iter=300, activation="logistic",solver="sgd")
svmClassifier=svm.SVC(kernel="rbf",gamma="auto")


X,Y = GL.get_iris()


NUM_DATA=len(X)
train_perc=0.75
print("prima riga pre normalizzazione",X[0],"   ",Y[0])
#features=normalize(X, norm="l1",axis=0)
features=X
print("prima riga post normalizzazione", features[0],"   ", Y[0])
if NUM_DATA==len(features): print("corrette dimesioni")

# print tutti i dati
# for x,y in zip(X, Y):
#     print(f"x = {x}, y = {y}")        

print(len(features))

feats_train, feats_val, Y_train, Y_val = train_test_split(
    features, Y, train_size=train_perc
)
inizio =time.time()
equivClassifier.fit(feats_train, Y_train)
svmClassifier.fit(feats_train, Y_train)

predictions = equivClassifier.predict(feats_val)
predictions2 = svmClassifier.predict(feats_val)

fine = time.time()
print(accuracy_score(Y_val, predictions))
print(f1_score(Y_val, predictions))
print(precision_score(Y_val, predictions))
print(recall_score(Y_val, predictions))
print(predictions)

print("----------------")

print(fine-inizio)
# print(accuracy_score(Y_val, predictions2))
# print(f1_score(Y_val, predictions2))
# print(precision_score(Y_val, predictions2))
# print(recall_score(Y_val, predictions2))
# print(predictions2)
