

import numpy as np
 
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

FAST = True
RANDOM_STATE = 42

mnist = fetch_openml("mnist_784", version=1, as_frame=False)

X_all = mnist.data.astype("float32") / 255.0
y_all = mnist.target.astype("int64")

if FAST:
    X_all, _, y_all,_= train_test_split(
        X_all,
        y_all,
        train_size=10000,
        random_state=RANDOM_STATE,
        stratify=y_all)

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_all, y_all, test_size=0.30, random_state=RANDOM_STATE, stratify=y_all
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.28, random_state=RANDOM_STATE, stratify=y_train_val
)


pipe_knn= Pipeline([("scaler",MinMaxScaler()),("knn",KNeighborsClassifier())])

grid_knn = GridSearchCV(
    pipe_knn,
    {
        "knn__n_neighbors":[3, 5, 7, 9],
        "knn__weights":["uniform", "distance"],
        "knn__metric": ["euclidean", "manhattan"],},
    scoring="accuracy",
    cv= 2, n_jobs =-1)

grid_knn.fit(X_train,y_train)

y_val_pred_knn = grid_knn.predict(X_val)
acc_val_knn = accuracy_score(y_val, y_val_pred_knn)

print("--- KNN ---")
print("Best params (cv) : ", grid_knn.best_params_)
print("Val accuracy: ", acc_val_knn)

pipe_mlp = Pipeline([("scaler", StandardScaler()),
    ("mlp", MLPClassifier(max_iter=300, random_state=RANDOM_STATE))])

param_mlp= {
  "mlp__hidden_layer_sizes": [(64,), (128,), (64,32)],"mlp__alpha": [1e-4, 1e-3, 1e-2], "mlp__activation": ["relu", "tanh"]

}

grid_mlp = GridSearchCV(estimator=pipe_mlp,param_grid=param_mlp,
                        scoring="accuracy",
                        cv=2,
                        n_jobs=-1)

grid_mlp.fit(X_train, y_train)
y_val_pred_mlp =grid_mlp.predict(X_val )
acc_val_mlp = accuracy_score( y_val, y_val_pred_mlp)

pipe_svm = Pipeline([
    ("scaler", StandardScaler()) ,
    ("svm", SVC())])

grid_svm = GridSearchCV(
    pipe_svm,
    {
        "svm__kernel":["linear"],
        "svm__C":[1, 10]

    },
    scoring="accuracy",
    cv=2 ,
    n_jobs=-1,

)

grid_svm.fit(X_train, y_train)
y_val_pred_svm=grid_svm.predict(X_val)
acc_val_svm = accuracy_score(y_val,y_val_pred_svm)


print("VAL acc-KNN:", acc_val_knn,"best:", grid_knn.best_params_)
print("VAL acc- MLP:", acc_val_mlp,"best: ",grid_mlp.best_params_)
print("VAL acc- SVM:", acc_val_svm,"best: ", grid_svm.best_params_ )

models=[
    ("KNN", grid_knn, acc_val_knn),
    ("MLP", grid_mlp, acc_val_mlp),
    ("SVM", grid_svm, acc_val_svm),]

models.sort(key = lambda t:t[2] , reverse=True)
winner_name , winner_grid, _= models[0]

print("\nModelul selectat:", winner_name)

best_model= winner_grid.best_estimator_
best_model.fit(np.vstack([X_train, X_val]),np.hstack([y_train, y_val]))

y_test_pred= best_model.predict(X_test)
acc_test = accuracy_score(y_test, y_test_pred)



print("\n*** Evaluare Finala(Test)**")
print("Model:",winner_name)
print("Accuracy:", acc_test)
print(classification_report(y_test,y_test_pred, digits=4))
print("Confusion matrix : ")
print(confusion_matrix(y_test,y_test_pred))
