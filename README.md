# MNIST Model Comparison

Proiectul acesta compara mai multe modele de Machine Learning pe datasetul MNIST,care contine imagini cu cifre scrise de mana.

Scopul acestui proiectului este clasificarea imaginilor in una dintre cele 10 clase posibile: cifrele de la 0 la 9.

## Modele folosite
In proiectul acesta sunt comparate urmatoarele modele:

- K-Nearest Neighbors
- Multi-Layer Perceptron
- Support Vector Machine

## Concepte utilizate

- impartirea datelor in train,validation si test
- scalarea datelor
- Pipeline in scikit-learn
- GridSearchCV pentru alegerea hiperparametrilor
- compararea modelelor pe setul de validare
- evaluarea finala pe setul de test
- accuracy
- classification report
- confusion matri

## Dataset
Proiectul foloseste datasetul MNIST incarcat prin functia `fetch_openml` din scikit-learn.

Fiecare imagine are dimensiunea 28x28 pixeli, adica este reprezentata prin 784 de valori numerice.

## Structura proiectului


```text
MNIST-Model-Comparison/
│
├── main.py
├── README.md
├── requirements.txt
└── .gitignore
