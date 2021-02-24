## Library calls
import pandas as pd
from sklearn.model_selection import train_test_split
## Pre Requisitos

# Upload Dataset
data = pd.read_excel(r'data1.xlsx', engine='openpyxl', header=None)
data

# Divide Dataset

x_train, x_test = train_test_split(data.values.tolist(), test_size=0.20)

x_train
x_test


# Exercicio 3
def gradiente()
    

def gradiente_batch(x,y)
    n = len(x) - 1 if x else None;
    if n == None : raise Exception("Empty Array")
    for i in range(1,n):
        sum_gradiente= gradiente(x,y,w) + sum_gradiente
    return (1/n)*sum_gradiente

def stop_condition()
    print("Calcular stop condition")


# Exercise 4

# Exercise 5

# Exercise 6