## Library calls
import pandas as pd
import numpy as np
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
def gradiente():
    print()


def gradienteE(w,X,Y,G):
    print()


def gradiente_batch(x,y):
    n = len(x) - 1 if x else None;
    if n == None : raise Exception("Empty Array")
    for i in range(1,n):
        sum_gradiente += (gradiente(w,x[i],I) - y[i]) * pow(x[i],0:I)
    sum_gradiente = sum_gradiente / n
    return sum_gradiente


def stop_condition():
    print("Calcular stop condition")

# escolher grau de polinomio a ajustar.
I=7

# Método de gradiente completo (gradiente de batch)
# Vamos utilizar algoritmo de Armijo, para a procura
# ou seja, memso método de G=1 e P=1 (segundo o exemplo da prof.)


# definimos ponto inicial como w^(1) = (0,0,0,...0)
w1 = np.full((1,I),0);

# temos que definir um valor de tolerancia
tol = 1e-4
# definimos também o nr max de iterações
nt = len (x_train) #onde nt é o nr de obs do treino
maxit = 10*Nt

def principal():
    while(norm(gradiente_batch() > tol && k< max it)): # criterios de paragem
        # calcular gradente no ponto wk

        # calcular direcção de procura

        # calcular comprimento do passo

        # calcular Função objectivo no ponto wk

        # Calcular o gradiente completo no ponto wk

        # Calcular novo ponto

        # Inserir algoritmo para calcular ponto necessários a calcular gráfico

    print("A solução ótima")

    print("In-Sample Error")

    print("Out-Sample Error")

    #Print do grafico com valor da Função CUsto ao longo do processo Iterativo



# Exercise 4

# Exercise 5

# Exercise 6