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
def backtrackingArmijo():
    delta=0.1
    eta_k=1
    while( fun_Custo(w+eta_k*s_k,X,Y)> Custo_k+delta*eta_k*grad_full*s_k):
        eta_k=eta_k/2
        if eta_k*norm(s_k)<=1e-8:
            eta_k=1
    return eta_k


def passo(w,Custo_k,grad_full,s_k,X,Y,k,nt):
    eta_k = backtrackingArmijo(w,Custo_k,grad_full,s_k,X,Y)
    return eta_k

def fun_Custo(w,X,Y):
    I= len(w)-1
    nt=len(X)
    for i in range(1,nt):
        val += pow((gradiente(w,X[i],I) -Y[i]),2)
    val= val/(2*nt)
    return(val)

def gradiente(w,x,I):
    for i in range(0,I):
        sum += np.dot(w,pow(x,i)) # must have 2 np arrays.
    return sum


def gradiente_batch(w,I,x,y):
    n = len(x) - 1 if x else None;
    if n == None : raise Exception("Empty Array")
    for i in range(1,n):
        for j in range(0,I):
            sum_gradiente += (gradiente(w,x[i],I) - y[i]) * pow(x[i],j)
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
maxit = 10*nt
k = 1 # ponto inicial
def principal():
    while(norm(gradiente_batch() > tol && k< max it)): # criterios de paragem
        # calcular gradente no ponto wk
        grad_k = gradiente_batch(wk,x,y)
        # calcular direcção de procura
        s_k = -grad_k
        # calcular comprimento do passo

        # calcular Função objectivo no ponto wk
        Custo_k = fun_Custo(w,X,Y)
        # Calcular o gradiente completo no ponto wk
        grad_full = gradiente_batch(w,X,Y)
        # Calcular comprimento do passo
        eta_k = passo(w,Custo_k,grad_full,s_k,X,Y,k,nt)
        # Calcular novo ponto
        wk = wk + eta_k*sk
        k = k+1
        # Inserir algoritmo para calcular ponto necessários a calcular gráfico

    print("A solução ótima")

    print("In-Sample Error")

    print("Out-Sample Error")

    #Print do grafico com valor da Função CUsto ao longo do processo Iterativo



# Exercise 4

# Exercise 5

# Exercise 6