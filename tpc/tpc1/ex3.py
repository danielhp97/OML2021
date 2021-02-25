## Library calls
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from numpy import linalg as LA
## Pre Requisitos

# Upload Dataset
data = pd.read_excel(r'data1.xlsx', engine='openpyxl', header=None)
data

# Divide Dataset

x_train, x_test = train_test_split(data.values.tolist(), test_size=0.20)
x_train
x_test
# Dividindo dados
def Extract(lst,i):
    return [item[i] for item in lst]

X_data_train = Extract(x_train,0)
X_data_train
X_data_test = Extract(x_test,0)
X_data_test

Y_data_train = Extract(x_train,1)
Y_data_test = Extract(x_test,1)


# Exercicio 3
def backtrackingArmijo():
    delta=0.1
    eta_k=1
    while( fun_Custo(w+eta_k*s_k,X,Y)> Custo_k+delta*eta_k*np.dot(grad_full.T,s_k)):
        eta_k=eta_k/2
        if eta_k*LA.norm(s_k)<=1e-8:
            eta_k=1
    return eta_k


def passo(w,Custo_k,grad_full,s_k,X,Y,k,nt):
    eta_k = backtrackingArmijo(w,Custo_k,grad_full,s_k,X,Y)
    return eta_k

def fun_Custo(w,X,Y):
    val = 0
    I= len(w)-1
    nt=len(X)
    for i in range(1,nt):
        val += pow((gradiente(w,X[i],I) -Y[i]),2)
    val= val/(2*nt)
    return(val)

def gradiente(w,x,I):
    sum = 0
    for i in range(0,I):
        sum += np.dot(w,pow(x,i)) # must have 2 np arrays.
    return sum


def gradiente_batch(w,x,y):
    sum_gradiente = 0
    n = len(x) - 1 if x else None;
    if n == None : raise Exception("Empty Array")
    for i in range(1,n):
        for j in range(0,I):
            sum_gradiente += (gradiente(w,x[i],I) - y[i]) * pow(x[i],j)
    sum_gradiente = sum_gradiente / n
    return sum_gradiente



I=7
# Método de gradiente completo (gradiente de batch)
# Vamos utilizar algoritmo de Armijo, para a procura
# ou seja, mesmo método de G=1 e P=1 (segundo o exemplo da prof.)
w1 = np.full((1,I),0); # definimos ponto inicial como w^(1) = (0,0,0,...0)
tol = 1e-4 # temos que definir um valor de tolerancia
nt = len (x_train) #onde nt é o nr de obs do treino
maxit = 10*nt # definimos também o nr max de iterações
k = 1 # ponto inicial

# escolher grau de polinomio a ajustar.

def principal():
    wk = 0
    Custo_k = 0
    print(gradiente_batch(w1,X_data_train,Y_data_train))
    while(LA.norm(gradiente_batch(w1,X_data_train,Y_data_train) > tol) and k< maxit): # criterios de paragem
        grad_k = gradiente_batch(w1,X_data_train,Y_data_train) # calcular gradente no ponto wk
        s_k = -grad_k # calcular direcção de procura
        Custo_k = fun_Custo(w1,X_data_train,Y_data_train) # calcular Função objectivo no ponto wk
        grad_full = gradiente_batch(w1,X_data_train,Y_dat_train) # Calcular o gradiente completo no ponto wk
        eta_k = passo(w1,Custo_k,grad_full,s_k,X_data_train,Y_dat_train,k,nt) # Calcular comprimento do passo
        wk = wk + eta_k*sk # Calcular novo ponto
        k = k+1 # Inserir algoritmo para calcular ponto necessários a calcular gráfico
        print(wk)
    #print("A solução ótima:", wk)
    #E_dt = Custo_k(wk,X_data_train,Y_data_train)
    #print("In-Sample Error: ", E_dt)
    #E_dv = Custo_k(wk,X_data_teste,X_data_teste)
    #print("Out-Sample Error: ", E_dv)

principal()
    #Print do grafico com valor da Função CUsto ao longo do processo Iterativo
