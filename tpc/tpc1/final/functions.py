
## Library calls
import pandas as pd
import numpy as np

# ∇F(w(k)) =1N∑Nn=1∇fn(w(k);xn,yn)

def gradiente_batch(w,I,x,y):
    sum_gradiente=float(0)
    n = len(x) - 1 if x else None;
    if n == None : raise Exception("Empty Array")
    for i in range(1,n):
        for j in range(0,I):
            sum_gradiente += (gradiente(w,x[i],I) - y[i]) * pow(x[i],j)
    sum_gradiente = sum_gradiente / n
    return sum_gradiente

# ∇fn(w(k);xn,yn)

def gradiente(w,x,I):
    sum = float(0)
    for i in range(0,I):
        sum += np.dot(w,pow(x,i)) # must have 2 np arrays.
    return sum

def Extract(lst,i):
    return [item[i] for item in lst]


 # falta inserir parametros a entrar na função
def calcular_passo():
    eta_k = 0
    eta_k = alg_armijo()
    return eta_k

# falta perceber se o linalg norm é o ideal e implementar a flag para parar
# (com as restrições que estao na main funcion)
def alg_armijo(delta):
    flag = 1
    while (flag == 1):
        eta_k = eta_k / 2
        if eta_k * linalg.norm() <=