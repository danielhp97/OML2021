 # Dataset Preparation
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


#import functions
from functions import *


data = pd.read_excel(r'data1.xlsx', engine='openpyxl', header=None)
train, test = train_test_split(data.values.tolist(), test_size=0.20)
train_x = Extract(train,0)
train_y = Extract(train,1)
# ex 3
w = (0,0,0)
I = 3
# implementar metodo gradiente de batch
result = gradiente_batch(w,I,x=train_x,y=train_y)
print(f'O resultado do gradiente de batch é: {result}')

# define criterios de paragem
grad_const = 1e-4
k = 10*len(train)
print(f'O valor de k é: {k}\nO valor de max de grad é : {grad_const}')


# calcular comprimento do passo



# calcular para varios graus

# para cada polinomio calcular:
#   w*
#   erro de teste (in-sample)
#   erro de validação (out-sample)

# plots.
