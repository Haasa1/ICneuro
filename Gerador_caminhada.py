import random
import matplotlib.pyplot as plt
import numpy as np


x0 = 0 #posicao inicial
p = 0.5 #probabilidade de passo p/ direita e 1-p p/ esquerda


def Caminho_aleatorio(X0,T,p):#primeiro termo é a condição inicial, 2a termo é o numero de iteraçoes
    lista = []
    x = X0
    lista.append(x)
    for i in range(0,T):
        if random.random() < p:
            x = x + 1
        else:
            x = x - 1
        lista.append(x)
    return np.array(lista)

def Caminhadas_aleatorias_de_max_T(N,x0,p,T_values): #modificado para receber os paramentros do sigma quadrado
    x_lista = []
    x_soma_quadrado = []
    x_soma = []
    soma_aux = 0
    soma_aux_quadrado = 0
    for i in range(len(T_values)):
        x_of_T = []
        for _ in range(N):
            x = Caminho_aleatorio(x0,T_values[i],p)
            x_of_T.append(x[-1])
            soma_aux += x[-1]
            soma_aux_quadrado += (x[-1]**2)
        x_lista.append(x_of_T)
        x_soma.append(soma_aux)
        x_soma_quadrado.append(soma_aux_quadrado)
        soma_aux = 0
        soma_aux_quadrado = 0
    return np.array(x_lista,dtype=float), np.array(x_soma), np.array(x_soma_quadrado)

T_values = [5,10,100,1000,10000]
N = 100000
x_lista,x_soma,x_quadrado = Caminhadas_aleatorias_de_max_T(N,x0,p,T_values)

x_lista.tofile('Sample_caminhada_1D.csv', sep = ',')
x_soma.tofile('Sample_soma_caminhada_1D.csv', sep = ',')
x_quadrado.tofile('Sample_quadrado_caminhada_1D.csv', sep = ',')