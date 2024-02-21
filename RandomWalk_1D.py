import random
import matplotlib.pyplot as plt
import numpy as np

#parametros
x0 = 0 #posicao inicial
p = 0.5 #probabilidade de passo p/ direita e 1-p p/ esquerda
t = 10 # numero de iteracoes

"""
###### Adicionado por Maurício
######
###### não entendi pra que são essas variáveis... boa prática de programação é nomear
###### as variáveis com nomes intuitivos e mnemônicos, que descrevam a sua
###### "função no código" de maneira sucinta
"""

"""
###### Adicionado por Matheus 20/08
######
###### vou tentar melhorar a nomeclatura das variaveis
"""
numbersx = np.arange(t+1) #numero de iterações
numbersy = np.arange(-t,t+1)