import random
import matplotlib.pyplot as plt
import numpy as np

#parametros
x0 = 0 #posicao inicial
p = 0.5 #probabilidade de passo p/ direita e 1-p p/ esquerda
t = 10 # numero de iteracoes


numbersx = np.arange(t+1) #numero de iterações
numbersy = np.arange(-t,t+1)

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


T = 1000

x1 = Caminho_aleatorio(x0,T,p)
x2 = Caminho_aleatorio(x0,T,p)
x3 = Caminho_aleatorio(x0,T,p)
x4 = Caminho_aleatorio(x0,T,p)
x5 = Caminho_aleatorio(x0,T,p)


t_plot = np.arange(T+1)


plt.plot(t_plot,x1,'r-',linewidth=0.5)
plt.plot(t_plot,x2,'b-',linewidth=0.5)
plt.plot(t_plot,x3,'g-',linewidth=0.5)
plt.plot(t_plot,x4,'c-',linewidth=0.5)
plt.plot(t_plot,x5,'y-',linewidth=0.5)
plt.grid(True)
plt.axis([0, T, -100, 100])
plt.xlabel('time')
plt.ylabel('x(distance)')
plt.title('Caminhada Aleatória')
plt.text(10, 95, f'p = {p} \nx$_0$ = {x0}',va='top',ha='left')
#plt.xticks(numbersx)
#plt.yticks(numbersy)
plt.show()

def Caminhadas_aleatorias_de_max_T(N,x0,p,T_values):
    x_lista = []
    for i in range(len(T_values)):
        x_of_T = []
        for _ in range(N):
            x = Caminho_aleatorio(x0,T_values[i],p)
            x_of_T.append(x[-1])
        x_lista.append(x_of_T)
    return x_lista

T_values = [5,10,100,1000,10000]
N = 100000
x_lista = Caminhadas_aleatorias_de_max_T(N,x0,p,T_values)
x_lista_copy = np.array(x_lista,dtype=float)

print(x_lista_copy)

def sigma_quadrado(x_list, T_values,N):
   sigma_2 = []
   x_2_med = 0
   x_med = 0
   for i in range(len(T_values)):
       for j in range(N):
           x_med = x_med + x_list[i][j]
           x_2_med = x_2_med + (x_list[i][j])**2
       sigma = x_2_med/N - (x_med/N)**2
       sigma_2.append(sigma)
       x_med = 0
       x_2_med = 0
   return sigma_2
print(sigma_quadrado (x_lista_copy, T_values, N))
print("___________")


