import numpy as np                             #RED DE KOHONEN PACHECO ARELLANO ATZIN NATANAEL
import matplotlib.pyplot as plt
from numpy.random.mtrand import rand

#La BMU es la celda cuyos pesos están más cerca del ejemplo de entrenamiento
def find_BMU(SOM, x):
    distSq = (np.square(SOM - x)).sum(axis=2)
    return np.unravel_index(np.argmin(distSq, axis=None), distSq.shape)


#Actualiza los pesos del SOM recibiendo el training data
def update_weights(SOM, train_ex, learn_rate, radius_sq,
                   BMU_coord, step=3):
    g, h = BMU_coord
    #si el radio está cerca de cero, se cambia BMU
    if radius_sq < 1e-3: #0.000001
        SOM[g, h, :] += learn_rate * (train_ex - SOM[g, h, :])
        return SOM
    # Cambia todas las celdas del BMU
    for i in range(max(0, g - step), min(SOM.shape[0], g + step)):
        for j in range(max(0, h - step), min(SOM.shape[1], h + step)):
            dist_sq = np.square(i - g) + np.square(j - h)
            dist_func = np.exp(-dist_sq / 2 / radius_sq)
            SOM[i, j, :] += learn_rate * dist_func * (train_ex - SOM[i, j, :])
    return SOM


# Entrenamiento del SOM. Requiere la cuadricula inicializada
def train_SOM(SOM, train_data, learn_rate=.1, radius_sq=1,
              lr_decay=.1, radius_decay=.1, epochs=10):
    learn_rate_0 = learn_rate
    radius_0 = radius_sq
    for epoch in np.arange(0, epochs):
        rand.shuffle(train_data)
        for train_ex in train_data:
            g, h = find_BMU(SOM, train_ex)
            SOM = update_weights(SOM, train_ex,
                                 learn_rate, radius_sq, (g, h))
        #Actualizando learning rate y radio
        learn_rate = learn_rate_0 * np.exp(-epoch * lr_decay)
        radius_sq = radius_0 * np.exp(-epoch * radius_decay)
    return SOM
#Dimensiones del SOM
m = 10
n = 10
# Numero de entrenamientos
n_x = 2500
rand = np.random.RandomState(0)
# Inicializando training data
train_data = (rand.randint(0, 255, (n_x, 3)))
# Inicializando el SOM de manera aleatoria
SOM = rand.randint(0, 255, (m, n, 3)).astype(float)
# Ejecucion del training data y SOM
fig, ax = plt.subplots(
    nrows=1, ncols=2, figsize=(12, 3.5),
    subplot_kw=dict(xticks=[], yticks=[]))
ax[0].imshow(train_data.reshape(50, 50, 3)) #dimension del mapeo
ax[0].title.set_text('Training Data')
ax[1].imshow(SOM.astype(int))
ax[1].title.set_text('SOM Grid Inicializado de manera aleatoria')
fig, ax = plt.subplots(
    nrows=2, ncols=5, figsize=(15, 3.5),
    subplot_kw=dict(xticks=[], yticks=[]))
total_epochs = 0
for epochs, i in zip([1, 4, 5, 10, 10,10, 10, 10, 10, 10], range(0,10)):
    total_epochs += epochs
    SOM = train_SOM(SOM, train_data, epochs=epochs)
    if(i<=4):
        ax[0][i].imshow(SOM.astype(int))
        ax[0][i].title.set_text('Epochs = ' + str(total_epochs))
    else:
        ax[1][i-5].imshow(SOM.astype(int))
        ax[1][i-5].title.set_text('Epochs = ' + str(total_epochs))
fig, ax = plt.subplots(
    nrows=3, ncols=3, figsize=(15, 15),
    subplot_kw=dict(xticks=[], yticks=[]))

# 3 SOMS inicializados de manera aleatoria

for learn_rate, i in zip([0.001, 0.5, 0.99], [0, 1, 2]):
    for radius_sq, j in zip([0.01, 1, 10], [0, 1, 2]):
        rand = np.random.RandomState(0)
        SOM = train_SOM(SOM, train_data, epochs = 5,
                        learn_rate = learn_rate,
                        radius_sq = radius_sq)
        ax[i][j].imshow(SOM.astype(int))
        ax[i][j].title.set_text('$\eta$ = ' + str(learn_rate) +
                                ', $\sigma^2$ = ' + str(radius_sq))
plt.show()