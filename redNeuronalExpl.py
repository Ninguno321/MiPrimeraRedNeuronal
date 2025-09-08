import tensorflow as tf 
# TensorFlow es una biblioteca de código abierto creada por googlepara el aprendizaje automático.

import numpy as np
# NumPy es una biblioteca de Python que se utiliza para trabajar con matrices y realizar cálculos numéricos de manera eficiente.
# También cuenta con el número pi

# Declaraciones
crad = np.pi/180

grados = np.array([10, 100, -75, 0, 824, 120, -120], dtype=float)
radiantes = np.array([10*crad, 100*crad, -75*crad, 0*crad, 824*crad, 120*crad, -120*crad], dtype=float)

# Ahora vamos a diseñar nuestro modelo de red neuronal explicado antes.
# Nos apoyaremos de keras, que nos permite crear redes neuronales de forma sencilla.
# Podemos especificar los dos capas de entrada y salida por separado o podemos ahorrarnos un paso
# y solo especificar la capa de salida ya que la de entrada se deduce de los datos que le pasemos.

# Creamos una capa densa (conexiones entre todas las neuronas de una capa con todas las de la siguiente)
# La capa No densa no tiene todas las conexiones entre neuronas. 

# units es el número de neuronas que tendrá la capa. Nosotros solo tenemos una neurona por capa
# input_shape es la forma de los datos de entrada. Nosotros solo tenemos un dato de entrada (un grado)
                                    # nos autoregistra la capa de entrada con una neurona.
capa = tf.keras.layers.Dense(units=1, input_shape=[1])

# Estas capas ahora están volando en el espacio. Necesitamos un modelo que las contenga.
# En este caso usaremos un modelo secuencial porque solo tenemos una entrada y una salida.

modelo = tf.keras.Sequential([capa])

# Ahora tenemos que compilar el modelo.
# Tenemos que especificar cómo queremos que aprenda (procese los datos).

modelo.compile(
    optimizer = tf.keras.optimizers.Adam(0.1), 
    # Adam es un algoritmo de optimización que ajusta los pesos y sesgos de la red neuronal para minimizar el error en las predicciones.
    # El 0.1 es la tasa de aprendizaje, que controla qué tan rápido o lento se ajustan los pesos y sesgos. A menor número más lento aprende pero si es muy grande puede pasarse.
    loss = 'mean_squared_error'
    # La función de pérdida mide qué tan bien está funcionando la red neuronal. Mean Squared Error (MSE) es una función comúnmente utilizada para problemas de regresión.
    # Una poca cantidad de errores grandes es peor que muchos errores pequeños.
)

# Ahora ya podemos entrenar el modelo.
print("Empezando el entrenamiento...")

#Para entrenarlo usamos el método fit. Le pasamos los datos de entrada y salida. Y le indicamos cuántas veces queremos que vea los datos (epochs).
# Verbose es para que no nos imprima todo el proceso de entrenamiento.
#Tendremos que ajustar el número de epochs para que aprenda bien pero sin pasarse.

historial = modelo.fit(grados, radiantes, epochs=500, verbose=False)

print("Modelo entrenado")

# Ahora podemos ver cómo de bien ha aprendido.













