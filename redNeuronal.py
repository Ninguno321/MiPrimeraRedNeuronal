import tensorflow as tf 
import numpy as np

crad = np.pi/180
AdamOptimizerValue = 0.05
epochsValue = 1000
exampleValue = 180.0

grados = np.array([10, 100, -75, 0, 824, 120, -120, 155, -40, -900, 414, 37, 64], dtype=float)
radiantes = np.array([10*crad, 100*crad, -75*crad, 0*crad, 824*crad, 120*crad, -120*crad, 155*crad, -40*crad, -900*crad, 414*crad, 37*crad, 64*crad], dtype=float)
capa = tf.keras.layers.Dense(units=1, input_shape=[1])

modelo = tf.keras.Sequential([capa])

modelo.compile(
    optimizer = tf.keras.optimizers.Adam(AdamOptimizerValue),
    loss = 'mean_squared_error'
)
print("Empezando el entrenamiento...")

historial = modelo.fit(grados, radiantes, epochs=epochsValue, verbose=False)

print("Modelo entrenado")


import matplotlib.pyplot as plt
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(historial.history['loss'])
plt.show()

print("Hagamos una predicción")
resultado = modelo.predict(np.array([exampleValue]))
print("El resultado es: " + str(resultado) + " radianes")
print("El valor correcto es: " + str(exampleValue*crad) + " radianes")
print("Peso: " + str(capa.get_weights()))






