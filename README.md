# MiPrimeraRedNeuronal

Creación de una red neuronal desde cero usando Python y Tensorflow.

Anlizaremos cómo crearla y explicaremos cómo y por qué funciona, también veremos las diferencias entre el # APRENDIZAJE AUTOMÁTICO # y la # PROGRAMACIÓN REGULAR #.


-PROGRAMACIÓN REGULAR
Normalmente programámos algoritmos para que a partir de entradas generen resultados. Nosotros somos los encargados de escribir las reglas e instrucciones a seguir para llegar a esos resultados.

-APRENDIZAJE AUTOMÁTICO
Poseemos las entradas y los resultados pero no necesariamente sabemos cómo a partir de esas entradas obtenemos esos resultados. Es decir no conocemos el algoritmo usado.


OBJETIVO: 
Lo que pretendemos hacer en este proyecto es crear un modelo que pueda a partir de ciertas entradas obtener los resultados esperados de cada una y que pueda aprender por si solo el algoritmo necesario para obtenerlos.

Por ejemplo, queremos convertir grados a radianes. Radianes = grados x pi/180.

# En programación regular podríamos usar una funcion como esta:

    import math
    pi = math.pi

    def function(G):
        return G * (pi/180)

# Usando aprendizaje automático:
Suponemos que no conocemos la fórmula de conversión, solo contamos con un número de entradas con sus correspondientes salidas.

                                    | GRADOS  |RADIANES |
                                    |---------|---------|
                                    | 10      | 0.1745  |
                                    | 100     | 1.7453  |
                                    | -75     | -1.3089 |
                                    | 0       | 0       |
                                    | 824     | 14.3815 |









Creditos: @Ringa Tech