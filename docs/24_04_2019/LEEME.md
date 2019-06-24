## Modelo v6

### Arquitectura

Se tienen 3 modelos:

* **Modelo LIKE:** Se aprende el _me gusta / no me gusta_ de los usuarios respecto a los restaurantes. Se hace el producto escalar de los embeddigs de ambos. Estos embeddigs se calculan con factorización de matrices no negativas.
* **Modelo TAKE:** Se aprende el _la hizo / no la hizo_ de las fotos respecto de los usuarios. Se hace producto escalar enntre embedding de usuario (como modelo anterior) y embedding de imagen con una capa lineal y bias.
* **Modelo BOTH:** Se aprenden ambas cosas compartiendo los pesos de la parte de los usuarios.

### Conjunto de datos

A partir de todo el conjunto de datos:

1. Si un usuario fué varias veces a un mismo restaurante, se deja la última.
2. Se separa el conjunto en 2, las reviews con imagen y las que no:

    2.1. Con las que tienen imagen:
        
        2.1.1. Se divide en TRAIN_DEV y TEST utilizando (a).
        2.1.2. Se divide TRAIN_DEV en TRAIN y DEV utilizando (a).
        2.1.3. Para los conjuntos TRAIN y TRAIN_DEV, se crean nuevos ejemplos utilizando (b) lo que genera TRAIN_IMG y TRAIN_DEV_IMG
        2.1.4. Para los conjuntos DEV y TEST, se crean los conjuntos DEV_IMG y TEST_IMG utilizando (c).

    2.2. Con las que no tienen imagen:
        
        2.2.1. Se divide en TRAIN_DEV y TEST utilizando (d).
        2.2.2. Se divide TRAIN_DEV en TRAIN y DEV utilizando (d).
        2.1.3. Para los conjuntos TRAIN y TRAIN_DEV, se crean nuevos ejemplos utilizando (e) lo que genera TRAIN_LIKE y TRAIN_DEV_LIKE
        2.1.4. Para los conjuntos DEV y TEST, se crean los conjuntos DEV_LIKE y TEST_LIKE utilizando (f).

        
##### Algoritmo (a)
> **Para cada usuario:** si tiene una review va a TRAIN, si tiene más, 1 para TEST y el resto a TRAIN.  

##### Algoritmo (b)
> **Para cada usuario, restaurante, imagen:**
> * Se obtienen `d` imágenes (o menos si no hay `d`) de otros usuarios del mismo restaurante.
> * Se obtienen `f` imágenes (o menos si no hay `f`) de otros usuarios en otros restaurantes (a los que no fué el usuario).
> * Se repite la imágen real `d+f` veces para compensar la distribución de ejemplos positivos y negativos.

##### Algoritmo (c)
> **Para cada usuario, restaurante, imagen:**
> * Se obtienen el resto de imágenes del restaurante.
> * Se modifica su usuario para que sea el actual.
> * Se marca el item actual para saber cual es el real.
> * Si el usuario tiene 4 imágenes se crean 4 casos distintos para evitar que el random se vea beneficiado.


##### Algoritmo (d)
> **Para cada usuario:** si tiene una review va a TRAIN, si tiene más y alguna es positiva, la positiva para TEST y el resto a TRAIN. Si no tiene positivas, todas a TRAIN.

##### Algoritmo (e)
> **Para el conjunto de TRAIN:** Se añade el conjunto de TRAIN de la parte de las imágenes (pero sin incluir estas).  

##### Algoritmo (f)
> **Para el conjunto de DEV o TEST:** 
> * Se añade el conjunto de DEV o TEST de la parte de las imágenes (**SOLO LAS POSITIVAS**).  
> * Una vez unidos, para cada positivo, se generan 99 ejemplos negativos con restaurantes a los que ese usuario concreto:
>   * No tiene en TRAIN + TRAIN de las imágenes si se está en el conjunto de DEV
>   * No tiene en TRAIN_DEV + TRAIN_DEV de las imágenes si se está en el conjunto de TEST

### Baselines

##### Parte TAKE
* **Random:** Se asigna una probabilidad aleatoria a cada imagen del restaurante en DEV|TEST y se obtiene la posición en funcion de esta probabilidad. Se hacen 10 repeticiones para obtener cerca del 50% (una sola podría salir muy bien).
* **Centroide:** Se calcula la imagen centroide de cada restaurante con los datos de TRAIN y en DEV|TEST se calculan las distancias con este.

##### Parte LIKE
* **Positivos:** Se ordenan los restaurantes en función del número de reviews positivas. Más positivas más alto en TOP.

### Entrenamiento

A tener en cuenta:
* Se hace un grid-search con LR-DECAY.
* En el caso del modelo `BOTH` se entrena primero un batch de LIKE y luego uno de TAKE.

### Evaluación

Se obtienen los resultados desagregados en función de los datos de los que se dispone en TRAIN (tanto para usuarios como restaurantes).

### Resultados

La parte de TAKE parece funcionar correctamente, pero en la parte de LIKE, se ve que el baseline nos iguala o mejora.

Ver `Resultados.xlsx`
