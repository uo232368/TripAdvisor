# 1.-Arquitecturas

En todos los casos se utiliza el optimizador **Adam**.

## 1.1 Arquitectura V1 (13/09/2018)
### 1.1.1 Descripción
Utilizando solamente aquellas reviews que posean imágen, se entrenaría un modelo con la siguiente arquitectura:

<center>![](modelv3_net.png)</center>

##### Entrada:
* Usuario (`one-hot`)
* Restaurante (`one-hot`)

##### Salida:
* Clasificación binaria (`DOT` con embeddings de usr y rest)
* Multiregresión (Concatenar embeddings y transfomar a espacio de imágenes)
 
### 1.1.2 Características
* Loss: `c·MSE + (1-c)·BIN_XENTROPY`
* Embedding de imágenes procedente de 3 formas de métodos posibles:
	* Autoencoder
	* Red preentrenada (InceptionResNetV2) 
	* Fine-Tuning de red preentrenada añadiendo clasificación binaria a la salida y usuario a la entrada.
* Loss de imágenes (`MSE`) normalizada (dividida entre la `MSE` media de todas las imágenes con la imagen centrioide). Esto es necesario para que la loss de la clasificación y la de la multiregresión se muevan en rangos de valores similares.
* Oversampling en TRAIN
* `c` en loss

### 1.1.3 Problemas
* Pocas reviews con imágen.
* Mayoría de usuarios con una sola review.
* División `TRAIN`/`DEV`/`TEST` compleja.

___
## 1.2 Arquitectura V2 (21/09/2018)
### 1.2.1 Descripción
Utilizando las valoraciones de aquellos **usuarios con 10 o más**, se entrenaría un modelo con la siguiente arquitectura:

<center>![](modelv2_net.png)</center>

##### Entrada:
* Usuario (`one-hot`)
* Imágen (Embedding obtenido mediante cualquiera de las 3 formas del punto 1.2)
* Restaurante (`one-hot`)

##### Salida:
* Clasificación binaria (Red profunda con 5 capas (ReLu) con tamaño de capa `n=(n-1)/2`)

### 1.2.2 Funcionamiento

Se dividirá el proceso de aprendizaje en dos fases:
##### Fase 1:
En esta fase se entrena el modelo con todas las reviews disponibles (se coloca un vector de 0's en el vector de la imágen independientemente de si tienen o no imagen real).
Para crear los conjuntos de `TRAIN`/`DEV`/`TEST` se realiza el siguiente procedimiento:

* Agrupar las reviews **con imágen** por usuario-restaurante y separar aleatoriamente `0.9`,`0.05`,`0.05`
* Agrupar las reviews **sin imágen** por usuario-restaurante y separar aleatoriamente `0.9`,`0.05`,`0.05`
* Tener en cuenta que un usuario puede tener varias reviews en un restaurante (una con imágen y otra sin ella) 

##### Fase 2:
En la segunda fase se refina el modelo anterior **partiendo de los pesos aprendidos en la fase 1 (sin fijarlos)** con la idea de ver si mejora la clasificación al añadir las imágenes.

Se ha de entrenar con un **learning-rate menor `lr/2` al de la fase 1** para evitar cambios bruscos que alteren en gran medida lo aprendido previamente.

El **conjunto de datos** ha de ser exactamente el mismo **de la fase 1** pero en el **TRAIN** hay que eliminar aquellas valoraciones **sin imágenes**.

### 1.2.3 Características
* Loss: `BIN_XENTROPY`
* Embedding de imágenes procedente de 3 formas de métodos posibles:
	* Autoencoder
	* Red preentrenada (InceptionResNetV2) 
	* Fine-Tuning de red preentrenada añadiendo clasificación binaria a la salida y usuario a la entrada.
* Oversampling en TRAIN
* Sin `c` en loss

### 1.2.4 Cambios
* El tamaño de las capas ocultas pasa a ser `4096 , 2048 , 1024 , 512 , 256` con el fin de reducir parámetros.
* Se pasa a usuarios con 5 (en vez de 10) o más reviews para evitar reducir en gran medida el conjunto.

### 1.2.5 Problemas
Este modelo **no modifica la representación de usuarios y restaurantes en función de las imágenes**. Estas se tienen en cuenta para la clasificación pero no en los embeddings.
___
## 1.3 Arquitectura V3 (24/09/2018)
### 1.3.1 Descripción
Esta aquitectura es idéntica a la anterior con la salvedad del modelo a utilizar. En este caso el nuevo modelo es:

<center>![](modelv3_net.png)</center>

##### Entrada:
* Usuario (`one-hot`)
* Restaurante (`one-hot`)

##### Salida:
* Clasificación binaria (`DOT` con embeddings de usr y rest)
* Multiregresión (Concatenar embeddings y transfomar a espacio de imágenes)
 ___
 
# 2.-Experimentación

## 2.1 Grid-Search [24/10/2018] (fase 1)
Utilizando las arquitecturas v2 y v3 se procede inicialmente a realizar un `Grid-Search` para cada uno de los modelos.
* Inicialmente, el `batch-size` **no se considera** como un hiperparámetro a optimizar y se fija en 512.
* Se prueba para ambos modelos unos `learning-rates` salteados  con el fin de ver en que rangos funciona mejor el modelo y realizar posteriormente otra ejecución con estos valores.
* En cuanto al `emb-size` se probarán los valores: 512,1024,2048 (solo en el modelo v3 dado que el v2 no posee embeddings como tal)
* Se realizará un **early-stopping** utilizando las **10** epochs anteriores deteniendose en el caso de que la pendiente sea mayor que **-1e-8**.
* El máximo de epochs establecido en todas las pruebas es de **5000**.
### 2.1.1 Resultados
En el **modelo v2** se retornan los siguientes resultados:

| Learning Rate |   DEV loss  | Epochs |
|:-------------:|:-----------:|:------:|
|    1.00E-07   | 0.555192433 |    140 |
|    1.00E-05   | 0.958121745 |     10 |
|    1.00E-03   | 1.262997863 |     10 |
|    1.00E-01   | 3.165766139 |     10 |

Estos resultados parecen indicar que es necesaria una nueva ejecución con `learning-rates` menores o en torno a **1e-7**.

## Cambios [02/10/2018]
Se descarta el Grid-Seach anterior.  
Utilizando las arquitecturas v2 y v3 se procede inicialmente a realizar un `Random-Search` para cada uno de los modelos.
* Se utilizarán para ambos modelos 5 `learning-rates`: (1e-9, 1e-7, 1e-5, 1e-3, 1e-1).
* En cuanto al `emb-size` se probarán 4 valores en el intervalo `[128,256,512,1024]` (solo en el modelo v3 dado que el v2 no posee embeddings como tal)

El `batch-size` **no se considera** como un hiperparámetro a optimizar y se fija en 512.  

El máximo de epochs establecido en todas las pruebas es de **500**.  

#### Acelerar la búsqueda de hiperparámetros
Dada la baja velocidad, se deciden implementar los siguientes cambios:

* Se pasa de un `OverSampling` de proporciones 50/50 a utilizar todos los de clase 1 , todos los de clase cero y finalmente otros tantos de clase 0 como los ya existentes (escogidos de forma aleatoria)
* Reducir los tamaños de las capas del modelo 2 (1024/512/256/128/64)
* Se realizará un **early-stopping** utilizando las **7** epochs anteriores deteniendose en el caso de que la pendiente sea mayor que **-1e-8**.  
* Si la loss entre las 2 primeras epochs es exactamente igual, detener.

## Cambios [03/10/2018]
* El early stopping se realizará esperando 5 epochs y utilizando las 5 siguientes. Como mínimo se realizarán 10 epochs a no ser que no exista mejora alguna.
* La métrica a utilizar durante la fase de búsqueda de hiperparámetros será el `Area Under ROC` del conjunto de validación.
* En el modelo V2 se realizará un `grid-search` variando:
    * Learning-rate: `(1e-1, 1e-3, 1e-5, 1e-7, 1e-9)`
* En el modelo V3 se realizará nuevamente un `grid-search` variando:
    * Learning-rate: `(1e-1, 1e-3, 1e-5)`
    * Embedding: `(256,512)`

### Resultados
#### Modelo v1
|    LR    | AUC-ROC | Epochs |
|:--------:|:-------:|:------:|
| 1,00E-09 |  0,4976 |     33 |
| 1,00E-07 |  0,6992 |    404 |
| 1,00E-06 |  0,7092 |    126 |
| 1,00E-05 |  0,6463 |     10 |
| 1,00E-03 |  0,5000 |      2 |
| 1,00E-01 |  0,5000 |      2 |

#### Modelo v2

|          |    EMB   |        |         |        |         |        |          |        |
|:--------:|:--------:|:------:|---------|--------|---------|--------|----------|--------|
|          |      **128** |        | **256**     |        | **512**     |        | **1024**     |        |
|    **LR**    |  **AUC-ROC** | **EPOCHS** | **AUC-ROC** | **EPOCHS** | **AUC-ROC** | **EPOCHS** | **AUC-ROC**  | **EPOCHS** |
| 1,00E-07 | 0,542115 |    500 | 0,4671  | 32     | 0,6023  | 500    | 0,615969 | 500    |
| 1,00E-05 | 0,660177 |     24 | 0,6691  | 16     | 0,6632  | 15     | 0,659717 | 13     |
| 1,00E-03 | 0,713721 |     24 | 0,6907  | 21     | 0,6872  | 17     | 0,690673 | 16     |
| 1,00E-01 |      0,5 |      2 | 0,5000  | 2      | 0,5000  | 2      | 0,5      | 2      |


## Cambios [04/10/2018]
Analizando los resultados y probando a realizar entrenamientos con parámetros concretos se ve que existe un alto sobreajuste en TRAIN.
Para evitar esto y mejorar los resutados en DEV, se deciden los siguientes cambios:

* Añadir `DropOut` en las 5 capas intermedias del modelo v2 y en los embeddings previa realización del producto escalar en el modelo v3.
* Cambio del tamaño del batch de 512 a 128 con el fin de reducir el número de epochs y no alcanzar el máximo establecido (500)
* Gridsearch con mejores valores y dropout de 0.5.
* Probar con oversampling con y sin dropout y sin oversampling con y sin dropout
* Asegurarse del funcionamiento del dropout en el modelo v3

#### Resultados

Ver fichero `docs/04_10_2018.xlsx`

## Cambios [10/10/2018]

Se añade el cálculo de la métrica AUC (en DEV y TRAIN) utilizando las probabilidades binarizadas de antemano para evitar resutados "muy optimistas" y cambios de AUC ante la misma matriz de confusión.

## Cambios [11/10/2018]
Ante los problemas vistos utilizando el AUC, se vuelve a realizar la optimización de hiperparámetros utilizando la **loss** en DEV `binary_crossentropy`.

Se realizará un nuevo Grid-Search utilizando los siguientes parámetros:

### Modelo 2

* **LR**: [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
* **DPOUT**: No
* **OVERSAMPLING**: Duplicar ejemplos de la clase 0. 

### Modelo 3

* **LR**:  [1e-3, 1e-5, 1e-7, 1e-9]
* **EMB**: [128, 256, 512, 1024]
* **DPOUT**: No
* **OVERSAMPLING**: Duplicar ejemplos de la clase 0. 

### Otros
Además de minimizar la loss en DEV, también se mostrará para cada epoch la siguiente información:
* AUC_TRAIN
* AUC_BIN_TRAIN
* TRAIN_LOSS
* AUC_DEV
* AUC_BIN_DEV
* DEV_LOSS (A minimizar)
* DEV_F1 (Seleccionando correctamente la clase minoritaria)
* DEV_ACCURACY ( TP+TN / TOTAL EJEMPLOS )
* TP
* FP
* FN
* TN

#### Resultados

Ver fichero `docs/11_10_2018.xlsx`

## Cambios [16/10/2018]
Analizando las epochs a fondo (batch a batch) (ver `docs/16_10_2018_batch.xml`) se descubrió un error en los datos de entrenamiento.
Los datos de la clase 1 y 0 no se encontraban mezclaban.

Se repiten los Grid-searchs con los datos mezclados.

Se añaden métodos para obtener más información sobre los conjuntos.
* Valoraciones por usuario (positivas, negativas y no valoradas)

#### Resultados

Ver fichero `docs/16_10_2018_gs.xml`

## Cambios [19/10/2018]
Se cambia la forma de separar datos y evaluar el modelo para adaptalo a un problema similar como el descrito por **Yehuda Koren** en **"Performance of Recommender Algorithms
on Top-N Recommendation Tasks"**.

### Generación de datos
Utilizar aquellos usuarios que tenagan como mínimo 5 reviews (positivas o negativas) **CON IMAGEN**.  

De los anteriores, aquellos que tengan como mínimo 5 (analisis de distribución de usuarios `docs/19_10_2018_stats.xlsx`) reviews positivas.  

* De todas las reviews positivas, separar para cada usuario `(3,1,1 => TRAIN, DEV, TEST)`.  
* Todas las negativas para TRAIN.
* Crear, para cada usuario, `n` (para compensar distribución de positivos 50/50) ejemplos negativos con los restaurantes no vistos por usuario.  
* En este caso `n = ((N_POSITIVOS_TRAIN - N_NEGATIVOS_TRAIN)/N_USUARIOS)+1`.

La razon de añadir nuevos elementos es reforzar la evaluación (ver apartado de evaluación).

Por tanto los conjuntos finales son:
* **TRAIN_V1**: Reviews separadas para TRAIN con la imágen puesta a 0
* **TRAIN_V2**: Reviews separadas para TRAIN con el embedding de la imágen.
* **DEV**: Reviews separadas para DEV y sin multiplicar por el número de imágenes
* **TEST**: Reviews separadas para TEST y sin multiplicar por el número de imágenes

### Evaluación
Para cada positivo de cada usuario en DEV o TRAIN, se escogen 1000 aleatorios no vistos y se consideran como negativos. Tras valorar cada uno de ellos se ordenan y se considera acierto si el item real está en el **TOP-N**.

Para obtener el precisión y recall, previo entrenamiento del modelo con los datos de TRAIN, para cada item `i` positivo de cada usuario:

* Se seleccionan 1000 items no vistos por el usuario (No se pueden añadir los ya añadidos en entrenamiento).
* Se obtiene la recomendación asociada a cada uno de los 1001 items y se ordenan de mayor a menor rating.
* Si el item positivo `i` se encuentra en el TOP-N de esa lista, se cuenta como un acierto.

#### Precision y Recall
Siendo `|T|` el número de positivos del conjunto de TEST y `N` el número de items del TOP-N:

* **Precision:** `precision(N): # hits/|T|`
* **Recall:**  `recall(N): recall(N)/N`

## Cambios [29/10/2018]

### Modelo

Se utiliza ahora solamente el modelo v3* (ahora v1), entrenado de la siguiente forma:
* Un batch solo con `valoraciones` para la parte de clasificación binaria.
* Un batch solo con `imagenes` para la parte de multiregresión.

De esta forma, se pretenede entrenar cada parte de forma independiente compartiendo la codificación de usuarios y restaurantes.  

En el futuro, es posible que se añada una tercera salida de la red con los comentarios en formato `D2V` o `LSTM` lo que implicariía otro paso más en el entrenamiento.

*se probó una versión con el producto escalar previamente, pero los resultados eran peores (Ver fichero `docs/29_10_2018/deep_vs_dot.xlsx`). También se probó la posibilidad de añadir capas ocultas o en la parte de clasificación binaria sin mejoras relevantes (Ver fichero `docs/29_10_2018/deep_hidden.xlsx`)


### Conjunto de datos
Usuarios con más de 5 revews positivas, separando de la siguiente forma:
* **TEST:** 1 review positiva + 1000  nuevas reviews (distintas de DEV y TRAIN)
* **DEV:** 1 review positiva  + 100 nuevas reviews (distintas de TRAIN)
* **TRAIN_V1:** Resto reviews positivas  + reviews negativas + n nuevas reviews para compensar distribución de 1s y 0s
* **TRAIN_V2:** Igual que TRAIN_V1 (sin n reviews nuevas) y sólo con imágenes

### Entrenamiento
Dos fases:
* Primera fase entrenado solamente la clasificación binaria con **TRAIN_V1**.
* Segunda fase entrenado la clasificación binaria con **TRAIN_V1** y la multiregresión con **TRAIN_V2**.

### Evaluación
Se mantiene la evaluación TOP-N anterior.

### Pruebas
Los mejores resultados conseguidos rondan los 4200 hits en TOP5 con dropout en el modelo deep.

### Análisis de datos

* Se generó un gráfico para ver la proporción entre restaurantes y reviews emulando al de Koren. (Ver fichero `docs/29_10_2018/tail_graph.xlsx`)
* Para ver la distribución de ejemplos por usuario, se generó otro gráfico (Ver fichero `docs/29_10_2018/users_graph.xlsx`)

Analizando lo anterior, se realizan un cambios en la generación de datos descritos a continuación.


## Cambios [12/11/2018]

### Conjunto de datos
Usuarios con más de 3 revews positivas, separando de la siguiente forma:
* **TEST:** 1 review positiva + 99  nuevas reviews (distintas de DEV y TRAIN). Se coje la última del usuario (ver aclaraciones).
* **TEST_V2:** De las anteriores, las que tienen imagen y repetidas por el número de imágenes.
* **DEV:** 1 review positiva  + 99 nuevas reviews (distintas de TRAIN). Se coje la penúltima del usuario (ver aclaraciones).
* **DEV_V2:** De las anteriores, las que tienen imagen y repetidas por el número de imágenes.
* **TRAIN_V1:** Resto reviews positivas  + reviews negativas + n* nuevas reviews para compensar distribución de 1s y 0s
* **TRAIN_V2:** Igual que TRAIN_V1 (sin n reviews nuevas) , sólo con imágenes y compensado.

```
* Se vió que añadir más de n items no vistos por usuario mejora los resultados.
Por tanto N = (N_RST/N_USR)*100
Ver: docs/12_11_2018/12_11_2018_no_vistos.xlsx

No añadir items es bueno, pero peor que lo anterior.
```
#### Compensar TRAIN1 Y TRAIN2
Para igualar el tamaño de los datos de TRAINv1 y TRAINv2, se rellena este último (dado que es de menor tamaño) con ejemplos seleccionados aleatoriamente de este propio conjunto.
De esta forma, a la hora de incluir imagenes, es posible crear batches de igual tamaño (en TRAINv1 y TRAINv2) así como obtener el mismo número de estos.

### Regularización L2 (descartado)
Se probó a añadir regularización L2, pero requiere de numerosos parámetros. Mejor dropout.

### Selecciona de "no vistos" según probabilidad (descartado)
Empeora el resultado del TOP-N.

### Análisis de datos
* Se generó un gráfico para ver la proporción entre restaurantes y reviews emulando al de Koren. (Ver fichero `docs/12_11_2018/tail_graph.xlsx`)
* Para ver la distribución de ejemplos por usuario, se generó otro gráfico (Ver fichero `docs/12_11_2018/users_graph.xlsx`)
* Un resumen de lo anterior se puede ver en: `docs/12_11_2018/Gráficas.pdf`

### Aclaraciones 

Se ordenan los ejemplos positivos de cada usuario de más antiguo a más nuevo y se seleccionan los ejemplos del final para dev y test

#### ¿Por qué se ordenan para desordenar?
Si no se ordenaran, las reviews positivas se encontrarían agrupadas por el restaurante (dada la operación previa). Ej:

| INDEX | USER | RESTAURANT |
|-------|------|------------|
| 0     | 28   | **55**     |
| 1     | 456  | **55**     |
| 2     | 12   | **55**     |
| 3     | 9    | **55**     |
| 4     | 75   | **55**     |
| 5     | 96   | **55**     |
| 6     | 777  | 897        |
| 7     | 25   | 897        |

Si se hace un GroupBy por usuario de estos datos, el restaurante 55 SIEMPRE aparecerá en la primera fila de cada usuario `(28,456,12,9,75,96,777)` y por tanto siempre irá o bien a TRAIN o a DEV o a TEST en función de la política de selección de ejemplos.

#### ¿Por qué va mejor >=3 que >=20 añadiendo +100/+200/+300?

* De los usuarios >=20 se tiene mucha información y añadir más no aporta mucho
* De los usuarios de >=3, como muchos tienen pocos datos, añadir nueva información facilita el aprendizaje de la fn para esos usuarios (que son muchos)

### Pruebas
Se probaron diferentes tamaños de embedding y de capas ocultas (resumen en `docs/12_11_2018/12_11_2018_arch.xlsx`):
* 512+512 -> ~1500 `out/12_11_2018/arch/img_bin_512.out` 
* 512+512 -> 128 -> ~1500 `out/12_11_2018/arch/img_bin_512_128.out` 
* 512+512 -> 1024 -> ~1500 `out/12_11_2018/arch/img_bin_512_128.out` 
* 1024+1024 -> 1024 -> ~1500 `out/12_11_2018/arch/img_bin_512_128.out`

### Evaluación de las imágenes en DEV
No se cuentan los restaurantes que no aparecen en TRAIN y si en DEV (no se aprendió nada de ellos)

Para los que si tienen datos en TRAIN:
* Se busca, entre los vectores (de las imágenes) del restaurante en TRAIN, el más cercano a la predicción y se utiliza este como predicción del modelo.
* Con la "predicción" anterior, se calcula la distancia con las imágenes de ese usuario y restaurante en DEV.
* Se toma la menor distancia y se calcula finalmente la media de todas las distancias mínimas.


## Actualización [17/12/2018]
Tras ver que el rendimiento de la predicción de imágenes no es muy bueno (utlizar un modelo "tonto" que predice siempre la imágen centroide del restaurante mejora los resultados del nuestro), se decide simplificar la tarea a resolver, siendo ahora:  

`¿Añadir las imágenes mejora la tarea de recomendación?`

Se cree que una mejor forma para realizar una recomendación de imágenes es lo indicado en `docs/17_12_2018/recomendar_imágenes.pdf`

Este nuevo método (Modelv3) se desarollará en una actualización futura.

### Pruebas
Se hacen diferentes pruebas variando usuarios, restaurantes y lugares con el fin de ver si añadir o no imágenes mejora la clasificación binaria.  
Estas pruebas se pueden ver en `docs/17_12_2018/17_12_2018_tests.xlsx` y están hechas con el mismo modelo (con mismos hiper-parámetros) sobre el conjunto de validación.  
En todas ellas se puede ver una mejora del rendimiento a la hora de añadir las imágenes. También se ve un decremento del rendimiento cuando se reduce el número de usuarios y restaurantes.

## Actualización [16/01/2019]
Nuevo modelo v3

Train v3 => Al final no se añaden imágenes en los "no vistos" para no confundir al modelo. Al añadirlas, la mayoría de ejemplos que tenían imágen eran estos ejemplos falsos, lo cual decrementaba el rendimiento.

Se hace un gs para ver el mejor LR con y sin imágenes.

## Análisis

El nuevo conjunto de entrenamiento TRAIN_v3:
* El 8,13% de las reviews tiene imagen (77% + / 22% -)
* El 91,8% de las reviews no tiene imagen (12% + / 87% -)

### Evaluación

La recomendación de restaurantes como hasta ahora (TOP N entre 100 ejemplos (1+ y 99-))

La recomendación de imágenes de restaurantes:

* Con los ejemplos positivos de DEV que tienen foto
* Obtener una probabilidad para TODAS (TRAIN + DEV + TEST) las fotos del restaurante y quedarse con la de mayor probabilidad.
* Calcular las distancias entre la foto recomendada con la o las de DEV que se estén evaluando.
* Quedarse con la distancia mínima.
* Obtener tambien la distancia mínima a la foto más cercana al centroide, a una aleatoria completamente, a la de mayor y a la de menor distancias (4 sistemas "tontos")
* Finalmente se obtiene una distancia mínima media.

No se evaluarán todos los restaurantes de DEV con foto, solo aquellos que, en total, tengan más de 5 fotos.

### Resultados

Durante el grid search se obtuvieron mejores resultados para el sistema "tonto" del centroide que para el de nuestra tarea.
Se probó a entrenar solamente con los ejemplos con imágen para ver si se reducía la distancia mínima, pero nada:

| MEAN    | MEDIAN | MIN_D   | RNDM    | CNTR   | MAX     | MIN |
|---------|--------|---------|---------|--------|---------|-----|
| 33.1981 | 27.0   | 12.6405 | 12.0552 | 9.9794 | 18.4548 | 0.0 |
| 33.1732 | 27.0   | 12.5208 | 12.0901 | 9.9794 | 18.4548 | 0.0 |
| 34.3079 | 29.0   | 12.349  | 11.9695 | 9.9794 | 18.4548 | 0.0 |
| 35.5677 | 30.0   | 12.1322 | 12.0573 | 9.9794 | 18.4548 | 0.0 |
| 36.1027 | 31.0   | 12.064  | 12.0371 | 9.9794 | 18.4548 | 0.0 |
| 36.8609 | 32.0   | 12.1859 | 12.0895 | 9.9794 | 18.4548 | 0.0 |
| 36.6867 | 32.0   | 11.982  | 12.1196 | 9.9794 | 18.4548 | 0.0 |
| 37.4171 | 33.0   | 12.0466 | 12.0924 | 9.9794 | 18.4548 | 0.0 |
| 37.4498 | 33.0   | 11.9151 | 12.0216 | 9.9794 | 18.4548 | 0.0 |
| 38.6722 | 34.0   | 12.0348 | 12.0939 | 9.9794 | 18.4548 | 0.0 |
| 39.161  | 34.0   | 11.9669 | 12.1394 | 9.9794 | 18.4548 | 0.0 |
| 39.4348 | 35.0   | 11.9934 | 12.1029 | 9.9794 | 18.4548 | 0.0 |
| 39.4147 | 35.0   | 11.9448 | 12.0788 | 9.9794 | 18.4548 | 0.0 |
| 39.1515 | 35.0   | 11.9733 | 12.0675 | 9.9794 | 18.4548 | 0.0 |
| 40.0274 | 36.0   | 12.046  | 12.1146 | 9.9794 | 18.4548 | 0.0 |

## Actualización [31/01/2019]
Se decide hacer una prueba "interna" para ver si se puede conseguir aprender la clasificación binaria (me gusta y no me gusta).

Partiendo de los datos anteriores (`TRAIN_v3`, `DEV_v3` y `TEST_v3`) se plantea:
* Juntar `TRAIN_v3` (sólo los que tienen imagen) y los positivos de `DEV_v3`.
* Dividir el conjunto anterior en `TRAIN_v3_1` y `DEV_v3_1` (70%/30%).
* Triplicar los negativos del `TRAIN_v3_1`.
* Hacer grid-search con `TRAIN_v3_1` y `DEV_v3_1`.
* Entrenar con los mejores parámetros combinando `TRAIN_v3_1` y `DEV_v3_1`.
* Finalmente, hacer test con el conjunto `TEST_v3` calculando la distancia mínima (Ver evaluación en actualización previa).

### Resultados

Tras evaluar la acuraccy en `TRAIN_v3_1` y `DEV_v3_1` se obtuvieron unos resultados de 99% y 92% respectivamente.
Tras evaluar con el conjunto `TEST_v3` se obtuvieron los mismos resultados que en la actualización previa.

Analizando este comportamiento se descubrió que el modelo no tenía en cuenta las imágenes (valor de pesos reducido; ver imágen `out/31_01_2019/R0.png`) y se vió que eliminando las imágenes (poniendo 0's) se obtenían los mismos e incluso mejores resultados en acurracy.

## Actualización [08/02/2019]
Ante los resultados previos se decide cambiar el concepto del aprendizaje, pasando de aprender `(u,r,foto) => Me gusta / No me gusta` a aprender `(u,r,foto de u en r, foto de otro u en r)`.
Esto implica pasar de un problema de clasificación binaria a un problema de aprendizaje de preferencias. Destacar en este caso que no se aprende el `este usuario que hizo esta foto en este restaurante le gusta / no le gusta` sinó que se aprende `este usurio prefiere esta foto a esta`.

### Modelo
Inicialmente se creará un modelo lineal (ModelV4) en el que se obtendrá un embedding para el usuario y otro para el restaurante en espacio `512`.
Este modelo recibirá tambien 2 imágenes, las cuales se cambiarán de espacio mediante una matriz que las llevará a espacio `1024`.

A partir de los embeddigs anteriores (usuario,restaurante, imágen +, imágen -) se calcurarán 2 productos escalares:
* **h(m) = < concat(u,r), imagen mejor >**  
* **h(p) =< concat(u,r), imagen peor >** 
 
y la loss a minimizar será: `loss = max(0,1-(h(m)-h(p)))`

### Creación de ejemplos

Para cada terna (usuario, restaurante, imagen) única:
* Crear 5 ejemplos con las 5 fotos más distantes del restaurante respecto de la actual.

## Actualización [15/02/2019]
Se vio que los vectores de las fotos son mejores cuando se utilizan las imánenes de menor dimensión, por tanto, se volvieron a generar los vectores con estas imágenes de menor tamaño.
Este comportamiento es devido, probablemente, a el reescalado que re realiza en la entrada de la red, que al partir de una imágen más reducida no afecta tanto.

## Actualización [20/02/2019]
Se descubrió que los modelos básicos "centroide" y "random" partian con ventaja sobre nuestro modelo. El modelo utiliza solo las fotos en TRAIN para aprender y nosotros estabamos calculando centroide y random con todas las fotos disponibles.
Para solucionar este comportamiento, se ha de calcular los centroides de cada restaurante y la foto random de cada uno con los datos que hay en TRAIN y solamente esos.

Al hacer esto se vio que puede haber restaurantes que estén en DEV y no en TRAIN, por tanto hay que cambiar el planteamiento realizado hasta ahora.

# Modelo final [25/03/2019]
Finalmente se opta por utilizar una red neuronal (ModelV4D2) para resolver el problema de preferencias.
Cada una de las preferencias tendrá los siguientes componentes:
* Usuario (u).
* Restaurante al que fué (rb).
* Foto que hizo (fb).
* Restaurante al que fué o no (rm).
* Foto que no hizo (fm).

La idea es aplicar la misma red a cada uno de los componentes de cada preferencia, por tanto se tendrá un valor para la entrada (u,rb,fb) y otro para (u,rm,fm).
La red se entrenará con el objetivo de que la salida de la red para la entrada buena (u,rb,fb) sea mayor (+1 de margen) respecto de la mala.

Por tanto la pérdida del modelo sería: `loss = max(0 , 1-(salida_buena-salida_mala)`

### Arquitectura
La red realizará el siguiente procedimiento:
* Entrada: un usuario (one-hot), un restaurante (one-hot) y una foto (1536).
* Obtener embedding tanto de usuario como de restaurante (a 512 y 256 respectivamente)
* Transformar el vector de la imagen a un espacio de 512 dimensiones (con ReLu y DropOut) y finalmente a uno de 256.
En este punto se tienen 3 embeddings, uno para cada componente de la entrada de tamaños 512, 256 y 256 para usuario, restaurante e imagen.
* Estos 3 embeddings se concatenan y se les aplica un DropOut
* Finalmente, esta concatenación se transforma, mediante varias capas ocultas de 1024 a 512, de 512 a 256, de 256 a 128 y de 128 a la salida de dimensión 1. 
* Todas las capas anteriores poseen Bias, ReLu y DropOut a excepción de la última.

### Evaluación
El conjunto de evaluación (tanto `DEV` como `TEST`) tendrá, para cada review:
* Las fotos de la review del usuario `u` en el restaurante `r`.
* El resto de fotos de `r` que no son del usuario. En estas reviews se modifica el usuario y se pone `u`.

El objetivo es ver que valoración da el modelo a cada una de las fotos del restaurante incluidas las del usuario (`n` en total) y, ordenando las valoraciones de mayor a menor, obtener la posición de la primera foto del usuario `p`.
Con esta posición se calculará el percentil (`p\n`) y el percentil comenzando en cero (`(p-1)\n`).


Destacar que en los conjuntos de evaluación, cada usuario está solamente una vez dada la forma de separar; ver siguiente apartado.

### Creación de conjuntos

##### Separar TRAIN, DEV y TEST:
* Se utilizan solamente las reviews que tengan foto.
* Se agrupan las reviews de un usuario que fue más de una vez al mismo restaurante.
* Se divide el conjunto en 2: `TRAIN_DEV` y `TEST`.
    * Para ello, por cada usuario, se mueve una de sus reviews a `TEST` y el resto a `TRAIN_DEV`.
    * Si solo tiene una, esta va para `TRAIN_DEV`.
    * Al finalizar, se comprueba que no existan restaurantes en `TEST` que no estén en `TRAIN_DEV`. Si los hay se trasladan a este último conjunto.
* Finalmente, se utiliza el mismo procedimiento para dividir el conjunto de `TRAIN_DEV` en `TRAIN` y `DEV`.

De este procedimiento resultan los conjuntos `TRAIN_DEV` y `TEST` utilizados para el entrenamiento final y `TRAIN` y `DEV` utilizados para la fase de búsqueda de hiperparámetros.

##### Generar conjuntos adaptados al modelo:
Partiendo de los conjuntos anteriores, es necesario llevar a cabo una adaptación a la forma de entrenar y evaluar.

En el caso del conjunto de entrenamiento, partiendo del conjunto de `TRAIN` o `TRAIN_DEV` anteriores se crearán nuevos de la siguiente forma:
* Para cada **fila** (usuario, restaurante, foto 1)
* Crear preferencias con las 10 fotos (o menos, si no hay 10) más alejadas a la actual (foto 1) de otros usuarios dentro del **restaurante actual**.
* Crear preferencias con 10 fotos (o menos, si no hay 10) de otros usuarios en **otros restaurantes**.

Finalmente, para la evaluación se han de crear nuevos conjuntos partiendo de `DEV` y `TEST` de la siguiente forma:
* Para cada **review** ((usuario, restaurante, foto 1), (usuario, restaurante, foto 2)... ), marcar las imágenes existentes como 'fotos reales del usuario `u`' 
* Añadir el resto de imágenes del restaurante (las de otros usuarios) cambiando su usuario por `u`


### Baselines
Con el fin de comparar los resultados de nuestro sistema contra otros más elementales y sencillos, se calcularon los siguientes baselines:
* **Baseline Random:** Para cada uno de los ejemplos de `DEV` o `TEST` donde se tienen todas las fotos de un restaurante, incluidas las del usuario, se asigna una probabilidad aleatoria uniforme a cada una de ellas, se ordenan de mayor a menor y se busca la posición de la primera foto real del usuario.
* **Baseline Centroide:** Se calcula previamente en el conjunto de `TRAIN` o `TRAIN_DEV` el centroide de cada restaurante (a partir de sus fotos). En `DEV` o `TEST` se calcula la distancia ente el centroide correspondiente y las fotos de test, se ordenan de menor a mayor distancia y se obtiene la posicion de la primera foto real del usuario.

### Experimentación

##### Validación
Sobre los conjuntos de `TRAIN` y `DEV` adaptados al modelo y partiendo de los siguientes hiper-parámetros prefijados:
* **Epochs:** 100
* **DropOut:** 0.8
* **Descenso LR:** Linear cosine decay

Se realizó un GRID-SEARCH **con 5 repeticiones** de cada experimento variando:
* **El LR inicial:** 1e-3, 1e-4 y 1e-5
* **La ciudad:** Gijón, Barcelona y Madrid

Resultados en `out/20_02_2019` y en `docs/20_02_2019/Resultados_*`

##### Test
Con la mejor combinación obtenida previamente para cada ciudad, se realizaron nuevamente 5 repeticiones utilizando en este caso los conjuntos `TRAIN_DEV` y `TEST`.

Además de obtener los valores de percentil y percentil comenzando en cero para todo el conjunto, en la evaluación final se desagregaron los resultados en los siguientes grupos:
* **Usuarios con 9 fotos o más en `TRAIN_DEV`:** 20% aprox de los ejemplos
* **Usuarios con entre 5 y 8 fotos en `TRAIN_DEV`:** 40% aprox de los ejemplos
* **Usuarios con 4 fotos en `TRAIN_DEV`:** 60% aprox de los ejemplos
* **Usuarios con entre 2 y 3 fotos en `TRAIN_DEV`:** 80% aprox de los ejemplos
* **Usuarios con 1 foto en `TRAIN_DEV`:** 100% de los ejemplos

Para cada uno de estos grupos se calculó:
* El resultado del modelo
* La mejora del modelo respecto del Baseline Random (RND-MOD)
* La mejora del modelo respecto del Baseline Centroide (CNT-MOD)







