# Generador de dataset sintético
Se utilizó un generador aleatorio con make_blobs de scikit-learn. El número de centroides es elegido al azar entre 1 y 20 en cada ejecución, y se garantiza una distancia mínima de 5 unidades entre cada par de centroides para asegurar que los clusters sean visualmente distinguibles.
Parámetros importantes del generador:
• n_centers: entero pseudo aleatorio entre 1 y 20 (random.randint)
• Distancia mínima entre centroides: 5 unidades (np.linalg.norm + all())
• Dispersión por cluster: blob_std = 0.8 para los clusters compactos
• Número de muestras: 2.000
• Rango de coordenadas: [-10, 10] en ambos ejes
# Entrenamiento de KMeans
Se utilizó KMeans de scikit-learn con k igual al número de centroides generados. El modelo asigna cada punto al centroide más cercano utilizando la distancia euclídea, y repite este proceso hasta alcanzar la convergencia.
Se dibujaron las fronteras de decisión del modelo mediante un meshgrid de resolución 1000x1000, coloreando cada región según el cluster asignado y marcando los centroides encontrados con un círculo blanco y una cruz negra.
# Método del Codo
Se calculó la inercia (suma de distancias cuadradas al centroide más cercano) para valores de k desde 1 hasta min(20, n_centers). El gráfico resultante muestra el punto de inflexión donde añadir más clusters ya no reduce significativamente la inercia.
# Silhouette Score
Se calculó el Silhouette Score para cada valor de k desde 2 hasta el máximo. Este indicador mide qué tan bien está agrupada cada muestra: valores cercanos a 1 indican buena separación, valores cercanos a 0 indican solapamiento, y valores negativos indican mala asignación.
Adicionalmente se generaron diagramas de silueta para valores específicos de k, donde se visualiza el coeficiente de silueta de cada muestra ordenada por cluster, permitiendo identificar clusters bien definidos versus clusters problemáticos.
# Descripción del Dataset
El dataset GTSRB (German Traffic Sign Recognition Benchmark) contiene imágenes de señales de tránsito reales tomadas en condiciones variables de iluminación, ángulo y resolución. Características principales:
•	Total de imágenes usadas: 12.000 (muestra aleatoria del conjunto de entrenamiento)
•	Número de clases: 43 tipos de señales de tránsito
•	Resolución normalizada: 32x32 píxeles en escala de grises
•	Características por imagen: 1024 (vectores aplanados, normalizados a [0,1])
•	Etiquetas: presentes en Train.csv como ClassId — NO usadas para entrenar

# Preprocesamiento
Cada imagen fue cargada con OpenCV, convertida a escala de grises y redimensionada a 32x32 píxeles. Los valores de píxel se normalizaron dividiendo por 255 para obtener valores en el rango [0, 1]. Posteriormente se aplicó StandardScaler para centrar y escalar los datos antes de PCA.
# KMeans en el dataset GTSRB 
Se aplicó K-Means con k=43 (igual al número real de clases) sobre las representaciones en 2D y 3D obtenidas con PCA. Se visualizaron las fronteras de decisión y se calcularon el Método del Codo y el Silhouette Score para distintos valores de k.