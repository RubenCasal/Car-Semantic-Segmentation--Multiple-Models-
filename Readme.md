# DEMO
<p align="center">
  <img src="./demo_video.gif" alt="Demo Video" width="500" />
</p>

<p align="center">
  <a href="https://youtu.be/qO9DAAGrzR4" target="_blank">Watch the full video on YouTube</a>
</p>


# Proyecto de Segmentación de Video para Conducción Autónoma

Este proyecto aplica modelos de segmentación semántica para identificar elementos clave en videos de conducción, apoyando la percepción de sistemas de vehículos autónomos. Utilizamos una red UNET para detectar cuatro clases en la escena, cada una con un color específico:

- **Vehículos:** Azul `(255, 0, 0)`
- **Peatones:** Amarillo `(255, 255, 0)`
- **Carretera:** Rojo `(0, 0, 255)`
- **Marcas en la carretera:** Verde `(0, 255, 0)`

## Modelo/Arquitectura

El modelo de segmentación se basa en una arquitectura **UNET** que consta de:

1. **Encoder (Downsampling):** Reduce la resolución y extrae características clave mediante bloques de convolución y max-pooling.
2. **Bottleneck:** Actúa como un cuello de botella, capturando las características más relevantes en baja resolución.
3. **Decoder (Upsampling):** Restaura la resolución de la imagen utilizando transposiciones de convolución y conexiones "skip" para retener detalles importantes.
4. **Capa de Salida:** Genera una máscara segmentada con un canal para cada clase de interés.

La arquitectura UNET permite una segmentación precisa, crucial para la interpretación en tiempo real de las escenas de conducción.

## Entrenamiento

Para optimizar la segmentación, se ha decidido utilizar **cuatro modelos independientes** en lugar de uno solo que detecte todas las clases. Este enfoque permite que cada modelo esté especializado en su propia clase (vehículos, peatones, carretera, y marcas en la carretera), mejorando la precisión y el rendimiento, especialmente en situaciones donde las clases tienen diferencias visuales marcadas. Al reducir la complejidad de las predicciones, cada modelo puede enfocarse en patrones específicos de su clase, resultando en una segmentación más precisa.

### Detalles de Entrenamiento

- **Augmentaciones:** Se han utilizado diversas transformaciones para aumentar la robustez del modelo frente a variaciones en las escenas de conducción. Estas incluyen:
  - Escalado, rotación y desplazamiento aleatorio.
  - Efectos climáticos como niebla y lluvia.
  - Modificación de brillo y contraste.
  - Transformaciones geométricas como el flip horizontal y vertical.

- **Parámetros de Entrenamiento:**
  - **Tamaño de imagen:** 360x480 píxeles para optimizar el balance entre resolución y rendimiento.
  - **Batch Size:** 8, adecuado para el tamaño de imagen y la capacidad de la GPU.
  - **Learning Rate:** 1e-4, seleccionado para asegurar una convergencia estable.
  - **Función de pérdida:** Se ha implementado una pérdida combinada con `BCEWithLogitsLoss` y `JaccardLoss`, balanceando la precisión por píxel y la similaridad estructural entre predicciones y etiquetas.

Esta configuración permite obtener un modelo de segmentación robusto, capaz de adaptarse a diferentes condiciones de conducción y detectar con precisión cada clase relevante en la escena.
## Resultados y Modo de Uso

### Resultados

Para evaluar la efectividad de los modelos, se han probado en videos aleatorios que no forman parte del dataset de entrenamiento, garantizando que las predicciones sean generalizables y robustas. Los resultados muestran que cada modelo puede identificar y colorear de manera precisa las distintas clases: vehículos, peatones, carretera y marcas viales.

La función `generate_segmented_video` aplica la segmentación en cada fotograma del video de entrada y combina los colores definidos para cada clase, generando un video final en el que cada elemento de interés se resalta en tiempo real.

### Modo de Uso

Para ejecutar la segmentación en un video, se debe importar y llamar a la función `generate_segmented_video` con los parámetros adecuados, incluyendo las rutas de los modelos entrenados y los archivos de video de entrada y salida.

Ejemplo de uso en el archivo `main.py`:

```python
from video_segmentation_multi_model_function import generate_segmented_video

# Rutas de entrada y salida de video
INPUT_PATH = './input_videos/video2.mp4'
OUTPUT_PATH = './output_videos/segmentation2.mp4'

# Rutas de los modelos entrenados
VEHICLES_MODEL_PATH = './models/vehicles/vehicles2_model.pth.tar'
PEDESTRIANS_MODEL_PATH = './models/pedestrians/pedestrians3_model.pth.tar'
ROAD_MODEL_PATH = './models/road/road_model.pth.tar'
ROAD_MARKS_MODEL_PATH = './models/road_marks/road_marks3.pth.tar'

# Generación de video segmentado
generate_segmented_video(INPUT_PATH, OUTPUT_PATH,
                         vehicles_model_path=VEHICLES_MODEL_PATH,
                         pedestrian_model_path=PEDESTRIANS_MODEL_PATH,
                         road_model_path=ROAD_MODEL_PATH,
                         road_marks_model_path=ROAD_MARKS_MODEL_PATH)
