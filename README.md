[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Lj3YlzJp)
# Proyecto Final 2025-1: AI Neural Network
## **CS2013 Programación III** · Informe Final

### **Descripción**

> Ejemplo: Implementación de una red neuronal multicapa en C++ para clasificación de dígitos manuscritos.

### Contenidos

1. [Datos generales](#datos-generales)
2. [Requisitos e instalación](#requisitos-e-instalación)
3. [Investigación teórica](#1-investigación-teórica)
4. [Diseño e implementación](#2-diseño-e-implementación)
5. [Ejecución](#3-ejecución)
6. [Análisis del rendimiento](#4-análisis-del-rendimiento)
7. [Trabajo en equipo](#5-trabajo-en-equipo)
8. [Conclusiones](#6-conclusiones)
9. [Bibliografía](#7-bibliografía)
10. [Licencia](#licencia)
---

### Datos generales

* **Tema**: Redes Neuronales en AI
* **Grupo**: `grupo_3_Daros`
* **Integrantes**:

  * José Daniel Huamán Rosales – 209900001 (Responsable de investigación teórica)
  * Juan Carlos Ticlia Malqui – 209900002 (Desarrollo de la arquitectura)
  * Paulo Isael Miranda Barrietos – 209900003 (Implementación del modelo)
  * José Daniel Huamán Rosales – 209900004 (Pruebas y benchmarking)
  * Elmer José Villegas Suarez – 209900005 (Documentación y demo)

> *Nota: Reemplazar nombres y roles reales.*

---

### Requisitos e instalación

1. **Compilador**: GCC 11 o superior
2. **Dependencias**:

   * CMake 3.18+
   * Eigen 3.4
   * \[Otra librería opcional]
3. **Instalación**:

   ```bash
   git clone https://github.com/EJEMPLO/proyecto-final.git
   cd proyecto-final
   mkdir build && cd build
   cmake ..
   make
   ```

> *Ejemplo de repositorio y comandos, ajustar según proyecto.*

---

### 1. Investigación teórica

* **Objetivo**: Explorar fundamentos y arquitecturas de redes neuronales.
  Historia y evolución de las redes neuronales. Una red neuronal artificial (RNA) es, según Haykin (1994), “un procesador paralelo masivamente distribuido que tiene una facilidad natural para el almacenamiento de conocimiento obtenido de la experiencia para luego hacer éste disponible para su uso”
  biblus.us.es
  . La historia de estos modelos se remonta a la década de 1940: la primera ola de investigación (cibernética, 1940s-1960s) introdujo teorías de aprendizaje neuronal (McCulloch & Pitts, 1943; Hebb, 1949) junto con los primeros modelos como la neurona de McCulloch-Pitts y el perceptrón de Rosenblatt (1958)
  deeplearningbook.org
  . Tras un periodo de desilusión en los 1970s (p.ej., las limitaciones señaladas por Minsky y Papert en 1969), surgió una segunda ola en los 1980s bajo el paradigma conexionista, impulsada por el algoritmo de retropropagación del error (Rumelhart et al., 1986) que permitió entrenar redes con una o dos capas ocultas de neuronas
  deeplearningbook.org
  . Finalmente, a partir de 2006 inicia la tercera ola o era del aprendizaje profundo (deep learning), caracterizada por redes neuronales con muchas capas (redes profundas) capaces de aprender representaciones jerárquicas a gran escala
  deeplearningbook.org
  .
  Principales arquitecturas: Perceptrón Multicapa (MLP), Redes Convolucionales (CNN), Redes Recurrentes (RNN). Las arquitecturas fundamentales de RNA incluyen:
  Perceptrón Multicapa (MLP): Es una red neuronal feed-forward (de propagación hacia adelante) con múltiples capas de neuronas totalmente conectadas (una capa de entrada, capas ocultas y una capa de salida). Una MLP se puede ver como una función matemática que transforma un conjunto de entradas en un conjunto de salidas mediante la composición de muchas funciones más simples
  mnassar.github.io
  . De hecho, el término “multilayer perceptron” es algo impreciso, pues el modelo realmente consta de múltiples capas de modelos de regresión logística con activaciones continuas, más que de múltiples perceptrones binarios con funciones de activación discontinuas
  wiki.eecs.yorku.ca
  . Las MLP son útiles para aproximar funciones complejas en tareas de clasificación y regresión, aprendiendo pesos sinápticos que capturan las relaciones entre las variables de entrada y salida.
  Red Neuronal Convolucional (CNN): Arquitectura diseñada para procesar datos con estructura de rejilla (por ejemplo, imágenes pixeles en 2D o series temporales 1D), aplicando operaciones de convolución para extraer características locales y espaciales
  slideplayer.com
  . En una CNN, al menos una capa usa convolución en lugar de la multiplicación matricial densa típica de las capas plenamente conectadas, lo que reduce drásticamente el número de parámetros al emplear conexiones locales y pesos compartidos
  slideplayer.com
  . Las CNN han logrado gran éxito en visión por computador al explotar propiedades como la invariancia a traslaciones y la detección de patrones locales repetitivos en las imágenes
  slideplayer.com
  .
  Red Neuronal Recurrente (RNN): Tipo de red neuronal que incorpora conexiones recurrentes (retroalimentación), lo cual la hace adecuada para datos secuenciales como texto, voz o series temporales. Las RNN mantienen un estado interno de memoria que se actualiza a medida que procesan una secuencia, permitiendo modelar dependencias a lo largo del tiempo
  arxiv.org
  . Este estado oculto actúa como un resumen de lo “recordado” hasta el momento en la secuencia, influyendo en la salida mientras avanzan los pasos temporales. Las variantes modernas de RNN, como LSTM y GRU, introducen mecanismos de puerta en las neuronas recurrentes para facilitar el aprendizaje de dependencias de largo plazo mitigando problemas como el desvanecimiento del gradiente.
  Algoritmos de entrenamiento: retropropagación y optimizadores. El algoritmo clave para entrenar redes neuronales es la retropropagación del error (backpropagation), que permite calcular de forma eficiente el gradiente de la función de pérdida con respecto a todos los pesos de la red, propagando el error desde la capa de salida hacia las capas anteriores
  deeplearningbook.org
  . Dicho gradiente se utiliza luego en un esquema de optimización por descenso de gradiente para ajustar los pesos. En la práctica, se emplea típicamente descenso por gradiente estocástico (stochastic gradient descent, SGD) sobre minibatches de datos, junto con mejoras como el método de momento (momentum) para acelerar la convergencia. Más recientemente, se han desarrollado optimizadores adaptativos (p. ej., AdaGrad, RMSProp, Adam) que ajustan automáticamente la tasa de aprendizaje durante el entrenamiento, logrando convergencias más rápidas y estables
  mnassar.github.io
  . Estas técnicas de optimización permiten entrenar redes neuronales profundas de manera efectiva incluso con grandes conjuntos de datos.
---

### 2. Diseño e implementación

#### 2.1 Arquitectura de la solución

* **Patrones de diseño**: ejemplo: Factory para capas, Strategy para optimizadores.
* **Estructura de carpetas (ejemplo)**:

  ```
  proyecto-final/
  ├── src/
  │   ├── layers/
  │   ├── optimizers/
  │   └── main.cpp
  ├── tests/
  └── docs/
  ```

#### 2.2 Manual de uso y casos de prueba

* **Cómo ejecutar**: `./build/neural_net_demo input.csv output.csv`
* **Casos de prueba**:

  * Test unitario de capa densa.
  * Test de función de activación ReLU.
  * Test de convergencia en dataset de ejemplo.

> *Personalizar rutas, comandos y casos reales.*

---

### 3. Ejecución

> **Demo de ejemplo**: Video/demo alojado en `docs/demo.mp4`.
> Pasos:
>
> 1. Preparar datos de entrenamiento (formato CSV).
> 2. Ejecutar comando de entrenamiento.
> 3. Evaluar resultados con script de validación.

---

### 4. Análisis del rendimiento

* **Métricas de ejemplo**:

  * Iteraciones: 1000 épocas.
  * Tiempo total de entrenamiento: 2m30s.
  * Precisión final: 92.5%.
* **Ventajas/Desventajas**:

  * * Código ligero y dependencias mínimas.
  * – Sin paralelización, rendimiento limitado.
* **Mejoras futuras**:

  * Uso de BLAS para multiplicaciones (Justificación).
  * Paralelizar entrenamiento por lotes (Justificación).

---

### 5. Trabajo en equipo

| Tarea                     | Miembro  | Rol                       |
| ------------------------- | -------- | ------------------------- |
| Investigación teórica     | Alumno A | Documentar bases teóricas |
| Diseño de la arquitectura | Alumno B | UML y esquemas de clases  |
| Implementación del modelo | Alumno C | Código C++ de la NN       |
| Pruebas y benchmarking    | Alumno D | Generación de métricas    |
| Documentación y demo      | Alumno E | Tutorial y video demo     |

> *Actualizar con tareas y nombres reales.*

---

### 6. Conclusiones

* **Logros**: Implementar NN desde cero, validar en dataset de ejemplo.
* **Evaluación**: Calidad y rendimiento adecuados para propósito académico.
* **Aprendizajes**: Profundización en backpropagation y optimización.
* **Recomendaciones**: Escalar a datasets más grandes y optimizar memoria.

---

### 7. Bibliografía

Referencias (formato APA 7.ª ed.)
Haykin, S. (1994). Neural networks: A comprehensive foundation (1.ª ed.). Prentice Hall.

McCulloch, W. S., & Pitts, W. (1943). A logical calculus of the ideas immanent in nervous activity. The Bulletin of Mathematical Biophysics, 5(4), 115-133. https://doi.org/10.1007/BF02478259

Hebb, D. O. (1949). The organization of behavior: A neuropsychological theory. Wiley.

Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain. Psychological Review, 65(6), 386-408. https://doi.org/10.1037/h0042519

Minsky, M., & Papert, S. (1969). Perceptrons: An introduction to computational geometry. MIT Press.

Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. Nature, 323, 533-536. https://doi.org/10.1038/323533a0

Hinton, G. E., Osindero, S., & Teh, Y.-W. (2006). A fast learning algorithm for deep belief nets. Neural Computation, 18(7), 1527-1554. https://doi.org/10.1162/neco.2006.18.7.1527

Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT Press.

Nielsen, M. A. (2015). Neural networks and deep learning. Determination Press. https://neuralnetworksanddeeplearning.com

LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324. https://doi.org/10.1109/5.726791

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735-1780. https://doi.org/10.1162/neco.1997.9.8.1735

Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder–decoder for statistical machine translation. Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (pp. 1724-1734). Association for Computational Linguistics.

Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research, 12, 2121-2159.

Tieleman, T., & Hinton, G. (2012). Lecture 6.5—RMSProp: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural Networks for Machine Learning. University of Toronto.

Kingma, D. P., & Ba, J. L. (2015). Adam: A method for stochastic optimization. International Conference on Learning Representations (ICLR 2015). https://arxiv.org/abs/1412.6980
---

### Licencia

Este proyecto usa la licencia **MIT**. Ver [LICENSE](LICENSE) para detalles.

---
