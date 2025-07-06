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

# 1. Investigación teórica

*Fundamentos y arquitectura de las redes neuronales con foco en un clasificador de spam en C++*

---

## 1.1 Historia y evolución de las redes neuronales

Las **redes neuronales artificiales (RNA)** nacen en la década de 1940 dentro de la cibernética.

* **McCulloch & Pitts (1943)** modelaron la primera neurona lógica –un sumador binario con umbral– y **Hebb (1949)** formuló la primera regla de aprendizaje sináptico (“las neuronas que se disparan juntas, se conectan”).
* **Rosenblatt (1958)** presentó el **perceptrón**, un clasificador lineal entrenable en hardware.
* El optimismo cesó cuando **Minsky y Papert (1969)** demostraron que un perceptrón de capa única no resuelve XOR ⇒ *primer invierno de la IA*.
* A mediados de los 80, **Rumelhart, Hinton y Williams (1986)** publican el algoritmo de **retropropagación del error** (backpropagation), reactivando el campo bajo el paraguas *conexionista* y permitiendo entrenar el **perceptrón multicapa (MLP)**.
* En los 90 se afianzan arquitecturas especializadas: **LeNet-5** (LeCun et al., 1998) para imágenes y **LSTM** (Hochreiter & Schmidhuber, 1997) para secuencias.
* Desde 2006, con el pre-entrenamiento de Hinton et al. y la llegada de la GPU, se inaugura la era del **aprendizaje profundo**. Hoy, modelos como **transformers** dominan la PNL, aunque para un proyecto de spam en C++ un MLP o una CNN 1-D ya son prácticos.

---

## 1.2 Arquitecturas clave y su relación con la detección de spam

| Arquitectura                     | Idea esencial                                                                                   | Por qué sirve en un clasificador de spam                                                                                                          |
| -------------------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MLP** (feed-forward denso)     | Multiplica vectores de características por matrices de pesos y aplica activaciones no lineales. | Nuestro **Bag-of-Words vectorizado** produce vectores densos (o dispersos) de tamaño \|vocab\|; un MLP aprende directamente la frontera ham/spam. |
| **CNN 1-D**                      | Convolución en la dimensión de la secuencia; detecta *n-gramas* locales con pesos compartidos.  | Frases como “free \$\$\$” o “click aquí” se detectan como patrones locales independientemente de su posición.                                     |
| **RNN / LSTM / GRU**             | Mantienen un estado oculto que se actualiza token a token.                                      | Útil si queremos modelar la **dependencia de orden** entre palabras en vez de tratarlas como bolsa.                                               |
| *(Mención)* **Transformer/BERT** | Auto-atención bidireccional; se pre-entrena con corpus masivo.                                  | En producción podríamos exportar a ONNX y usar `onnx-runtime`, pero supera el alcance de un laboratorio C++ puro.                                 |

---

## 1.3 Algoritmos de entrenamiento

1. **Retropropagación**: aplica la regla de la cadena para computar $\partial\mathcal L/\partial\theta$ capa por capa desde la salida hasta la entrada.
2. **Descenso de gradiente estocástico (SGD)** con *mini-batch* ajusta los pesos $\theta \leftarrow \theta - \eta\nabla\mathcal L$.
3. **Optimizadores adaptativos**:

  * **AdaGrad** (2011) adapta la tasa de aprendizaje por parámetro; útil si el vector de entrada es muy disperso como en BoW.
  * **RMSProp** (2012) suaviza AdaGrad con una media exponencial de gradientes al cuadrado.
  * **Adam** (2015) combina momento y RMSProp; converge rápido y es el estándar en proyectos de spam académicos.

---

## 1.4 Vectorización: puente entre teoría y código C++

> **Objetivo práctico**: convertir cada correo en un vector numérico y cada *forward/backward pass* en una operación de álgebra lineal soportada por BLAS/cuBLAS.

### 1.4.1 Vectorización del texto

* **Bag-of-Words (BoW)**: vector $v\in\mathbb R^{|V|}$ donde $v_i$ es la frecuencia de la palabra $i$.
* **TF-IDF**: $v_i = \text{tf}_i\cdot\log\frac{N}{\text{df}_i}$ realza términos raros e informativos.
* **Embeddings** (word2vec/FastText): mapa $E\in\mathbb R^{|V|\times d}$; el correo se representa como promedio o concatenación de embeddings → entrada densa de bajo rango.

### 1.4.2 Vectorización numérica

* En un **MLP**: `Matrix X (B×|V|)` × `Matrix Wᵀ (hidden×|V|)` → `Z`.
* Con **Eigen** o **OpenBLAS** (`cblas_sgemm`) esta multiplicación usa SIMD y multihilo; en GPU, cuBLAS/cuDNN.
* Las CNN 1-D convierten la convolución en *im2col* + GEMM, reutilizando la misma infraestructura.

---

## 1.5 Pasos para el proyecto de spam

1. **Carga y preprocesamiento** (`TextLoader`)

  * Tokenización + stop-words + minúsculas.
  * Construcción de vocabulario con **min\_freq≥5**.
2. **Vectorización**

  * Empezar con BoW; añadir TF-IDF como mejora.
3. **Modelo base**

  * `Dense(|V|→128) → ReLU → Dropout(0.3) → Dense(128→1) → Sigmoid`.
4. **Entrenamiento**

  * `Batch=64`, `Adam, η=1e-3`, 10-20 épocas, pérdida BCE.
5. **Métricas**

  * Accuracy, *precision*, *recall* y **F1** (más relevante en clases desequilibradas).
6. **Persistencia**

  * Serializar pesos (`.bin`) + vocabulario (`.json`) para inferencia.
7. **Escalabilidad**

  * Para >100 k correos, compilar con OpenBLAS multihilo o exportar a ONNX y usar GPU.

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

## Referencias en español (APA 7)

* Haykin, S. (1994). *Redes neuronales: Un enfoque integral*. Prentice Hall.
* Hebb, D. O. (1949). *La organización del comportamiento*. Wiley.
* Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735-1780. [https://doi.org/10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
* Kingma, D. P., & Ba, J. L. (2015). Adam: A method for stochastic optimization. *International Conference on Learning Representations*. [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
* LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE, 86*(11), 2278-2324. [https://doi.org/10.1109/5.726791](https://doi.org/10.1109/5.726791)
* McCulloch, W. S., & Pitts, W. (1943). A logical calculus of the ideas immanent in nervous activity. *The Bulletin of Mathematical Biophysics, 5*(4), 115-133. [https://doi.org/10.1007/BF02478259](https://doi.org/10.1007/BF02478259)
* Minsky, M., & Papert, S. (1969). *Perceptrones: Una introducción a la geometría computacional*. MIT Press.
* Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature, 323*(6088), 533-536. [https://doi.org/10.1038/323533a0](https://doi.org/10.1038/323533a0)
* Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain. *Psychological Review, 65*(6), 386-408. [https://doi.org/10.1037/h0042519](https://doi.org/10.1037/h0042519)
* Tieleman, T., & Hinton, G. (2012). Lecture 6.5—RMSProp: Divide the gradient by a running average of its recent magnitude. *COURSERA: Neural Networks for Machine Learning*. University of Toronto.
* Jurafsky, D., & Martin, J. H. (2023). *Procesamiento del lenguaje natural y comprensión de voz* (borrador 3.ª ed.). Stanford.
* Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. *arXiv*. [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781)



### Licencia

Este proyecto usa la licencia **MIT**. Ver [LICENSE](LICENSE) para detalles.

---
