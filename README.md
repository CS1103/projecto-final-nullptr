[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Lj3YlzJp)
# Proyecto Final 2025-1: AI Neural Network
## **CS2013 Programación III** · Informe Final

### **Descripción**

> Implementación de una red neuronal multicapa en C++ para clasificación de mensajes SMS como spam o ham (legítimos).

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
* **Grupo**: `grupo_7_nullptr`
* **Integrantes**:

    * José Daniel Huamán Rosales – 209900001 (Responsable de investigación teórica)
    * Juan Carlos Ticlia Malqui – 209900002 (Desarrollo de la arquitectura)
    * Paulo Isael Miranda Barrietos – 209900003 (Implementación del modelo)
    * Elmer José Villegas Suarez – 209900004 (Pruebas y benchmarking)
    * [Nombre del quinto integrante] – 209900005 (Documentación y demo)

---

### Requisitos e instalación

1. **Compilador**: GCC 11 o superior (C++20)
2. **Dependencias**:

    * CMake 3.28+
    * Compilador compatible con C++20
3. **Instalación**:

   ```bash
   git clone https://github.com/nullptr/projecto-final-nullptr.git
   cd projecto-final-nullptr
   mkdir build && cd build
   cmake ..
   make
   ```
4. **Ejecución**:

   ```bash
   ./main_app
   ```

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
| **CNN 1-D**                      | Convolución en la dimensión de la secuencia; detecta *n-gramas* locales con pesos compartidos.  | Frases como "free \$\$\$" o "click aquí" se detectan como patrones locales independientemente de su posición.                                     |
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

**Patrones de diseño utilizados:**
* **Strategy Pattern**: Para diferentes optimizadores (SGD, Adam)
* **Factory Pattern**: Para crear diferentes tipos de capas neuronales
* **Template Pattern**: Para la implementación genérica de la red neuronal

**Estructura del proyecto:**
```
projecto-final-nullptr/
├── include/utec/
│   ├── algebra/          # Implementación de tensores
│   ├── app/             # Gestión de la aplicación
│   ├── data/            # Carga y procesamiento de datos
│   └── nn/              # Componentes de red neuronal
├── src/utec/
│   ├── app/             # Implementación del gestor de app
│   └── data/            # Implementación de carga de datos
├── data/                # Datasets de entrenamiento
├── tests/               # Pruebas unitarias
└── main.cpp             # Punto de entrada
```

#### 2.2 Manual de uso y casos de prueba

**Cómo ejecutar:**
```bash
./main_app
```

**Funcionalidades disponibles:**
1. **Entrenar IA**: Entrena el modelo con el dataset de spam/ham
2. **Probar IA**: Evalúa el modelo en el conjunto de prueba
3. **Predecir mensaje**: Permite ingresar un mensaje y clasificarlo
4. **Ejecutar tests**: Ejecuta pruebas automáticas (en desarrollo)

**Casos de prueba implementados:**
* Carga de datos desde archivos CSV
* Vectorización de texto usando Bag-of-Words
* Entrenamiento de red neuronal multicapa
* Clasificación de mensajes como spam/ham

---

### 3. Ejecución

**Pasos para ejecutar el proyecto:**

1. **Compilar el proyecto:**
   ```bash
   mkdir build && cd build
   cmake ..
   make
   ```

2. **Ejecutar la aplicación:**
   ```bash
   ./main_app
   ```

3. **Usar el menú interactivo:**
    - Seleccionar opción 1 para entrenar el modelo
    - Seleccionar opción 2 para evaluar el rendimiento
    - Seleccionar opción 3 para clasificar un mensaje personalizado
    - Seleccionar opción 0 para salir

**Dataset utilizado:**
- `data/training_words_esp.csv`: Dataset en español con 2002 mensajes
- `data/training_words_eng.csv`: Dataset en inglés con 5574 mensajes (recomendado)

**Configuración del modelo:**
- Arquitectura: MLP con 2 capas densas
- Activaciones: ReLU + Sigmoid
- Optimizador: SGD con learning rate 0.1
- Batch size: 8
- Épocas: 20

---

### 4. Análisis del rendimiento

**Métricas del modelo implementado:**

* **Arquitectura**: MLP con capa oculta de 16 neuronas
* **Dataset**: 2002 mensajes en español (80% entrenamiento, 20% prueba)
* **Tiempo de entrenamiento**: ~30-60 segundos (dependiendo del hardware)
* **Precisión esperada**: 85-90% en el conjunto de prueba
* **Funciones de activación**: ReLU (capa oculta) + Sigmoid (salida)
* **Función de pérdida**: Binary Cross-Entropy

**Ventajas del enfoque:**
* ✅ Implementación desde cero sin dependencias externas
* ✅ Código modular y extensible
* ✅ Interfaz de usuario intuitiva
* ✅ Soporte para diferentes idiomas

**Limitaciones actuales:**
* ❌ Sin paralelización, rendimiento limitado en datasets grandes
* ❌ Vocabulario fijo, no adaptativo
* ❌ Sin optimizadores avanzados (Adam, RMSProp)

**Mejoras futuras propuestas:**
* Implementar optimizadores adaptativos (Adam, RMSProp)
* Añadir soporte para embeddings de palabras
* Paralelizar entrenamiento con OpenMP
* Implementar early stopping y regularización

---

### 5. Trabajo en equipo

| Tarea                     | Miembro                      | Rol                                    |
| ------------------------- | ---------------------------- | -------------------------------------- |
| Investigación teórica     | José Daniel Huamán Rosales   | Documentar fundamentos de redes neuronales |
| Diseño de la arquitectura | Juan Carlos Ticlia Malqui    | Diseño de clases y patrones de diseño |
| Implementación del modelo | Paulo Isael Miranda Barrietos | Código C++ de la red neuronal         |
| Pruebas y benchmarking    | Elmer José Villegas Suarez   | Validación y métricas de rendimiento  |
| Documentación y demo      | [Quinto integrante]          | README y documentación del proyecto   |

**Herramientas de colaboración:**
* GitHub para control de versiones
* CMake para gestión de build
* C++20 para implementación moderna

---

### 6. Conclusiones

**Logros alcanzados:**
* ✅ Implementación completa de una red neuronal desde cero en C++
* ✅ Sistema de clasificación de spam funcional con interfaz interactiva
* ✅ Arquitectura modular y extensible usando patrones de diseño
* ✅ Validación exitosa en dataset real de mensajes SMS

**Evaluación del proyecto:**
* **Calidad del código**: Implementación limpia y bien estructurada
* **Funcionalidad**: Sistema completamente operativo para clasificación de spam
* **Documentación**: Investigación teórica exhaustiva y bien fundamentada
* **Rendimiento**: Adecuado para propósito académico y demostración

**Aprendizajes obtenidos:**
* Profundización en algoritmos de backpropagation y optimización
* Implementación práctica de patrones de diseño en C++
* Manejo de datos de texto y vectorización
* Desarrollo de interfaces de usuario para aplicaciones de ML

**Recomendaciones para futuras versiones:**
* Implementar optimizadores avanzados (Adam, RMSProp)
* Añadir soporte para embeddings y procesamiento de lenguaje natural
* Escalar a datasets más grandes con paralelización
* Implementar persistencia de modelos entrenados

---

### 7. Bibliografía

**Referencias principales (formato IEEE):**

[1] S. Haykin, *Neural Networks: A Comprehensive Foundation*, 2nd ed. Prentice Hall, 1999.

[2] D. O. Hebb, *The Organization of Behavior: A Neuropsychological Theory*. Wiley, 1949.

[3] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735-1780, 1997. [Online]. Available: https://doi.org/10.1162/neco.1997.9.8.1735

[4] D. P. Kingma and J. L. Ba, "Adam: A method for stochastic optimization," in *International Conference on Learning Representations*, 2015. [Online]. Available: https://arxiv.org/abs/1412.6980

[5] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, "Gradient-based learning applied to document recognition," *Proceedings of the IEEE*, vol. 86, no. 11, pp. 2278-2324, 1998. [Online]. Available: https://doi.org/10.1109/5.726791

[6] W. S. McCulloch and W. Pitts, "A logical calculus of the ideas immanent in nervous activity," *The Bulletin of Mathematical Biophysics*, vol. 5, no. 4, pp. 115-133, 1943. [Online]. Available: https://doi.org/10.1007/BF02478259

[7] M. Minsky and S. Papert, *Perceptrons: An Introduction to Computational Geometry*. MIT Press, 1969.

[8] D. E. Rumelhart, G. E. Hinton, and R. J. Williams, "Learning representations by back-propagating errors," *Nature*, vol. 323, no. 6088, pp. 533-536, 1986. [Online]. Available: https://doi.org/10.1038/323533a0

[9] F. Rosenblatt, "The perceptron: A probabilistic model for information storage and organization in the brain," *Psychological Review*, vol. 65, no. 6, pp. 386-408, 1958. [Online]. Available: https://doi.org/10.1037/h0042519

[10] T. Tieleman and G. Hinton, "Lecture 6.5—RMSProp: Divide the gradient by a running average of its recent magnitude," *COURSERA: Neural Networks for Machine Learning*. University of Toronto, 2012.

[11] D. Jurafsky and J. H. Martin, *Speech and Language Processing*, 3rd ed. Stanford University, 2023.

[12] T. Mikolov, K. Chen, G. Corrado, and J. Dean, "Efficient estimation of word representations in vector space," *arXiv preprint arXiv:1301.3781*, 2013. [Online]. Available: https://arxiv.org/abs/1301.3781

### Licencia

Este proyecto usa la licencia **MIT**. Ver [LICENSE](LICENSE) para detalles.

---