[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Lj3YlzJp)
# Proyecto Final 2025-1: AI Neural Network
## **CS2013 Programación III** · Informe Final

### **Descripción**

> Implementación de una red neuronal multicapa en C++ para la detección automática de mensajes SMS spam, usando técnicas modernas de preprocesamiento y entrenamiento supervisado. El objetivo es explorar y demostrar la eficacia de arquitecturas simples y optimizadores modernos en la clasificación binaria de texto.

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
```
      ALUMNO                        CODIGO       ROL
    José Daniel Huamán Rosales     209900001   (Responsable de investigación teórica)
    Juan Carlos Ticlia Malqui      202410584   (Desarrollo de la arquitectura)
    Paulo Isael Miranda Barrietos  202410580   (Implementación del modelo, documentación y demo)
    Elmer José Villegas Suarez     202410032   (Pruebas y benchmarking)
```
---

### Requisitos e instalación

1. **Compilador**: GCC 11 o superior (C++20)
2. **Dependencias**:

    * CMake 3.20+
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
| *(Mención)* **CNN 1-D**                      | Convolución en la dimensión de la secuencia; detecta *n-gramas* locales con pesos compartidos.  | Frases como "free \$\$\$" o "click aquí" se detectan como patrones locales independientemente de su posición.                                     |
| *(Mención)* **RNN / LSTM / GRU**             | Mantienen un estado oculto que se actualiza token a token.                                      | Útil si queremos modelar la **dependencia de orden** entre palabras en vez de tratarlas como bolsa.                                               |
| *(Mención)* **Transformer/BERT** | Auto-atención bidireccional; se pre-entrena con corpus masivo.                                  | En producción podríamos exportar a ONNX y usar `onnx-runtime`, pero supera el alcance de un laboratorio C++ puro.                                 |

---

## 1.3 Algoritmos de entrenamiento

1. **Retropropagación**: aplica la regla de la cadena para computar $\partial\mathcal L/\partial\theta$ capa por capa desde la salida hasta la entrada.
2. **Descenso de gradiente estocástico (SGD)** con *mini-batch* ajusta los pesos $\theta \leftarrow \theta - \eta\nabla\mathcal L$.
3. **Optimizadores implementados en el proyecto**:

* **SGD** (implementado): Optimizador básico con tasa de aprendizaje fija; útil para problemas simples y como base de comparación.
* **Adam** (implementado): Combina momento y adaptación de tasa de aprendizaje; converge rápido y es el estándar en proyectos de spam académicos.

4. **Optimizadores mencionados para referencia teórica**:

* **AdaGrad** (2011) adapta la tasa de aprendizaje por parámetro; útil si el vector de entrada es muy disperso como en BoW.
* **RMSProp** (2012) suaviza AdaGrad con una media exponencial de gradientes al cuadrado.

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
* *(Mención)* Las CNN 1-D convierten la convolución en *im2col* + GEMM, reutilizando la misma infraestructura.

---

## 1.5 Pasos para el proyecto de spam

1. **Carga y preprocesamiento** (`TextLoader`)

* Tokenización + stop-words + minúsculas.
* Construcción de vocabulario a partir de todas las palabras presentes en el dataset.
2. **Vectorización**

* Empezar con BoW; añadir TF-IDF como mejora.
3. **Modelo base**

* `Dense(|V|→128) → ReLU → Dropout(0.3) → Dense(128→1) → Sigmoid`.
4. **Entrenamiento**

* `Batch=8`, `Adam, η=0.01`, 20 épocas, pérdida BCE (configuración actual del proyecto).
5. **Métricas**

* Accuracy, *precision*, *recall* y **F1** (más relevante en clases desequilibradas).
6. **Persistencia**

* Serializar pesos (models/model.txt) + vocabulario (models/vocabulary.txt) para inferencia.
7. **Escalabilidad**

* Para >100 k correos, compilar con OpenBLAS multihilo o exportar a ONNX y usar GPU.

---
### 2. Diseño e implementación

#### 2.1 Arquitectura de la solución

##### 2.1.1 Patrones de diseño implementados

La arquitectura de la red neuronal desarrollada integra varios patrones de diseño fundamentales, los cuales aportan flexibilidad, extensibilidad y mantenibilidad al sistema. A continuación, se describen los principales patrones empleados, su propósito general y la forma en que se materializan en la implementación:

**a) Strategy Pattern (Patrón Estrategia)**

Este patrón define una familia de algoritmos intercambiables, permitiendo seleccionar la estrategia más adecuada en tiempo de ejecución sin modificar el contexto en el que se utilizan. Su implementación se refleja principalmente en `nn_interfaces.h`, donde se abstraen comportamientos clave del entrenamiento:

* **Optimizadores**: A través de la interfaz `IOptimizer<T>`, que define un contrato común para algoritmos como SGD y Adam. Gracias a esta interfaz, es posible intercambiar optimizadores dinámicamente, favoreciendo la experimentación con distintas técnicas de actualización de parámetros.

* **Funciones de pérdida**: La interfaz `ILoss<T,DIMS>` encapsula distintas métricas de error (MSELoss, BCELoss), lo que permite sustituir la función objetivo sin modificar la lógica general del entrenamiento.

* **Capas de activación**: Las clases ReLU y Sigmoid implementan la interfaz `ILayer<T>`, permitiendo utilizar diferentes funciones de activación en la misma arquitectura, con la posibilidad de combinarlas de forma flexible.

**b) Template Method Pattern (Patrón Método Plantilla)**

El patrón Template Method establece el esqueleto de un algoritmo en un método base, delegando a subclases o parámetros la definición de ciertos pasos específicos, permitiendo así personalizar partes del comportamiento sin modificar la estructura general. En este caso, se encuentra implementado en el método `train()` de la clase `NeuralNetwork`, que define el flujo completo del proceso de entrenamiento:

* Validación de parámetros y de la estructura de la red.
* Inicialización del optimizador.
* Iteración a lo largo de las épocas.
* Procesamiento por lotes, incluyendo la propagación hacia adelante, cálculo de la pérdida y retropropagación.
* Actualización de parámetros mediante el optimizador.
* Cálculo de métricas y registro de resultados.

La especificidad de las funciones de pérdida y del optimizador se inyecta mediante parámetros de plantilla, lo que permite modificar el comportamiento del entrenamiento con alta flexibilidad, manteniendo la estructura global del algoritmo intacta.

**c) Factory Pattern (Patrón Fábrica)**

El patrón Factory tiene como objetivo delegar la creación de objetos a clases especializadas, evitando así acoplar el código cliente a implementaciones concretas. En este proyecto, se encuentra implementado en el archivo `app_manager.cpp` mediante la función `build_model()`, que encapsula la lógica de construcción e inicialización de los componentes principales:

* **Construcción de capas**: La función `build_model()` crea instancias de capas Dense con inicialización personalizada de pesos y bias, utilizando lambdas para especificar valores iniciales (0.01f para pesos, 0.0f para bias).

* **Configuración de arquitectura**: Establece una arquitectura específica (input_size → 16 → 1) con ReLU como activación intermedia y Sigmoid como activación final, optimizada para clasificación binaria de spam.

* **Inicialización de pesos:** En este proyecto, los pesos de las capas densas se inicializan a 0.01 y los bias a 0.0. Esta inicialización simple permite un arranque estable del entrenamiento, aunque en proyectos más grandes se recomienda el uso de técnicas como Xavier o He para redes profundas.

##### 2.1.2 Estructura de carpetas

Para garantizar la organización, la escalabilidad y la mantenibilidad del proyecto, se ha adoptado una estructura de carpetas modular que facilita la separación lógica de responsabilidades y la integración de los diferentes componentes del detector de spam. A continuación, se detalla la jerarquía de directorios y archivos que conforman el proyecto SMS Spam Detector, donde se distribuyen los módulos de procesamiento de datos de texto, definición de la arquitectura neuronal para clasificación binaria, funciones de pérdida, optimizadores, pruebas unitarias, documentación y otros recursos necesarios para el correcto funcionamiento y la extensibilidad del sistema:

```
projecto_final_nullptr/
├── data/                         # Dataset CSV
│   ├── stopwords_eng.csv
│   ├── stopwords_test.csv
│   ├── training_words_eng.csv
│   └── training_words_esp.csv
├── include/                      # Cabeceras principales
│   └── utec/
│       ├── algebra/              # Álgebra: Tensor<T, Rank>
│       │   └── tensor.h
│       ├── app/                  # Controlador general
│       │   └── app_manager.h
│       ├── data/                 # Carga y preprocesamiento de datos
│       │   ├── dataset_utils.h
│       │   └── test_loader.h
│       └── nn/                   # Red neuronal: capas y arquitectura
│           ├── neural_network.h
│           ├── nn_activation.h
│           ├── nn_dense.h
│           ├── nn_interfaces.h
│           ├── nn_loss.h
│           └── nn_optimizer.h
├── models/                       # Modelos y vocabularios guardados
│   ├── model.txt
│   └── vocabulary.txt
├── src/                          # Implementación fuente
│   └── utec/
│       ├── app/
│       │   └── app_manager.cpp
│       └── data/
│           ├── dataset_utils.cpp
│           └── text_loader.cpp
├── tests/                        # Tests
│   ├── catch_main.cpp
│   ├── test_app_manager_load_ai.cpp
│   ├── test_convergence_adam.cpp
│   ├── test_neural_network.cpp
│   ├── test_stopwords.cpp
│   ├── test_tensor.cpp
│   └── test_text_loader.cpp
└── tools/                        # catch2
│   └── catch/
│       └── catch.hpp.cpp
├── .gitignore
├── CMakeLists.txt
├── LICENSE.md
├── README.md
└── main.cpp            # Punto de entrada del programa
```

#### 2.2 Manual de uso y casos de prueba

##### 2.2.1 Cómo ejecutar

El sistema de red neuronal se puede ejecutar de múltiples maneras según las necesidades del usuario. Los ejecutables se encuentran en la carpeta build/ después de la compilación, y también pueden ejecutarse directamente desde IDEs como CLion usando las configuraciones de ejecución disponibles.

**Experimentos principales**

```bash
# Ejecutar el sistema principal de experimentos
./build/main_app
```

Este comando lanza un sistema interactivo que permite:

* **Entrenar IA**: Ejecuta `train_model()` que carga datos SMS con `TextLoader::load_data()`, construye el modelo del detector con `build_model()`, entrena con `model.train<BCELoss, Adam>()` y guarda el modelo con `save_model()` y `save_vocabulary()`

* **Probar IA**: Ejecuta `test_model()` que evalúa el detector de spam en el conjunto de prueba usando `DatasetUtils::split_dataset()`, `vector_to_tensor()`, `labels_to_tensor()` y `model.predict()`

* **Predecir mensaje**: Ejecuta `predict_message()` que permite ingresar un mensaje SMS, lo vectoriza con `loader.vectorize()`, hace predicción con `model.predict()` y clasifica como spam/ham

* **Cargar IA entrenada**: Ejecuta `load_trained_model()` que carga vocabulario y modelo del detector en paralelo usando threads, restaurando el estado completo del sistema

**Ejecución de tests**

```bash
# Ejecutar todos los tests unitarios en secuencia
./build/run_all_tests

# Ejecutar tests específicos para validación individual
./build/test_neural_network      # Valida funcionamiento de redes neuronales
./build/test_tensor              # Verifica operaciones de tensores
./build/test_text_loader         # Prueba carga de datos
```

##### 2.2.2 Casos de prueba detallados

El sistema incluye una suite completa de tests unitarios que valida el correcto funcionamiento de todos los componentes críticos de la red neuronal:

**a) Test unitario de red neuronal (test_neural_network)**

* **Ejecutable**: `./build/test_neural_network`
* **Propósito**: Valida el funcionamiento completo de las redes neuronales multicapa para clasificación de spam
* **Casos específicos cubiertos**:
  * **Problema XOR**: Prueba la capacidad de la red para resolver problemas no linealmente separables (simulando la complejidad de clasificación spam/ham)
  * **Arquitectura**: 2→4→1 con Sigmoid en ambas capas (similar a la arquitectura del detector de spam)
  * **Entrenamiento**: 4000 épocas con BCELoss y tasa de aprendizaje 0.08
  * **Validación**: Verifica que las predicciones sean correctas para todos los casos XOR

**b) Test de operaciones de tensores (test_tensor)**

* **Ejecutable**: `./build/test_tensor`
* **Propósito**: Verifica la implementación correcta de las operaciones de álgebra lineal
* **Casos específicos cubiertos**:
  * **Creación y forma**: Verificación de dimensiones y tamaño de tensores
  * **Asignación y acceso**: Lectura y escritura de elementos individuales
  * **Operaciones aritméticas**: Suma de tensores y multiplicación por escalar
  * **Redimensionamiento**: Cambio de forma de tensores con `reshape()`
  * **Producto matricial**: Multiplicación de matrices con `matrix_product()`

**c) Test de carga de datos (test_text_loader)**

* **Ejecutable**: `./build/test_text_loader`
* **Propósito**: Valida el procesamiento y vectorización de datos de texto para el detector de spam
* **Casos específicos cubiertos**:
  * **Carga de datasets SMS**: Verificación de lectura correcta de archivos CSV con mensajes spam/ham
  * **Construcción de vocabulario**: Creación de mapeo palabra-índice con `build_vocabulary()` para palabras frecuentes en spam
  * **Tokenización**: Procesamiento de texto con `tokenize()` (normalización, minúsculas, eliminación de puntuación)
  * **Vectorización BoW**: Conversión de mensajes SMS a vectores numéricos con `vectorize()`
  * **Persistencia**: Guardado y carga de vocabularios con `save_vocabulary()` y `load_vocabulary()`

**d) Test de carga de IA pre-entrenada (test_app_manager_load_ai)**

* **Ejecutable**: `./build/test_app_manager_load_ai`
* **Propósito**: Verifica la funcionalidad de persistencia y carga de modelos del detector de spam
* **Casos específicos cubiertos**:
  * **Carga de vocabulario**: Restauración del mapeo palabra-índice desde archivo para procesar nuevos mensajes
  * **Carga de modelo**: Restauración de pesos y arquitectura con `load_model()` del detector entrenado
  * **Inferencia post-carga**: Validación de predicciones consistentes con `predict()` para clasificación spam/ham
  * **Integración completa**: Verificación del flujo completo de carga y predicción de mensajes SMS

**e) Test de convergencia con Adam (test_convergence_adam)**

* **Ejecutable**: `./build/test_convergence_adam`
* **Propósito**: Verifica que el optimizador Adam logra convergencia en una tarea lógica simple (OR).
* **Casos específicos cubiertos**:
  * Entrenamiento de una red neuronal pequeña con Adam en el problema OR.
  * Validación de que la pérdida final es baja (convergencia).

**Criterios de éxito**

Un sistema funcionando correctamente debe cumplir:

* **Tasa de éxito**: 100% en todos los tests unitarios
* **Convergencia**: Reducción demostrable de la función de pérdida
* **Precisión**: Mejora significativa en métricas de evaluación
* **Estabilidad**: Ausencia de errores numéricos o desbordamientos
* **Rendimiento**: Tiempos de ejecución razonables

**Estructura de archivos (entrada/salida)**

* **Archivos de entrada esperados**:
  * Entrenamiento: `data/training_words_eng.csv` (recomendado) o `data/training_words_esp.csv` (base de datos débil)
  * Formato CSV: `label,message` donde label es "spam" o "ham" y message contiene el texto del SMS
* **Archivos de salida generados**:
  * Modelos: `models/model.txt` (pesos y arquitectura del detector de spam)
  * Vocabularios: `models/vocabulary.txt` (mapeo palabra-índice para vectorización de mensajes SMS)

**Ejemplo de flujo de trabajo**

```bash
# 1. Compilar el proyecto
mkdir build && cd build
cmake ..
make

# 2. Verificar funcionamiento del sistema
./run_all_tests

# 3. Ejecutar aplicación principal
./main_app

# 4. Analizar resultados
# Los archivos de modelo contienen parámetros entrenados
```

**Interpretación de resultados**

Los tests proporcionan información detallada sobre:

* **Estado de componentes**: Cada test indica si los componentes del detector de spam funcionan correctamente
* **Rendimiento**: Tiempos de ejecución y eficiencia computacional para procesamiento de mensajes SMS
* **Calidad de convergencia**: Métricas de pérdida y precisión en la clasificación spam/ham
* **Robustez**: Capacidad del sistema para manejar diferentes tipos de mensajes SMS y arquitecturas

Una ejecución exitosa de todos los tests garantiza que el detector de spam está listo para procesar mensajes reales y datasets de SMS.

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
    - Seleccionar opción 1 para entrenar IA
    - Seleccionar opción 2 para probar IA
    - Seleccionar opción 3 para predecir mensaje
    - Seleccionar opción 4 para cargar IA entrenada
    - Seleccionar opción 0 para salir

**Dataset utilizado:**
- `data/training_words_esp.csv`: Dataset en español con 2002 mensajes
- `data/training_words_eng.csv`: Dataset en inglés con 5574 mensajes (recomendado)

**Configuración del modelo:**
- Arquitectura: MLP con 2 capas densas
- Activaciones: ReLU + Sigmoid
- Optimizador: Adam con learning rate 0.01
- Batch size: 8
- Épocas: 20

---

### 4. Análisis del rendimiento

#### Métricas del modelo implementado

- **Arquitectura:** MLP con capa oculta de 16 neuronas
- **Dataset:** 2002 mensajes en español y 5574 mensajes en inglés (80% entrenamiento, 20% prueba)
- **Funciones de activación:** ReLU (capa oculta) + Sigmoid (salida)
- **Función de pérdida:** Binary Cross-Entropy (BCELoss)
- **Optimizadores probados:** SGD y Adam
- **Batch size:** 8
- **Épocas:** 20
- **Precisión esperada:** >95% (en la mayoría de casos superior al 99%)
- **Tiempo de entrenamiento:** ver tabla comparativa abajo

#### Resultados comparativos de entrenamiento

| Optimizador | Stopwords | Tiempo de entrenamiento | Precisión (dataset)         |
|-------------|-----------|------------------------|-----------------------------|
| SGD         | No        | 20.48 min              | >95% (usualmente >99%)      |
| SGD         | Sí        | 14.34 min              | >95% (usualmente >99%)      |
| Adam        | No        | 48.74 min              | >95% (usualmente >99%)      |
| Adam        | Sí        | 34.12 min              | >95% (usualmente >99%)      |

**Justificación de Adam y BCELoss:**
Se utiliza el optimizador Adam [4] por su eficiencia y robustez en tareas de clasificación, ya que ajusta la tasa de aprendizaje de manera adaptativa y acelera la convergencia en comparación con SGD tradicional. Para la función de pérdida, se emplea Binary Cross-Entropy (BCELoss) [4], que es la opción estándar para problemas de clasificación binaria, ya que penaliza fuertemente las predicciones incorrectas y modela la probabilidad de pertenencia a la clase spam de manera adecuada.

#### Ventajas del enfoque

- ✅ Implementación desde cero sin dependencias externas
- ✅ Código modular y extensible
- ✅ Interfaz de usuario intuitiva
- ✅ Soporte para diferentes idiomas
- ✅ Preprocesamiento con eliminación de stopwords para mejorar la calidad de los datos

#### Limitaciones actuales

- ❌ Sin paralelización, rendimiento limitado en datasets grandes
- ❌ Vocabulario fijo, no adaptativo
- ❌ Sin optimizadores avanzados adicionales (RMSProp, AdaGrad)

#### Mejoras futuras propuestas

- Implementar optimizadores adaptativos adicionales (RMSProp, AdaGrad)
- Añadir soporte para embeddings de palabras
- Implementar vectorización TF-IDF
- Implementar vectorización basada en n-gramas
- Paralelizar entrenamiento con OpenMP
- Implementar early stopping y regularización

---

### 5. Trabajo en equipo

| Tarea                     | Miembro                      | Rol                                    |
| ------------------------- | ---------------------------- | -------------------------------------- |
| Investigación teórica     | José Daniel Huamán Rosales   | Documentar fundamentos de redes neuronales |
| Diseño de la arquitectura | Juan Carlos Ticlia Malqui    | Diseño de clases y patrones de diseño |
| Implementación del modelo | Paulo Isael Miranda Barrietos | Código C++ de la red neuronal, demo y documentación   |
| Pruebas y benchmarking    | Elmer José Villegas Suarez   | Validación y métricas de rendimiento  |

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
* Implementar optimizadores avanzados adicionales (RMSProp, AdaGrad)
* Añadir soporte para embeddings y procesamiento de lenguaje natural
* Implementar vectorización TF-IDF
* Implementar vectorización basada en n-gramas
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

### Recursos de datos

- Stopwords en inglés: [NLTK's list of english stopwords (sebleier)](https://gist.github.com/sebleier/554280)
- Dataset SMS Spam (inglés): [Kaggle - SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

### Licencia

Este proyecto usa la licencia **MIT**. Ver [LICENSE](LICENSE) para detalles.