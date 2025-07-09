//
// Created by paulo on 5/07/2025.
//

#include "utec/app/app_manager.h"

#include "utec/data/text_loader.h"
#include "utec/data/dataset_utils.h"
#include "utec/nn/neural_network.h"
#include "utec/nn/nn_dense.h"
#include "utec/nn/nn_activation.h"
#include "utec/nn/nn_loss.h"
#include "utec/algebra/tensor.h"
#include <iostream>
#include <iomanip>
#include <memory>
#include <thread>
#include <chrono>

using namespace std;
using namespace utec::app;
using namespace utec::data;
using namespace utec::neural_network;
using namespace utec::algebra;

namespace {
    // Crear variables globales y privadas a nivel de archivo

    NeuralNetwork<float> model;

    // dos opciones de lenguaje (no recomendado cambiar)
    // - training_words_esp.csv (database debil)
    // - training_words_eng.csv (preferible)
    TextLoader loader("../data/training_words_eng.csv");


    size_t input_size = 0;
    bool model_trained = false;

    void build_model() {
        model = NeuralNetwork<float>(); // reset del modelo
        model.add_layer(make_unique<Dense<float>>(input_size, 16,
            [](utec::algebra::Tensor<float, 2>& W) { W.fill(0.01f); },  // pesos
            [](utec::algebra::Tensor<float, 2>& b) { b.fill(0.0f); })); // bias
        model.add_layer(make_unique<ReLU<float>>());

        model.add_layer(make_unique<Dense<float>>(16, 1,
            [](utec::algebra::Tensor<float, 2>& W) { W.fill(0.01f); },  // pesos
            [](utec::algebra::Tensor<float, 2>& b) { b.fill(0.0f); })); // bias
        model.add_layer(make_unique<Sigmoid<float>>());
    }
}


void AppManager::show_menu() {
    int option = -1;

    while (option) { // option != 0
        cout << "\n=== SMS Spam Detector ===" << endl;
        cout << "1. Entrenar IA" << endl;
        cout << "2. Probar IA" << endl;
        cout << "3. Predecir mensaje" << endl;
        cout << "4. Cargar IA entrenada" << endl;
        cout << "0. Salir" << endl;
        cout << "Seleccione una opcion: ";
        cin >> option;
        cin.ignore();

        switch (option) {
            case 1: train_model(); break;
            case 2: test_model(); break;
            case 3: predict_message(); break;
            case 4: load_trained_model(); break;
            case 0: cout << "Saliendo..." << endl; break;
            default: cout << "Opcion invalida" << endl; break;
        }
    }
}


void AppManager::train_model() {
    cout << "\nCargando datos y entrenando IA..." << endl;

    loader.load_stopwords("../data/stopwords_eng.txt");
    loader.load_data();
    input_size = loader.get_vocabulary_size();

    vector<TextExample> train_set, test_set;
    DatasetUtils::split_dataset(loader.get_dataset(), train_set, test_set);

    auto X_train = DatasetUtils::vector_to_tensor(train_set);
    auto Y_train = DatasetUtils::labels_to_tensor(train_set);

    build_model();

    // Cronometrar entrenamiento
    auto start = chrono::high_resolution_clock::now();

    model.train<BCELoss, Adam>(X_train, Y_train, 20, 8, 0.01f);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, std::ratio<60>> elapsed_minutes = end - start;


    model_trained = true;

    cout << fixed << setprecision(2);
    cout << "Entrenamiendo completado en " << elapsed_minutes.count() << " minutos." << endl;

    loader.save_vocabulary("../models/vocabulary.txt");
    model.save_model("../models/model.txt");
}


void AppManager::test_model() {
    if (!model_trained) {
        cout << "Primero debe entrenar la IA." << endl;
        return;
    }

    cout << "\nEvaluando modelo..." << endl;

    vector<TextExample> train_set, test_set;
    DatasetUtils::split_dataset(loader.get_dataset(), train_set, test_set);

    auto X_test = DatasetUtils::vector_to_tensor(test_set);
    auto Y_test = DatasetUtils::labels_to_tensor(test_set);

    auto Y_pred = model.predict(X_test);

    int correct = 0;
    int total = Y_test.shape()[0];

    for (int i = 0; i < total; ++i) {
        float predicted = Y_pred(i, 0) >= 0.5f ? 1.0f : 0.0f;
        float actual = Y_test(i, 0);
        if (predicted == actual) ++correct;
    }

    float accuracy = static_cast<float>(correct) / total * 100.0f;
    cout << fixed << setprecision(2);
    cout << "Precision de la IA en el conjunto de datos: " << accuracy << "%" << "\n";
}



void AppManager::predict_message() {
    if (!model_trained) {
        cout << "Primero debe entrenar la IA." << endl;
        return;
    }

    cout << "\nIngrese el mensaje a evaluar: ";
    string message;
    cin.ignore();
    getline(cin, message);

    auto vectorized = loader.vectorize(message);
    utec::algebra::Tensor<float, 2> input(2, input_size);
    for (size_t i = 0; i < input_size; ++i)
        input(0, i) = vectorized[i];

    auto prediction = model.predict(input);

    if (prediction(0, 0) >= 0.5f)
        cout << "El mensaje es SPAM." << endl;
    else
        cout << "El mensaje NO es SPAM" << endl;
}

void AppManager::load_trained_model() {
    cout << "\nCargando vocabulario y modelo entrenado..." << endl;

    // Hilos en paralelo: cargar vocabulario y cargar modelo pre-entrenado
    thread vocab_thread([&]() {
        loader.load_stopwords("../data/stopwords_eng.txt");
        loader.load_vocabulary("../models/vocabulary.txt");
        input_size = loader.get_vocabulary_size();
        loader.load_data();
    });

    thread model_thread([&]() {
        build_model();
        model.load_model("../models/model.txt");
    });

    // join de hilos
    vocab_thread.join();
    model_thread.join();

    model_trained = true;

    cout << " [ Modelo listo para ser usado ]" << endl;
}
