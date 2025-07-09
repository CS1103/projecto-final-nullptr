#include "catch.hpp"
#include "utec/app/app_manager.h"
#include "utec/data/text_loader.h"
#include "utec/nn/neural_network.h"
#include "utec/nn/nn_dense.h"
#include "utec/nn/nn_activation.h"
#include "utec/algebra/tensor.h"

#include <memory>

using namespace utec::app;
using namespace utec::data;
using namespace utec::neural_network;
using namespace utec::algebra;

TEST_CASE("Test: Load trained IA and make a prediction", "[Integration]") {
    // cargar vocabulario
    TextLoader loader("../data/training_words_eng.csv");
    loader.load_vocabulary("../models/vocabulary.txt");
    size_t input_size = loader.get_vocabulary_size();

    // cargar dataset
    loader.load_data();
    auto dataset = loader.get_dataset();

    // tomamos el primer ejemplo de prueba
    REQUIRE(!dataset.empty());  // Verificamos que el dataset no esté vacío

    auto example = dataset[0];

    // Contruimos el modelo y cargamos pesos
    NeuralNetwork<float> model;

    model.add_layer(std::make_unique<Dense<float>>(input_size, 16,
        [](utec::algebra::Tensor<float, 2>& W) { W.fill(0.01f); },
        [](utec::algebra::Tensor<float, 2>& b) { b.fill(0.0f); }));

    model.add_layer(std::make_unique<ReLU<float>>());

    model.add_layer(std::make_unique<Dense<float>>(16, 1,
        [](utec::algebra::Tensor<float, 2>& W) { W.fill(0.01f); },
        [](utec::algebra::Tensor<float, 2>& b) { b.fill(0.0f); }));

    model.add_layer(std::make_unique<Sigmoid<float>>());

    model.load_model("../models/model.txt");

    // input para prediccion
    utec::algebra::Tensor<float, 2> input(2, input_size);
    for (size_t i = 0; i < input_size; ++i) {
        input(0, i) = example.vectorized_text[i];
    }

    auto prediction = model.predict(input);

    REQUIRE(prediction(0, 0) >= 0.0f);
    REQUIRE(prediction(0, 0) <= 1.0f);
}
