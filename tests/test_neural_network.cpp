#include "catch.hpp"
#include "utec/nn/neural_network.h"
#include "utec/nn/nn_dense.h"
#include "utec/nn/nn_activation.h"
#include "utec/nn/nn_loss.h"
#include "utec/algebra/tensor.h"

#include <random>
#include <iomanip>

using namespace utec::neural_network;
using namespace utec::algebra;

TEST_CASE("Neural Network XOR Problem", "[NeuralNetwork]") {
    // Preparacion de datos (XOR)
    constexpr size_t batch_size = 4;
    utec::algebra::Tensor<double, 2> X(batch_size, 2);
    utec::algebra::Tensor<double, 2> Y(batch_size, 1);

    X = { 1, 0,
          0, 1,
          0, 0,
          1, 1 };

    Y = { 1, 1, 0, 0 };

    // Inicializacion Xavier
    std::mt19937 gen(4);
    auto xavier_init = [&](auto& parameter) {
        const double limit = std::sqrt(6.0 / (parameter.shape()[0] + parameter.shape()[1]));
        std::uniform_real_distribution<> dist(-limit, limit);
        for (auto& v : parameter) v = dist(gen);
    };

    // Construcción de la red neuronal
    NeuralNetwork<double> net;
    net.add_layer(std::make_unique<Dense<double>>(2, 4, xavier_init, xavier_init));
    net.add_layer(std::make_unique<Sigmoid<double>>());
    net.add_layer(std::make_unique<Dense<double>>(4, 1, xavier_init, xavier_init));
    net.add_layer(std::make_unique<Sigmoid<double>>());

    // Entrenamiento
    constexpr size_t epochs = 4000;
    constexpr double lr = 0.08;
    net.train<BCELoss>(X, Y, epochs, batch_size, lr);

    // Prediccion
    utec::algebra::Tensor<double, 2> Y_pred = net.predict(X);

    // Validacion
    for (size_t i = 0; i < batch_size; ++i) {
        double prediction = Y_pred(i, 0);
        double expected = Y(i, 0);

        if (expected < 0.5) {
            REQUIRE(prediction < 0.5); // Esperamos que la salida sea cercana a 0
        } else {
            REQUIRE(prediction >= 0.6); // Esperamos que la salida sea cercana a 1
        }
    }
}