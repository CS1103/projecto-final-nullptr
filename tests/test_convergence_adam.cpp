#include "catch.hpp"
#include "utec/nn/neural_network.h"
#include "utec/nn/nn_dense.h"
#include "utec/nn/nn_activation.h"
#include "utec/nn/nn_loss.h"
#include "utec/nn/nn_optimizer.h"
#include "utec/algebra/tensor.h"

using namespace utec::neural_network;
using namespace utec::algebra;

TEST_CASE("Adam optimizer converges on simple OR task", "[NeuralNetwork][Adam]") {
    NeuralNetwork<float> model;

    model.add_layer(std::make_unique<Dense<float>>(2, 4,
        [](utec::algebra::Tensor<float, 2>& W) { W.fill(0.1f); },
        [](utec::algebra::Tensor<float, 2>& b) { b.fill(0.0f); }));

    model.add_layer(std::make_unique<ReLU<float>>());

    model.add_layer(std::make_unique<Dense<float>>(4, 1,
        [](utec::algebra::Tensor<float, 2>& W) { W.fill(0.1f); },
        [](utec::algebra::Tensor<float, 2>& b) { b.fill(0.0f); }));

    model.add_layer(std::make_unique<Sigmoid<float>>());

    // Ejemplo con OR logico
    // Datos del OR
    utec::algebra::Tensor<float, 2> X(4, 2);
    utec::algebra::Tensor<float, 2> Y(4, 1);
    X(0, 0) = 0; X(0, 1) = 0; Y(0, 0) = 0;
    X(1, 0) = 0; X(1, 1) = 1; Y(1, 0) = 1;
    X(2, 0) = 1; X(2, 1) = 0; Y(2, 0) = 1;
    X(3, 0) = 1; X(3, 1) = 1; Y(3, 0) = 1;

    // Entrenar con Adam
    model.train<BCELoss, Adam>(X, Y, 50, 4, 0.05f);

    // Verificar que la perdida final sea baja
    auto y_pred = model.predict(X);
    BCELoss<float> loss(y_pred, Y);

    REQUIRE(loss.loss() < 0.1f);  // convergencia aceptable
}
