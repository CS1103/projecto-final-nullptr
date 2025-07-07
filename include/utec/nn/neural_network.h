//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H


#include "utec/nn/nn_interfaces.h"
#include "utec/nn/nn_optimizer.h"
#include "utec/nn/nn_loss.h"
#include "utec/nn/nn_dense.h"
#include <vector>
#include <memory>
#include <algorithm>

#include "nn_activation.h"

namespace utec::neural_network {

    template<typename T>
    class NeuralNetwork {
        std::vector<std::unique_ptr<ILayer<T>>> layers_;
        Tensor<T, 2> last_output_;

    public:
        void add_layer(std::unique_ptr<ILayer<T>> layer) {
            layers_.emplace_back(std::move(layer));
        }

        Tensor<T, 2> forward(const Tensor<T, 2>& x) {
            last_output_ = x;
            for (auto& layer : layers_) {
                last_output_ = layer->forward(last_output_);
            }
            return last_output_;
        }

        void backward(const Tensor<T, 2>& grad) {
            Tensor<T, 2> g = grad;
            for (int i = layers_.size() - 1; i >= 0; --i) {
                g = layers_[i]->backward(g);
            }
        }

        void optimize(T learning_rate) {
            SGD<T> opt(learning_rate);  // Por defecto
            for (auto& layer : layers_) {
                layer->update_params(opt);
            }
        }

        Tensor<T, 2> predict(const Tensor<T, 2>& X) {
            return forward(X);
        }

        template <
            template <typename> class LossType,
            template <typename> class OptimizerType = SGD
        >
        void train(const Tensor<T,2>& X, const Tensor<T,2>& Y,
                   const size_t epochs, const size_t batch_size, T learning_rate) {
            OptimizerType<T> optimizer(learning_rate);
            const size_t n = X.shape()[0];

            for (size_t epoch = 0; epoch < epochs; ++epoch) {
                for (size_t i = 0; i < n; i += batch_size) {
                    size_t actual_batch_size = std::min(batch_size, n - i);

                    // Crear mini-batch
                    Tensor<T, 2> x_batch(actual_batch_size, X.shape()[1]);
                    Tensor<T, 2> y_batch(actual_batch_size, Y.shape()[1]);
                    for (size_t j = 0; j < actual_batch_size; ++j) {
                        for (size_t k = 0; k < X.shape()[1]; ++k)
                            x_batch(j, k) = X(i + j, k);
                        for (size_t k = 0; k < Y.shape()[1]; ++k)
                            y_batch(j, k) = Y(i + j, k);
                    }

                    Tensor<T, 2> y_pred = forward(x_batch);
                    LossType<T> loss(y_pred, y_batch);
                    Tensor<T, 2> dL = loss.loss_gradient();

                    backward(dL);

                    for (auto& layer : layers_)
                        layer->update_params(optimizer);

                    optimizer.step(); // para Adam, ignorado por SGD
                }
            }
        }

        // Funciones para la IA pre-entrenada
        void save_model(const std::string& path) const {
            std::ofstream out(path);
            if (!out.is_open()) {
                std::cerr << "No se pudo abrir el archivo para guardar el modelo: " << path << std::endl;
                return;
            }

            for (const auto& layer: layers_) {
                auto* dense_layer = dynamic_cast<Dense<T>*>(layer.get());
                if (dense_layer) {
                    out << "Dense\n";
                    dense_layer -> save_layer(out);
                    continue;
                }

                auto* relu_layer = dynamic_cast<ReLU<T>*>(layer.get());
                if (relu_layer) {
                    out << "ReLU\n";
                    continue;
                }

                auto* sigmoid_layer = dynamic_cast<Sigmoid<T>*>(layer.get());
                if (sigmoid_layer) {
                    out << "Sigmoid\n";
                    continue;
                }
            }

            out.close();
            std::cout << "Modelo guardado." << std::endl;
        }

        void load_model(const std::string& path) {
            std::ifstream in(path);
            if (!in.is_open()) {
                std::cerr << "No se pudo abrir el archivo para cargar el modelo." << std::endl;
                return;
            }

            std::string layer_type;

            for (auto& layer: layers_) {
                in >> layer_type;

                if (layer_type == "Dense") {
                    auto* dense_layer = dynamic_cast<Dense<T>*>(layer.get());
                    if (dense_layer) {
                        dense_layer -> load_layer(in);
                    } else {
                        std::cerr << "Error: se esperaba Dense pero se encontro otra capa." << std::endl;
                        return;
                    }
                } else if (layer_type == "ReLU") {
                    if (dynamic_cast<ReLU<T>*>(layer.get()) == nullptr) {
                        std::cerr << "Error: se esperaba ReLU pero se encontró otra capa." << std::endl;
                        return;
                    }
                } else if (layer_type == "Sigmoid") {
                    if (dynamic_cast<Sigmoid<T>*>(layer.get()) == nullptr) {
                        std::cerr << "Error: se esperaba Sigmoid pero se encontro otra capa." << std::endl;
                        return;
                    }
                } else {
                    std::cerr << "Error: capa desconocida: " << layer_type << std::endl;
                    return;
                }
            }
            in.close();
            std::cout << "Modelo cargado correctamente." << std::endl;
        }
    };

}


#endif //PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
