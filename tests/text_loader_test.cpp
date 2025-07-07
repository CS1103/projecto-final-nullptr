#include "catch.hpp"
#include "utec/data/text_loader.h"

using namespace utec::data;

TEST_CASE("TextLoader loads dataset and builds vocabulary", "[TextLoader]") {
    TextLoader loader("../data/training_words_esp.csv");

    loader.load_data();

    REQUIRE(loader.get_vocabulary_size() > 0);
    REQUIRE(loader.get_dataset().size() > 0);
}

TEST_CASE("TextLoader vectorizes simple message correctly", "[TextLoader]") {
    TextLoader loader("../data/training_words_esp.csv");

    loader.load_data();

    auto vector = loader.vectorize("dinero gratis");
    REQUIRE(vector.size() == loader.get_vocabulary_size());

    const auto& vocab = loader.get_vocabulary_list();
    bool found_dinero = false;
    bool found_gratis = false;

    for (size_t i = 0; i < vocab.size(); ++i) {
        if (vocab[i] == "dinero" && vector[i] > 0) found_dinero = true;
        if (vocab[i] == "gratis" && vector[i] > 0) found_gratis = true;
    }

    REQUIRE(found_dinero);
    REQUIRE(found_gratis);
}

TEST_CASE("TextLoader tokenizes correctly", "[TextLoader]") {
    TextLoader loader;

    auto tokens = loader.tokenize("Hola, mundo! Hola.");

    REQUIRE(tokens.size() == 3); // ["hola", "mundo", "hola"]
    REQUIRE(tokens[0] == "hola");
    REQUIRE(tokens[1] == "mundo");
    REQUIRE(tokens[2] == "hola");
}
