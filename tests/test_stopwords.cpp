#include "catch.hpp"
#include "utec/data/text_loader.h"
#include <fstream>

using namespace utec::data;

TEST_CASE("Stopwords are removed from tokenized text", "[TextLoader][Stopwords]") {
    TextLoader loader;

    std::ofstream stopwords_file("../data/stopwords_test.txt");
    stopwords_file << "el\nla\nlos\nlas\nde\ndel\ny\na\n";
    stopwords_file.close();

    loader.load_stopwords("../data/stopwords_test.txt");

    std::string mensaje = "La oferta del dia es gratis y sin compromiso";

    auto tokens = loader.tokenize(mensaje);

    // no deberia contener "la", "del", "y".
    REQUIRE(std::find(tokens.begin(), tokens.end(), "la") == tokens.end());
    REQUIRE(std::find(tokens.begin(), tokens.end(), "y") == tokens.end());
    REQUIRE(std::find(tokens.begin(), tokens.end(), "del") == tokens.end());

    // deberia mantener las demas "oferta", "gratis", "compromiso"
    REQUIRE(std::find(tokens.begin(), tokens.end(), "oferta") != tokens.end());
    REQUIRE(std::find(tokens.begin(), tokens.end(), "gratis") != tokens.end());
    REQUIRE(std::find(tokens.begin(), tokens.end(), "compromiso") != tokens.end());
}
