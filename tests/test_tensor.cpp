#include "catch.hpp"
#include "utec/algebra/tensor.h"

using namespace utec::algebra;

TEST_CASE("Tensor creation and basic shape", "[Tensor]") {
    Tensor<float, 2> t(2, 3);

    REQUIRE(t.shape()[0] == 2);
    REQUIRE(t.shape()[1] == 3);
    REQUIRE(t.size() == 6);
}

TEST_CASE("Tensor element assignment and access", "[Tensor]") {
    Tensor<float, 2> t(2, 2);
    t(0, 0) = 1.0f;
    t(0, 1) = 2.0f;
    t(1, 0) = 3.0f;
    t(1, 1) = 4.0f;

    REQUIRE(t(0, 0) == Approx(1.0f));
    REQUIRE(t(0, 1) == Approx(2.0f));
    REQUIRE(t(1, 0) == Approx(3.0f));
    REQUIRE(t(1, 1) == Approx(4.0f));
}

TEST_CASE("Tensor addition", "[Tensor]") {
    Tensor<float, 2> a(2, 2);
    a(0, 0) = 1.0f; a(0, 1) = 2.0f;
    a(1, 0) = 3.0f; a(1, 1) = 4.0f;

    Tensor<float, 2> b(2, 2);
    b(0, 0) = 5.0f; b(0, 1) = 6.0f;
    b(1, 0) = 7.0f; b(1, 1) = 8.0f;

    auto c = a + b;

    REQUIRE(c(0, 0) == Approx(6.0f));
    REQUIRE(c(0, 1) == Approx(8.0f));
    REQUIRE(c(1, 0) == Approx(10.0f));
    REQUIRE(c(1, 1) == Approx(12.0f));
}

TEST_CASE("Tensor scalar multiplication", "[Tensor]") {
    Tensor<float, 2> t(2, 2);
    t(0, 0) = 1.0f; t(0, 1) = 2.0f;
    t(1, 0) = 3.0f; t(1, 1) = 4.0f;

    auto result = t * 2.0f;

    REQUIRE(result(0, 0) == Approx(2.0f));
    REQUIRE(result(0, 1) == Approx(4.0f));
    REQUIRE(result(1, 0) == Approx(6.0f));
    REQUIRE(result(1, 1) == Approx(8.0f));
}

TEST_CASE("Tensor reshape", "[Tensor]") {
    Tensor<float, 2> t(2, 3);
    t.reshape(3, 2);

    REQUIRE(t.shape()[0] == 3);
    REQUIRE(t.shape()[1] == 2);
    REQUIRE(t.size() == 6);
}

TEST_CASE("Tensor matrix product", "[Tensor]") {
    Tensor<float, 2> A(2, 2);
    A(0, 0) = 1; A(0, 1) = 2;
    A(1, 0) = 3; A(1, 1) = 4;

    Tensor<float, 2> B(2, 2);
    B(0, 0) = 5; B(0, 1) = 6;
    B(1, 0) = 7; B(1, 1) = 8;

    auto C = matrix_product(A, B);

    REQUIRE(C(0, 0) == Approx(19.0f)); // 1*5 + 2*7
    REQUIRE(C(0, 1) == Approx(22.0f)); // 1*6 + 2*8
    REQUIRE(C(1, 0) == Approx(43.0f)); // 3*5 + 4*7
    REQUIRE(C(1, 1) == Approx(50.0f)); // 3*6 + 4*8
}
