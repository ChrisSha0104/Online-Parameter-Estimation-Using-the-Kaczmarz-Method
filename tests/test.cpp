#include <param_id/lib.hpp>
#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>

TEST_CASE("test_hello_world") {
  std::string value = param_id::test();
  REQUIRE(value == std::string("Hello world!"));
}
