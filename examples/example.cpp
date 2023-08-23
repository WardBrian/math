#include <stan/math.hpp>
#include <iostream>

int main() {
  stan::math::var x = 5.0;
  stan::math::var y = 2.0;
  stan::math::var z = x * y;
  z.grad();
  std::cout << "z = " << z.val() << std::endl;
  std::cout << "dz/dx = " << x.adj() << std::endl;
  std::cout << "dz/dy = " << y.adj() << std::endl;
  stan::math::recover_memory();
  return 0;
}
