#ifndef STAN_MATH_REV_FUN_FFT_HPP
#define STAN_MATH_REV_FUN_FFT_HPP

#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/fun/typedefs.hpp>
#include <stan/math/rev/meta.hpp>
#include <stan/math/prim/fun/fft.hpp>
#include <Eigen/Dense>
#include <complex>
#include <type_traits>
#include <vector>

namespace stan {
namespace math {

template <typename V, require_eigen_vector_vt<is_complex, V>* = nullptr,
          require_var_t<base_type_t<value_type_t<V>>>* = nullptr>
inline plain_type_t<V> fft(const V& v) {
  if (unlikely(v.size() < 1)) {
    return plain_type_t<V>(v);
  }

  arena_t<V> arena_v = v;
  arena_t<V> res = fft(arena_v.val());

  reverse_pass_callback([arena_v, res]() mutable {
    arena_v.adj() += inv_fft(res.adj());
  });

  return plain_type_t<V>(res);
}

}  // namespace math
}  // namespace stan
#endif
