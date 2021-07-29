#ifndef STAN_MATH_PRIM_FUN_UB_OM_FREE_HPP
#define STAN_MATH_PRIM_FUN_UB_OM_FREE_HPP

#include <stan/math/prim/fun/offset_multiplier_free.hpp>
#include <stan/math/prim/fun/ub_free.hpp>
#include <stan/math/prim/fun/eval.hpp>

namespace stan {
namespace math {
template <typename T, typename U, typename M, typename S>
inline auto ub_offset_multiplier_free(const T& y, const U& ub, const M& mu,
                                       const S& sigma) {
  return eval(offset_multiplier_free(ub_free(y, ub), mu, sigma));
}

}  // namespace math
}  // namespace stan
#endif
