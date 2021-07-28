#ifndef STAN_MATH_PRIM_FUN_LUB_OM_FREE_HPP
#define STAN_MATH_PRIM_FUN_LUB_OM_FREE_HPP

#include <stan/math/prim/fun/offset_multiplier_free.hpp>
#include <stan/math/prim/fun/lub_free.hpp>
#include <stan/math/prim/fun/eval.hpp>

namespace stan {
namespace math {
template <typename T, typename L, typename U, typename M, typename S>
inline auto lub_offset_multiplier_free(const T& y, const L& lb, const U& ub,
                                       const M& mu, const S& sigma) {
  return eval(offset_multiplier_free(lub_free(y, lb, ub), mu, sigma));
}
}  // namespace math
}  // namespace stan
#endif
