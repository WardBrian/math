#ifndef STAN_MATH_PRIM_FUN_UB_OM_CONSTRAIN_HPP
#define STAN_MATH_PRIM_FUN_UB_OM_CONSTRAIN_HPP

#include <stan/math/prim/fun/offset_multiplier_constrain.hpp>
#include <stan/math/prim/fun/ub_constrain.hpp>

namespace stan {
namespace math {

template <typename T, typename U, typename M, typename S>
inline auto ub_offset_multiplier_constrain(const T& y, const U& ub, const M& mu,
                                           const S& sigma) {
  return ub_constrain(offset_multiplier_constrain(y, mu, sigma), ub);
}

template <typename T, typename U, typename M, typename S>
inline auto ub_offset_multiplier_constrain(const T& y, const U& ub, const M& mu,
                                           const S& sigma,
                                           return_type_t<T, U, M, S>& lp) {
  return ub_constrain(offset_multiplier_constrain(y, mu, sigma, lp), ub, lp);
}

}  // namespace math
}  // namespace stan
#endif
