#ifndef STAN_MATH_PRIM_FUN_LB_OM_CONSTRAIN_HPP
#define STAN_MATH_PRIM_FUN_LB_OM_CONSTRAIN_HPP

#include <stan/math/prim/fun/offset_multiplier_constrain.hpp>
#include <stan/math/prim/fun/lb_constrain.hpp>

namespace stan {
namespace math {

template <typename T, typename L, typename M, typename S>
inline auto lb_offset_multiplier_constrain(const T& y, const L& lb, const M& mu,
                                           const S& sigma) {
  return lb_constrain(offset_multiplier_constrain(y, mu, sigma), lb);
}

template <typename T, typename L, typename M, typename S>
inline auto lb_offset_multiplier_constrain(const T& y, const L& lb, const M& mu,
                                           const S& sigma,
                                           return_type_t<T, L, M, S>& lp) {
  return lb_constrain(offset_multiplier_constrain(y, mu, sigma, lp), lb, lp);
}

}  // namespace math
}  // namespace stan
#endif
