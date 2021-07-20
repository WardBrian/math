#ifndef STAN_MATH_PRIM_FUN_LUB_OM_CONSTRAIN_HPP
#define STAN_MATH_PRIM_FUN_LUB_OM_CONSTRAIN_HPP

#include <stan/math/prim/fun/offset_multiplier_constrain.hpp>
#include <stan/math/prim/fun/lub_constrain.hpp>

namespace stan {
namespace math {

template <typename T, typename L, typename U, typename M, typename S>
inline auto lub_offset_multiplier_constrain(const T& y, const L& lb,
                                            const U& ub, const M& mu,
                                            const S& sigma) {
  return lub_constrain(offset_multiplier_constrain(y, mu, sigma), lb, ub);
}

template <typename T, typename L, typename U, typename M, typename S>
inline auto lub_offset_multiplier_constrain(const T& y, const L& lb,
                                            const U& ub, const M& mu,
                                            const S& sigma,
                                            return_type_t<T, L, U, M, S>& lp) {
  return lub_constrain(offset_multiplier_constrain(y, mu, sigma, lp), lb, ub,
                       lp);
}

}  // namespace math
}  // namespace stan
#endif
