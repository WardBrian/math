#include <stan/math/prim.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>

TEST(prob_transform, lub_om) {
  for (double L : std::vector<double>{-1, 0.5, 2, 10}) {
    for (double U : std::vector<double>{-1, 0.5, 2, 10}) {
      if (L >= U)
        continue;

      for (double O : std::vector<double>{-1, 0.5, 2, 10}) {
        for (double M : std::vector<double>{0.5, 2, 10}) {
          for (double x : std::vector<double>{-20, -15, 0.1, 3, 45.2}) {
            EXPECT_FLOAT_EQ(
                L + (U - L) * stan::math::inv_logit(x * M + O),
                stan::math::lub_offset_multiplier_constrain(x, L, U, O, M));
          }
        }
      }
    }
  }
}

TEST(prob_transform, lub_om_underflow) {
  EXPECT_EQ(0, stan::math::lub_offset_multiplier_constrain(-1000, 0, 1, 0, 1));
  double lp = 0;
  EXPECT_EQ(0,
            stan::math::lub_offset_multiplier_constrain(-1000, 0, 1, 0, 1, lp));
  EXPECT_EQ(0, stan::math::lub_offset_multiplier_constrain(0, 0, 1, -1000, 1));
  lp = 0;
  EXPECT_EQ(0,
            stan::math::lub_offset_multiplier_constrain(0, 0, 1, -1000, 1, lp));
  EXPECT_EQ(0, stan::math::lub_offset_multiplier_constrain(-1, 0, 1, 0, 1000));
  lp = 0;
  EXPECT_EQ(0,
            stan::math::lub_offset_multiplier_constrain(-1, 0, 1, 0, 1000, lp));
}

TEST(prob_transform, lub_om_vec) {
  Eigen::VectorXd input(2);
  input << -1.0, 1.1;
  Eigen::VectorXd lbv(2);
  lbv << -1.0, 0.5;
  Eigen::VectorXd ubv(2);
  ubv << 2.0, 3.0;
  Eigen::VectorXd muv(2);
  muv << 2.0, 3.0;
  Eigen::VectorXd sigmav(2);
  sigmav << 2.0, 0.5;
  double lb = 1.0;
  double ub = 2.0;
  double mu = 0.5;
  double sigma = 5.0;

  Eigen::VectorXd resvvvv(2);
  resvvvv << (2.0 + 1.0) * stan::math::inv_logit(-1.0 * 2.0 + 2.0) - 1.0,
      (3.0 - 0.5) * stan::math::inv_logit(1.1 * 0.5 + 3.0) + 0.5;
  Eigen::VectorXd ressvvv(2);
  ressvvv << (2.0 - 1.0) * stan::math::inv_logit(-1.0 * 2.0 + 2.0) + 1.0,
      (3.0 - 1.0) * stan::math::inv_logit(1.1 * 0.5 + 3.0) + 1.0;
  Eigen::VectorXd resvsvv(2);
  resvsvv << (2.0 + 1.0) * stan::math::inv_logit(-1.0 * 2.0 + 2.0) - 1.0,
      (2.0 - 0.5) * stan::math::inv_logit(1.1 * 0.5 + 3.0) + 0.5;
  Eigen::VectorXd resvvsv(2);
  resvvsv << (2.0 + 1.0) * stan::math::inv_logit(-1.0 * 2.0 + 0.5) - 1.0,
      (3.0 - 0.5) * stan::math::inv_logit(1.1 * 0.5 + 0.5) + 0.5;
  Eigen::VectorXd resvvvs(2);
  resvvvs << (2.0 + 1.0) * stan::math::inv_logit(-1.0 * 5.0 + 2.0) - 1.0,
      (3.0 - 0.5) * stan::math::inv_logit(1.1 * 5.0 + 3.0) + 0.5;
  Eigen::VectorXd resssvv(2);
  resssvv << (2.0 - 1.0) * stan::math::inv_logit(-1.0 * 2.0 + 2.0) + 1.0,
      (2.0 - 1.0) * stan::math::inv_logit(1.1 * 0.5 + 3.0) + 1.0;
  Eigen::VectorXd resvssv(2);
  resvssv << (2.0 + 1.0) * stan::math::inv_logit(-1.0 * 2.0 + 0.5) - 1.0,
      (2.0 - 0.5) * stan::math::inv_logit(1.1 * 0.5 + 0.5) + 0.5;
  Eigen::VectorXd resvvss(2);
  resvvss << (2.0 + 1.0) * stan::math::inv_logit(-1.0 * 5.0 + 0.5) - 1.0,
      (3.0 - 0.5) * stan::math::inv_logit(1.1 * 5.0 + 0.5) + 0.5;
  Eigen::VectorXd ressvsv(2);
  ressvsv << (2.0 - 1.0) * stan::math::inv_logit(-1.0 * 2.0 + 0.5) + 1.0,
      (3.0 - 1.0) * stan::math::inv_logit(1.1 * 0.5 + 0.5) + 1.0;
  Eigen::VectorXd resvsvs(2);
  resvsvs << (2.0 + 1.0) * stan::math::inv_logit(-1.0 * 5.0 + 2.0) - 1.0,
      (2.0 - 0.5) * stan::math::inv_logit(1.1 * 5.0 + 3.0) + 0.5;
  Eigen::VectorXd ressvvs(2);
  ressvvs << (2.0 - 1.0) * stan::math::inv_logit(-1.0 * 5.0 + 2.0) + 1.0,
      (3.0 - 1.0) * stan::math::inv_logit(1.1 * 5.0 + 3.0) + 1.0;
  Eigen::VectorXd ressssv(2);
  ressssv << (2.0 - 1.0) * stan::math::inv_logit(-1.0 * 2.0 + 0.5) + 1.0,
      (2.0 - 1.0) * stan::math::inv_logit(1.1 * 0.5 + 0.5) + 1.0;
  Eigen::VectorXd resssvs(2);
  resssvs << (2.0 - 1.0) * stan::math::inv_logit(-1.0 * 5.0 + 2.0) + 1.0,
      (2.0 - 1.0) * stan::math::inv_logit(1.1 * 5.0 + 3.0) + 1.0;
  Eigen::VectorXd ressvss(2);
  ressvss << (2.0 - 1.0) * stan::math::inv_logit(-1.0 * 5.0 + 0.5) + 1.0,
      (3.0 - 1.0) * stan::math::inv_logit(1.1 * 5.0 + 0.5) + 1.0;
  Eigen::VectorXd resvsss(2);
  resvsss << (2.0 + 1.0) * stan::math::inv_logit(-1.0 * 5.0 + 0.5) - 1.0,
      (2.0 - 0.5) * stan::math::inv_logit(1.1 * 5.0 + 0.5) + 0.5;
  Eigen::VectorXd res(2);
  res << (2.0 - 1.0) * stan::math::inv_logit(-1.0 * 5.0 + 0.5) + 1.0,
      (2.0 - 1.0) * stan::math::inv_logit(1.1 * 5.0 + 0.5) + 1.0;

  EXPECT_MATRIX_EQ(resvvvv, stan::math::lub_offset_multiplier_constrain(
                                input, lbv, ubv, muv, sigmav));
  EXPECT_MATRIX_EQ(ressvvv, stan::math::lub_offset_multiplier_constrain(
                                input, lb, ubv, muv, sigmav));
  EXPECT_MATRIX_EQ(resvsvv, stan::math::lub_offset_multiplier_constrain(
                                input, lbv, ub, muv, sigmav));
  EXPECT_MATRIX_EQ(resvvsv, stan::math::lub_offset_multiplier_constrain(
                                input, lbv, ubv, mu, sigmav));
  EXPECT_MATRIX_EQ(resvvvs, stan::math::lub_offset_multiplier_constrain(
                                input, lbv, ubv, muv, sigma));
  EXPECT_MATRIX_EQ(resssvv, stan::math::lub_offset_multiplier_constrain(
                                input, lb, ub, muv, sigmav));
  EXPECT_MATRIX_EQ(resvssv, stan::math::lub_offset_multiplier_constrain(
                                input, lbv, ub, mu, sigmav));
  EXPECT_MATRIX_EQ(resvvss, stan::math::lub_offset_multiplier_constrain(
                                input, lbv, ubv, mu, sigma));
  EXPECT_MATRIX_EQ(ressvsv, stan::math::lub_offset_multiplier_constrain(
                                input, lb, ubv, mu, sigmav));
  EXPECT_MATRIX_EQ(resvsvs, stan::math::lub_offset_multiplier_constrain(
                                input, lbv, ub, muv, sigma));
  EXPECT_MATRIX_EQ(ressvvs, stan::math::lub_offset_multiplier_constrain(
                                input, lb, ubv, muv, sigma));
  EXPECT_MATRIX_EQ(ressssv, stan::math::lub_offset_multiplier_constrain(
                                input, lb, ub, mu, sigmav));
  EXPECT_MATRIX_EQ(resssvs, stan::math::lub_offset_multiplier_constrain(
                                input, lb, ub, muv, sigma));
  EXPECT_MATRIX_EQ(ressvss, stan::math::lub_offset_multiplier_constrain(
                                input, lb, ubv, mu, sigma));
  EXPECT_MATRIX_EQ(resvsss, stan::math::lub_offset_multiplier_constrain(
                                input, lbv, ub, mu, sigma));
  EXPECT_MATRIX_EQ(res, stan::math::lub_offset_multiplier_constrain(
                            input, lb, ub, mu, sigma));

  double lp = 0.0;
  EXPECT_MATRIX_EQ(resvvvv, stan::math::lub_offset_multiplier_constrain(
                                input, lbv, ubv, muv, sigmav, lp));
  EXPECT_FLOAT_EQ(((ubv - lbv).array().log()
                   + (input.array() * sigmav.array() + muv.array())
                   - 2
                         * (input.array() * sigmav.array() + muv.array())
                               .exp()
                               .log1p()
                               .array()
                   + sigmav.array().log())
                      .sum(),
                  lp);
  lp = 0.0;
  EXPECT_MATRIX_EQ(ressvvv, stan::math::lub_offset_multiplier_constrain(
                                input, lb, ubv, muv, sigmav, lp));
  EXPECT_FLOAT_EQ(
      ((ubv.array() - lb).log() + (input.array() * sigmav.array() + muv.array())
       - 2
             * (input.array() * sigmav.array() + muv.array())
                   .exp()
                   .log1p()
                   .array()
       + sigmav.array().log())
          .sum(),
      lp);
  lp = 0.0;
  EXPECT_MATRIX_EQ(resvsvv, stan::math::lub_offset_multiplier_constrain(
                                input, lbv, ub, muv, sigmav, lp));
  EXPECT_FLOAT_EQ(
      ((ub - lbv.array()).log() + (input.array() * sigmav.array() + muv.array())
       - 2
             * (input.array() * sigmav.array() + muv.array())
                   .exp()
                   .log1p()
                   .array()
       + sigmav.array().log())
          .sum(),
      lp);
  lp = 0.0;
  EXPECT_MATRIX_EQ(resvvsv, stan::math::lub_offset_multiplier_constrain(
                                input, lbv, ubv, mu, sigmav, lp));
  EXPECT_FLOAT_EQ(
      ((ubv - lbv).array().log() + (input.array() * sigmav.array() + mu)
       - 2 * (input.array() * sigmav.array() + mu).exp().log1p().array())
          .sum(),
      lp);
  lp = 0.0;
  EXPECT_MATRIX_EQ(resvvvs, stan::math::lub_offset_multiplier_constrain(
                                input, lbv, ubv, muv, sigma, lp));
  EXPECT_FLOAT_EQ(
      ((ubv - lbv).array().log() + ((input * sigma).array() + muv.array())
       - 2 * ((input * sigma).array() + muv.array()).exp().log1p().array()
       + std::log(sigma))
          .sum(),
      lp);
  lp = 0.0;
  EXPECT_MATRIX_EQ(resssvv, stan::math::lub_offset_multiplier_constrain(
                                input, lb, ub, muv, sigmav, lp));
  EXPECT_FLOAT_EQ(
      (std::log(ub - lb) + (input.array() * sigmav.array() + muv.array())
       - 2
             * (input.array() * sigmav.array() + muv.array())
                   .exp()
                   .log1p()
                   .array()
       + sigmav.array().log())
          .sum(),
      lp);
  lp = 0.0;
  EXPECT_MATRIX_EQ(resvssv, stan::math::lub_offset_multiplier_constrain(
                                input, lbv, ub, mu, sigmav, lp));
  EXPECT_FLOAT_EQ(
      ((ub - lbv.array()).log() + (input.array() * sigmav.array() + mu)
       - 2 * (input.array() * sigmav.array() + mu).exp().log1p().array()
       + sigmav.array().log())
          .sum(),
      lp);
  lp = 0.0;
  EXPECT_MATRIX_EQ(resvvss, stan::math::lub_offset_multiplier_constrain(
                                input, lbv, ubv, mu, sigma, lp));
  EXPECT_FLOAT_EQ(((ubv - lbv).array().log() + (input.array() * sigma + mu)
                   - 2 * (input.array() * sigma + mu).exp().log1p().array()
                   + std::log(sigma))
                      .sum(),
                  lp);
  lp = 0.0;
  EXPECT_MATRIX_EQ(ressvsv, stan::math::lub_offset_multiplier_constrain(
                                input, lb, ubv, mu, sigmav, lp));
  EXPECT_FLOAT_EQ(
      ((ubv.array() - lb).log() + (input.array() * sigmav.array() + mu)
       - 2 * (input.array() * sigmav.array() + mu).exp().log1p().array()
       + sigmav.array().log())
          .sum(),
      lp);
  lp = 0.0;
  EXPECT_MATRIX_EQ(resvsvs, stan::math::lub_offset_multiplier_constrain(
                                input, lbv, ub, muv, sigma, lp));
  EXPECT_FLOAT_EQ(
      ((ub - lbv.array()).log() + (input.array() * sigma + muv.array())
       - 2 * (input.array() * sigma + muv.array()).exp().log1p().array()
       + std::log(sigma))
          .sum(),
      lp);
  lp = 0.0;
  EXPECT_MATRIX_EQ(ressvvs, stan::math::lub_offset_multiplier_constrain(
                                input, lb, ubv, muv, sigma, lp));
  EXPECT_FLOAT_EQ(
      ((ubv.array() - lb).log() + (input.array() * sigma + muv.array())
       - 2 * (input.array() * sigma + muv.array()).exp().log1p().array()
       + std::log(sigma))
          .sum(),
      lp);
  lp = 0.0;
  EXPECT_MATRIX_EQ(ressssv, stan::math::lub_offset_multiplier_constrain(
                                input, lb, ub, mu, sigmav, lp));
  EXPECT_FLOAT_EQ(
      (std::log(ub - lb) + (input.array() * sigmav.array() + mu)
       - 2 * (input.array() * sigmav.array() + mu).exp().log1p().array()
       + sigmav.array().log())
          .sum(),
      lp);
  lp = 0.0;
  EXPECT_MATRIX_EQ(resssvs, stan::math::lub_offset_multiplier_constrain(
                                input, lb, ub, muv, sigma, lp));
  EXPECT_FLOAT_EQ(
      (std::log(ub - lb) + (input.array() * sigma + muv.array())
       - 2 * (input.array() * sigma + muv.array()).exp().log1p().array()
       + std::log(sigma))
          .sum(),
      lp);
  lp = 0.0;
  EXPECT_MATRIX_EQ(ressvss, stan::math::lub_offset_multiplier_constrain(
                                input, lb, ubv, mu, sigma, lp));
  EXPECT_FLOAT_EQ(((ubv.array() - lb).log() + (input.array() * sigma + mu)
                   - 2 * (input.array() * sigma + mu).exp().log1p().array()
                   + std::log(sigma))
                      .sum(),
                  lp);
  lp = 0.0;
  EXPECT_MATRIX_EQ(resvsss, stan::math::lub_offset_multiplier_constrain(
                                input, lbv, ub, mu, sigma, lp));
  EXPECT_FLOAT_EQ(((ub - lbv.array()).log() + (input.array() * sigma + mu)
                   - 2 * (input.array() * sigma + mu).exp().log1p().array()
                   + std::log(sigma))
                      .sum(),
                  lp);
  lp = 0.0;
  EXPECT_MATRIX_EQ(res, stan::math::lub_offset_multiplier_constrain(
                            input, lb, ub, mu, sigma, lp));
  EXPECT_FLOAT_EQ((std::log(ub - lb) + (input.array() * sigma + mu)
                   - 2 * (input.array() * sigma + mu).exp().log1p().array()
                   + std::log(sigma))
                      .sum(),
                  lp);
}

TEST(prob_transform, lub_om_constrain_matrix) {
  Eigen::VectorXd x(4);
  x << -1.0, 1.1, 3.0, 4.0;
  Eigen::VectorXd ub(4);
  ub << stan::math::INFTY, stan::math::INFTY, 6.0, 7.0;
  Eigen::VectorXd lb(4);
  lb << 2.0, stan::math::NEGATIVE_INFTY, stan::math::NEGATIVE_INFTY, 2.0;
  Eigen::VectorXd sigma(4);
  sigma << 1.1, 0.3, 6.0, 3.0;
  Eigen::VectorXd offset(4);
  offset << -2.0, 0.0, 0.2, 2.0;

  double sigmad = 3.0;
  double offsetd = -2.0;
  double ubd = 8;
  double lbd = -2;

  Eigen::VectorXd sigma_bad(3);
  Eigen::VectorXd offset_bad(3);
  Eigen::VectorXd ub_bad(3);
  Eigen::VectorXd lb_bad(3);

  // matrix, real, real, real, real
  {
    Eigen::VectorXd result = stan::math::lub_offset_multiplier_constrain(
        x, lbd, ubd, offsetd, sigmad);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::lub_offset_multiplier_constrain(
                                     x(i), lbd, ubd, offsetd, sigmad));
    }
    Eigen::VectorXd x_free = stan::math::lub_offset_multiplier_free(
        result, lbd, ubd, offsetd, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
  }

  // matrix, matrix, real, real, real
  {
    Eigen::VectorXd result = stan::math::lub_offset_multiplier_constrain(
        x, lb, ubd, offsetd, sigmad);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::lub_offset_multiplier_constrain(
                                     x(i), lb(i), ubd, offsetd, sigmad));
    }
    Eigen::VectorXd x_free = stan::math::lub_offset_multiplier_free(
        result, lb, ubd, offsetd, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(x, lb_bad, ubd,
                                                             offsetd, sigmad),
                 std::invalid_argument);
  }

  // matrix, real, matrix, real, real
  {
    Eigen::VectorXd result = stan::math::lub_offset_multiplier_constrain(
        x, lbd, ub, offsetd, sigmad);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::lub_offset_multiplier_constrain(
                                     x(i), lbd, ub(i), offsetd, sigmad));
    }
    Eigen::VectorXd x_free = stan::math::lub_offset_multiplier_free(
        result, lbd, ub, offsetd, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(x, lbd, ub_bad,
                                                             offsetd, sigmad),
                 std::invalid_argument);
  }

  // matrix, real, real, matrix, real
  {
    Eigen::VectorXd result = stan::math::lub_offset_multiplier_constrain(
        x, lbd, ubd, offset, sigmad);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::lub_offset_multiplier_constrain(
                                     x(i), lbd, ubd, offset(i), sigmad));
    }
    Eigen::VectorXd x_free = stan::math::lub_offset_multiplier_free(
        result, lbd, ubd, offset, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lbd, ubd, offset_bad, sigmad),
                 std::invalid_argument);
  }

  // matrix, real, real, real, matrix
  {
    Eigen::VectorXd result = stan::math::lub_offset_multiplier_constrain(
        x, lbd, ubd, offsetd, sigma);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::lub_offset_multiplier_constrain(
                                     x(i), lbd, ubd, offsetd, sigma(i)));
    }
    Eigen::VectorXd x_free = stan::math::lub_offset_multiplier_free(
        result, lbd, ubd, offsetd, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lbd, ubd, offsetd, sigma_bad),
                 std::invalid_argument);
  }

  // matrix, matrix, matrix, real, real
  {
    Eigen::VectorXd result = stan::math::lub_offset_multiplier_constrain(
        x, lb, ub, offsetd, sigmad);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::lub_offset_multiplier_constrain(
                                     x(i), lb(i), ub(i), offsetd, sigmad));
    }
    Eigen::VectorXd x_free = stan::math::lub_offset_multiplier_free(
        result, lb, ub, offsetd, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(x, lb_bad, ub_bad,
                                                             offsetd, sigmad),
                 std::invalid_argument);
  }

  // matrix, matrix, real, matrix, real
  {
    Eigen::VectorXd result = stan::math::lub_offset_multiplier_constrain(
        x, lb, ubd, offset, sigmad);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::lub_offset_multiplier_constrain(
                                     x(i), lb(i), ubd, offset(i), sigmad));
    }
    Eigen::VectorXd x_free = stan::math::lub_offset_multiplier_free(
        result, lb, ubd, offset, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lb_bad, ubd, offset_bad, sigmad),
                 std::invalid_argument);
  }

  // matrix, matrix, real, real, matrix
  {
    Eigen::VectorXd result = stan::math::lub_offset_multiplier_constrain(
        x, lb, ubd, offsetd, sigma);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::lub_offset_multiplier_constrain(
                                     x(i), lb(i), ubd, offsetd, sigma(i)));
    }
    Eigen::VectorXd x_free = stan::math::lub_offset_multiplier_free(
        result, lb, ubd, offsetd, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lb_bad, ubd, offsetd, sigma_bad),
                 std::invalid_argument);
  }

  // matrix, real, matrix, matrix, real
  {
    Eigen::VectorXd result = stan::math::lub_offset_multiplier_constrain(
        x, lbd, ub, offset, sigmad);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::lub_offset_multiplier_constrain(
                                     x(i), lbd, ub(i), offset(i), sigmad));
    }
    Eigen::VectorXd x_free = stan::math::lub_offset_multiplier_free(
        result, lbd, ub, offset, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lbd, ub_bad, offset_bad, sigmad),
                 std::invalid_argument);
  }

  // matrix, real, matrix, real, matrix
  {
    Eigen::VectorXd result = stan::math::lub_offset_multiplier_constrain(
        x, lbd, ub, offsetd, sigma);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::lub_offset_multiplier_constrain(
                                     x(i), lbd, ub(i), offsetd, sigma(i)));
    }
    Eigen::VectorXd x_free = stan::math::lub_offset_multiplier_free(
        result, lbd, ub, offsetd, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lbd, ub_bad, offsetd, sigma_bad),
                 std::invalid_argument);
  }
  // matrix, real, real, matrix, matrix
  {
    Eigen::VectorXd result = stan::math::lub_offset_multiplier_constrain(
        x, lbd, ubd, offset, sigma);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::lub_offset_multiplier_constrain(
                                     x(i), lbd, ubd, offset(i), sigma(i)));
    }
    Eigen::VectorXd x_free = stan::math::lub_offset_multiplier_free(
        result, lbd, ubd, offset, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lbd, ubd, offset_bad, sigma_bad),
                 std::invalid_argument);
  }

  // matrix, real, matrix, matrix, matrix
  {
    Eigen::VectorXd result = stan::math::lub_offset_multiplier_constrain(
        x, lbd, ub, offset, sigma);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::lub_offset_multiplier_constrain(
                                     x(i), lbd, ub(i), offset(i), sigma(i)));
    }
    Eigen::VectorXd x_free = stan::math::lub_offset_multiplier_free(
        result, lbd, ub, offset, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lbd, ub_bad, offset_bad, sigma_bad),
                 std::invalid_argument);
  }

  // matrix, matrix, real, matrix, matrix
  {
    Eigen::VectorXd result = stan::math::lub_offset_multiplier_constrain(
        x, lb, ubd, offset, sigma);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::lub_offset_multiplier_constrain(
                                     x(i), lb(i), ubd, offset(i), sigma(i)));
    }
    Eigen::VectorXd x_free = stan::math::lub_offset_multiplier_free(
        result, lb, ubd, offset, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lb_bad, ubd, offset_bad, sigma_bad),
                 std::invalid_argument);
  }

  // matrix, matrix, matrix, real, matrix
  {
    Eigen::VectorXd result = stan::math::lub_offset_multiplier_constrain(
        x, lb, ub, offsetd, sigma);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::lub_offset_multiplier_constrain(
                                     x(i), lb(i), ub(i), offsetd, sigma(i)));
    }
    Eigen::VectorXd x_free = stan::math::lub_offset_multiplier_free(
        result, lb, ub, offsetd, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lb_bad, ub_bad, offsetd, sigma_bad),
                 std::invalid_argument);
  }

  // matrix, matrix, matrix, matrix, real
  {
    Eigen::VectorXd result = stan::math::lub_offset_multiplier_constrain(
        x, lb, ub, offset, sigmad);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::lub_offset_multiplier_constrain(
                                     x(i), lb(i), ub(i), offset(i), sigmad));
    }
    Eigen::VectorXd x_free = stan::math::lub_offset_multiplier_free(
        result, lb, ub, offset, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lb_bad, ub_bad, offset_bad, sigmad),
                 std::invalid_argument);
  }

  // matrix, matrix, matrix, matrix, matrix
  {
    Eigen::VectorXd result
        = stan::math::lub_offset_multiplier_constrain(x, lb, ub, offset, sigma);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::lub_offset_multiplier_constrain(
                                     x(i), lb(i), ub(i), offset(i), sigma(i)));
    }
    Eigen::VectorXd x_free
        = stan::math::lub_offset_multiplier_free(result, lb, ub, offset, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lb_bad, ub_bad, offset_bad, sigma_bad),
                 std::invalid_argument);
  }

  // matrix, real, real, real, real, lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    Eigen::VectorXd result = stan::math::lub_offset_multiplier_constrain(
        x, lbd, ubd, offsetd, sigmad, lp0);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::lub_offset_multiplier_constrain(
                                     x(i), lbd, ubd, offsetd, sigmad, lp1));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);
    Eigen::VectorXd x_free = stan::math::lub_offset_multiplier_free(
        result, lbd, ubd, offsetd, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
  }

  // matrix, matrix, real, real, real, lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    Eigen::VectorXd result = stan::math::lub_offset_multiplier_constrain(
        x, lb, ubd, offsetd, sigmad, lp0);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::lub_offset_multiplier_constrain(
                                     x(i), lb(i), ubd, offsetd, sigmad, lp1));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    Eigen::VectorXd x_free = stan::math::lub_offset_multiplier_free(
        result, lb, ubd, offsetd, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lb_bad, ubd, offsetd, sigmad, lp0),
                 std::invalid_argument);
  }

  // matrix, real, matrix, real, real, lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    Eigen::VectorXd result = stan::math::lub_offset_multiplier_constrain(
        x, lbd, ub, offsetd, sigmad);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::lub_offset_multiplier_constrain(
                                     x(i), lbd, ub(i), offsetd, sigmad));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    Eigen::VectorXd x_free = stan::math::lub_offset_multiplier_free(
        result, lbd, ub, offsetd, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lbd, ub_bad, offsetd, sigmad, lp0),
                 std::invalid_argument);
  }

  // matrix, real, real, matrix, real, lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    Eigen::VectorXd result = stan::math::lub_offset_multiplier_constrain(
        x, lbd, ubd, offset, sigmad);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::lub_offset_multiplier_constrain(
                                     x(i), lbd, ubd, offset(i), sigmad));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    Eigen::VectorXd x_free = stan::math::lub_offset_multiplier_free(
        result, lbd, ubd, offset, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lbd, ubd, offset_bad, sigmad, lp0),
                 std::invalid_argument);
  }

  // matrix, real, real, real, matrix, lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    Eigen::VectorXd result = stan::math::lub_offset_multiplier_constrain(
        x, lbd, ubd, offsetd, sigma);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::lub_offset_multiplier_constrain(
                                     x(i), lbd, ubd, offsetd, sigma(i)));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    Eigen::VectorXd x_free = stan::math::lub_offset_multiplier_free(
        result, lbd, ubd, offsetd, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lbd, ubd, offsetd, sigma_bad, lp0),
                 std::invalid_argument);
  }

  // matrix, matrix, matrix, real, real, lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    Eigen::VectorXd result = stan::math::lub_offset_multiplier_constrain(
        x, lb, ub, offsetd, sigmad);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::lub_offset_multiplier_constrain(
                                     x(i), lb(i), ub(i), offsetd, sigmad));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    Eigen::VectorXd x_free = stan::math::lub_offset_multiplier_free(
        result, lb, ub, offsetd, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lb_bad, ub_bad, offsetd, sigmad, lp0),
                 std::invalid_argument);
  }

  // matrix, matrix, real, matrix, real, lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    Eigen::VectorXd result = stan::math::lub_offset_multiplier_constrain(
        x, lb, ubd, offset, sigmad);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::lub_offset_multiplier_constrain(
                                     x(i), lb(i), ubd, offset(i), sigmad));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    Eigen::VectorXd x_free = stan::math::lub_offset_multiplier_free(
        result, lb, ubd, offset, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lb_bad, ubd, offset_bad, sigmad, lp0),
                 std::invalid_argument);
  }

  // matrix, matrix, real, real, matrix, lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    Eigen::VectorXd result = stan::math::lub_offset_multiplier_constrain(
        x, lb, ubd, offsetd, sigma);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::lub_offset_multiplier_constrain(
                                     x(i), lb(i), ubd, offsetd, sigma(i)));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    Eigen::VectorXd x_free = stan::math::lub_offset_multiplier_free(
        result, lb, ubd, offsetd, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lb_bad, ubd, offsetd, sigma_bad, lp0),
                 std::invalid_argument);
  }

  // matrix, real, matrix, matrix, real, lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    Eigen::VectorXd result = stan::math::lub_offset_multiplier_constrain(
        x, lbd, ub, offset, sigmad);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::lub_offset_multiplier_constrain(
                                     x(i), lbd, ub(i), offset(i), sigmad));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    Eigen::VectorXd x_free = stan::math::lub_offset_multiplier_free(
        result, lbd, ub, offset, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lbd, ub_bad, offset_bad, sigmad, lp0),
                 std::invalid_argument);
  }

  // matrix, real, matrix, real, matrix, lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    Eigen::VectorXd result = stan::math::lub_offset_multiplier_constrain(
        x, lbd, ub, offsetd, sigma);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::lub_offset_multiplier_constrain(
                                     x(i), lbd, ub(i), offsetd, sigma(i)));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    Eigen::VectorXd x_free = stan::math::lub_offset_multiplier_free(
        result, lbd, ub, offsetd, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lbd, ub_bad, offsetd, sigma_bad, lp0),
                 std::invalid_argument);
  }
  // matrix, real, real, matrix, matrix, lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    Eigen::VectorXd result = stan::math::lub_offset_multiplier_constrain(
        x, lbd, ubd, offset, sigma);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::lub_offset_multiplier_constrain(
                                     x(i), lbd, ubd, offset(i), sigma(i)));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    Eigen::VectorXd x_free = stan::math::lub_offset_multiplier_free(
        result, lbd, ubd, offset, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lbd, ubd, offset_bad, sigma_bad, lp0),
                 std::invalid_argument);
  }

  // matrix, real, matrix, matrix, matrix, lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    Eigen::VectorXd result = stan::math::lub_offset_multiplier_constrain(
        x, lbd, ub, offset, sigma);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::lub_offset_multiplier_constrain(
                                     x(i), lbd, ub(i), offset(i), sigma(i)));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    Eigen::VectorXd x_free = stan::math::lub_offset_multiplier_free(
        result, lbd, ub, offset, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lbd, ub_bad, offset_bad, sigma_bad, lp0),
                 std::invalid_argument);
  }

  // matrix, matrix, real, matrix, matrix, lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    Eigen::VectorXd result = stan::math::lub_offset_multiplier_constrain(
        x, lb, ubd, offset, sigma);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::lub_offset_multiplier_constrain(
                                     x(i), lb(i), ubd, offset(i), sigma(i)));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    Eigen::VectorXd x_free = stan::math::lub_offset_multiplier_free(
        result, lb, ubd, offset, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lb_bad, ubd, offset_bad, sigma_bad, lp0),
                 std::invalid_argument);
  }

  // matrix, matrix, matrix, real, matrix, lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    Eigen::VectorXd result = stan::math::lub_offset_multiplier_constrain(
        x, lb, ub, offsetd, sigma);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::lub_offset_multiplier_constrain(
                                     x(i), lb(i), ub(i), offsetd, sigma(i)));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    Eigen::VectorXd x_free = stan::math::lub_offset_multiplier_free(
        result, lb, ub, offsetd, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lb_bad, ub_bad, offsetd, sigma_bad, lp0),
                 std::invalid_argument);
  }

  // matrix, matrix, matrix, matrix, real, lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    Eigen::VectorXd result = stan::math::lub_offset_multiplier_constrain(
        x, lb, ub, offset, sigmad);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::lub_offset_multiplier_constrain(
                                     x(i), lb(i), ub(i), offset(i), sigmad));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    Eigen::VectorXd x_free = stan::math::lub_offset_multiplier_free(
        result, lb, ub, offset, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lb_bad, ub_bad, offset_bad, sigmad, lp0),
                 std::invalid_argument);
  }

  // matrix, matrix, matrix, matrix, matrix, lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    Eigen::VectorXd result
        = stan::math::lub_offset_multiplier_constrain(x, lb, ub, offset, sigma);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::lub_offset_multiplier_constrain(
                                     x(i), lb(i), ub(i), offset(i), sigma(i)));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    Eigen::VectorXd x_free
        = stan::math::lub_offset_multiplier_free(result, lb, ub, offset, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lb_bad, ub_bad, offset_bad, sigma_bad, lp0),
                 std::invalid_argument);
  }
}

TEST(prob_transform, lub_om_constrain_array) {
  std::vector<double> x{-1.0, 1.1, 3.0, 4.0};
  std::vector<double> ub{stan::math::INFTY, stan::math::INFTY, 6.0, 7.0};
  std::vector<double> lb{2.0, stan::math::NEGATIVE_INFTY,
                         stan::math::NEGATIVE_INFTY, 2.0};
  std::vector<double> sigma{1.1, 0.3, 6.0, 3.0};
  std::vector<double> offset{-2.0, 0.0, 0.2, 2.0};

  double sigmad = 3.0;
  double offsetd = -2.0;
  double ubd = 8;
  double lbd = -2;

  std::vector<double> offset_bad{-2, -3};
  std::vector<double> sigma_bad{8, 9};
  std::vector<double> lb_bad{-2, -3};
  std::vector<double> ub_bad{8, 9};

  // array[], real, real, real, real
  {
    auto result = stan::math::lub_offset_multiplier_constrain(x, lbd, ubd,
                                                              offsetd, sigmad);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::lub_offset_multiplier_constrain(
                                     x[i], lbd, ubd, offsetd, sigmad));
    }
    auto x_free = stan::math::lub_offset_multiplier_free(result, lbd, ubd,
                                                         offsetd, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
  }

  // array[], array[], real, real, real
  {
    auto result = stan::math::lub_offset_multiplier_constrain(x, lb, ubd,
                                                              offsetd, sigmad);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::lub_offset_multiplier_constrain(
                                     x[i], lb[i], ubd, offsetd, sigmad));
    }
    auto x_free = stan::math::lub_offset_multiplier_free(result, lb, ubd,
                                                         offsetd, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(x, lb_bad, ubd,
                                                             offsetd, sigmad),
                 std::invalid_argument);
  }

  // array[], real, array[], real, real
  {
    auto result = stan::math::lub_offset_multiplier_constrain(x, lbd, ub,
                                                              offsetd, sigmad);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::lub_offset_multiplier_constrain(
                                     x[i], lbd, ub[i], offsetd, sigmad));
    }
    auto x_free = stan::math::lub_offset_multiplier_free(result, lbd, ub,
                                                         offsetd, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(x, lbd, ub_bad,
                                                             offsetd, sigmad),
                 std::invalid_argument);
  }

  // array[], real, real, array[], real
  {
    auto result = stan::math::lub_offset_multiplier_constrain(x, lbd, ubd,
                                                              offset, sigmad);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::lub_offset_multiplier_constrain(
                                     x[i], lbd, ubd, offset[i], sigmad));
    }
    auto x_free = stan::math::lub_offset_multiplier_free(result, lbd, ubd,
                                                         offset, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lbd, ubd, offset_bad, sigmad),
                 std::invalid_argument);
  }

  // array[], real, real, real, array[]
  {
    auto result = stan::math::lub_offset_multiplier_constrain(x, lbd, ubd,
                                                              offsetd, sigma);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::lub_offset_multiplier_constrain(
                                     x[i], lbd, ubd, offsetd, sigma[i]));
    }
    auto x_free = stan::math::lub_offset_multiplier_free(result, lbd, ubd,
                                                         offsetd, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lbd, ubd, offsetd, sigma_bad),
                 std::invalid_argument);
  }

  // array[], array[], array[], real, real
  {
    auto result = stan::math::lub_offset_multiplier_constrain(x, lb, ub,
                                                              offsetd, sigmad);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::lub_offset_multiplier_constrain(
                                     x[i], lb[i], ub[i], offsetd, sigmad));
    }
    auto x_free = stan::math::lub_offset_multiplier_free(result, lb, ub,
                                                         offsetd, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(x, lb_bad, ub_bad,
                                                             offsetd, sigmad),
                 std::invalid_argument);
  }

  // array[], array[], real, array[], real
  {
    auto result = stan::math::lub_offset_multiplier_constrain(x, lb, ubd,
                                                              offset, sigmad);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::lub_offset_multiplier_constrain(
                                     x[i], lb[i], ubd, offset[i], sigmad));
    }
    auto x_free = stan::math::lub_offset_multiplier_free(result, lb, ubd,
                                                         offset, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lb_bad, ubd, offset_bad, sigmad),
                 std::invalid_argument);
  }

  // array[], array[], real, real, array[]
  {
    auto result = stan::math::lub_offset_multiplier_constrain(x, lb, ubd,
                                                              offsetd, sigma);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::lub_offset_multiplier_constrain(
                                     x[i], lb[i], ubd, offsetd, sigma[i]));
    }
    auto x_free = stan::math::lub_offset_multiplier_free(result, lb, ubd,
                                                         offsetd, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lb_bad, ubd, offsetd, sigma_bad),
                 std::invalid_argument);
  }

  // array[], real, array[], array[], real
  {
    auto result = stan::math::lub_offset_multiplier_constrain(x, lbd, ub,
                                                              offset, sigmad);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::lub_offset_multiplier_constrain(
                                     x[i], lbd, ub[i], offset[i], sigmad));
    }
    auto x_free = stan::math::lub_offset_multiplier_free(result, lbd, ub,
                                                         offset, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lbd, ub_bad, offset_bad, sigmad),
                 std::invalid_argument);
  }

  // array[], real, array[], real, array[]
  {
    auto result = stan::math::lub_offset_multiplier_constrain(x, lbd, ub,
                                                              offsetd, sigma);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::lub_offset_multiplier_constrain(
                                     x[i], lbd, ub[i], offsetd, sigma[i]));
    }
    auto x_free = stan::math::lub_offset_multiplier_free(result, lbd, ub,
                                                         offsetd, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lbd, ub_bad, offsetd, sigma_bad),
                 std::invalid_argument);
  }
  // array[], real, real, array[], array[]
  {
    auto result = stan::math::lub_offset_multiplier_constrain(x, lbd, ubd,
                                                              offset, sigma);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::lub_offset_multiplier_constrain(
                                     x[i], lbd, ubd, offset[i], sigma[i]));
    }
    auto x_free = stan::math::lub_offset_multiplier_free(result, lbd, ubd,
                                                         offset, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lbd, ubd, offset_bad, sigma_bad),
                 std::invalid_argument);
  }

  // array[], real, array[], array[], array[]
  {
    auto result = stan::math::lub_offset_multiplier_constrain(x, lbd, ub,
                                                              offset, sigma);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::lub_offset_multiplier_constrain(
                                     x[i], lbd, ub[i], offset[i], sigma[i]));
    }
    auto x_free = stan::math::lub_offset_multiplier_free(result, lbd, ub,
                                                         offset, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lbd, ub_bad, offset_bad, sigma_bad),
                 std::invalid_argument);
  }

  // array[], array[], real, array[], array[]
  {
    auto result = stan::math::lub_offset_multiplier_constrain(x, lb, ubd,
                                                              offset, sigma);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::lub_offset_multiplier_constrain(
                                     x[i], lb[i], ubd, offset[i], sigma[i]));
    }
    auto x_free = stan::math::lub_offset_multiplier_free(result, lb, ubd,
                                                         offset, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lb_bad, ubd, offset_bad, sigma_bad),
                 std::invalid_argument);
  }

  // array[], array[], array[], real, array[]
  {
    auto result = stan::math::lub_offset_multiplier_constrain(x, lb, ub,
                                                              offsetd, sigma);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::lub_offset_multiplier_constrain(
                                     x[i], lb[i], ub[i], offsetd, sigma[i]));
    }
    auto x_free = stan::math::lub_offset_multiplier_free(result, lb, ub,
                                                         offsetd, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lb_bad, ub_bad, offsetd, sigma_bad),
                 std::invalid_argument);
  }

  // array[], array[], array[], array[], real
  {
    auto result = stan::math::lub_offset_multiplier_constrain(x, lb, ub, offset,
                                                              sigmad);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::lub_offset_multiplier_constrain(
                                     x[i], lb[i], ub[i], offset[i], sigmad));
    }
    auto x_free = stan::math::lub_offset_multiplier_free(result, lb, ub, offset,
                                                         sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lb_bad, ub_bad, offset_bad, sigmad),
                 std::invalid_argument);
  }

  // array[], array[], array[], array[], array[]
  {
    auto result
        = stan::math::lub_offset_multiplier_constrain(x, lb, ub, offset, sigma);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::lub_offset_multiplier_constrain(
                                     x[i], lb[i], ub[i], offset[i], sigma[i]));
    }
    auto x_free
        = stan::math::lub_offset_multiplier_free(result, lb, ub, offset, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lb_bad, ub_bad, offset_bad, sigma_bad),
                 std::invalid_argument);
  }

  // array[], real, real, real, real, lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    auto result = stan::math::lub_offset_multiplier_constrain(
        x, lbd, ubd, offsetd, sigmad, lp0);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::lub_offset_multiplier_constrain(
                                     x[i], lbd, ubd, offsetd, sigmad, lp1));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);
    auto x_free = stan::math::lub_offset_multiplier_free(result, lbd, ubd,
                                                         offsetd, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
  }

  // array[], array[], real, real, real, lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    auto result = stan::math::lub_offset_multiplier_constrain(
        x, lb, ubd, offsetd, sigmad, lp0);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::lub_offset_multiplier_constrain(
                                     x[i], lb[i], ubd, offsetd, sigmad, lp1));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    auto x_free = stan::math::lub_offset_multiplier_free(result, lb, ubd,
                                                         offsetd, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lb_bad, ubd, offsetd, sigmad, lp0),
                 std::invalid_argument);
  }

  // array[], real, array[], real, real, lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    auto result = stan::math::lub_offset_multiplier_constrain(x, lbd, ub,
                                                              offsetd, sigmad);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::lub_offset_multiplier_constrain(
                                     x[i], lbd, ub[i], offsetd, sigmad));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    auto x_free = stan::math::lub_offset_multiplier_free(result, lbd, ub,
                                                         offsetd, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lbd, ub_bad, offsetd, sigmad, lp0),
                 std::invalid_argument);
  }

  // array[], real, real, array[], real, lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    auto result = stan::math::lub_offset_multiplier_constrain(x, lbd, ubd,
                                                              offset, sigmad);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::lub_offset_multiplier_constrain(
                                     x[i], lbd, ubd, offset[i], sigmad));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    auto x_free = stan::math::lub_offset_multiplier_free(result, lbd, ubd,
                                                         offset, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lbd, ubd, offset_bad, sigmad, lp0),
                 std::invalid_argument);
  }

  // array[], real, real, real, array[], lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    auto result = stan::math::lub_offset_multiplier_constrain(x, lbd, ubd,
                                                              offsetd, sigma);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::lub_offset_multiplier_constrain(
                                     x[i], lbd, ubd, offsetd, sigma[i]));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    auto x_free = stan::math::lub_offset_multiplier_free(result, lbd, ubd,
                                                         offsetd, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lbd, ubd, offsetd, sigma_bad, lp0),
                 std::invalid_argument);
  }

  // array[], array[], array[], real, real, lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    auto result = stan::math::lub_offset_multiplier_constrain(x, lb, ub,
                                                              offsetd, sigmad);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::lub_offset_multiplier_constrain(
                                     x[i], lb[i], ub[i], offsetd, sigmad));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    auto x_free = stan::math::lub_offset_multiplier_free(result, lb, ub,
                                                         offsetd, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lb_bad, ub_bad, offsetd, sigmad, lp0),
                 std::invalid_argument);
  }

  // array[], array[], real, array[], real, lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    auto result = stan::math::lub_offset_multiplier_constrain(x, lb, ubd,
                                                              offset, sigmad);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::lub_offset_multiplier_constrain(
                                     x[i], lb[i], ubd, offset[i], sigmad));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    auto x_free = stan::math::lub_offset_multiplier_free(result, lb, ubd,
                                                         offset, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lb_bad, ubd, offset_bad, sigmad, lp0),
                 std::invalid_argument);
  }

  // array[], array[], real, real, array[], lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    auto result = stan::math::lub_offset_multiplier_constrain(x, lb, ubd,
                                                              offsetd, sigma);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::lub_offset_multiplier_constrain(
                                     x[i], lb[i], ubd, offsetd, sigma[i]));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    auto x_free = stan::math::lub_offset_multiplier_free(result, lb, ubd,
                                                         offsetd, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lb_bad, ubd, offsetd, sigma_bad, lp0),
                 std::invalid_argument);
  }

  // array[], real, array[], array[], real, lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    auto result = stan::math::lub_offset_multiplier_constrain(x, lbd, ub,
                                                              offset, sigmad);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::lub_offset_multiplier_constrain(
                                     x[i], lbd, ub[i], offset[i], sigmad));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    auto x_free = stan::math::lub_offset_multiplier_free(result, lbd, ub,
                                                         offset, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lbd, ub_bad, offset_bad, sigmad, lp0),
                 std::invalid_argument);
  }

  // array[], real, array[], real, array[], lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    auto result = stan::math::lub_offset_multiplier_constrain(x, lbd, ub,
                                                              offsetd, sigma);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::lub_offset_multiplier_constrain(
                                     x[i], lbd, ub[i], offsetd, sigma[i]));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    auto x_free = stan::math::lub_offset_multiplier_free(result, lbd, ub,
                                                         offsetd, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lbd, ub_bad, offsetd, sigma_bad, lp0),
                 std::invalid_argument);
  }
  // array[], real, real, array[], array[], lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    auto result = stan::math::lub_offset_multiplier_constrain(x, lbd, ubd,
                                                              offset, sigma);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::lub_offset_multiplier_constrain(
                                     x[i], lbd, ubd, offset[i], sigma[i]));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    auto x_free = stan::math::lub_offset_multiplier_free(result, lbd, ubd,
                                                         offset, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lbd, ubd, offset_bad, sigma_bad, lp0),
                 std::invalid_argument);
  }

  // array[], real, array[], array[], array[], lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    auto result = stan::math::lub_offset_multiplier_constrain(x, lbd, ub,
                                                              offset, sigma);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::lub_offset_multiplier_constrain(
                                     x[i], lbd, ub[i], offset[i], sigma[i]));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    auto x_free = stan::math::lub_offset_multiplier_free(result, lbd, ub,
                                                         offset, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lbd, ub_bad, offset_bad, sigma_bad, lp0),
                 std::invalid_argument);
  }

  // array[], array[], real, array[], array[], lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    auto result = stan::math::lub_offset_multiplier_constrain(x, lb, ubd,
                                                              offset, sigma);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::lub_offset_multiplier_constrain(
                                     x[i], lb[i], ubd, offset[i], sigma[i]));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    auto x_free = stan::math::lub_offset_multiplier_free(result, lb, ubd,
                                                         offset, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lb_bad, ubd, offset_bad, sigma_bad, lp0),
                 std::invalid_argument);
  }

  // array[], array[], array[], real, array[], lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    auto result = stan::math::lub_offset_multiplier_constrain(x, lb, ub,
                                                              offsetd, sigma);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::lub_offset_multiplier_constrain(
                                     x[i], lb[i], ub[i], offsetd, sigma[i]));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    auto x_free = stan::math::lub_offset_multiplier_free(result, lb, ub,
                                                         offsetd, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lb_bad, ub_bad, offsetd, sigma_bad, lp0),
                 std::invalid_argument);
  }

  // array[], array[], array[], array[], real, lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    auto result = stan::math::lub_offset_multiplier_constrain(x, lb, ub, offset,
                                                              sigmad);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::lub_offset_multiplier_constrain(
                                     x[i], lb[i], ub[i], offset[i], sigmad));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    auto x_free = stan::math::lub_offset_multiplier_free(result, lb, ub, offset,
                                                         sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lb_bad, ub_bad, offset_bad, sigmad, lp0),
                 std::invalid_argument);
  }

  // array[], array[], array[], array[], array[], lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    auto result
        = stan::math::lub_offset_multiplier_constrain(x, lb, ub, offset, sigma);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::lub_offset_multiplier_constrain(
                                     x[i], lb[i], ub[i], offset[i], sigma[i]));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    auto x_free
        = stan::math::lub_offset_multiplier_free(result, lb, ub, offset, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(stan::math::lub_offset_multiplier_constrain(
                     x, lb_bad, ub_bad, offset_bad, sigma_bad, lp0),
                 std::invalid_argument);
  }
}

TEST(prob_transform, lub_om_exception) {
  using stan::math::lub_offset_multiplier_constrain;
  using stan::math::lub_offset_multiplier_free;

  EXPECT_THROW(lub_offset_multiplier_constrain(5.0, 1.0, 1.0, 0, -1),
               std::domain_error);
  EXPECT_THROW(lub_offset_multiplier_constrain(5.0, 1.0, 1.0, 0, 0),
               std::domain_error);

  EXPECT_NO_THROW(lub_offset_multiplier_constrain(5.0, 1.0, 1.01, 0, 1));
  double lp = 12;
  EXPECT_THROW(lub_offset_multiplier_constrain(5.0, 1.0, 1.0, 0, -1, lp),
               std::domain_error);
  EXPECT_NO_THROW(lub_offset_multiplier_constrain(5.0, 1.0, 1.01, 0, 1, lp));

  EXPECT_THROW(lub_offset_multiplier_constrain(5.0, 0.0, 10.0, 1.0, 0.0),
               std::domain_error);
  EXPECT_THROW(
      lub_offset_multiplier_constrain(
          5.0, 0.0, 10.0, std::numeric_limits<double>::infinity(), 1.0),
      std::domain_error);
  EXPECT_THROW(lub_offset_multiplier_constrain(5.0, 0.0, 10.0, NAN, 1.0),
               std::domain_error);
  EXPECT_NO_THROW(lub_offset_multiplier_constrain(5.0, 0, 2, 1.0, 0.01));
  EXPECT_THROW(lub_offset_multiplier_free(5.0, 0, 2, 1.0, 0.0),
               std::domain_error);
  EXPECT_THROW(
      lub_offset_multiplier_free(5.0, 9.0, 10.0,
                                 std::numeric_limits<double>::infinity(), 1.0),
      std::domain_error);
  EXPECT_THROW(lub_offset_multiplier_free(10.0, 0, 1, 4.0, 2.0),
               std::domain_error);
  EXPECT_THROW(lub_offset_multiplier_free(
                   1.0, -std::numeric_limits<double>::infinity(),
                   std::numeric_limits<double>::infinity(), 2.0, 0.0),
               std::domain_error);
  EXPECT_THROW(lub_offset_multiplier_free(
                   -10.0 - 0.1, -std::numeric_limits<double>::infinity(),
                   std::numeric_limits<double>::infinity(), -10.0, -27.0),
               std::domain_error);
  EXPECT_THROW(lub_offset_multiplier_free(5.0, 0.0, 10.0, NAN, 1.0),
               std::domain_error);
  EXPECT_NO_THROW(lub_offset_multiplier_free(5.0, 0.0, 10.0, 1.0, 0.01));
  EXPECT_THROW(lub_offset_multiplier_free(5.0, 11.0, 10.0, 1.0, 0.01),
               std::domain_error);

  lp = 12;
  EXPECT_THROW(lub_offset_multiplier_constrain(5.0, 0, 1, 1.0, 0.0, lp),
               std::domain_error);
  EXPECT_THROW(
      lub_offset_multiplier_constrain(
          5.0, 0, 10, std::numeric_limits<double>::infinity(), 1.0, lp),
      std::domain_error);
  EXPECT_THROW(lub_offset_multiplier_constrain(5.0, 0, 10, NAN, 1.0, lp),
               std::domain_error);
  EXPECT_NO_THROW(lub_offset_multiplier_constrain(5.0, 0, 2, 1.0, 0.01, lp));
}

TEST(prob_transform, lub_om_j) {
  for (double L : std::vector<double>{-1, 0.5, 2, 10}) {
    for (double U : std::vector<double>{0.5, 2, 5, 10}) {
      if (L >= U)
        continue;

      for (double O : std::vector<double>{-1, 0.5, 2, 10}) {
        for (double M : std::vector<double>{0.5, 1, 2, 10, 100}) {
          for (double x : std::vector<double>{-20, -15, 0.1, 3, 45.2}) {
            double lp = -17.0;

            EXPECT_FLOAT_EQ(
                L + (U - L) * stan::math::inv_logit(x * M + O),
                stan::math::lub_offset_multiplier_constrain(x, L, U, O, M, lp));

            EXPECT_FLOAT_EQ(-17.0 + log(U - L) + log(M)
                                + stan::math::log_inv_logit(x * M + O)
                                + stan::math::log1m_inv_logit(x * M + O),
                            lp);
          }
        }
      }
    }
  }
}

TEST(prob_transform, lub_om_f) {
  for (double L : std::vector<double>{-1, 0.5, 2, 10}) {
    for (double U : std::vector<double>{-1, 0.5, 2, 10}) {
      if (L >= U)
        continue;

      for (double O : std::vector<double>{-1, 0.5, 2, 10}) {
        for (double M : std::vector<double>{0.5, 2, 10}) {
          double y = L + U / 2;
          EXPECT_FLOAT_EQ(
              ((stan::math::logit((y - L) / (U - L)) - O) / M),
              stan::math::lub_offset_multiplier_free(y, L, U, O, M));
        }
      }
    }
  }
}

TEST(prob_transform, lub_om_rt) {
  double x = 5.0;
  double lb = -2.0;
  double ub = 8.0;
  double off = -2;
  double sigma = 3.0;
  double xc
      = stan::math::lub_offset_multiplier_constrain(x, lb, ub, off, sigma);
  double xcf = stan::math::lub_offset_multiplier_free(xc, lb, ub, off, sigma);
  EXPECT_FLOAT_EQ(x, xcf);
  double xcfc
      = stan::math::lub_offset_multiplier_constrain(xcf, lb, ub, off, sigma);
  EXPECT_FLOAT_EQ(xc, xcfc);
}
