#include <stan/math/prim.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>

TEST(prob_transform, ub_om) {
  for (double U : std::vector<double>{-1, 0.5, 2, 10}) {
    for (double O : std::vector<double>{-1, 0.5, 2, 10}) {
      for (double M : std::vector<double>{0.5, 2, 10}) {
        for (double x : std::vector<double>{-20, -15, 0.1, 3, 45.2}) {
          EXPECT_FLOAT_EQ(
              U - exp(x * M + O),
              stan::math::ub_offset_multiplier_constrain(x, U, O, M));
        }
      }
    }
  }
}

TEST(prob_transform, ub_om_underflow) {
  EXPECT_EQ(0, stan::math::ub_offset_multiplier_constrain(-1000, 0, 0, 1));
  double lp = 0;
  EXPECT_EQ(0, stan::math::ub_offset_multiplier_constrain(-1000, 0, 0, 1, lp));
  EXPECT_EQ(0, stan::math::ub_offset_multiplier_constrain(0, 0, -1000, 1));
  lp = 0;
  EXPECT_EQ(0, stan::math::ub_offset_multiplier_constrain(0, 0, -1000, 1, lp));
  EXPECT_EQ(0, stan::math::ub_offset_multiplier_constrain(-1, 0, 0, 1000));
  lp = 0;
  EXPECT_EQ(0, stan::math::ub_offset_multiplier_constrain(-1, 0, 0, 1000, lp));
}

TEST(prob_transform, ub_om_vec) {
  Eigen::VectorXd input(2);
  input << -1.0, 1.1;
  Eigen::VectorXd ubv(2);
  ubv << 2.0, 3.0;

  Eigen::VectorXd muv(2);
  muv << 2.0, 3.0;
  Eigen::VectorXd sigmav(2);
  sigmav << 2.0, 0.5;
  double ub = 2.0;
  double mu = 0.5;
  double sigma = 5.0;

  Eigen::VectorXd resvvv(2);
  resvvv << 2.0 - exp(-1.0 * 2.0 + 2.0), 3.0 - exp(1.1 * 0.5 + 3.0);
  Eigen::VectorXd ressvv(2);
  ressvv << 2.0 - exp(-1.0 * 2.0 + 2.0), 2.0 - exp(1.1 * 0.5 + 3.0);
  Eigen::VectorXd resvsv(2);
  resvsv << 2.0 - exp(-1.0 * 2.0 + 0.5), 3.0 - exp(1.1 * 0.5 + 0.5);
  Eigen::VectorXd resvvs(2);
  resvvs << 2.0 - exp(-1.0 * 5.0 + 2.0), 3.0 - exp(1.1 * 5.0 + 3.0);
  Eigen::VectorXd resssv(2);
  resssv << 2.0 - exp(-1.0 * 2.0 + 0.5), 2.0 - exp(1.1 * 0.5 + 0.5);
  Eigen::VectorXd ressvs(2);
  ressvs << 2.0 - exp(-1.0 * 5.0 + 2.0), 2.0 - exp(1.1 * 5.0 + 3.0);
  Eigen::VectorXd resvss(2);
  resvss << 2.0 - exp(-1.0 * 5.0 + 0.5), 3.0 - exp(1.1 * 5.0 + 0.5);
  Eigen::VectorXd res(2);
  res << 2.0 - exp(-1.0 * 5.0 + 0.5), 2.0 - exp(1.1 * 5.0 + 0.5);

  EXPECT_MATRIX_EQ(resvvv, stan::math::ub_offset_multiplier_constrain(
                               input, ubv, muv, sigmav));
  EXPECT_MATRIX_EQ(ressvv, stan::math::ub_offset_multiplier_constrain(
                               input, ub, muv, sigmav));
  EXPECT_MATRIX_EQ(resvsv, stan::math::ub_offset_multiplier_constrain(
                               input, ubv, mu, sigmav));
  EXPECT_MATRIX_EQ(resvvs, stan::math::ub_offset_multiplier_constrain(
                               input, ubv, muv, sigma));
  EXPECT_MATRIX_EQ(resssv, stan::math::ub_offset_multiplier_constrain(
                               input, ub, mu, sigmav));
  EXPECT_MATRIX_EQ(ressvs, stan::math::ub_offset_multiplier_constrain(
                               input, ub, muv, sigma));
  EXPECT_MATRIX_EQ(resvss, stan::math::ub_offset_multiplier_constrain(
                               input, ubv, mu, sigma));
  EXPECT_MATRIX_EQ(
      res, stan::math::ub_offset_multiplier_constrain(input, ub, mu, sigma));

  double lp = 0.0;
  EXPECT_MATRIX_EQ(resvvv, stan::math::ub_offset_multiplier_constrain(
                               input, ubv, muv, sigmav, lp));
  EXPECT_FLOAT_EQ(
      (muv.array() + (sigmav.array() * input.array()) + sigmav.array().log())
          .sum(),
      lp);
  lp = 0.0;
  EXPECT_MATRIX_EQ(ressvv, stan::math::ub_offset_multiplier_constrain(
                               input, ub, muv, sigmav, lp));
  EXPECT_FLOAT_EQ(
      (muv.array() + (sigmav.array() * input.array()) + sigmav.array().log())
          .sum(),
      lp);
  lp = 0.0;
  EXPECT_MATRIX_EQ(resvsv, stan::math::ub_offset_multiplier_constrain(
                               input, ubv, mu, sigmav, lp));
  EXPECT_FLOAT_EQ(
      (mu + (sigmav.array() * input.array()) + sigmav.array().log()).sum(), lp);
  lp = 0.0;
  EXPECT_MATRIX_EQ(resvvs, stan::math::ub_offset_multiplier_constrain(
                               input, ubv, muv, sigma, lp));
  EXPECT_FLOAT_EQ(
      (muv.array() + (sigma * input.array()) + std::log(sigma)).sum(), lp);
  lp = 0.0;
  EXPECT_MATRIX_EQ(resssv, stan::math::ub_offset_multiplier_constrain(
                               input, ub, mu, sigmav, lp));
  EXPECT_FLOAT_EQ(
      (mu + (sigmav.array() * input.array()) + sigmav.array().log()).sum(), lp);
  lp = 0.0;
  EXPECT_MATRIX_EQ(ressvs, stan::math::ub_offset_multiplier_constrain(
                               input, ub, muv, sigma, lp));
  EXPECT_FLOAT_EQ(
      (muv.array() + (sigma * input.array()) + std::log(sigma)).sum(), lp);
  lp = 0.0;
  EXPECT_MATRIX_EQ(resvss, stan::math::ub_offset_multiplier_constrain(
                               input, ubv, mu, sigma, lp));
  EXPECT_FLOAT_EQ((mu + (sigma * input.array()) + std::log(sigma)).sum(), lp);
  lp = 0.0;
  EXPECT_MATRIX_EQ(res, stan::math::ub_offset_multiplier_constrain(
                            input, ub, mu, sigma, lp));
  EXPECT_FLOAT_EQ((mu + (sigma * input.array()) + std::log(sigma)).sum(), lp);
}

TEST(prob_transform, ub_om_constrain_matrix) {
  Eigen::VectorXd x(4);
  x << -1.0, 1.1, 3.0, 4.0;

  Eigen::VectorXd ub(4);
  ub << stan::math::INFTY, stan::math::INFTY, 6.0, 7.0;
  Eigen::VectorXd sigma(4);
  sigma << 1.1, 0.3, 6.0, 3.0;
  Eigen::VectorXd offset(4);
  offset << -2.0, 0.0, 0.2, 2.0;

  double sigmad = 3.0;
  double offsetd = -2.0;
  double ubd = 8;

  Eigen::VectorXd sigma_bad(3);
  Eigen::VectorXd offset_bad(3);
  Eigen::VectorXd ub_bad(3);

  // matrix, real, real, real
  {
    Eigen::VectorXd result
        = stan::math::ub_offset_multiplier_constrain(x, ubd, offsetd, sigmad);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::ub_offset_multiplier_constrain(
                                     x(i), ubd, offsetd, sigmad));
    }
    Eigen::VectorXd x_free
        = stan::math::ub_offset_multiplier_free(result, ubd, offsetd, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
  }

  // matrix, matrix, real, real
  {
    Eigen::VectorXd result
        = stan::math::ub_offset_multiplier_constrain(x, ub, offsetd, sigmad);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::ub_offset_multiplier_constrain(
                                     x(i), ub(i), offsetd, sigmad));
    }
    Eigen::VectorXd x_free
        = stan::math::ub_offset_multiplier_free(result, ub, offsetd, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(
        stan::math::ub_offset_multiplier_constrain(x, ub_bad, offsetd, sigmad),
        std::invalid_argument);
  }
  // matrix, real, matrix, real
  {
    Eigen::VectorXd result
        = stan::math::ub_offset_multiplier_constrain(x, ubd, offset, sigmad);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::ub_offset_multiplier_constrain(
                                     x(i), ubd, offset(i), sigmad));
    }
    Eigen::VectorXd x_free
        = stan::math::ub_offset_multiplier_free(result, ubd, offset, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(
        stan::math::ub_offset_multiplier_constrain(x, ubd, offset_bad, sigmad),
        std::invalid_argument);
  }

  // matrix, real, real, matrix
  {
    Eigen::VectorXd result
        = stan::math::ub_offset_multiplier_constrain(x, ubd, offsetd, sigma);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::ub_offset_multiplier_constrain(
                                     x(i), ubd, offsetd, sigma(i)));
    }
    Eigen::VectorXd x_free
        = stan::math::ub_offset_multiplier_free(result, ubd, offsetd, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(
        stan::math::ub_offset_multiplier_constrain(x, ubd, offsetd, sigma_bad),
        std::invalid_argument);
  }

  // matrix, matrix, matrix, real
  {
    Eigen::VectorXd result
        = stan::math::ub_offset_multiplier_constrain(x, ub, offset, sigmad);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::ub_offset_multiplier_constrain(
                                     x(i), ub(i), offset(i), sigmad));
    }
    Eigen::VectorXd x_free
        = stan::math::ub_offset_multiplier_free(result, ub, offset, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(x, ub_bad,
                                                            offset_bad, sigmad),
                 std::invalid_argument);
  }

  // matrix, matrix, real, matrix
  {
    Eigen::VectorXd result
        = stan::math::ub_offset_multiplier_constrain(x, ub, offsetd, sigma);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::ub_offset_multiplier_constrain(
                                     x(i), ub(i), offsetd, sigma(i)));
    }
    Eigen::VectorXd x_free
        = stan::math::ub_offset_multiplier_free(result, ub, offsetd, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(x, ub_bad, offsetd,
                                                            sigma_bad),
                 std::invalid_argument);
  }

  // matrix, real, matrix, matrix
  {
    Eigen::VectorXd result
        = stan::math::ub_offset_multiplier_constrain(x, ubd, offset, sigma);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::ub_offset_multiplier_constrain(
                                     x(i), ubd, offset(i), sigma(i)));
    }
    Eigen::VectorXd x_free
        = stan::math::ub_offset_multiplier_free(result, ubd, offset, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(x, ubd, offset_bad,
                                                            sigma_bad),
                 std::invalid_argument);
  }

  // matrix, matrix, matrix, matrix
  {
    Eigen::VectorXd result
        = stan::math::ub_offset_multiplier_constrain(x, ub, offset, sigma);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::ub_offset_multiplier_constrain(
                                     x(i), ub(i), offset(i), sigma(i)));
    }
    Eigen::VectorXd x_free
        = stan::math::ub_offset_multiplier_free(result, ub, offset, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(
                     x, ub_bad, offset_bad, sigma_bad),
                 std::invalid_argument);
  }

  // matrix, real, real, real, lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    Eigen::VectorXd result = stan::math::ub_offset_multiplier_constrain(
        x, ubd, offsetd, sigmad, lp0);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::ub_offset_multiplier_constrain(
                                     x(i), ubd, offsetd, sigmad, lp1));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);
    Eigen::VectorXd x_free
        = stan::math::ub_offset_multiplier_free(result, ubd, offsetd, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
  }

  // matrix, matrix, real, real, lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    Eigen::VectorXd result = stan::math::ub_offset_multiplier_constrain(
        x, ub, offsetd, sigmad, lp0);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::ub_offset_multiplier_constrain(
                                     x(i), ub(i), offsetd, sigmad, lp1));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    Eigen::VectorXd x_free
        = stan::math::ub_offset_multiplier_free(result, ub, offsetd, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(x, ub_bad, offsetd,
                                                            sigmad, lp0),
                 std::invalid_argument);
  }
  // matrix, real, matrix, real, lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    Eigen::VectorXd result = stan::math::ub_offset_multiplier_constrain(
        x, ubd, offset, sigmad, lp0);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::ub_offset_multiplier_constrain(
                                     x(i), ubd, offset(i), sigmad, lp1));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    Eigen::VectorXd x_free
        = stan::math::ub_offset_multiplier_free(result, ubd, offset, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(x, ubd, offset_bad,
                                                            sigmad, lp0),
                 std::invalid_argument);
  }

  // matrix, real, real, matrix, lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    Eigen::VectorXd result = stan::math::ub_offset_multiplier_constrain(
        x, ubd, offsetd, sigma, lp0);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::ub_offset_multiplier_constrain(
                                     x(i), ubd, offsetd, sigma(i), lp1));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    Eigen::VectorXd x_free
        = stan::math::ub_offset_multiplier_free(result, ubd, offsetd, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(x, ubd, offsetd,
                                                            sigma_bad, lp0),
                 std::invalid_argument);
  }

  // matrix, matrix, matrix, real, lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    Eigen::VectorXd result = stan::math::ub_offset_multiplier_constrain(
        x, ub, offset, sigmad, lp0);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::ub_offset_multiplier_constrain(
                                     x(i), ub(i), offset(i), sigmad, lp1));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    Eigen::VectorXd x_free
        = stan::math::ub_offset_multiplier_free(result, ub, offset, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(
                     x, ub_bad, offset_bad, sigmad, lp0),
                 std::invalid_argument);
  }

  // matrix, matrix, real, matrix, lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    Eigen::VectorXd result = stan::math::ub_offset_multiplier_constrain(
        x, ub, offsetd, sigma, lp0);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::ub_offset_multiplier_constrain(
                                     x(i), ub(i), offsetd, sigma(i), lp1));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    Eigen::VectorXd x_free
        = stan::math::ub_offset_multiplier_free(result, ub, offsetd, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(x, ub_bad, offsetd,
                                                            sigma_bad, lp0),
                 std::invalid_argument);
  }

  // matrix, real, matrix, matrix, lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    Eigen::VectorXd result = stan::math::ub_offset_multiplier_constrain(
        x, ubd, offset, sigma, lp0);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::ub_offset_multiplier_constrain(
                                     x(i), ubd, offset(i), sigma(i), lp1));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    Eigen::VectorXd x_free
        = stan::math::ub_offset_multiplier_free(result, ubd, offset, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(x, ubd, offset_bad,
                                                            sigma_bad),
                 std::invalid_argument);
  }

  // matrix, matrix, matrix, matrix, lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    Eigen::VectorXd result
        = stan::math::ub_offset_multiplier_constrain(x, ub, offset, sigma, lp0);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result(i), stan::math::ub_offset_multiplier_constrain(
                                     x(i), ub(i), offset(i), sigma(i), lp1));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    Eigen::VectorXd x_free
        = stan::math::ub_offset_multiplier_free(result, ub, offset, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x.coeff(i), x_free.coeff(i));
    }
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(
                     x, ub_bad, offset_bad, sigma_bad),
                 std::invalid_argument);
  }
}

TEST(prob_transform, ub_om_constrain_array) {
  std::vector<double> x{-1.0, 1.1, 3.0, 4.0};
  std::vector<double> ub{stan::math::INFTY, stan::math::INFTY, 6.0, 7.0};

  std::vector<double> sigma{1.1, 0.3, 6.0, 3.0};
  std::vector<double> offset{-2.0, 0.0, 0.2, 2.0};

  double sigmad = 3.0;
  double offsetd = -2.0;
  double ubd = 8;

  std::vector<double> offset_bad{-2, -3};
  std::vector<double> sigma_bad{8, 9};
  std::vector<double> ub_bad{8, 9};

  // array[], real, real, real
  {
    auto result
        = stan::math::ub_offset_multiplier_constrain(x, ubd, offsetd, sigmad);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::ub_offset_multiplier_constrain(
                                     x[i], ubd, offsetd, sigmad));
    }
    auto x_free
        = stan::math::ub_offset_multiplier_free(result, ubd, offsetd, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
  }

  // array[], array[], real, real
  {
    auto result
        = stan::math::ub_offset_multiplier_constrain(x, ub, offsetd, sigmad);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::ub_offset_multiplier_constrain(
                                     x[i], ub[i], offsetd, sigmad));
    }
    auto x_free
        = stan::math::ub_offset_multiplier_free(result, ub, offsetd, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(
        stan::math::ub_offset_multiplier_constrain(x, ub_bad, offsetd, sigmad),
        std::invalid_argument);
  }
  // array[], real, array[], real
  {
    auto result
        = stan::math::ub_offset_multiplier_constrain(x, ubd, offset, sigmad);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::ub_offset_multiplier_constrain(
                                     x[i], ubd, offset[i], sigmad));
    }
    auto x_free
        = stan::math::ub_offset_multiplier_free(result, ubd, offset, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(
        stan::math::ub_offset_multiplier_constrain(x, ubd, offset_bad, sigmad),
        std::invalid_argument);
  }

  // array[], real, real, array[]
  {
    auto result
        = stan::math::ub_offset_multiplier_constrain(x, ubd, offsetd, sigma);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::ub_offset_multiplier_constrain(
                                     x[i], ubd, offsetd, sigma[i]));
    }
    auto x_free
        = stan::math::ub_offset_multiplier_free(result, ubd, offsetd, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(
        stan::math::ub_offset_multiplier_constrain(x, ubd, offsetd, sigma_bad),
        std::invalid_argument);
  }

  // array[], array[], array[], real
  {
    auto result
        = stan::math::ub_offset_multiplier_constrain(x, ub, offset, sigmad);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::ub_offset_multiplier_constrain(
                                     x[i], ub[i], offset[i], sigmad));
    }
    auto x_free
        = stan::math::ub_offset_multiplier_free(result, ub, offset, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(x, ub_bad,
                                                            offset_bad, sigmad),
                 std::invalid_argument);
  }

  // array[], array[], real, array[]
  {
    auto result
        = stan::math::ub_offset_multiplier_constrain(x, ub, offsetd, sigma);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::ub_offset_multiplier_constrain(
                                     x[i], ub[i], offsetd, sigma[i]));
    }
    auto x_free
        = stan::math::ub_offset_multiplier_free(result, ub, offsetd, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(x, ub_bad, offsetd,
                                                            sigma_bad),
                 std::invalid_argument);
  }

  // array[], real, array[], array[]
  {
    auto result
        = stan::math::ub_offset_multiplier_constrain(x, ubd, offset, sigma);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::ub_offset_multiplier_constrain(
                                     x[i], ubd, offset[i], sigma[i]));
    }
    auto x_free
        = stan::math::ub_offset_multiplier_free(result, ubd, offset, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(x, ubd, offset_bad,
                                                            sigma_bad),
                 std::invalid_argument);
  }

  // array[], array[], array[], array[]
  {
    auto result
        = stan::math::ub_offset_multiplier_constrain(x, ub, offset, sigma);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::ub_offset_multiplier_constrain(
                                     x[i], ub[i], offset[i], sigma[i]));
    }
    auto x_free
        = stan::math::ub_offset_multiplier_free(result, ub, offset, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(
                     x, ub_bad, offset_bad, sigma_bad),
                 std::invalid_argument);
  }

  // array[], real, real, real, lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    auto result = stan::math::ub_offset_multiplier_constrain(x, ubd, offsetd,
                                                             sigmad, lp0);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::ub_offset_multiplier_constrain(
                                     x[i], ubd, offsetd, sigmad, lp1));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);
    auto x_free
        = stan::math::ub_offset_multiplier_free(result, ubd, offsetd, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
  }

  // array[], array[], real, real, lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    auto result = stan::math::ub_offset_multiplier_constrain(x, ub, offsetd,
                                                             sigmad, lp0);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::ub_offset_multiplier_constrain(
                                     x[i], ub[i], offsetd, sigmad, lp1));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    auto x_free
        = stan::math::ub_offset_multiplier_free(result, ub, offsetd, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(x, ub_bad, offsetd,
                                                            sigmad, lp0),
                 std::invalid_argument);
  }
  // array[], real, array[], real, lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    auto result = stan::math::ub_offset_multiplier_constrain(x, ubd, offset,
                                                             sigmad, lp0);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::ub_offset_multiplier_constrain(
                                     x[i], ubd, offset[i], sigmad, lp1));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    auto x_free
        = stan::math::ub_offset_multiplier_free(result, ubd, offset, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(x, ubd, offset_bad,
                                                            sigmad, lp0),
                 std::invalid_argument);
  }

  // array[], real, real, array[], lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    auto result = stan::math::ub_offset_multiplier_constrain(x, ubd, offsetd,
                                                             sigma, lp0);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::ub_offset_multiplier_constrain(
                                     x[i], ubd, offsetd, sigma[i], lp1));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    auto x_free
        = stan::math::ub_offset_multiplier_free(result, ubd, offsetd, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(x, ubd, offsetd,
                                                            sigma_bad, lp0),
                 std::invalid_argument);
  }

  // array[], array[], array[], real, lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    auto result = stan::math::ub_offset_multiplier_constrain(x, ub, offset,
                                                             sigmad, lp0);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::ub_offset_multiplier_constrain(
                                     x[i], ub[i], offset[i], sigmad, lp1));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    auto x_free
        = stan::math::ub_offset_multiplier_free(result, ub, offset, sigmad);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(
                     x, ub_bad, offset_bad, sigmad, lp0),
                 std::invalid_argument);
  }

  // array[], array[], real, array[], lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    auto result = stan::math::ub_offset_multiplier_constrain(x, ub, offsetd,
                                                             sigma, lp0);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::ub_offset_multiplier_constrain(
                                     x[i], ub[i], offsetd, sigma[i], lp1));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    auto x_free
        = stan::math::ub_offset_multiplier_free(result, ub, offsetd, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(x, ub_bad, offsetd,
                                                            sigma_bad, lp0),
                 std::invalid_argument);
  }

  // array[], real, array[], array[], lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    auto result = stan::math::ub_offset_multiplier_constrain(x, ubd, offset,
                                                             sigma, lp0);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::ub_offset_multiplier_constrain(
                                     x[i], ubd, offset[i], sigma[i], lp1));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    auto x_free
        = stan::math::ub_offset_multiplier_free(result, ubd, offset, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(x, ubd, offset_bad,
                                                            sigma_bad),
                 std::invalid_argument);
  }

  // array[], array[], array[], array[], lp
  {
    double lp0 = 0.0;
    double lp1 = 0.0;
    auto result
        = stan::math::ub_offset_multiplier_constrain(x, ub, offset, sigma, lp0);
    for (size_t i = 0; i < result.size(); ++i) {
      EXPECT_FLOAT_EQ(result[i], stan::math::ub_offset_multiplier_constrain(
                                     x[i], ub[i], offset[i], sigma[i], lp1));
    }
    EXPECT_FLOAT_EQ(lp0, lp1);

    auto x_free
        = stan::math::ub_offset_multiplier_free(result, ub, offset, sigma);
    for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_FLOAT_EQ(x[i], x_free[i]);
    }
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(
                     x, ub_bad, offset_bad, sigma_bad),
                 std::invalid_argument);
  }
}

TEST(prob_transform, ub_om_constrain_matrix_array) {
  Eigen::VectorXd x(4);
  x << -1.0, 1.1, 3.0, 4.0;
  Eigen::VectorXd bad_x(4);
  bad_x << 3, 3, 3, 3;
  Eigen::VectorXd ub(4);
  ub << stan::math::INFTY, stan::math::INFTY, 6.0, 7.0;
  Eigen::VectorXd sigma(4);
  sigma << 1.1, 0.3, 6.0, 3.0;
  Eigen::VectorXd offset(4);
  offset << -2.0, 0.0, 0.2, 2.0;

  double sigmad = 3.0;
  double offsetd = -2.0;
  double ubd = 8;

  Eigen::VectorXd sigma_bad(3);
  Eigen::VectorXd offset_bad(3);
  Eigen::VectorXd ub_bad(3);
  std::vector<Eigen::VectorXd> x_vec{x, x};
  std::vector<Eigen::VectorXd> ub_vec{ub, ub};
  std::vector<Eigen::VectorXd> sigma_vec{sigma, sigma};
  std::vector<Eigen::VectorXd> offset_vec{offset, offset};
  std::vector<Eigen::VectorXd> x_bad_vec{bad_x, bad_x};

  std::vector<Eigen::VectorXd> sigma_bad_vec{sigma_bad, sigma_bad, sigma_bad};
  std::vector<Eigen::VectorXd> offset_bad_vec{offset_bad, offset_bad,
                                              sigma_bad};
  std::vector<Eigen::VectorXd> ub_bad_vec{ub_bad, ub_bad, ub_bad};

  // array[] matrix, array[] matrix, array[] matrix, array[] matrix
  {
    auto result = stan::math::ub_offset_multiplier_constrain(
        x_vec, ub_vec, offset_vec, sigma_vec);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(
            result[i](j),
            stan::math::ub_offset_multiplier_constrain(
                x_vec[i](j), ub_vec[i](j), offset_vec[i](j), sigma_vec[i](j)));
      }
    }
    auto x_vec_free = stan::math::ub_offset_multiplier_free(
        result, ub_vec, offset_vec, sigma_vec);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(x_vec[i].coeff(j), x_vec_free[i].coeff(j));
      }
    }
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(
                     x_bad_vec, ub_bad_vec, offset_vec, sigma_vec),
                 std::invalid_argument);
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(
                     x_bad_vec, ub_vec, offset_bad_vec, sigma_vec),
                 std::invalid_argument);
    EXPECT_THROW(stan::math::ub_offset_multiplier_free(
                     x_bad_vec, ub_vec, offset_bad_vec, sigma_vec),
                 std::invalid_argument);
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(
                     x_bad_vec, ub_vec, offset_vec, sigma_bad_vec),
                 std::invalid_argument);
    EXPECT_THROW(stan::math::ub_offset_multiplier_free(
                     x_bad_vec, ub_vec, offset_vec, sigma_bad_vec),
                 std::invalid_argument);
  }
  // array[] matrix, array[] matrix, array[] matrix, matrix
  {
    auto result = stan::math::ub_offset_multiplier_constrain(x_vec, ub_vec,
                                                             offset_vec, sigma);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(
            result[i](j),
            stan::math::ub_offset_multiplier_constrain(
                x_vec[i](j), ub_vec[i](j), offset_vec[i](j), sigma(j)));
      }
    }
    auto x_vec_free = stan::math::ub_offset_multiplier_free(result, ub_vec,
                                                            offset_vec, sigma);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(x_vec[i].coeff(j), x_vec_free[i].coeff(j));
      }
    }
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(
                     x_bad_vec, ub_bad_vec, offset_vec, sigma),
                 std::invalid_argument);
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(
                     x_bad_vec, ub_vec, offset_bad_vec, sigma),
                 std::invalid_argument);
    EXPECT_THROW(stan::math::ub_offset_multiplier_free(x_bad_vec, ub_vec,
                                                       offset_bad_vec, sigma),
                 std::invalid_argument);
  }
  // array[] matrix, array[] matrix, array[] matrix, real
  {
    auto result = stan::math::ub_offset_multiplier_constrain(
        x_vec, ub_vec, offset_vec, sigmad);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(
            result[i](j),
            stan::math::ub_offset_multiplier_constrain(
                x_vec[i](j), ub_vec[i](j), offset_vec[i](j), sigmad));
      }
    }
    auto x_vec_free = stan::math::ub_offset_multiplier_free(result, ub_vec,
                                                            offset_vec, sigmad);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(x_vec[i].coeff(j), x_vec_free[i].coeff(j));
      }
    }
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(
                     x_bad_vec, ub_bad_vec, offset_vec, sigmad),
                 std::invalid_argument);
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(
                     x_bad_vec, ub_vec, offset_bad_vec, sigmad),
                 std::invalid_argument);
    EXPECT_THROW(stan::math::ub_offset_multiplier_free(x_bad_vec, ub_vec,
                                                       offset_bad_vec, sigmad),
                 std::invalid_argument);
  }
  // array[] matrix, array[] matrix, matrix, array[] matrix
  {
    auto result = stan::math::ub_offset_multiplier_constrain(x_vec, ub_vec,
                                                             offset, sigma_vec);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(
            result[i](j),
            stan::math::ub_offset_multiplier_constrain(
                x_vec[i](j), ub_vec[i](j), offset(j), sigma_vec[i](j)));
      }
    }
    auto x_vec_free = stan::math::ub_offset_multiplier_free(result, ub_vec,
                                                            offset, sigma_vec);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(x_vec[i].coeff(j), x_vec_free[i].coeff(j));
      }
    }
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(
                     x_bad_vec, ub_bad_vec, offset, sigma_vec),
                 std::invalid_argument);
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(
                     x_bad_vec, ub_vec, offset, sigma_bad_vec),
                 std::invalid_argument);
    EXPECT_THROW(stan::math::ub_offset_multiplier_free(x_bad_vec, ub_vec,
                                                       offset, sigma_bad_vec),
                 std::invalid_argument);
  }
  // array[] matrix, array[] matrix, matrix, matrix
  {
    auto result = stan::math::ub_offset_multiplier_constrain(x_vec, ub_vec,
                                                             offset, sigma);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(result[i](j),
                        stan::math::ub_offset_multiplier_constrain(
                            x_vec[i](j), ub_vec[i](j), offset(j), sigma(j)));
      }
    }
    auto x_vec_free
        = stan::math::ub_offset_multiplier_free(result, ub_vec, offset, sigma);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(x_vec[i].coeff(j), x_vec_free[i].coeff(j));
      }
    }
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(
                     x_bad_vec, ub_bad_vec, offset, sigma),
                 std::invalid_argument);
  }
  // array[] matrix, array[] matrix, matrix, real
  {
    auto result = stan::math::ub_offset_multiplier_constrain(x_vec, ub_vec,
                                                             offset, sigmad);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(result[i](j),
                        stan::math::ub_offset_multiplier_constrain(
                            x_vec[i](j), ub_vec[i](j), offset(j), sigmad));
      }
    }
    auto x_vec_free
        = stan::math::ub_offset_multiplier_free(result, ub_vec, offset, sigmad);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(x_vec[i].coeff(j), x_vec_free[i].coeff(j));
      }
    }
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(
                     x_bad_vec, ub_bad_vec, offset, sigmad),
                 std::invalid_argument);
  }
  // array[] matrix, array[] matrix, real, array[] matrix
  {
    auto result = stan::math::ub_offset_multiplier_constrain(
        x_vec, ub_vec, offsetd, sigma_vec);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(
            result[i](j),
            stan::math::ub_offset_multiplier_constrain(
                x_vec[i](j), ub_vec[i](j), offsetd, sigma_vec[i](j)));
      }
    }
    auto x_vec_free = stan::math::ub_offset_multiplier_free(result, ub_vec,
                                                            offsetd, sigma_vec);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(x_vec[i].coeff(j), x_vec_free[i].coeff(j));
      }
    }
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(
                     x_bad_vec, ub_bad_vec, offsetd, sigma_vec),
                 std::invalid_argument);
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(
                     x_bad_vec, ub_vec, offsetd, sigma_bad_vec),
                 std::invalid_argument);
    EXPECT_THROW(stan::math::ub_offset_multiplier_free(x_bad_vec, ub_vec,
                                                       offsetd, sigma_bad_vec),
                 std::invalid_argument);
  }
  // array[] matrix, array[] matrix, real, matrix
  {
    auto result = stan::math::ub_offset_multiplier_constrain(x_vec, ub_vec,
                                                             offsetd, sigma);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(result[i](j),
                        stan::math::ub_offset_multiplier_constrain(
                            x_vec[i](j), ub_vec[i](j), offsetd, sigma(j)));
      }
    }
    auto x_vec_free
        = stan::math::ub_offset_multiplier_free(result, ub_vec, offsetd, sigma);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(x_vec[i].coeff(j), x_vec_free[i].coeff(j));
      }
    }
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(
                     x_bad_vec, ub_bad_vec, offsetd, sigma),
                 std::invalid_argument);
  }
  // array[] matrix, array[] matrix, real, real
  {
    auto result = stan::math::ub_offset_multiplier_constrain(x_vec, ub_vec,
                                                             offsetd, sigmad);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(result[i](j),
                        stan::math::ub_offset_multiplier_constrain(
                            x_vec[i](j), ub_vec[i](j), offsetd, sigmad));
      }
    }
    auto x_vec_free = stan::math::ub_offset_multiplier_free(result, ub_vec,
                                                            offsetd, sigmad);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(x_vec[i].coeff(j), x_vec_free[i].coeff(j));
      }
    }
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(
                     x_bad_vec, ub_bad_vec, offsetd, sigmad),
                 std::invalid_argument);
  }
  // array[] matrix, matrix, array[] matrix, array[] matrix
  {
    auto result = stan::math::ub_offset_multiplier_constrain(
        x_vec, ub, offset_vec, sigma_vec);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(
            result[i](j),
            stan::math::ub_offset_multiplier_constrain(
                x_vec[i](j), ub(j), offset_vec[i](j), sigma_vec[i](j)));
      }
    }
    auto x_vec_free = stan::math::ub_offset_multiplier_free(
        result, ub, offset_vec, sigma_vec);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(x_vec[i].coeff(j), x_vec_free[i].coeff(j));
      }
    }
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(
                     x_bad_vec, ub, offset_bad_vec, sigma_vec),
                 std::invalid_argument);
    EXPECT_THROW(stan::math::ub_offset_multiplier_free(
                     x_bad_vec, ub, offset_bad_vec, sigma_vec),
                 std::invalid_argument);
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(
                     x_bad_vec, ub, offset_vec, sigma_bad_vec),
                 std::invalid_argument);
    EXPECT_THROW(stan::math::ub_offset_multiplier_free(
                     x_bad_vec, ub, offset_vec, sigma_bad_vec),
                 std::invalid_argument);
  }
  // array[] matrix, matrix, array[] matrix, matrix
  {
    auto result = stan::math::ub_offset_multiplier_constrain(x_vec, ub,
                                                             offset_vec, sigma);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(result[i](j),
                        stan::math::ub_offset_multiplier_constrain(
                            x_vec[i](j), ub(j), offset_vec[i](j), sigma(j)));
      }
    }
    auto x_vec_free
        = stan::math::ub_offset_multiplier_free(result, ub, offset_vec, sigma);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(x_vec[i].coeff(j), x_vec_free[i].coeff(j));
      }
    }
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(
                     x_bad_vec, ub, offset_bad_vec, sigma),
                 std::invalid_argument);
    EXPECT_THROW(stan::math::ub_offset_multiplier_free(x_bad_vec, ub,
                                                       offset_bad_vec, sigma),
                 std::invalid_argument);
  }
  // array[] matrix, matrix, array[] matrix, real
  {
    auto result = stan::math::ub_offset_multiplier_constrain(
        x_vec, ub, offset_vec, sigmad);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(result[i](j),
                        stan::math::ub_offset_multiplier_constrain(
                            x_vec[i](j), ub(j), offset_vec[i](j), sigmad));
      }
    }
    auto x_vec_free
        = stan::math::ub_offset_multiplier_free(result, ub, offset_vec, sigmad);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(x_vec[i].coeff(j), x_vec_free[i].coeff(j));
      }
    }
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(
                     x_bad_vec, ub, offset_bad_vec, sigmad),
                 std::invalid_argument);
    EXPECT_THROW(stan::math::ub_offset_multiplier_free(x_bad_vec, ub,
                                                       offset_bad_vec, sigmad),
                 std::invalid_argument);
  }
  // array[] matrix, matrix, matrix, array[] matrix
  {
    auto result = stan::math::ub_offset_multiplier_constrain(x_vec, ub, offset,
                                                             sigma_vec);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(result[i](j),
                        stan::math::ub_offset_multiplier_constrain(
                            x_vec[i](j), ub(j), offset(j), sigma_vec[i](j)));
      }
    }
    auto x_vec_free
        = stan::math::ub_offset_multiplier_free(result, ub, offset, sigma_vec);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(x_vec[i].coeff(j), x_vec_free[i].coeff(j));
      }
    }
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(
                     x_bad_vec, ub, offset, sigma_bad_vec),
                 std::invalid_argument);
    EXPECT_THROW(stan::math::ub_offset_multiplier_free(x_bad_vec, ub, offset,
                                                       sigma_bad_vec),
                 std::invalid_argument);
  }
  // array[] matrix, matrix, matrix, matrix
  {
    auto result
        = stan::math::ub_offset_multiplier_constrain(x_vec, ub, offset, sigma);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(result[i](j),
                        stan::math::ub_offset_multiplier_constrain(
                            x_vec[i](j), ub(j), offset(j), sigma(j)));
      }
    }
    auto x_vec_free
        = stan::math::ub_offset_multiplier_free(result, ub, offset, sigma);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(x_vec[i].coeff(j), x_vec_free[i].coeff(j));
      }
    }
  }
  // array[] matrix, matrix, matrix, real
  {
    auto result
        = stan::math::ub_offset_multiplier_constrain(x_vec, ub, offset, sigmad);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(result[i](j),
                        stan::math::ub_offset_multiplier_constrain(
                            x_vec[i](j), ub(j), offset(j), sigmad));
      }
    }
    auto x_vec_free
        = stan::math::ub_offset_multiplier_free(result, ub, offset, sigmad);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(x_vec[i].coeff(j), x_vec_free[i].coeff(j));
      }
    }
  }
  // array[] matrix, matrix, real, array[] matrix
  {
    auto result = stan::math::ub_offset_multiplier_constrain(x_vec, ub, offsetd,
                                                             sigma_vec);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(result[i](j),
                        stan::math::ub_offset_multiplier_constrain(
                            x_vec[i](j), ub(j), offsetd, sigma_vec[i](j)));
      }
    }
    auto x_vec_free
        = stan::math::ub_offset_multiplier_free(result, ub, offsetd, sigma_vec);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(x_vec[i].coeff(j), x_vec_free[i].coeff(j));
      }
    }
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(
                     x_bad_vec, ub, offsetd, sigma_bad_vec),
                 std::invalid_argument);
    EXPECT_THROW(stan::math::ub_offset_multiplier_free(x_bad_vec, ub, offsetd,
                                                       sigma_bad_vec),
                 std::invalid_argument);
  }
  // array[] matrix, matrix, real, matrix
  {
    auto result
        = stan::math::ub_offset_multiplier_constrain(x_vec, ub, offsetd, sigma);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(result[i](j),
                        stan::math::ub_offset_multiplier_constrain(
                            x_vec[i](j), ub(j), offsetd, sigma(j)));
      }
    }
    auto x_vec_free
        = stan::math::ub_offset_multiplier_free(result, ub, offsetd, sigma);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(x_vec[i].coeff(j), x_vec_free[i].coeff(j));
      }
    }
  }
  // array[] matrix, matrix, real, real
  {
    auto result = stan::math::ub_offset_multiplier_constrain(x_vec, ub, offsetd,
                                                             sigmad);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(result[i](j),
                        stan::math::ub_offset_multiplier_constrain(
                            x_vec[i](j), ub(j), offsetd, sigmad));
      }
    }
    auto x_vec_free
        = stan::math::ub_offset_multiplier_free(result, ub, offsetd, sigmad);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(x_vec[i].coeff(j), x_vec_free[i].coeff(j));
      }
    }
  }
  // array[] matrix, real, array[] matrix, array[] matrix
  {
    auto result = stan::math::ub_offset_multiplier_constrain(
        x_vec, ubd, offset_vec, sigma_vec);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(
            result[i](j),
            stan::math::ub_offset_multiplier_constrain(
                x_vec[i](j), ubd, offset_vec[i](j), sigma_vec[i](j)));
      }
    }
    auto x_vec_free = stan::math::ub_offset_multiplier_free(
        result, ubd, offset_vec, sigma_vec);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(x_vec[i].coeff(j), x_vec_free[i].coeff(j));
      }
    }
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(
                     x_bad_vec, ubd, offset_bad_vec, sigma_vec),
                 std::invalid_argument);
    EXPECT_THROW(stan::math::ub_offset_multiplier_free(
                     x_bad_vec, ubd, offset_bad_vec, sigma_vec),
                 std::invalid_argument);
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(
                     x_bad_vec, ubd, offset_vec, sigma_bad_vec),
                 std::invalid_argument);
    EXPECT_THROW(stan::math::ub_offset_multiplier_free(
                     x_bad_vec, ubd, offset_vec, sigma_bad_vec),
                 std::invalid_argument);
  }
  // array[] matrix, real, array[] matrix, matrix
  {
    auto result = stan::math::ub_offset_multiplier_constrain(x_vec, ubd,
                                                             offset_vec, sigma);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(result[i](j),
                        stan::math::ub_offset_multiplier_constrain(
                            x_vec[i](j), ubd, offset_vec[i](j), sigma(j)));
      }
    }
    auto x_vec_free
        = stan::math::ub_offset_multiplier_free(result, ubd, offset_vec, sigma);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(x_vec[i].coeff(j), x_vec_free[i].coeff(j));
      }
    }
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(
                     x_bad_vec, ubd, offset_bad_vec, sigma),
                 std::invalid_argument);
    EXPECT_THROW(stan::math::ub_offset_multiplier_free(x_bad_vec, ubd,
                                                       offset_bad_vec, sigma),
                 std::invalid_argument);
  }
  // array[] matrix, real, array[] matrix, real
  {
    auto result = stan::math::ub_offset_multiplier_constrain(
        x_vec, ubd, offset_vec, sigmad);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(result[i](j),
                        stan::math::ub_offset_multiplier_constrain(
                            x_vec[i](j), ubd, offset_vec[i](j), sigmad));
      }
    }
    auto x_vec_free = stan::math::ub_offset_multiplier_free(result, ubd,
                                                            offset_vec, sigmad);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(x_vec[i].coeff(j), x_vec_free[i].coeff(j));
      }
    }
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(
                     x_bad_vec, ubd, offset_bad_vec, sigmad),
                 std::invalid_argument);
    EXPECT_THROW(stan::math::ub_offset_multiplier_free(x_bad_vec, ubd,
                                                       offset_bad_vec, sigmad),
                 std::invalid_argument);
  }
  // array[] matrix, real, matrix, array[] matrix
  {
    auto result = stan::math::ub_offset_multiplier_constrain(x_vec, ubd, offset,
                                                             sigma_vec);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(result[i](j),
                        stan::math::ub_offset_multiplier_constrain(
                            x_vec[i](j), ubd, offset(j), sigma_vec[i](j)));
      }
    }
    auto x_vec_free
        = stan::math::ub_offset_multiplier_free(result, ubd, offset, sigma_vec);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(x_vec[i].coeff(j), x_vec_free[i].coeff(j));
      }
    }
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(
                     x_bad_vec, ubd, offset, sigma_bad_vec),
                 std::invalid_argument);
    EXPECT_THROW(stan::math::ub_offset_multiplier_free(x_bad_vec, ubd, offset,
                                                       sigma_bad_vec),
                 std::invalid_argument);
  }
  // array[] matrix, real, matrix, matrix
  {
    auto result
        = stan::math::ub_offset_multiplier_constrain(x_vec, ubd, offset, sigma);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(result[i](j),
                        stan::math::ub_offset_multiplier_constrain(
                            x_vec[i](j), ubd, offset(j), sigma(j)));
      }
    }
    auto x_vec_free
        = stan::math::ub_offset_multiplier_free(result, ubd, offset, sigma);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(x_vec[i].coeff(j), x_vec_free[i].coeff(j));
      }
    }
  }
  // array[] matrix, real, matrix, real
  {
    auto result = stan::math::ub_offset_multiplier_constrain(x_vec, ubd, offset,
                                                             sigmad);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(result[i](j),
                        stan::math::ub_offset_multiplier_constrain(
                            x_vec[i](j), ubd, offset(j), sigmad));
      }
    }
    auto x_vec_free
        = stan::math::ub_offset_multiplier_free(result, ubd, offset, sigmad);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(x_vec[i].coeff(j), x_vec_free[i].coeff(j));
      }
    }
  }
  // array[] matrix, real, real, array[] matrix
  {
    auto result = stan::math::ub_offset_multiplier_constrain(
        x_vec, ubd, offsetd, sigma_vec);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(result[i](j),
                        stan::math::ub_offset_multiplier_constrain(
                            x_vec[i](j), ubd, offsetd, sigma_vec[i](j)));
      }
    }
    auto x_vec_free = stan::math::ub_offset_multiplier_free(result, ubd,
                                                            offsetd, sigma_vec);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(x_vec[i].coeff(j), x_vec_free[i].coeff(j));
      }
    }
    EXPECT_THROW(stan::math::ub_offset_multiplier_constrain(
                     x_bad_vec, ubd, offsetd, sigma_bad_vec),
                 std::invalid_argument);
    EXPECT_THROW(stan::math::ub_offset_multiplier_free(x_bad_vec, ubd, offsetd,
                                                       sigma_bad_vec),
                 std::invalid_argument);
  }
  // array[] matrix, real, real, matrix
  {
    auto result = stan::math::ub_offset_multiplier_constrain(x_vec, ubd,
                                                             offsetd, sigma);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(result[i](j),
                        stan::math::ub_offset_multiplier_constrain(
                            x_vec[i](j), ubd, offsetd, sigma(j)));
      }
    }
    auto x_vec_free
        = stan::math::ub_offset_multiplier_free(result, ubd, offsetd, sigma);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(x_vec[i].coeff(j), x_vec_free[i].coeff(j));
      }
    }
  }
  // array[] matrix, real, real, real
  {
    auto result = stan::math::ub_offset_multiplier_constrain(x_vec, ubd,
                                                             offsetd, sigmad);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(result[i](j),
                        stan::math::ub_offset_multiplier_constrain(
                            x_vec[i](j), ubd, offsetd, sigmad));
      }
    }
    auto x_vec_free
        = stan::math::ub_offset_multiplier_free(result, ubd, offsetd, sigmad);
    for (size_t i = 0; i < x_vec.size(); ++i) {
      for (size_t j = 0; j < x_vec[i].size(); ++j) {
        EXPECT_FLOAT_EQ(x_vec[i].coeff(j), x_vec_free[i].coeff(j));
      }
    }
  }
}

TEST(prob_transform, ub_om_exception) {
  using stan::math::ub_offset_multiplier_constrain;
  using stan::math::ub_offset_multiplier_free;

  EXPECT_THROW(ub_offset_multiplier_constrain(5.0, 1.0, 0, -1),
               std::domain_error);
  EXPECT_THROW(ub_offset_multiplier_constrain(5.0, 1.0, 0, 0),
               std::domain_error);

  EXPECT_NO_THROW(ub_offset_multiplier_constrain(5.0, 1.0, 0, 1));
  double lp = 12;
  EXPECT_THROW(ub_offset_multiplier_constrain(5.0, 1.0, 0, -1, lp),
               std::domain_error);
  EXPECT_NO_THROW(ub_offset_multiplier_constrain(5.0, 1.0, 0, 1, lp));

  EXPECT_THROW(ub_offset_multiplier_constrain(
                   5.0, 0.0, std::numeric_limits<double>::infinity(), 1.0),
               std::domain_error);
  EXPECT_THROW(ub_offset_multiplier_constrain(5.0, 0.0, NAN, 1.0),
               std::domain_error);
  EXPECT_NO_THROW(ub_offset_multiplier_constrain(5.0, 0, 1.0, 0.01));
  EXPECT_THROW(ub_offset_multiplier_free(5.0, 0, 1.0, 0.0), std::domain_error);
  EXPECT_THROW(ub_offset_multiplier_free(
                   5.0, 9.0, std::numeric_limits<double>::infinity(), 1.0),
               std::domain_error);
  EXPECT_THROW(ub_offset_multiplier_free(4, 0, 4.0, 2.0), std::domain_error);
  EXPECT_THROW(ub_offset_multiplier_free(
                   1.0, -std::numeric_limits<double>::infinity(), 2.0, 0.0),
               std::domain_error);
  EXPECT_THROW(
      ub_offset_multiplier_free(
          -10.0 - 0.1, -std::numeric_limits<double>::infinity(), -10.0, -27.0),
      std::domain_error);
  EXPECT_THROW(ub_offset_multiplier_free(5.0, 0.0, NAN, 1.0),
               std::domain_error);
  EXPECT_NO_THROW(ub_offset_multiplier_free(5.0, 6.0, 1.0, 0.01));
  EXPECT_THROW(ub_offset_multiplier_free(5.0, 1.0, 1.0, 0.01),
               std::domain_error);

  lp = 12;
  EXPECT_THROW(ub_offset_multiplier_constrain(5.0, 0, 1.0, 0.0, lp),
               std::domain_error);
  EXPECT_THROW(ub_offset_multiplier_constrain(
                   5.0, 0, std::numeric_limits<double>::infinity(), 1.0, lp),
               std::domain_error);
  EXPECT_THROW(ub_offset_multiplier_constrain(5.0, 0, NAN, 1.0, lp),
               std::domain_error);
  EXPECT_NO_THROW(ub_offset_multiplier_constrain(5.0, 0, 1.0, 0.01, lp));
}

TEST(prob_transform, ub_om_j) {
  for (double U : std::vector<double>{0.5, 2, 5, 10}) {
    for (double O : std::vector<double>{-1, 0.5, 2, 10}) {
      for (double M : std::vector<double>{0.5, 1, 2, 10, 100}) {
        for (double x : std::vector<double>{-20, -15, 0.1, 3, 45.2}) {
          double lp = -17.0;

          EXPECT_FLOAT_EQ(
              U - exp(x * M + O),
              stan::math::ub_offset_multiplier_constrain(x, U, O, M, lp));

          EXPECT_FLOAT_EQ(-17.0 + O + (M * x) + std::log(M), lp);
        }
      }
    }
  }
}

TEST(prob_transform, ub_om_f) {
  for (double U : std::vector<double>{0.5, 2, 5, 10}) {
    for (double O : std::vector<double>{-1, 0.5, 2, 10}) {
      for (double M : std::vector<double>{0.5, 2, 10}) {
        double y = U - 3;
        EXPECT_FLOAT_EQ(((std::log((U - y)) - O) / M),
                        stan::math::ub_offset_multiplier_free(y, U, O, M));
      }
    }
  }
}

TEST(prob_transform, ub_om_rt) {
  double x = 5.0;
  double ub = 8.0;
  double off = -2;
  double sigma = 3.0;
  double xc = stan::math::ub_offset_multiplier_constrain(x, ub, off, sigma);
  double xcf = stan::math::ub_offset_multiplier_free(xc, ub, off, sigma);
  EXPECT_FLOAT_EQ(x, xcf);
  double xcfc = stan::math::ub_offset_multiplier_constrain(xcf, ub, off, sigma);
  EXPECT_FLOAT_EQ(xc, xcfc);
}
