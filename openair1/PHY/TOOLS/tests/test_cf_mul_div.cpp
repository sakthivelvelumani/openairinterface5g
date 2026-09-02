/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include <gtest/gtest.h>
#include <array>
#include <complex>
#include <random>
#include <vector>

extern "C" {
#include "openair1/PHY/TOOLS/tools_defs.h"
}

namespace {

std::complex<double> to_cplx(cf_t x)
{
  return {x.r, x.i};
}

// float has 24 bits of mantissa; allow a few ulp for the multiply/divide chains
constexpr double TOLERANCE = 8.0 / (1 << 24);

// One of the three operations under test, with its scalar and vector implementation and a double
// precision reference, so that the same checks can run on all of them
struct Op {
  const char *name;
  void (*vector)(const cf_t *, const cf_t *, cf_t *, uint32_t);
  cf_t (*scalar)(cf_t, cf_t);
  std::complex<double> (*ref)(std::complex<double>, std::complex<double>);
};

const Op MUL = {"mult_cf_vector",
                mult_cf_vector,
                cf_mul,
                [](std::complex<double> a, std::complex<double> b) { return a * b; }};
const Op MUL_CONJ = {"mult_conj_cf_vector",
                     mult_conj_cf_vector,
                     cf_mul_conj,
                     [](std::complex<double> a, std::complex<double> b) { return std::conj(a) * b; }};
const Op DIV = {"div_cf_vector",
                div_cf_vector,
                cf_div,
                [](std::complex<double> a, std::complex<double> b) { return a / b; }};

// Relative error of a result w.r.t. the double precision reference, in units of the magnitude of the
// expected value: both components are checked against the same scale, so a component that should
// cancel to 0 is allowed an absolute error only.
testing::AssertionResult is_close(const Op &op, cf_t x1, cf_t x2, cf_t got, double tolerance = TOLERANCE)
{
  const std::complex<double> expected = op.ref(to_cplx(x1), to_cplx(x2));
  const double scale = std::abs(expected);
  const double diff = std::abs(to_cplx(got) - expected);
  const double err = scale == 0.0 ? diff : diff / scale;
  if (err <= tolerance)
    return testing::AssertionSuccess();
  return testing::AssertionFailure() << op.name << "(" << x1.r << " + " << x1.i << "i, " << x2.r << " + " << x2.i
                                     << "i) = " << got.r << " + " << got.i << "i, expected " << expected.real() << " + "
                                     << expected.imag() << "i, relative error " << err << " > " << tolerance;
}

// The vector versions use explicit FMAs, the scalar ones plain C expressions, so the two are not
// necessarily bit exact; they must stay within the same tolerance of each other.
testing::AssertionResult matches_scalar(const Op &op, cf_t x1, cf_t x2, cf_t got)
{
  const cf_t expected = op.scalar(x1, x2);
  const std::complex<double> ref = op.ref(to_cplx(x1), to_cplx(x2));
  const double scale = std::abs(ref);
  const double diff = std::abs(to_cplx(got) - to_cplx(expected));
  const double err = scale == 0.0 ? diff : diff / scale;
  if (err <= TOLERANCE)
    return testing::AssertionSuccess();
  return testing::AssertionFailure() << op.name << " and its scalar version disagree on " << x1.r << " + " << x1.i
                                     << "i, " << x2.r << " + " << x2.i << "i: " << got.r << " + " << got.i << "i vs "
                                     << expected.r << " + " << expected.i << "i, relative difference " << err;
}

std::vector<cf_t> random_vector(std::mt19937 &gen, uint32_t n, float min_exp, float max_exp)
{
  std::uniform_real_distribution<float> exponent(min_exp, max_exp);
  std::uniform_real_distribution<float> mantissa(-1.0f, 1.0f);
  std::vector<cf_t> v(n);
  for (auto &e : v) {
    const float scale = std::pow(2.0f, exponent(gen));
    e = (cf_t){mantissa(gen) * scale, mantissa(gen) * scale};
  }
  return v;
}

// Elements of magnitude 2^exp2 within a factor of 2, so that none of them lands in the denormal
// range when exp2 is close to the lower end of the float exponent range
std::vector<cf_t> random_vector_scaled(std::mt19937 &gen, uint32_t n, float exp2)
{
  std::uniform_real_distribution<float> mantissa(0.5f, 1.0f);
  std::uniform_int_distribution<int> sign(0, 1);
  std::vector<cf_t> v(n);
  for (auto &e : v) {
    const float scale = std::pow(2.0f, exp2);
    e = (cf_t){(sign(gen) ? 1.0f : -1.0f) * mantissa(gen) * scale,
               (sign(gen) ? 1.0f : -1.0f) * mantissa(gen) * scale};
  }
  return v;
}

// Runs the vector version on every length from 0 to 40, i.e. on every combination of full SIMD
// blocks and scalar tail lengths, plus one long vector
void check_random(const Op &op, float min_exp, float max_exp)
{
  std::mt19937 gen(42);

  for (uint32_t n = 0; n <= 40; n++) {
    const std::vector<cf_t> x1 = random_vector(gen, n, min_exp, max_exp);
    const std::vector<cf_t> x2 = random_vector(gen, n, min_exp, max_exp);
    std::vector<cf_t> y(n, (cf_t){NAN, NAN});
    op.vector(x1.data(), x2.data(), y.data(), n);

    for (uint32_t i = 0; i < n; i++) {
      EXPECT_TRUE(is_close(op, x1[i], x2[i], y[i])) << "index " << i << " of " << n;
      EXPECT_TRUE(matches_scalar(op, x1[i], x2[i], y[i])) << "index " << i << " of " << n;
    }
  }

  const uint32_t n = 100000;
  const std::vector<cf_t> x1 = random_vector(gen, n, min_exp, max_exp);
  const std::vector<cf_t> x2 = random_vector(gen, n, min_exp, max_exp);
  std::vector<cf_t> y(n, (cf_t){NAN, NAN});
  op.vector(x1.data(), x2.data(), y.data(), n);

  for (uint32_t i = 0; i < n; i++) {
    EXPECT_TRUE(is_close(op, x1[i], x2[i], y[i])) << "index " << i;
    EXPECT_TRUE(matches_scalar(op, x1[i], x2[i], y[i])) << "index " << i;
  }
}

// y may be either input; both must give the same result as an out of place run
void check_in_place(const Op &op)
{
  std::mt19937 gen(3);
  const uint32_t n = 37; // 4 SIMD blocks plus a 5 element tail
  const std::vector<cf_t> x1 = random_vector(gen, n, -10.0f, 10.0f);
  const std::vector<cf_t> x2 = random_vector(gen, n, -10.0f, 10.0f);

  std::vector<cf_t> expected(n, (cf_t){NAN, NAN});
  op.vector(x1.data(), x2.data(), expected.data(), n);

  std::vector<cf_t> first = x1;
  op.vector(first.data(), x2.data(), first.data(), n);
  std::vector<cf_t> second = x2;
  op.vector(x1.data(), second.data(), second.data(), n);

  for (uint32_t i = 0; i < n; i++) {
    EXPECT_EQ(first[i].r, expected[i].r) << op.name << " in place on input 1, index " << i;
    EXPECT_EQ(first[i].i, expected[i].i) << op.name << " in place on input 1, index " << i;
    EXPECT_EQ(second[i].r, expected[i].r) << op.name << " in place on input 2, index " << i;
    EXPECT_EQ(second[i].i, expected[i].i) << op.name << " in place on input 2, index " << i;
  }
}

// The loads and stores must not assume any alignment
void check_unaligned(const Op &op)
{
  std::mt19937 gen(5);
  const uint32_t n = 33;
  const std::vector<cf_t> x1 = random_vector(gen, n + 1, -10.0f, 10.0f);
  const std::vector<cf_t> x2 = random_vector(gen, n + 1, -10.0f, 10.0f);

  std::vector<cf_t> y(n + 1, (cf_t){NAN, NAN});
  op.vector(x1.data() + 1, x2.data(), y.data() + 1, n); // shifted input 1 and output

  for (uint32_t i = 0; i < n; i++)
    EXPECT_TRUE(is_close(op, x1[i + 1], x2[i], y[i + 1])) << "index " << i;
}

void check_known_values(const Op &op, const std::vector<std::array<cf_t, 3>> &cases)
{
  std::vector<cf_t> x1, x2, expected;
  for (const auto &c : cases) {
    x1.push_back(c[0]);
    x2.push_back(c[1]);
    expected.push_back(c[2]);
  }
  std::vector<cf_t> y(cases.size(), (cf_t){NAN, NAN});
  op.vector(x1.data(), x2.data(), y.data(), cases.size());

  for (size_t i = 0; i < cases.size(); i++) {
    // exactly representable inputs and results, so both implementations must be exact
    EXPECT_FLOAT_EQ(y[i].r, expected[i].r) << op.name << " real part, index " << i;
    EXPECT_FLOAT_EQ(y[i].i, expected[i].i) << op.name << " imag part, index " << i;
    const cf_t s = op.scalar(x1[i], x2[i]);
    EXPECT_FLOAT_EQ(s.r, expected[i].r) << op.name << " scalar real part, index " << i;
    EXPECT_FLOAT_EQ(s.i, expected[i].i) << op.name << " scalar imag part, index " << i;
  }
}

} // namespace

TEST(mult_cf_vector, known_values)
{
  // 10 cases: one full SIMD block of 8 plus a 2 element scalar tail
  check_known_values(MUL,
                     {{{{0.0f, 0.0f}, {3.0f, 4.0f}, {0.0f, 0.0f}}},
                      {{{1.0f, 0.0f}, {3.0f, 4.0f}, {3.0f, 4.0f}}}, // 1 is neutral
                      {{{0.0f, 1.0f}, {0.0f, 1.0f}, {-1.0f, 0.0f}}}, // i * i = -1
                      {{{0.0f, 1.0f}, {3.0f, 4.0f}, {-4.0f, 3.0f}}}, // multiplying by i rotates by 90 degrees
                      {{{2.0f, 0.0f}, {3.0f, 4.0f}, {6.0f, 8.0f}}},
                      {{{1.0f, 2.0f}, {3.0f, 4.0f}, {-5.0f, 10.0f}}},
                      {{{1.0f, -2.0f}, {3.0f, 4.0f}, {11.0f, -2.0f}}},
                      {{{-1.0f, 2.0f}, {3.0f, -4.0f}, {5.0f, 10.0f}}},
                      {{{1.0f, 2.0f}, {1.0f, -2.0f}, {5.0f, 0.0f}}}, // x * conj(x) = |x|^2
                      {{{0.5f, 0.25f}, {4.0f, 8.0f}, {0.0f, 5.0f}}}});
}

TEST(mult_cf_vector, random_values)
{
  check_random(MUL, -60.0f, 60.0f);
}

TEST(mult_cf_vector, in_place)
{
  check_in_place(MUL);
}

TEST(mult_cf_vector, unaligned)
{
  check_unaligned(MUL);
}

TEST(mult_conj_cf_vector, known_values)
{
  // the first operand is the conjugated one
  check_known_values(MUL_CONJ,
                     {{{{0.0f, 0.0f}, {3.0f, 4.0f}, {0.0f, 0.0f}}},
                      {{{1.0f, 0.0f}, {3.0f, 4.0f}, {3.0f, 4.0f}}},
                      {{{0.0f, 1.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}}}, // conj(i) * i = 1
                      {{{0.0f, 1.0f}, {3.0f, 4.0f}, {4.0f, -3.0f}}},
                      {{{2.0f, 0.0f}, {3.0f, 4.0f}, {6.0f, 8.0f}}},
                      {{{1.0f, 2.0f}, {3.0f, 4.0f}, {11.0f, -2.0f}}},
                      {{{1.0f, -2.0f}, {3.0f, 4.0f}, {-5.0f, 10.0f}}},
                      {{{1.0f, 2.0f}, {1.0f, 2.0f}, {5.0f, 0.0f}}}, // conj(x) * x = |x|^2
                      {{{3.0f, 4.0f}, {3.0f, 4.0f}, {25.0f, 0.0f}}},
                      {{{0.5f, 0.25f}, {4.0f, 8.0f}, {4.0f, 3.0f}}}});
}

TEST(mult_conj_cf_vector, random_values)
{
  check_random(MUL_CONJ, -60.0f, 60.0f);
}

TEST(mult_conj_cf_vector, in_place)
{
  check_in_place(MUL_CONJ);
}

TEST(mult_conj_cf_vector, unaligned)
{
  check_unaligned(MUL_CONJ);
}

TEST(div_cf_vector, known_values)
{
  check_known_values(DIV,
                     {{{{3.0f, 4.0f}, {1.0f, 0.0f}, {3.0f, 4.0f}}}, // 1 is neutral
                      {{{3.0f, 4.0f}, {2.0f, 0.0f}, {1.5f, 2.0f}}}, // real denominator
                      {{{3.0f, 4.0f}, {0.0f, 1.0f}, {4.0f, -3.0f}}}, // dividing by i rotates by -90 degrees
                      {{{0.0f, 0.0f}, {3.0f, 4.0f}, {0.0f, 0.0f}}},
                      {{{-5.0f, 10.0f}, {3.0f, 4.0f}, {1.0f, 2.0f}}}, // inverse of the multiplication case
                      {{{-5.0f, 10.0f}, {1.0f, 2.0f}, {3.0f, 4.0f}}},
                      {{{25.0f, 0.0f}, {3.0f, 4.0f}, {3.0f, -4.0f}}}, // |x|^2 / x = conj(x)
                      {{{3.0f, 4.0f}, {3.0f, 4.0f}, {1.0f, 0.0f}}},
                      {{{3.0f, 4.0f}, {-3.0f, -4.0f}, {-1.0f, 0.0f}}},
                      {{{0.0f, 5.0f}, {4.0f, 8.0f}, {0.5f, 0.25f}}}});
}

TEST(div_cf_vector, random_values)
{
  // exponents kept moderate so that the quotients themselves stay in the float range
  check_random(DIV, -30.0f, 30.0f);
}

// Magnitudes where the direct x * conj(y) / |y|^2 form breaks down: |y|^2 overflows to infinity
// above ~2^64 and underflows to zero below ~2^-75, while the quotients here are all around 1.
// Smith's algorithm never squares anything, so it keeps working.
TEST(div_cf_vector, extreme_denominators)
{
  std::mt19937 gen(9);
  // 2^-124 is the smallest magnitude that keeps all the operands normal floats
  for (float exp2 : {-124.0f, -100.0f, -70.0f, 70.0f, 100.0f, 126.0f}) {
    const uint32_t n = 19;
    const std::vector<cf_t> x1 = random_vector_scaled(gen, n, exp2);
    const std::vector<cf_t> x2 = random_vector_scaled(gen, n, exp2);
    std::vector<cf_t> y(n, (cf_t){NAN, NAN});
    div_cf_vector(x1.data(), x2.data(), y.data(), n);

    for (uint32_t i = 0; i < n; i++) {
      EXPECT_TRUE(is_close(DIV, x1[i], x2[i], y[i])) << "index " << i << ", exponent " << exp2;
      EXPECT_TRUE(matches_scalar(DIV, x1[i], x2[i], y[i])) << "index " << i << ", exponent " << exp2;
    }
  }
}

// A denormal of magnitude 2^e carries only e + 149 significant bits instead of 24, so the
// intermediate values of any algorithm lose precision there, down to a couple of bits just above
// the smallest denormal 2^-149. The tolerance below follows that available precision. Note that the
// scalar and the vector version drift far apart in this range, by much more than the ulp or so they
// differ by on normal operands: the vector version rounds a + b * r once, with an FMA, where the
// scalar version rounds the product to a denormal first and loses most of its bits doing so.
TEST(div_cf_vector, denormal_denominators)
{
  std::mt19937 gen(23);
  for (float exp2 : {-130.0f, -135.0f, -140.0f, -145.0f}) {
    const double DENORMAL_TOLERANCE = 8.0 * std::pow(2.0, -(exp2 + 149.0)); // a few ulp of what is left
    const uint32_t n = 19;
    const std::vector<cf_t> x1 = random_vector_scaled(gen, n, exp2);
    const std::vector<cf_t> x2 = random_vector_scaled(gen, n, exp2);
    std::vector<cf_t> y(n, (cf_t){NAN, NAN});
    div_cf_vector(x1.data(), x2.data(), y.data(), n);

    for (uint32_t i = 0; i < n; i++) {
      EXPECT_TRUE(std::isfinite(y[i].r) && std::isfinite(y[i].i)) << "index " << i << ", exponent " << exp2;
      EXPECT_TRUE(is_close(DIV, x1[i], x2[i], y[i], DENORMAL_TOLERANCE)) << "index " << i << ", exponent " << exp2;
      // the scalar version has to stay in range too, within the same reduced precision
      const cf_t s = cf_div(x1[i], x2[i]);
      EXPECT_TRUE(std::isfinite(s.r) && std::isfinite(s.i)) << "scalar, index " << i << ", exponent " << exp2;
      EXPECT_TRUE(is_close(DIV, x1[i], x2[i], s, DENORMAL_TOLERANCE))
          << "scalar, index " << i << ", exponent " << exp2;
    }
  }
}

// Denominators exactly on the real or the imaginary axis take the two branches of Smith's algorithm
// with a zero term, and are also where the swap condition |y.r| < |y.i| flips
TEST(div_cf_vector, degenerate_denominators)
{
  std::mt19937 gen(13);
  const uint32_t n = 20;
  const std::vector<cf_t> x1 = random_vector(gen, n, -10.0f, 10.0f);
  std::vector<cf_t> x2(n);
  for (uint32_t i = 0; i < n; i++) {
    const float v = (i % 2 == 0) ? 1.5f : -1.5f;
    switch (i % 4) {
      case 0: x2[i] = (cf_t){v, 0.0f}; break; // real
      case 1: x2[i] = (cf_t){0.0f, v}; break; // imaginary
      case 2: x2[i] = (cf_t){v, v}; break; // |y.r| == |y.i|
      case 3: x2[i] = (cf_t){v, -v}; break;
    }
  }

  std::vector<cf_t> y(n, (cf_t){NAN, NAN});
  div_cf_vector(x1.data(), x2.data(), y.data(), n);

  for (uint32_t i = 0; i < n; i++) {
    EXPECT_TRUE(is_close(DIV, x1[i], x2[i], y[i])) << "index " << i;
    EXPECT_TRUE(matches_scalar(DIV, x1[i], x2[i], y[i])) << "index " << i;
  }
}

TEST(div_cf_vector, in_place)
{
  check_in_place(DIV);
}

TEST(div_cf_vector, unaligned)
{
  check_unaligned(DIV);
}

// x / y and x * conj(y) differ by the |y|^2 factor: the conjugate multiplication alone is only the
// numerator of the division
TEST(div_cf_vector, is_not_a_conjugate_multiplication)
{
  std::mt19937 gen(17);
  const uint32_t n = 24;
  const std::vector<cf_t> x1 = random_vector(gen, n, -5.0f, 5.0f);
  const std::vector<cf_t> x2 = random_vector(gen, n, -5.0f, 5.0f);

  std::vector<cf_t> quotient(n, (cf_t){NAN, NAN});
  std::vector<cf_t> numerator(n, (cf_t){NAN, NAN});
  div_cf_vector(x1.data(), x2.data(), quotient.data(), n);
  mult_conj_cf_vector(x2.data(), x1.data(), numerator.data(), n); // x1 * conj(x2)

  for (uint32_t i = 0; i < n; i++) {
    const std::complex<double> q = to_cplx(quotient[i]);
    // the imaginary part of a quotient can be a near cancellation of the terms of the numerator, so
    // both components are compared against the magnitude of the quotient, not against themselves
    const std::complex<double> scaled = to_cplx(numerator[i]) / std::norm(to_cplx(x2[i]));
    EXPECT_LE(std::abs(q - scaled), std::abs(q) * TOLERANCE)
        << "index " << i << ": " << q.real() << " + " << q.imag() << "i vs " << scaled.real() << " + "
        << scaled.imag() << "i";
  }
}

TEST(cf_vector_ops, zero_length)
{
  for (const Op &op : {MUL, MUL_CONJ, DIV}) {
    cf_t canary = {1.0f, 2.0f};
    op.vector(&canary, &canary, &canary, 0);
    EXPECT_EQ(canary.r, 1.0f) << op.name;
    EXPECT_EQ(canary.i, 2.0f) << op.name;
  }
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
