/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include <gtest/gtest.h>
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

// Reference computed in double precision, so the comparison measures the error of cf_sqrt() only
std::complex<double> ref_sqrt(cf_t x)
{
  return std::sqrt(to_cplx(x));
}

// Relative error of the result w.r.t. the double precision reference, in units of the magnitude of
// sqrt(x): both components are checked against the same scale, so a component that should cancel to
// 0 is allowed an absolute error only.
double rel_error(cf_t x, cf_t got)
{
  const std::complex<double> expected = ref_sqrt(x);
  const double scale = std::abs(expected);
  if (scale == 0.0)
    return std::abs(to_cplx(got));
  return std::abs(to_cplx(got) - expected) / scale;
}

// Difference between two results for the same input, on the same relative scale as rel_error()
double rel_diff(cf_t a, cf_t b, cf_t x)
{
  const double scale = std::abs(ref_sqrt(x));
  const std::complex<double> diff = to_cplx(a) - to_cplx(b);
  return scale == 0.0 ? std::abs(diff) : std::abs(diff) / scale;
}

// float has 24 bits of mantissa; allow a few ulp for the hypot/sqrt/divide chain
constexpr double TOLERANCE = 8.0 / (1 << 24);

// All these values, and their expected results, are exactly representable in float, so both
// implementations must return them without any rounding error
const struct {
  cf_t in;
  cf_t out;
} KNOWN_VALUES[] = {
    {{0.0f, 0.0f}, {0.0f, 0.0f}},
    {{4.0f, 0.0f}, {2.0f, 0.0f}}, // positive real: real result
    {{-4.0f, 0.0f}, {0.0f, 2.0f}}, // negative real: purely imaginary result, on the positive branch
    {{0.25f, 0.0f}, {0.5f, 0.0f}},
    {{0.0f, 2.0f}, {1.0f, 1.0f}}, // 2i = (1 + i)^2
    {{0.0f, -2.0f}, {1.0f, -1.0f}},
    {{3.0f, 4.0f}, {2.0f, 1.0f}}, // 3 + 4i = (2 + i)^2
    {{3.0f, -4.0f}, {2.0f, -1.0f}},
    {{-3.0f, 4.0f}, {1.0f, 2.0f}}, // -3 + 4i = (1 + 2i)^2
    {{-3.0f, -4.0f}, {1.0f, -2.0f}},
};

// Returns a failed assertion carrying the values, or success, so callers can append their own context
testing::AssertionResult is_close(cf_t x, cf_t got)
{
  const double err = rel_error(x, got);
  if (err <= TOLERANCE)
    return testing::AssertionSuccess();
  return testing::AssertionFailure() << "sqrt(" << x.r << " + " << x.i << "i) = " << got.r << " + " << got.i
                                     << "i, expected " << ref_sqrt(x).real() << " + " << ref_sqrt(x).imag()
                                     << "i, relative error " << err << " > " << TOLERANCE;
}

} // namespace

TEST(cf_sqrt, known_values)
{
  for (const auto &c : KNOWN_VALUES) {
    const cf_t got = cf_sqrt(c.in);
    EXPECT_FLOAT_EQ(got.r, c.out.r) << "real part of sqrt(" << c.in.r << " + " << c.in.i << "i)";
    EXPECT_FLOAT_EQ(got.i, c.out.i) << "imag part of sqrt(" << c.in.r << " + " << c.in.i << "i)";
  }
}

// The principal square root always lies in the right half plane and keeps the sign of the imaginary
// part of the input, including for a signed zero imaginary part (as csqrtf() does).
TEST(cf_sqrt, principal_branch)
{
  for (float re : {-4.0f, -1.0f, 0.0f, 1.0f, 4.0f}) {
    for (float im : {-0.0f, 0.0f}) {
      const cf_t got = cf_sqrt((cf_t){re, im});
      EXPECT_GE(got.r, 0.0f) << "real part must be non-negative";
      EXPECT_EQ(std::signbit(got.i), std::signbit(im)) << "sign of imaginary part must be kept, in = " << re << " + "
                                                       << im << "i, out imag = " << got.i;
    }
  }
}

TEST(cf_sqrt, random_values)
{
  std::mt19937 gen(42);
  // Spans the full float exponent range, i.e. also the magnitudes where computing x.r^2 + x.i^2
  // would overflow or underflow
  std::uniform_real_distribution<float> exponent(-60.0f, 60.0f);
  std::uniform_real_distribution<float> mantissa(-1.0f, 1.0f);

  for (int i = 0; i < 100000; i++) {
    const float scale = std::pow(2.0f, exponent(gen));
    const cf_t x = {mantissa(gen) * scale, mantissa(gen) * scale};
    EXPECT_TRUE(is_close(x, cf_sqrt(x)));
  }
}

// Values close to the negative real axis are where the naive formula
// sqrt((|x| + x.r) / 2) + i * sqrt((|x| - x.r) / 2) loses all precision in the real part
TEST(cf_sqrt, near_negative_real_axis)
{
  for (int i = 0; i < 40; i++) {
    const float im = std::pow(2.0f, -i);
    for (float re : {-1.0f, -1e6f, -1e-6f}) {
      EXPECT_TRUE(is_close((cf_t){re, im}, cf_sqrt((cf_t){re, im})));
      EXPECT_TRUE(is_close((cf_t){re, -im}, cf_sqrt((cf_t){re, -im})));
    }
  }
}

// The vector version is expected to be as accurate as the scalar one, but not bit exact with it: it
// computes the magnitude of the input without hypotf(). Measured over the random inputs of
// sqrt_cf_vector.random_values, the two differ by up to ~2.6 ulp, and both stay ~2 ulp away from the
// double precision reference.
TEST(sqrt_cf_vector, known_values)
{
  // the table has 10 entries, i.e. one full SIMD block of 8 plus a 2 element scalar tail
  const uint32_t n = sizeof(KNOWN_VALUES) / sizeof(KNOWN_VALUES[0]);
  std::vector<cf_t> in(n);
  for (uint32_t i = 0; i < n; i++)
    in[i] = KNOWN_VALUES[i].in;

  std::vector<cf_t> out(n, (cf_t){NAN, NAN});
  sqrt_cf_vector(in.data(), out.data(), n);

  for (uint32_t i = 0; i < n; i++) {
    EXPECT_FLOAT_EQ(out[i].r, KNOWN_VALUES[i].out.r) << "real part, index " << i;
    EXPECT_FLOAT_EQ(out[i].i, KNOWN_VALUES[i].out.i) << "imag part, index " << i;
  }
}

TEST(sqrt_cf_vector, random_values)
{
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> exponent(-60.0f, 60.0f);
  std::uniform_real_distribution<float> mantissa(-1.0f, 1.0f);

  // Lengths from 0 to 40 to run the SIMD blocks with every possible scalar tail length
  for (uint32_t n = 0; n <= 40; n++) {
    std::vector<cf_t> in(n);
    for (auto &v : in) {
      const float scale = std::pow(2.0f, exponent(gen));
      v = (cf_t){mantissa(gen) * scale, mantissa(gen) * scale};
    }

    std::vector<cf_t> out(n, (cf_t){NAN, NAN});
    sqrt_cf_vector(in.data(), out.data(), n);

    for (uint32_t i = 0; i < n; i++) {
      EXPECT_TRUE(is_close(in[i], out[i])) << "index " << i << " of " << n;
      // the vector version must stay within a few ulp of the scalar one
      EXPECT_LE(rel_diff(out[i], cf_sqrt(in[i]), in[i]), TOLERANCE) << "index " << i << " of " << n;
    }
  }

  // one long vector, for the same input coverage as the scalar test
  const uint32_t n = 100000;
  std::vector<cf_t> in(n);
  for (auto &v : in) {
    const float scale = std::pow(2.0f, exponent(gen));
    v = (cf_t){mantissa(gen) * scale, mantissa(gen) * scale};
  }
  std::vector<cf_t> out(n, (cf_t){NAN, NAN});
  sqrt_cf_vector(in.data(), out.data(), n);
  for (uint32_t i = 0; i < n; i++) {
    EXPECT_TRUE(is_close(in[i], out[i])) << "index " << i;
    EXPECT_LE(rel_diff(out[i], cf_sqrt(in[i]), in[i]), TOLERANCE) << "index " << i;
  }
}

TEST(sqrt_cf_vector, near_negative_real_axis)
{
  std::vector<cf_t> in;
  for (int i = 0; i < 40; i++) {
    const float im = std::pow(2.0f, -i);
    for (float re : {-1.0f, -1e6f, -1e-6f}) {
      in.push_back((cf_t){re, im});
      in.push_back((cf_t){re, -im});
    }
  }

  std::vector<cf_t> out(in.size(), (cf_t){NAN, NAN});
  sqrt_cf_vector(in.data(), out.data(), in.size());

  for (size_t i = 0; i < in.size(); i++)
    EXPECT_TRUE(is_close(in[i], out[i])) << "index " << i;
}

// Same invariants as the scalar version: right half plane, sign of the imaginary part kept
TEST(sqrt_cf_vector, principal_branch)
{
  std::vector<cf_t> in;
  for (float re : {-4.0f, -1.0f, 0.0f, 1.0f, 4.0f})
    for (float im : {-0.0f, 0.0f})
      in.push_back((cf_t){re, im});

  std::vector<cf_t> out(in.size(), (cf_t){NAN, NAN});
  sqrt_cf_vector(in.data(), out.data(), in.size());

  for (size_t i = 0; i < in.size(); i++) {
    EXPECT_GE(out[i].r, 0.0f) << "real part must be non-negative, index " << i;
    EXPECT_EQ(std::signbit(out[i].i), std::signbit(in[i].i))
        << "sign of imaginary part must be kept, in = " << in[i].r << " + " << in[i].i
        << "i, out imag = " << out[i].i;
  }
}

// The loads and stores must not assume any alignment
TEST(sqrt_cf_vector, unaligned)
{
  std::mt19937 gen(5);
  std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

  const uint32_t n = 33;
  std::vector<cf_t> in(n + 1);
  for (auto &v : in)
    v = (cf_t){dist(gen), dist(gen)};

  std::vector<cf_t> aligned(n, (cf_t){NAN, NAN});
  std::vector<cf_t> unaligned(n + 1, (cf_t){NAN, NAN});
  sqrt_cf_vector(in.data() + 1, aligned.data(), n); // unaligned input
  sqrt_cf_vector(in.data(), unaligned.data() + 1, n); // unaligned output, different input offset

  for (uint32_t i = 0; i < n; i++) {
    EXPECT_TRUE(is_close(in[i + 1], aligned[i])) << "unaligned input, index " << i;
    EXPECT_TRUE(is_close(in[i], unaligned[i + 1])) << "unaligned output, index " << i;
  }
}

TEST(sqrt_cf_vector, in_place)
{
  std::mt19937 gen(11);
  std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

  const uint32_t n = 37;
  std::vector<cf_t> in(n);
  for (auto &v : in)
    v = (cf_t){dist(gen), dist(gen)};

  std::vector<cf_t> out(n, (cf_t){NAN, NAN});
  sqrt_cf_vector(in.data(), out.data(), n);

  std::vector<cf_t> inout = in;
  sqrt_cf_vector(inout.data(), inout.data(), n);

  // in place must give exactly the same result as out of place
  for (uint32_t i = 0; i < n; i++) {
    EXPECT_EQ(inout[i].r, out[i].r) << "index " << i;
    EXPECT_EQ(inout[i].i, out[i].i) << "index " << i;
  }
}

TEST(sqrt_cf_vector, zero_length)
{
  cf_t canary = {1.0f, 2.0f};
  sqrt_cf_vector(&canary, &canary, 0);
  EXPECT_EQ(canary.r, 1.0f);
  EXPECT_EQ(canary.i, 2.0f);
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
