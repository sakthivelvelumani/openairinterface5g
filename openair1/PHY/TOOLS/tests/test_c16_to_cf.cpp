/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <vector>

extern "C" {
#include "openair1/PHY/TOOLS/tools_defs.h"
}

namespace {

std::vector<c16_t> random_vector(std::mt19937 &gen, uint32_t n)
{
  std::uniform_int_distribution<int> dist(INT16_MIN, INT16_MAX);
  std::vector<c16_t> v(n);
  for (auto &e : v)
    e = (c16_t){(int16_t)dist(gen), (int16_t)dist(gen)};
  return v;
}

// int16 to float is exact and the division is a single rounded operation, so the result is the
// correctly rounded quotient and can be compared exactly against a double precision reference
void expect_exact(c16_t in, cf_t got, float scale, const char *what)
{
  EXPECT_FLOAT_EQ(got.r, (float)((double)in.r / (double)scale)) << what << ", real part, scale " << scale;
  EXPECT_FLOAT_EQ(got.i, (float)((double)in.i / (double)scale)) << what << ", imag part, scale " << scale;
}

} // namespace

TEST(c16_to_cf, known_values)
{
  // Q15 normalization, the use case of c16_to_ch()
  const cf_t q15 = c16_to_cf((c16_t){16384, -8192}, 32768.0f);
  EXPECT_FLOAT_EQ(q15.r, 0.5f);
  EXPECT_FLOAT_EQ(q15.i, -0.25f);

  // a scale of 1 is a plain widening
  const cf_t one = c16_to_cf((c16_t){INT16_MIN, INT16_MAX}, 1.0f);
  EXPECT_FLOAT_EQ(one.r, -32768.0f);
  EXPECT_FLOAT_EQ(one.i, 32767.0f);

  const cf_t zero = c16_to_cf((c16_t){0, 0}, 7.5f);
  EXPECT_FLOAT_EQ(zero.r, 0.0f);
  EXPECT_FLOAT_EQ(zero.i, 0.0f);
}

TEST(c16_to_cf_vector, matches_scalar)
{
  std::mt19937 gen(42);
  // every length from 0 to 40 runs the 4 element blocks with every possible scalar tail length
  for (uint32_t n = 0; n <= 40; n++) {
    const std::vector<c16_t> in = random_vector(gen, n);
    for (float scale : {1.0f, 32768.0f, 0.125f, 1234.5f}) {
      std::vector<cf_t> out(n, (cf_t){NAN, NAN});
      c16_to_cf_vector(in.data(), out.data(), n, scale);
      for (uint32_t i = 0; i < n; i++) {
        const cf_t ref = c16_to_cf(in[i], scale);
        // the vector path must be bit exact with the scalar one, both do the same divide
        EXPECT_EQ(out[i].r, ref.r) << "index " << i << " of " << n << ", scale " << scale;
        EXPECT_EQ(out[i].i, ref.i) << "index " << i << " of " << n << ", scale " << scale;
        expect_exact(in[i], out[i], scale, "vector");
      }
    }
  }
}

// The extremes of the int16 range, and the samples around zero, on a full block and on a tail
TEST(c16_to_cf_vector, extreme_samples)
{
  const std::vector<c16_t> in = {{INT16_MIN, INT16_MAX},
                                 {INT16_MAX, INT16_MIN},
                                 {0, -1},
                                 {-1, 0},
                                 {1, -1},
                                 {INT16_MIN, INT16_MIN}};
  for (float scale : {1.0f, 32768.0f, 1e-6f, 1e6f}) {
    std::vector<cf_t> out(in.size(), (cf_t){NAN, NAN});
    c16_to_cf_vector(in.data(), out.data(), in.size(), scale);
    for (size_t i = 0; i < in.size(); i++)
      expect_exact(in[i], out[i], scale, "extreme");
  }
}

// The loads and stores must not assume any alignment
TEST(c16_to_cf_vector, unaligned)
{
  std::mt19937 gen(7);
  const uint32_t n = 13;
  const std::vector<c16_t> in = random_vector(gen, n + 1);
  std::vector<cf_t> out(n + 1, (cf_t){NAN, NAN});
  c16_to_cf_vector(in.data() + 1, out.data() + 1, n, 100.0f);
  for (uint32_t i = 0; i < n; i++)
    expect_exact(in[i + 1], out[i + 1], 100.0f, "unaligned");
}

TEST(c16_to_cf_vector, zero_length)
{
  c16_t in = {1, 2};
  cf_t canary = {3.0f, 4.0f};
  c16_to_cf_vector(&in, &canary, 0, 2.0f);
  EXPECT_EQ(canary.r, 3.0f);
  EXPECT_EQ(canary.i, 4.0f);
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
