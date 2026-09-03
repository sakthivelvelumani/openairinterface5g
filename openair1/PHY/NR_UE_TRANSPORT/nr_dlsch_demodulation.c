/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/*!
 * \brief Top-level routines for demodulating the PDSCH physical channel from 38-211, V15.2 2018-06
 */

#include "PHY/NR_REFSIG/ptrs_nr.h"
#include "PHY/TOOLS/tools_defs.h"
#include "assertions.h"
#include "common/platform_constants.h"
#include "nr_phy_common.h"
#include "PHY/defs_nr_UE.h"
#include "nr_transport_proto_ue.h"
#include "PHY/sse_intrin.h"
#include "T.h"
#include "bits.h"
#include "openair1/PHY/NR_UE_ESTIMATION/nr_estimation.h"
#include "PHY/NR_REFSIG/nr_refsig.h"
#include "PHY/NR_REFSIG/dmrs_nr.h"
#include "common/utils/nr/nr_common.h"
#include "platform_types.h"
#include "utils.h"
#include <complex.h>
#include "openair1/PHY/TOOLS/phy_scope_interface.h"
#include "nfapi/open-nFAPI/nfapi/public_inc/nfapi_nr_interface.h"
#include <simde/x86/avx.h>
#include <simde/x86/sse2.h>
#include <simde/x86/avx512.h>
#include <simde/x86/avx2.h>
#include <avx512fintrin.h>

// #define DEBUG_HARQ(a...) printf(a)
#define DEBUG_HARQ(...)
//#define DEBUG_DLSCH_DEMOD
//#define DEBUG_PDSCH_RX

#define print_ints(s,x) printf("%s = %d %d %d %d\n",s,(x)[0],(x)[1],(x)[2],(x)[3])
#define print_shorts(s,x) printf("%s = [%d+j*%d, %d+j*%d, %d+j*%d, %d+j*%d]\n",s,(x)[0],(x)[1],(x)[2],(x)[3],(x)[4],(x)[5],(x)[6],(x)[7])

static bool overlap_csi_symbol(const fapi_nr_dl_config_csirs_pdu_rel15_t *csi_pdu, int symbol)
{
  int num_l0 [18] = {1, 1, 1, 1, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 4, 2, 2, 4};
  for (int s = 0; s < num_l0[csi_pdu->row - 1]; s++) {
    if (symbol == csi_pdu->symb_l0 + s)
      return true;
  }
  // check also l1 if relevant
  if (csi_pdu->row == 13 || csi_pdu->row == 14 || csi_pdu->row == 16 || csi_pdu->row == 17) {
    for (int s = 0; s < 2; s++) { // two consecutive symbols including l1
      if (symbol == csi_pdu->symb_l1 + s)
        return true;
    }
  }
  return false;
}

static uint32_t build_csi_overlap_bitmap(const fapi_nr_dl_config_dlsch_pdu_rel15_t *dlsch_config, int symbol)
{
  // LS 16 bits for even RBs, MS 16 bits for odd RBs
  uint32_t csi_res_bitmap = 0;
  int num_k[18] = {1, 1, 1, 1, 1, 4, 2, 2, 6, 3, 4, 4, 3, 3, 3, 4, 4, 4};
  for (int i = 0; i < dlsch_config->numCsiRsForRateMatching; i++) {
    const fapi_nr_dl_config_csirs_pdu_rel15_t *csi_pdu = &dlsch_config->csiRsForRateMatching[i];

    if (!overlap_csi_symbol(csi_pdu, symbol))
      continue;

    int num_kp = 1;
    int mult = 1;
    int k0_step = 0;
    int num_k0 = 1;
    switch (csi_pdu->row) {
      case 1:
        k0_step = 4;
        num_k0 = 3;
        break;
      case 2:
        break;
      case 4:
        num_kp = 2;
        mult = 4;
        k0_step = 2;
        num_k0 = 2;
        break;
      default:
        num_kp = 2;
        mult = 2;
    }
    int found = 0;
    int bit = 0;
    uint32_t temp_res_map = 0;
    while (found < num_k[csi_pdu->row - 1]) {
      if ((csi_pdu->freq_domain >> bit) & 0x01) {
        for (int k0 = 0; k0 < num_k0; k0++) {
          for (int kp = 0; kp < num_kp; kp++) {
            int re = (bit * mult) + (k0 * k0_step) + kp;
            temp_res_map |= (1 << re);
          }
        }
        found++;
      }
      bit++;
      AssertFatal(bit < 13,
                  "Couldn't find %d positive bits in bitmap %d for CSI freq. domain\n",
                  num_k[csi_pdu->row - 1],
                  csi_pdu->freq_domain);
    }
    if (csi_pdu->freq_density < 2)
      csi_res_bitmap |= (temp_res_map << (16 * csi_pdu->freq_density));
    else
      csi_res_bitmap |= (temp_res_map + (temp_res_map << 16));
  }
  return csi_res_bitmap;
}

//==============================================================================================
// Pre-processing for LLR computation
//==============================================================================================

static void nr_dlsch_channel_level_median(uint32_t rx_size_symbol,
                                          int32_t dl_ch_estimates_ext[][rx_size_symbol],
                                          int32_t median[MAX_ANT][MAX_ANT],
                                          int n_tx,
                                          int n_rx,
                                          int length)
{
  for (int aatx = 0; aatx < n_tx; aatx++) {
    for (int aarx = 0; aarx < n_rx; aarx++) {
      int64_t max = median[aatx][aarx]; // initialize the med point for max
      int64_t min = median[aatx][aarx]; // initialize the med point for min
      simde__m128i *dl_ch128 = (simde__m128i *)dl_ch_estimates_ext[aatx * n_rx + aarx];

      const int length2 = length >> 2; // length = number of REs, hence length2=nb_REs*(32/128) in SIMD loop

      for (int ii = 0; ii < length2; ii++) {
        simde__m128i norm128D =
            simde_mm_srai_epi32(simde_mm_madd_epi16(*dl_ch128, *dl_ch128), 2); //[|H_0|²/4 |H_1|²/4 |H_2|²/4 |H_3|²/4]
        int32_t *tmp = (int32_t *)&norm128D;
        int64_t norm_pack = (int64_t)tmp[0] + tmp[1] + tmp[2] + tmp[3];

        if (norm_pack > max)
          max = norm_pack;
        if (norm_pack < min)
          min = norm_pack;
        dl_ch128+=1;
      }

      median[aatx][aarx] = (max + min) >> 1;
      LOG_D(PHY, "Channel level  median [%d][%d]: %d max = %ld min = %ld\n", aatx, aarx, median[aatx][aarx], max, min);
    }
  }
}

//==============================================================================================
// Extraction functions
//==============================================================================================

static void nr_dlsch_extract_rbs(uint32_t rxdataF_sz,
                                 c16_t rxdataF[][rxdataF_sz],
                                 uint32_t rx_size_symbol,
                                 uint32_t pdsch_est_size,
                                 int32_t dl_ch_estimates[][pdsch_est_size],
                                 c16_t rxdataF_ext[][rx_size_symbol],
                                 int32_t dl_ch_estimates_ext[][rx_size_symbol],
                                 unsigned char symbol,
                                 uint8_t pilots,
                                 const fapi_nr_dl_config_dlsch_pdu_rel15_t *dlsch_config,
                                 const freq_alloc_bitmap_t *freq_alloc,
                                 uint8_t Nl,
                                 NR_DL_FRAME_PARMS *fp,
                                 uint32_t csi_res_bitmap,
                                 int chest_time_type)
{
  int config_type = dlsch_config->dmrsConfigType;
  int n_dmrs_cdm_groups = dlsch_config->n_dmrs_cdm_groups;
  if (config_type == NFAPI_NR_DMRS_TYPE1)
    AssertFatal(n_dmrs_cdm_groups == 1
                || n_dmrs_cdm_groups == 2,
                "n_dmrs_cdm_groups %d is illegal\n",
                n_dmrs_cdm_groups);
  else
    AssertFatal(n_dmrs_cdm_groups == 1
                || n_dmrs_cdm_groups == 2 
                || n_dmrs_cdm_groups == 3,
                "n_dmrs_cdm_groups %d is illegal\n",
                n_dmrs_cdm_groups);

  uint32_t dmrs_rb_bitmap = 0;
  if (pilots) {
    dmrs_rb_bitmap = 0xfff; // all REs taken by dmrs
    if (config_type == NFAPI_NR_DMRS_TYPE1 && n_dmrs_cdm_groups == 1)
      dmrs_rb_bitmap = 0x555; // alternating REs starting from 0
    if (config_type == NFAPI_NR_DMRS_TYPE2 && n_dmrs_cdm_groups == 1)
      dmrs_rb_bitmap = 0xc3;  // REs 0,1 and 6,7
    if (config_type == NFAPI_NR_DMRS_TYPE2 && n_dmrs_cdm_groups == 2)
      dmrs_rb_bitmap = 0x3cf;  // REs 0,1,2,3 and 6,7,8,9
  }

  // csi_res_bitmap LS 16 bits for even RBs, MS 16 bits for odd RBs
  uint32_t csi_res_even = csi_res_bitmap & 0xfff;
  uint32_t csi_res_odd = (csi_res_bitmap >> 16) & 0xfff;
  AssertFatal((dmrs_rb_bitmap & csi_res_even) == 0, "DMRS RE overlapping with CSI RE, it shouldn't happen\n");
  AssertFatal((dmrs_rb_bitmap & csi_res_odd) == 0, "DMRS RE overlapping with CSI RE, it shouldn't happen\n");
  uint32_t dmrs_csi_overlap_even = csi_res_even | dmrs_rb_bitmap;
  uint32_t dmrs_csi_overlap_odd = csi_res_odd | dmrs_rb_bitmap;
  int8_t validDmrsEst;
  if (chest_time_type == 0)
    validDmrsEst = get_valid_dmrs_idx_for_channel_est(dlsch_config->dlDmrsSymbPos, symbol);
  else
    validDmrsEst = get_next_dmrs_symbol_in_slot(dlsch_config->dlDmrsSymbPos, 0, 14); // get first dmrs symbol index

  int pos = 0;
  int block_start, block_end;
  int offset = 0;
  while (find_next_rb_block(freq_alloc->bitmap, dlsch_config->BWPSize, &pos, &block_start, &block_end)) {
    int start_rb = block_start + dlsch_config->BWPStart;
    int nb_rb = block_end - block_start + 1;
    const int start_re = (fp->first_carrier_offset + start_rb * NR_NB_SC_PER_RB) % fp->ofdm_symbol_size;
    for (int aarx = 0; aarx < fp->nb_antennas_rx; aarx++) {
      c16_t *rxF_ext = rxdataF_ext[aarx] + offset;
      c16_t *rxF = &rxdataF[aarx][symbol * fp->ofdm_symbol_size];
      for (int l = 0; l < Nl; l++) {
        int32_t *dl_ch0 = &dl_ch_estimates[(l * fp->nb_antennas_rx) + aarx][validDmrsEst * fp->ofdm_symbol_size];
        int32_t *dl_ch0_ext = dl_ch_estimates_ext[(l * fp->nb_antennas_rx) + aarx] + offset;
        if (pilots == 0 && csi_res_bitmap == 0) { // data symbol only
          if (l == 0) {
            if (start_re + nb_rb * NR_NB_SC_PER_RB <= fp->ofdm_symbol_size) {
              memcpy(rxF_ext, &rxF[start_re], nb_rb * NR_NB_SC_PER_RB * sizeof(int32_t));
            } else {
              int neg_length = fp->ofdm_symbol_size - start_re;
              int pos_length = nb_rb * NR_NB_SC_PER_RB - neg_length;
              memcpy(rxF_ext, &rxF[start_re], neg_length * sizeof(int32_t));
              memcpy(&rxF_ext[neg_length], rxF, pos_length * sizeof(int32_t));
            }
          }
          memcpy(dl_ch0_ext, dl_ch0, nb_rb * NR_NB_SC_PER_RB * sizeof(int32_t));
        } else {
          int j = 0;
          int k = start_re;
          for (int rb = start_rb; rb < start_rb + nb_rb; rb++) {
            uint32_t overlap_map = rb % 2 ?  dmrs_csi_overlap_odd : dmrs_csi_overlap_even;
            for (int re = 0; re < NR_NB_SC_PER_RB; re++) {
              if (((overlap_map >> re) & 0x01) == 0) {
                // DATA RE
                if (l == 0)
                  rxF_ext[j] = rxF[k];
                dl_ch0_ext[j] = dl_ch0[re];
                j++;
              }
              k++;
              if (k >= fp->ofdm_symbol_size)
                k -= fp->ofdm_symbol_size;
            }
            dl_ch0 += 12;
          }
        }
      }
    }
    offset += nb_rb * NR_NB_SC_PER_RB;
  }
}

/* Zero Forcing Rx function: nr_a_sum_b()
 * Compute the complex addition x=x+y
 *
 * */
void nr_a_sum_b(c16_t *input_x, c16_t *input_y, unsigned short nb_rb)
{
  unsigned short rb;
  simde__m128i *x = (simde__m128i *)input_x;
  simde__m128i *y = (simde__m128i *)input_y;

  for (rb=0; rb<nb_rb; rb++) {
    x[0] = simde_mm_adds_epi16(x[0], y[0]);
    x[1] = simde_mm_adds_epi16(x[1], y[1]);
    x[2] = simde_mm_adds_epi16(x[2], y[2]);
    x += 3;
    y += 3;
  }
}

static inline void element_sign_sse(c16_t *a, c16_t *b, int32_t sign)
{
  const int16_t nr_sign[8] __attribute__((aligned(16))) = {-1, -1, -1, -1, -1, -1, -1, -1};

  simde__m128i *a_128 = (simde__m128i *)a;
  simde__m128i *b_128 = (simde__m128i *)b;

  if (sign < 0)
    *b_128 = simde_mm_sign_epi16(*a_128, ((simde__m128i *)nr_sign)[0]);
  else
    *b_128 = *a_128;
}

/* Zero Forcing Rx function: nr_element_sign()
 * Compute b=sign*a */
static inline void nr_element_sign(c16_t *a, c16_t *b, unsigned short nb_rb, int32_t sign)
{
  const int16_t nr_sign[8] __attribute__((aligned(16))) = {-1, -1, -1, -1, -1, -1, -1, -1};
  simde__m128i *a_128,*b_128;

  a_128 = (simde__m128i *)a;
  b_128 = (simde__m128i *)b;

  for (int rb = 0; rb < 3 * nb_rb; rb++) {
    if (sign < 0)
      b_128[rb] = simde_mm_sign_epi16(a_128[rb], ((simde__m128i *)nr_sign)[0]);
    else
      b_128[rb] = a_128[rb];

#ifdef DEBUG_DLSCH_DEMOD
    print_shorts("b:", (int16_t *)b_128);
#endif
  }
}

static void nr_determin_sse(unsigned int size, c16_t *a44[][size], c16_t *ad_bc, int sign, unsigned int shift)
{
  if (size == 1) {
    element_sign_sse(a44[0][0], ad_bc, sign);
  } else {
    int16_t k, rr[size - 1], cc[size - 1];
    c16_t outtemp[4] __attribute__((aligned(16)));
    c16_t outtemp1[4] __attribute__((aligned(16)));
    c16_t *sub_matrix[size - 1][size - 1];
    for (int rtx = 0; rtx < size; rtx++) { // row calculation for determin
      int ctx = 0;
      // find the submatrix row and column indices
      k = 0;
      for (int rrtx = 0; rrtx < size; rrtx++)
        if (rrtx != rtx)
          rr[k++] = rrtx;
      k = 0;
      for (int cctx = 0; cctx < size; cctx++)
        if (cctx != ctx)
          cc[k++] = cctx;
      // fill out the sub matrix corresponds to this element

      for (int ridx = 0; ridx < (size - 1); ridx++)
        for (int cidx = 0; cidx < (size - 1); cidx++)
          sub_matrix[cidx][ridx] = a44[cc[cidx]][rr[ridx]];

      nr_determin_sse(size - 1, sub_matrix, outtemp, ((rtx & 1) == 1 ? -1 : 1) * ((ctx & 1) == 1 ? -1 : 1) * sign, shift);
      mult_complex_vectors(a44[ctx][rtx], outtemp, rtx == 0 ? ad_bc : outtemp1, sizeofArray(outtemp1), shift);

      if (rtx != 0)
        *(simde__m128i *)ad_bc = simde_mm_add_epi16(*(simde__m128i *)ad_bc, *(simde__m128i *)outtemp1);
    }
  }
}

/* Zero Forcing Rx function: nr_det_4x4()
 * Compute the matrix determinant for 4x4 Matrix
 *
 * */
static void nr_determin(int size,
                        c16_t *a44[][size], //
                        c16_t *ad_bc, // ad-bc
                        unsigned short nb_rb,
                        int32_t sign,
                        int32_t shift0)
{
  AssertFatal(size > 0, "");

  if(size==1) {
    nr_element_sign(a44[0][0], // a
                    ad_bc, // b
                    nb_rb,
                    sign);
  } else {
    int16_t k, rr[size - 1], cc[size - 1];
    c16_t outtemp[12 * nb_rb] __attribute__((aligned(32)));
    c16_t outtemp1[12 * nb_rb] __attribute__((aligned(32)));
    c16_t *sub_matrix[size - 1][size - 1];
    for (int rtx=0;rtx<size;rtx++) {//row calculation for determin
      int ctx=0;
      //find the submatrix row and column indices
      k=0;
      for(int rrtx=0;rrtx<size;rrtx++)
        if(rrtx != rtx) rr[k++] = rrtx;
      k=0;
      for(int cctx=0;cctx<size;cctx++)
        if(cctx != ctx) cc[k++] = cctx;
      // fill out the sub matrix corresponds to this element

      for (int ridx = 0; ridx < (size - 1); ridx++)
        for (int cidx = 0; cidx < (size - 1); cidx++)
          sub_matrix[cidx][ridx] = a44[cc[cidx]][rr[ridx]];

      nr_determin(size - 1,
                  sub_matrix, // a33
                  outtemp,
                  nb_rb,
                  ((rtx & 1) == 1 ? -1 : 1) * ((ctx & 1) == 1 ? -1 : 1) * sign,
                  shift0);
      mult_complex_vectors(a44[ctx][rtx], outtemp, rtx == 0 ? ad_bc : outtemp1, sizeofArray(outtemp1), shift0);

      if (rtx != 0)
        nr_a_sum_b(ad_bc, outtemp1, nb_rb);
    }
  }
}

static double complex nr_determin_cpx(int32_t size, // size
                                      double complex a44_cpx[][size], //
                                      int32_t sign)
{
  double complex outtemp, outtemp1;
  //Allocate the submatrix elements
  DevAssert(size > 0);
  if(size==1) {
    return (a44_cpx[0][0] * sign);
  }else {
    double complex sub_matrix[size - 1][size - 1];
    int16_t k, rr[size - 1], cc[size - 1];
    outtemp1 = 0;
    for (int rtx=0;rtx<size;rtx++) {//row calculation for determin
      int ctx=0;
      //find the submatrix row and column indices
      k=0;
      for(int rrtx=0;rrtx<size;rrtx++)
        if(rrtx != rtx) rr[k++] = rrtx;
      k=0;
      for(int cctx=0;cctx<size;cctx++)
        if(cctx != ctx) cc[k++] = cctx;
      //fill out the sub matrix corresponds to this element
       for (int ridx=0;ridx<(size-1);ridx++)
         for (int cidx=0;cidx<(size-1);cidx++)
           sub_matrix[cidx][ridx] = a44_cpx[cc[cidx]][rr[ridx]];

       outtemp = nr_determin_cpx(size - 1,
                                 sub_matrix, // a33
                                 ((rtx & 1) == 1 ? -1 : 1) * ((ctx & 1) == 1 ? -1 : 1) * sign);
       outtemp1 += a44_cpx[ctx][rtx] * outtemp;
    }

    return((double complex)outtemp1);
  }
}

void nr_matrix_inverse_sse(int32_t size,
                           c16_t *a44[][size], // Input matrix//conjH_H_elements[0]
                           c16_t *inv_H_h_H[][size], // Inverse
                           c16_t *ad_bc, // determin
                           int32_t shift0)
{
  DevAssert(size > 1);
  int16_t k, rr[size - 1], cc[size - 1];

  // Allocate the submatrix elements
  c16_t *sub_matrix[size - 1][size - 1];

  // Compute Matrix determinant
  nr_determin_sse(size,
                  a44, //
                  ad_bc, // determinant
                  +1,
                  shift0);
  // print_shorts("nr_det_",(int16_t*)&ad_bc[0]);

  // Compute Inversion of the H^*H matrix
  /* For 2x2 MIMO matrix, we compute
   * *        |(conj_H_00xH_00+conj_H_10xH_10)   (conj_H_00xH_01+conj_H_10xH_11)|
   * * H_h_H= |                                                                 |
   * *        |(conj_H_01xH_00+conj_H_11xH_10)   (conj_H_01xH_01+conj_H_11xH_11)|
   * *
   * *inv(H_h_H) =(1/det)*[d  -b
   * *                     -c  a]
   * **************************************************************************/
  for (int rtx = 0; rtx < size; rtx++) { // row
    k = 0;
    for (int rrtx = 0; rrtx < size; rrtx++)
      if (rrtx != rtx)
        rr[k++] = rrtx;
    for (int ctx = 0; ctx < size; ctx++) { // column
      k = 0;
      for (int cctx = 0; cctx < size; cctx++)
        if (cctx != ctx)
          cc[k++] = cctx;

      // fill out the sub matrix corresponds to this element
      for (int ridx = 0; ridx < (size - 1); ridx++)
        for (int cidx = 0; cidx < (size - 1); cidx++)
          // To verify
          sub_matrix[cidx][ridx] = a44[cc[cidx]][rr[ridx]];

      nr_determin_sse(size - 1, // size
                      sub_matrix,
                      inv_H_h_H[rtx][ctx], // out transpose
                      ((rtx & 1) == 1 ? -1 : 1) * ((ctx & 1) == 1 ? -1 : 1),
                      shift0);
    }
  }
}

/* Zero Forcing Rx function: nr_matrix_inverse()
 * Compute the matrix inverse and determinant up to 4x4 Matrix
 *
 * */
uint8_t nr_matrix_inverse(int32_t size,
                          c16_t *a44[][size], // Input matrix//conjH_H_elements[0]
                          c16_t *inv_H_h_H[][size], // Inverse
                          c16_t *ad_bc, // determin
                          unsigned short nb_rb,
                          int32_t flag, // fixed point or floating flag
                          int32_t shift0)
{
  DevAssert(size > 1);
  int16_t k,rr[size-1],cc[size-1];

  if(flag) {//fixed point SIMD calc.
    //Allocate the submatrix elements
    c16_t *sub_matrix[size - 1][size - 1];

    //Compute Matrix determinant
    nr_determin(size,
                a44, //
                ad_bc, // determinant
                nb_rb,
                +1,
                shift0);
    //print_shorts("nr_det_",(int16_t*)&ad_bc[0]);

    //Compute Inversion of the H^*H matrix
    /* For 2x2 MIMO matrix, we compute
     * *        |(conj_H_00xH_00+conj_H_10xH_10)   (conj_H_00xH_01+conj_H_10xH_11)|
     * * H_h_H= |                                                                 |
     * *        |(conj_H_01xH_00+conj_H_11xH_10)   (conj_H_01xH_01+conj_H_11xH_11)|
     * *
     * *inv(H_h_H) =(1/det)*[d  -b
     * *                     -c  a]
     * **************************************************************************/
    for (int rtx=0;rtx<size;rtx++) {//row
      k=0;
      for(int rrtx=0;rrtx<size;rrtx++)
        if(rrtx != rtx) rr[k++] = rrtx;
      for (int ctx=0;ctx<size;ctx++) {//column
        k=0;
        for(int cctx=0;cctx<size;cctx++)
          if(cctx != ctx) cc[k++] = cctx;

        //fill out the sub matrix corresponds to this element
        for (int ridx=0;ridx<(size-1);ridx++)
          for (int cidx=0;cidx<(size-1);cidx++)
            // To verify
            sub_matrix[cidx][ridx]=a44[cc[cidx]][rr[ridx]];

        nr_determin(size - 1, // size
                    sub_matrix,
                    inv_H_h_H[rtx][ctx], // out transpose
                    nb_rb,
                    ((rtx & 1) == 1 ? -1 : 1) * ((ctx & 1) == 1 ? -1 : 1),
                    shift0);
      }
    }
  }
  else {//floating point calc.
    //Allocate the submatrix elements
    double complex sub_matrix_cpx[size - 1][size - 1];
    //Convert the IQ samples (in Q15 format) to float complex
    double complex a44_cpx[size][size];
    double complex inv_H_h_H_cpx[size][size];
    double complex determin_cpx;
    for (int i=0; i<12*nb_rb; i++) {

      //Convert Q15 to floating point
      for (int rtx=0;rtx<size;rtx++) {//row
        for (int ctx=0;ctx<size;ctx++) {//column
          a44_cpx[ctx][rtx] =
              ((double)(a44[ctx][rtx])[i].r) / (1 << (shift0 - 1)) + I * ((double)(a44[ctx][rtx])[i].i) / (1 << (shift0 - 1));
        }
      }
      //Compute Matrix determinant (copy real value only)
      determin_cpx = nr_determin_cpx(size,
                                     a44_cpx, //
                                     +1);
      //if (i<4) printf("order %d nr_det_cpx = %lf+j%lf \n",log2_approx(creal(determin_cpx)),creal(determin_cpx),cimag(determin_cpx));

      //Round and convert to Q15 (Out in the same format as Fixed point).
      if (creal(determin_cpx)>0) {//determin of the symmetric matrix is real part only
        ((short *)ad_bc)[i << 1] = (short)((creal(determin_cpx) * (1 << (shift0))) + 0.5); //
      } else {
        ((short *)ad_bc)[i << 1] = (short)((creal(determin_cpx) * (1 << (shift0))) - 0.5); //
      }
      //Compute Inversion of the H^*H matrix (normalized output divide by determinant)
      for (int rtx=0;rtx<size;rtx++) {//row
        k=0;
        for(int rrtx=0;rrtx<size;rrtx++)
          if(rrtx != rtx) rr[k++] = rrtx;
        for (int ctx=0;ctx<size;ctx++) {//column
          k=0;
          for(int cctx=0;cctx<size;cctx++)
            if(cctx != ctx) cc[k++] = cctx;

          //fill out the sub matrix corresponds to this element
          for (int ridx=0;ridx<(size-1);ridx++)
            for (int cidx=0;cidx<(size-1);cidx++)
              sub_matrix_cpx[cidx][ridx] = a44_cpx[cc[cidx]][rr[ridx]];

          inv_H_h_H_cpx[rtx][ctx] = nr_determin_cpx(size - 1, // size,
                                                    sub_matrix_cpx, //
                                                    ((rtx & 1) == 1 ? -1 : 1) * ((ctx & 1) == 1 ? -1 : 1));
          //if (i==0) printf("H_h_H(r%d,c%d)=%lf+j%lf --> inv_H_h_H(%d,%d) = %lf+j%lf \n",rtx,ctx,creal(a44_cpx[ctx*size+rtx]),cimag(a44_cpx[ctx*size+rtx]),ctx,rtx,creal(inv_H_h_H_cpx[rtx*size+ctx]),cimag(inv_H_h_H_cpx[rtx*size+ctx]));

          if (creal(inv_H_h_H_cpx[rtx][ctx]) > 0)
            inv_H_h_H[rtx][ctx][i].r = (short)((creal(inv_H_h_H_cpx[rtx][ctx]) * (1 << (shift0 - 1))) + 0.5); // Convert to Q 18
          else
            inv_H_h_H[rtx][ctx][i].r = (short)((creal(inv_H_h_H_cpx[rtx][ctx]) * (1 << (shift0 - 1))) - 0.5); //

          if (cimag(inv_H_h_H_cpx[rtx][ctx]) > 0)
            inv_H_h_H[rtx][ctx][i].i = (short)((cimag(inv_H_h_H_cpx[rtx][ctx]) * (1 << (shift0 - 1))) + 0.5); //
          else
            inv_H_h_H[rtx][ctx][i].i = (short)((cimag(inv_H_h_H_cpx[rtx][ctx]) * (1 << (shift0 - 1))) - 0.5); //

          //if (i<4) printf("inv_H_h_H_FP(%d,%d)= %d+j%d \n",ctx,rtx, ((short *) inv_H_h_H[rtx*size+ctx])[i<<1],((short *) inv_H_h_H[rtx*size+ctx])[(i<<1)+1]);
        }
      }
    }
  }
  return(0);
}

/* Zero Forcing Rx function: nr_conjch0_mult_ch1()
 *
 *
 * */
// TODO: This function is just a wrapper, can be removed.
void nr_conjch0_mult_ch1(c16_t *ch0, c16_t *ch1, c16_t *ch0conj_ch1, unsigned short nb_rb, unsigned char output_shift0)
{
  //This function is used to compute multiplications in H_hermitian * H matrix
  mult_cpx_conj_vector(ch0, ch1, ch0conj_ch1, 12 * nb_rb, output_shift0);
}

/*
 * MMSE Rx function: up to 4 layers
 */
static void nr_dlsch_mmse(uint32_t rx_size_symbol,
                          unsigned char n_rx,
                          unsigned char nl, // number of layer
                          c16_t rxdataF_comp[][nl][rx_size_symbol],
                          c16_t dl_ch_mag[][rx_size_symbol],
                          c16_t dl_ch_magb[][rx_size_symbol],
                          c16_t dl_ch_magr[][rx_size_symbol],
                          int32_t dl_ch_estimates_ext[][rx_size_symbol],
                          unsigned char mod_order,
                          int shift,
                          unsigned char symbol,
                          int length,
                          uint32_t noise_var)
{
  uint32_t nb_rb_0 = (length + 11) / 12;
  c16_t determ_fin[12 * nb_rb_0] __attribute__((aligned(32)));

  ///Allocate H^*H matrix elements and sub elements
  c16_t conjH_H_elements_data[n_rx][nl][nl][12 * nb_rb_0];
  memset(conjH_H_elements_data, 0, sizeof(conjH_H_elements_data));
  c16_t *conjH_H_elements[n_rx][nl][nl];
  for (int aarx = 0; aarx < n_rx; aarx++)
    for (int rtx = 0; rtx < nl; rtx++)
      for (int ctx = 0; ctx < nl; ctx++)
        conjH_H_elements[aarx][rtx][ctx] = conjH_H_elements_data[aarx][rtx][ctx];

  //Compute H^*H matrix elements and sub elements:(1/2^log2_maxh)*conjH_H_elements
  for (int rtx = 0; rtx < nl; rtx++) {//row
    for (int ctx = 0; ctx < nl; ctx++) {//column
      for (int aarx = 0; aarx < n_rx; aarx++)  {
        c16_t *ch0r = (c16_t *)dl_ch_estimates_ext[rtx * n_rx + aarx];
        c16_t *ch0c = (c16_t *)dl_ch_estimates_ext[ctx * n_rx + aarx];
        nr_conjch0_mult_ch1(ch0r,
                            ch0c,
                            conjH_H_elements[aarx][ctx][rtx], // sic
                            nb_rb_0,
                            shift);
        if (aarx != 0)
          nr_a_sum_b(conjH_H_elements[0][ctx][rtx], conjH_H_elements[aarx][ctx][rtx], nb_rb_0);
      }
    }
  }

  // Add noise_var such that: H^h * H + noise_var * I
  if (noise_var != 0) {
    simde__m128i nvar_128i = simde_mm_set1_epi32(noise_var >> 3);
    for (int p = 0; p < nl; p++) {
      simde__m128i *conjH_H_128i = (simde__m128i *)conjH_H_elements[0][p][p];
      for (int k = 0; k < 3 * nb_rb_0; k++) {
        conjH_H_128i[0] = simde_mm_add_epi32(conjH_H_128i[0], nvar_128i);
        conjH_H_128i++;
      }
    }
  }

  //Compute the inverse and determinant of the H^*H matrix
  //Allocate the inverse matrix
  c16_t *inv_H_h_H[nl][nl];
  c16_t inv_H_h_H_data[nl][nl][12 * nb_rb_0];
  memset(inv_H_h_H_data, 0, sizeof(inv_H_h_H_data));
  for (int rtx = 0; rtx < nl; rtx++)
    for (int ctx = 0; ctx < nl; ctx++)
      inv_H_h_H[ctx][rtx] = inv_H_h_H_data[ctx][rtx];

  int fp_flag = 1;//0: float point calc 1: Fixed point calc
  nr_matrix_inverse(nl,
                    conjH_H_elements[0], // Input matrix
                    inv_H_h_H, // Inverse
                    determ_fin, // determin
                    nb_rb_0,
                    fp_flag, // fixed point flag
                    shift - (fp_flag == 1 ? 1 : 0)); // the out put is Q15

  // multiply Matrix inversion pf H_h_H by the rx signal vector
  c16_t outtemp[12 * nb_rb_0] __attribute__((aligned(32)));
  //Allocate rxdataF for zforcing out
  c16_t rxdataF_zforcing[nl][12 * nb_rb_0];
  memset(rxdataF_zforcing, 0, sizeof(rxdataF_zforcing));

  for (int rtx = 0; rtx < nl; rtx++) {//Output Layers row
    // loop over Layers rtx=0,...,N_Layers-1
    for (int ctx = 0; ctx < nl; ctx++) { // column multi
      // printf("Computing r_%d c_%d\n",rtx,ctx);
      // print_shorts(" H_h_H=",(int16_t*)&conjH_H_elements[ctx*nl+rtx][0][0]);
      // print_shorts(" Inv_H_h_H=",(int16_t*)&inv_H_h_H[ctx*nl+rtx][0]);
      mult_complex_vectors(inv_H_h_H[ctx][rtx],
                           rxdataF_comp[symbol][ctx],
                           outtemp,
                           sizeofArray(outtemp),
                           shift - (fp_flag == 1 ? 1 : 0));
      nr_a_sum_b(rxdataF_zforcing[rtx], outtemp, nb_rb_0); // a = a + b
    }
#ifdef DEBUG_DLSCH_DEMOD
    printf("Computing layer_%d \n", rtx);
    print_shorts(" Rx signal:=", (int16_t*)&rxdataF_zforcing[rtx][0]);
    print_shorts(" Rx signal:=", (int16_t*)&rxdataF_zforcing[rtx][4]);
    print_shorts(" Rx signal:=", (int16_t*)&rxdataF_zforcing[rtx][8]);
#endif
  }

  //Copy zero_forcing out to output array
  for (int rtx = 0; rtx < nl; rtx++)
    nr_element_sign(rxdataF_zforcing[rtx], rxdataF_comp[symbol][rtx], nb_rb_0, +1);

  //Update LLR thresholds with the Matrix determinant
  simde__m128i *dl_ch_mag128_0=NULL,*dl_ch_mag128b_0=NULL,*dl_ch_mag128r_0=NULL,*determ_fin_128;
  simde__m128i mmtmpD2,mmtmpD3;
  simde__m128i QAM_amp128={0},QAM_amp128b={0},QAM_amp128r={0};
  short nr_realpart[8]__attribute__((aligned(16))) = {1,0,1,0,1,0,1,0};
  determ_fin_128      = (simde__m128i *)&determ_fin[0];

  if (mod_order > 2) {
    if (mod_order == 4) {
      QAM_amp128 = simde_mm_set1_epi16(QAM16_n1);  //2/sqrt(10)
      QAM_amp128b = simde_mm_setzero_si128();
      QAM_amp128r = simde_mm_setzero_si128();
    } else if (mod_order == 6) {
      QAM_amp128  = simde_mm_set1_epi16(QAM64_n1); //4/sqrt{42}
      QAM_amp128b = simde_mm_set1_epi16(QAM64_n2); //2/sqrt{42}
      QAM_amp128r = simde_mm_setzero_si128();
    } else if (mod_order == 8) {
      QAM_amp128 = simde_mm_set1_epi16(QAM256_n1); //8/sqrt{170}
      QAM_amp128b = simde_mm_set1_epi16(QAM256_n2);//4/sqrt{170}
      QAM_amp128r = simde_mm_set1_epi16(QAM256_n3);//2/sqrt{170}
    }
    dl_ch_mag128_0 = (simde__m128i *)dl_ch_mag[0];
    dl_ch_mag128b_0 = (simde__m128i *)dl_ch_magb[0];
    dl_ch_mag128r_0 = (simde__m128i *)dl_ch_magr[0];

    for (int rb = 0; rb < 3 * nb_rb_0; rb++) {
      //for symmetric H_h_H matrix, the determinant is only real values
      mmtmpD2 = simde_mm_sign_epi16(determ_fin_128[0],*(simde__m128i*)&nr_realpart[0]);//set imag part to 0
      mmtmpD3 = simde_mm_shufflelo_epi16(mmtmpD2,SIMDE_MM_SHUFFLE(2,3,0,1));
      mmtmpD3 = simde_mm_shufflehi_epi16(mmtmpD3,SIMDE_MM_SHUFFLE(2,3,0,1));
      mmtmpD2 = simde_mm_add_epi16(mmtmpD2,mmtmpD3);

      dl_ch_mag128_0[0] = mmtmpD2;
      dl_ch_mag128b_0[0] = mmtmpD2;
      dl_ch_mag128r_0[0] = mmtmpD2;

      dl_ch_mag128_0[0] = simde_mm_mulhrs_epi16(dl_ch_mag128_0[0], QAM_amp128);
      dl_ch_mag128b_0[0] = simde_mm_mulhrs_epi16(dl_ch_mag128b_0[0],QAM_amp128b);
      dl_ch_mag128r_0[0] = simde_mm_mulhrs_epi16(dl_ch_mag128r_0[0],QAM_amp128r);

      determ_fin_128 += 1;
      dl_ch_mag128_0 += 1;
      dl_ch_mag128b_0 += 1;
      dl_ch_mag128r_0 += 1;
    }
  }
}

static void nr_dlsch_layer_demapping(const uint8_t Nl,
                                     const uint8_t mod_order,
                                     const int llrLayerSize,
                                     const int16_t llr_layers[NR_SYMBOLS_PER_SLOT][Nl][llrLayerSize],
                                     const fapi_nr_dl_config_dlsch_pdu_rel15_t *dlsch_config,
                                     const uint32_t re_len[NR_SYMBOLS_PER_SLOT],
                                     int16_t *llr)
{
  const int s0 = dlsch_config->start_symbol;
  const int s1 = dlsch_config->number_symbols;
  int k = 0;

  for (int i = s0; i < (s0 + s1); i++) {
    int16_t *p_layer[Nl];
    for (int l = 0; l < Nl; l++)
      p_layer[l] = (int16_t *)llr_layers[i][l];
    nr_layer_demapping(Nl, mod_order, re_len[i], p_layer, llr + k);
    k += re_len[i] * mod_order * Nl;
  }
}

/* Computes LLRs from compensated PDSCH signal per OFDM symbol for all layers */
static int nr_dlsch_llr(const NR_UE_DLSCH_t *dlsch,
                        const int len,
                        const int rx_size_symbol,
                        const c16_t dl_ch_mag[rx_size_symbol],
                        const c16_t dl_ch_magb[rx_size_symbol],
                        const c16_t dl_ch_magr[rx_size_symbol],
                        const int nb_antennas_rx,
                        const c16_t rxdataF_comp[dlsch->cw_info.Nl][rx_size_symbol],
                        const int llrSize,
                        int16_t layer_llr[dlsch->cw_info.Nl][llrSize])
{
  switch (dlsch->cw_info.qamModOrder) {
    case 2 :
      for (int l = 0; l < dlsch->cw_info.Nl; l++)
        nr_qpsk_llr(rxdataF_comp[l], layer_llr[l], len);
      break;

    case 4 :
      for (int l = 0; l < dlsch->cw_info.Nl; l++)
        nr_16qam_llr(rxdataF_comp[l], dl_ch_mag, layer_llr[l], len);
      break;

    case 6 :
      for(int l=0; l < dlsch->cw_info.Nl; l++)
        nr_64qam_llr(rxdataF_comp[l], dl_ch_mag, dl_ch_magb, layer_llr[l], len);
      break;

    case 8:
      for(int l=0; l < dlsch->cw_info.Nl; l++)
        nr_256qam_llr(rxdataF_comp[l], dl_ch_mag, dl_ch_magb, dl_ch_magr, layer_llr[l], len);
      break;

    default:
      AssertFatal(false, "Unknown mod_order!!!!\n");
      break;
  }

  return 0;
}
//==============================================================================================

/* Main Function */

int nr_rx_pdsch(PHY_VARS_NR_UE *ue,
                const UE_nr_rxtx_proc_t *proc,
                NR_UE_DLSCH_t *dlsch,
                const freq_alloc_bitmap_t *freq_alloc,
                fapi_nr_dl_config_dlsch_pdu_rel15_t *dlsch_config,
                NR_DL_UE_HARQ_t *dlsch_harq,
                unsigned char symbol,
                bool first_symbol_flag,
                unsigned char harq_pid,
                uint32_t pdsch_est_size,
                int32_t dl_ch_estimates[][pdsch_est_size],
                int16_t *llr,
                uint32_t dl_valid_re[NR_SYMBOLS_PER_SLOT],
                c16_t rxdataF[][ue->frame_parms.samples_per_slot_wCP],
                int32_t *log2_maxh,
                int rx_size_symbol,
                int nbRx,
                c16_t rxdataF_comp[][dlsch->cw_info.Nl][rx_size_symbol],
                c16_t dl_ch_mag[][dlsch->cw_info.Nl][rx_size_symbol],
                c16_t dl_ch_magb[][dlsch->cw_info.Nl][rx_size_symbol],
                c16_t dl_ch_magr[][dlsch->cw_info.Nl][rx_size_symbol],
                c16_t ptrs_phase_per_slot[][NR_SYMBOLS_PER_SLOT],
                int32_t ptrs_re_per_slot[][NR_SYMBOLS_PER_SLOT],
                uint32_t nvar,
                pdsch_scope_req_t *scope_req,
                c16_t rho_dl[][dlsch->cw_info.Nl * dlsch->cw_info.Nl][rx_size_symbol])
{
  NR_DL_FRAME_PARMS *fp = &ue->frame_parms;
  const int nl = dlsch->cw_info.Nl;
  const int matrixSz = nbRx * nl;
  __attribute__((aligned(32))) int32_t dl_ch_estimates_ext[matrixSz][rx_size_symbol];
  memset(dl_ch_estimates_ext, 0, sizeof(dl_ch_estimates_ext));

  // Use ML-based LLR for 2-layer MIMO with QPSK/16QAM/64QAM (nl==2, qamModOrder<=6).
  // Controlled by ue->do_ml (set via -E flag in dlsim, or ue->do_ml in the UE struct).
  // When false (default), MMSE equalization is used for all configurations.
  bool do_ml = ue->do_ml;

  // Reinterpret flat dl_ch_estimates_ext as [nl][nbRx][rx_size_symbol]
  c16_t(*chFext)[nbRx][rx_size_symbol] = (void *)dl_ch_estimates_ext;

  c16_t *p_rxComp[nl];
  for (int l = 0; l < nl; l++)
    p_rxComp[l] = rxdataF_comp[symbol][l];

  NR_UE_COMMON *common_vars  = &ue->common_vars;
  const int frame = proc->frame_rx;
  const int nr_slot_rx = proc->nr_slot_rx;
  const int gNB_id = proc->gNB_id;
  uint8_t slot = 0;

  uint32_t nb_re_pdsch = -1;
  DevAssert(dlsch_harq);

  if (gNB_id > 2) {
    LOG_E(PHY, "Illegal gNB_id %d\n", gNB_id);
    return(-1);
  }

  if (!common_vars) {
    LOG_E(PHY, "dlsch_demodulation.c: Null common_vars\n");
    return(-1);
  }

  if(symbol > fp->symbols_per_slot >> 1)
    slot = 1;

  uint8_t pilots = (dlsch_config->dlDmrsSymbPos >> symbol) & 1;
  uint8_t config_type = dlsch_config->dmrsConfigType;

  const bool need_rho = do_ml ? (nl == 2 && dlsch_config->cw_info->qamModOrder <= 6) : false;

  //----------------------------------------------------------
  //--------------------- RBs extraction ---------------------
  //----------------------------------------------------------
  const bool meas_enabled = cpumeas(CPUMEAS_GETSTATE);
  int nb_rb_pdsch = freq_alloc->num_rbs;

  start_meas_nr_ue_phy(ue, DLSCH_EXTRACT_RBS_STATS);
  __attribute__((aligned(32))) c16_t rxdataF_ext[nbRx][rx_size_symbol];
  memset(rxdataF_ext, 0, sizeof(rxdataF_ext));

  uint32_t csi_res_bitmap = build_csi_overlap_bitmap(dlsch_config, symbol);
  LOG_D(PHY, "%d.%d symbol %d csi overlap bitmap %d\n", frame, nr_slot_rx, symbol, csi_res_bitmap);

  nr_dlsch_extract_rbs(fp->samples_per_slot_wCP,
                       rxdataF,
                       rx_size_symbol,
                       pdsch_est_size,
                       dl_ch_estimates,
                       rxdataF_ext,
                       dl_ch_estimates_ext,
                       symbol,
                       pilots,
                       dlsch_config,
                       freq_alloc,
                       nl,
                       fp,
                       csi_res_bitmap,
                       ue->chest_time);
  stop_meas_nr_ue_phy(ue, DLSCH_EXTRACT_RBS_STATS);
  if (scope_req->copy_chanest_to_scope) {
    size_t size = sizeof(c16_t) * nb_rb_pdsch * NR_NB_SC_PER_RB;
    int copy_index = symbol - dlsch_config->start_symbol;
    int offset = copy_index * size;
    UEscopeCopyUnsafe(ue, pdschChanEstimates, dl_ch_estimates_ext[0], size, offset, copy_index);
  }
  if (meas_enabled) {
    LOG_D(PHY,
          "[AbsSFN %u.%d] Slot%d Symbol %d: Pilot/Data extraction %5.2f \n",
          frame,
          nr_slot_rx,
          slot,
          symbol,
          ue->phy_cpu_stats.cpu_time_stats[DLSCH_EXTRACT_RBS_STATS].p_time / (cpuf * 1000.0));
  }
  if (ue->phy_sim_pdsch_rxdataF_ext)
    memcpy(ue->phy_sim_pdsch_rxdataF_ext + symbol * sizeof(rxdataF_ext), rxdataF_ext, sizeof(rxdataF_ext));

  nb_re_pdsch = (pilots == 1) ?
                ((config_type == NFAPI_NR_DMRS_TYPE1) ? nb_rb_pdsch * (12 - 6 * dlsch_config->n_dmrs_cdm_groups) :
                nb_rb_pdsch * (12 - 4 * dlsch_config->n_dmrs_cdm_groups)):
                (nb_rb_pdsch * 12);
  // Subtract CSI-RS REs from PDSCH RE count
  if (csi_res_bitmap != 0) {
    uint32_t csi_re_count = 0;
    uint32_t csi_res_even = csi_res_bitmap & 0xfff;
    uint32_t csi_res_odd = (csi_res_bitmap >> 16) & 0xfff;
    uint32_t count_even = count_bits(&csi_res_even, 1);
    uint32_t count_odd  = count_bits(&csi_res_odd, 1);
    int start = freq_alloc->first_rb + dlsch_config->BWPStart;
    int end = freq_alloc->last_rb + 1;
    for (int rb = start; rb < end; rb++) {
      if ((freq_alloc->bitmap[rb / 32] >> (rb % 32)) & 0x01)
        csi_re_count += (rb % 2 == 0) ? count_even : count_odd;
    }
    nb_re_pdsch = (nb_re_pdsch > csi_re_count) ? (nb_re_pdsch - csi_re_count) : 0;
    if (csi_re_count > 0) {
      LOG_D(NR_PHY,
            "[CSI OVERLAP] Frame/Slot %d.%d Symbol %d: CSI-RS overlapping PDSCH - %d CSI-RS REs skipped, %d data REs extracted\n",
            frame,
            nr_slot_rx,
            symbol,
            csi_re_count,
            nb_re_pdsch);
    }
  }

  if (scope_req->copy_rxdataF_to_scope) {
    size_t size = sizeof(c16_t) * nb_re_pdsch;
    int copy_index = symbol - dlsch_config->start_symbol;
    UEscopeCopyUnsafe(ue, pdschRxdataF, rxdataF_ext[0], size, scope_req->scope_rxdataF_offset, copy_index);
    scope_req->scope_rxdataF_offset += size;
  }
  //----------------------------------------------------------
  //--------------------- Channel Scaling --------------------
  //----------------------------------------------------------
  start_meas_nr_ue_phy(ue, DLSCH_CHANNEL_SCALE_STATS);
  nr_scale_channel(rx_size_symbol, dl_ch_estimates_ext, 0, nb_re_pdsch, nl, nbRx, 0);
  stop_meas_nr_ue_phy(ue, DLSCH_CHANNEL_SCALE_STATS);
  if (meas_enabled) {
    LOG_D(PHY,
          "[AbsSFN %u.%d] Slot%d Symbol %d: Channel Scale  %5.2f \n",
          frame,
          nr_slot_rx,
          slot,
          symbol,
          ue->phy_cpu_stats.cpu_time_stats[DLSCH_CHANNEL_SCALE_STATS].p_time / (cpuf * 1000.0));
  }

  //----------------------------------------------------------
  //--------------------- Channel Level Calc. ----------------
  //----------------------------------------------------------
  start_meas_nr_ue_phy(ue, DLSCH_CHANNEL_LEVEL_STATS);
  if (first_symbol_flag) {
    int32_t avg[nl * nbRx];
    if (nb_re_pdsch)
      nr_channel_level(0, rx_size_symbol, (c16_t (*)[rx_size_symbol])dl_ch_estimates_ext, nbRx, nl, avg, nb_re_pdsch);
    else
      LOG_E(NR_PHY, "Average channel level is 0: nb_rb_pdsch = %d, nb_re_pdsch = %d\n", nb_rb_pdsch, nb_re_pdsch);
    int avgs = 0;
    int32_t median[MAX_ANT][MAX_ANT];
    for (int l = 0; l < nl; l++)
      for (int aarx = 0; aarx < nbRx; aarx++) {
        avgs = cmax(avgs, avg[l * nbRx + aarx]);
        LOG_D(PHY, "nb_rb %d avg_%d_%d Power per SC is %d\n", nb_rb_pdsch, aarx, l, avg[l * nbRx + aarx]);
        LOG_D(PHY, "avgs Power per SC is %d\n", avgs);
        median[l][aarx] = avg[l * nbRx + aarx];
      }

    if (nl > 1) {
      nr_dlsch_channel_level_median(rx_size_symbol, dl_ch_estimates_ext, median, nl, nbRx, nb_re_pdsch);
      for (int l = 0; l < nl; l++) {
        for (int aarx = 0; aarx < nbRx; aarx++) {
          avgs = cmax(avgs, median[l][aarx]);
        }
      }
    }
    // Output shift: half channel energy (log2|h|^2/2) + MRC antenna gain.
    // Single-layer adds +1 guard bit (raw peak); multi-layer uses median so no guard needed.
    if (nl == 1)
      *log2_maxh = (log2_approx(avgs) >> 1) + 1 + log2_approx(nbRx >> 1);
    else
      *log2_maxh = (log2_approx(avgs) >> 1) + log2_approx(nbRx >> 1);
    LOG_D(PHY, "[DLSCH] AbsSubframe %d.%d log2_maxh = %d (%d)\n", frame % 1024, nr_slot_rx, *log2_maxh, avgs);
#if T_TRACER
    T(T_UE_PHY_PDSCH_ENERGY,
      T_INT(gNB_id),
      T_INT(frame % 1024),
      T_INT(nr_slot_rx),
      T_INT(avg[0]), // layer 0, antenna 0
      T_INT(nbRx > 1 ? avg[1] : 0), // layer 0, antenna 1
      T_INT(nl > 1 ? avg[nbRx] : 0), // layer 1, antenna 0
      T_INT(nl > 1 && nbRx > 1 ? avg[nbRx + 1] : 0)); // layer 1, antenna 1
#endif
  }
  stop_meas_nr_ue_phy(ue, DLSCH_CHANNEL_LEVEL_STATS);
  if (meas_enabled) {
    LOG_D(PHY,
          "[AbsSFN %u.%d] Slot%d Symbol %d first_symbol_flag %d: Channel Level  %5.2f \n",
          frame,
          nr_slot_rx,
          slot,
          symbol,
          first_symbol_flag,
          ue->phy_cpu_stats.cpu_time_stats[DLSCH_CHANNEL_LEVEL_STATS].p_time / (cpuf * 1000.0));
  }

  //----------------------------------------------------------
  //--------------------- channel compensation ---------------
  //----------------------------------------------------------
  start_meas_nr_ue_phy(ue, DLSCH_CHANNEL_COMPENSATION_STATS);
  nr_channel_compensation(rx_size_symbol,
                          nbRx,
                          nl,
                          rxdataF_ext,
                          chFext,
                          dl_ch_mag[symbol],
                          dl_ch_magb[symbol],
                          dl_ch_magr[symbol],
                          p_rxComp,
                          need_rho ? (c16_t(*)[nl][rx_size_symbol])rho_dl[symbol] : NULL,
                          dlsch->cw_info.qamModOrder,
                          0, // symbol already baked into p_rxComp
                          *log2_maxh);
  stop_meas_nr_ue_phy(ue, DLSCH_CHANNEL_COMPENSATION_STATS);
  if (meas_enabled) {
    LOG_D(PHY,
          "[AbsSFN %u.%d] Slot%d Symbol %d log2_maxh %d Channel Comp  %5.2f \n",
          frame,
          nr_slot_rx,
          slot,
          symbol,
          *log2_maxh,
          ue->phy_cpu_stats.cpu_time_stats[DLSCH_CHANNEL_COMPENSATION_STATS].p_time / (cpuf * 1000.0));
  }
  // Please keep it: useful for debugging
#ifdef DEBUG_PDSCH_RX
  char filename[50];
  snprintf(filename, 50, "rxdataF0_symb_%d_nr_slot_rx_%d.m", symbol, nr_slot_rx);
  write_output(filename, "rxdataF0", &rxdataF[0][symbol * fp->ofdm_symbol_size], fp->ofdm_symbol_size, 1, 1);
  snprintf(filename, 50, "dl_ch_estimates0_symb_%d_nr_slot_rx_%d.m", symbol, nr_slot_rx);
  write_output(filename, "dl_ch_estimates0", &dl_ch_estimates[0][symbol * fp->ofdm_symbol_size], fp->ofdm_symbol_size, 1, 1);
  snprintf(filename, 50, "rxdataF_ext0_symb_%d_nr_slot_rx_%d.m", symbol, nr_slot_rx);
  write_output(filename, "rxdataF_ext0", &rxdataF_ext[0][0], rx_size_symbol, 1, 1);
  snprintf(filename, 50, "dl_ch_estimates_ext0_symb_%d_nr_slot_rx_%d.m", symbol, nr_slot_rx);
  write_output(filename, "dl_ch_estimates_ext0", &dl_ch_estimates_ext[0][0], rx_size_symbol, 1, 1);
  snprintf(filename, 50, "rxdataF_comp00_symb_%d_nr_slot_rx_%d.m", symbol, nr_slot_rx);
  write_output(filename, "rxdataF_comp00", &rxdataF_comp[0][0][symbol * rx_size_symbol], rx_size_symbol, 1, 1);
#endif

  // MRC is performed inline by nr_channel_compensation; apply MMSE for multi-layer
  start_meas_nr_ue_phy(ue, DLSCH_MRC_MMSE_STATS);
  if (nb_re_pdsch) {
    const uint8_t qamModOrder = dlsch->cw_info.qamModOrder;

    if ((nl > 2) || (nl == 2 && !do_ml)) {
      nr_dlsch_mmse(rx_size_symbol,
                    nbRx,
                    nl,
                    rxdataF_comp,
                    dl_ch_mag[symbol],
                    dl_ch_magb[symbol],
                    dl_ch_magr[symbol],
                    dl_ch_estimates_ext,
                    qamModOrder,
                    *log2_maxh,
                    symbol,
                    nb_re_pdsch,
                    nvar);
    } else if ((nl == 2) && (qamModOrder > 6) && do_ml) {
      nr_mmse_2layers(p_rxComp,
                      rx_size_symbol,
                      nbRx,
                      nl,
                      dl_ch_mag[symbol],
                      dl_ch_magb[symbol],
                      dl_ch_magr[symbol],
                      chFext,
                      freq_alloc->num_rbs,
                      qamModOrder,
                      *log2_maxh,
                      0,
                      nb_re_pdsch,
                      nvar);
    }
  }
  stop_meas_nr_ue_phy(ue, DLSCH_MRC_MMSE_STATS);

  if (meas_enabled) {
    LOG_D(PHY,
          "[AbsSFN %u.%d] Slot%d Symbol %d: Channel Combine and MMSE %5.2f \n",
          frame,
          nr_slot_rx,
          slot,
          symbol,
          ue->phy_cpu_stats.cpu_time_stats[DLSCH_MRC_MMSE_STATS].p_time / (cpuf * 1000.0));
  }

  /* Store the valid DL RE's */
  dl_valid_re[symbol] = nb_re_pdsch;
  int startSymbIdx = 0;
  int nbSymb = 0;
  int pduBitmap = 0;

  if(dlsch_harq->status == NR_ACTIVE) {
    startSymbIdx = dlsch_config->start_symbol;
    nbSymb = dlsch_config->number_symbols;
    pduBitmap = dlsch_config->pduBitmap;
  }

  /* PTRS processing for multiple antenna ports is broken because the following
  function estimates phase offset from and applies compensation to rxdataF_comp
  for each antenna port but rxdataF_comp has MRCed data. */
  /* TODO: Move PTRS phase estimation before immediately after DMRS channels
  estimation and apply PTRS phase compensation in nr_channel_compensationi() */
  /* Check for PTRS bitmap and process it respectively */
  if((pduBitmap & 0x1) && (dlsch->rnti_type == TYPE_C_RNTI_)) {
    nr_pdsch_ptrs_processing(1, // rxdataF_comp is MRCed so no point in processing all antenna ports. Fixme.
                             ptrs_phase_per_slot,
                             ptrs_re_per_slot,
                             rx_size_symbol,
                             nl,
                             rxdataF_comp,
                             fp,
                             dlsch_config,
                             nr_slot_rx,
                             symbol,
                             freq_alloc->num_rbs,
                             dlsch->rnti,
                             &dlsch->ptrs_symbols,
                             &dlsch->ptrs_symbol_index);
    dl_valid_re[symbol] -= ptrs_re_per_slot[0][symbol];
  }

  /* at last symbol in a slot calculate LLR's for whole slot */
  if (symbol == (startSymbIdx + nbSymb - 1)) {
    /* create LLR layer buffer */
    int max_symb_re = 0;
    GET_ARRAY_MAX(dl_valid_re, NR_SYMBOLS_PER_SLOT, max_symb_re);
    const int llr_per_symbol = max_symb_re * dlsch->cw_info.qamModOrder;
    __attribute__((aligned(32))) int16_t layer_llr[NR_SYMBOLS_PER_SLOT][nl][llr_per_symbol];

    // Generate LLR from PTRS compensated signal
    const uint8_t qamModOrder = dlsch->cw_info.qamModOrder;
    start_meas_nr_ue_phy(ue, DLSCH_LLR_STATS);
    for (int llr_sym = startSymbIdx; llr_sym < startSymbIdx + nbSymb; llr_sym++) {
      if (nl == 2 && qamModOrder <= 6 && do_ml) {
        // 2-layer QPSK/16QAM/64QAM: joint ML-LLR using inter-layer Tx correlation
        // rho_dl[llr_sym] is laid out as [nl*nl][rx_size_symbol]:
        // index 1 = rho[0][1], index nl (=2) = rho[1][0]
        nr_compute_ML_llr(rxdataF_comp[llr_sym][0],
                          rxdataF_comp[llr_sym][1],
                          dl_ch_mag[llr_sym][0],
                          dl_ch_mag[llr_sym][1],
                          layer_llr[llr_sym][0],
                          layer_llr[llr_sym][1],
                          rho_dl[llr_sym][1],
                          rho_dl[llr_sym][nl],
                          dl_valid_re[llr_sym],
                          qamModOrder);
      } else {
        nr_dlsch_llr(dlsch,
                     dl_valid_re[llr_sym],
                     rx_size_symbol,
                     dl_ch_mag[llr_sym][0],
                     dl_ch_magb[llr_sym][0],
                     dl_ch_magr[llr_sym][0],
                     nbRx,
                     rxdataF_comp[llr_sym],
                     llr_per_symbol,
                     layer_llr[llr_sym]);
      }
    }
    stop_meas_nr_ue_phy(ue, DLSCH_LLR_STATS);
    start_meas_nr_ue_phy(ue, DLSCH_LAYER_DEMAPPING);
    nr_dlsch_layer_demapping(nl, dlsch->cw_info.qamModOrder, llr_per_symbol, layer_llr, dlsch_config, dl_valid_re, llr);
    stop_meas_nr_ue_phy(ue, DLSCH_LAYER_DEMAPPING);

    if (UEScopeHasTryLock(ue)) {
      metadata mt = {.frame = proc->frame_rx, .slot = proc->nr_slot_rx };
      int total_valid_res = 0;
      for (int i = startSymbIdx; i < startSymbIdx + nbSymb; i++) {
        total_valid_res += dl_valid_re[i];
      }
      if (UETryLockScopeData(ue, pdschRxdataF_comp, sizeof(c16_t), 1,  total_valid_res, &mt)) {
        size_t offset = 0;
        for (int i = startSymbIdx; i < startSymbIdx + nbSymb; i++) {
          size_t data_size = sizeof(c16_t) * dl_valid_re[i];
          UEscopeCopyUnsafe(ue, pdschRxdataF_comp, &rxdataF_comp[i][0][0], data_size, offset, i);
          offset += data_size;
        }
        UEunlockScopeData(ue, pdschRxdataF_comp)
      }
    } else {
      UEscopeCopy(ue, pdschRxdataF_comp, rxdataF_comp[0], sizeof(c16_t), nl, rx_size_symbol, 0);
    }
  }

  if (meas_enabled) {
    LOG_D(PHY,
          "[AbsSFN %u.%d] Slot%d Symbol %d: LLR Computation  %5.2f \n",
          frame,
          nr_slot_rx,
          slot,
          symbol,
          ue->phy_cpu_stats.cpu_time_stats[DLSCH_LLR_STATS].p_time / (cpuf * 1000.0));
  }

#if T_TRACER
  T(T_UE_PHY_PDSCH_IQ,
    T_INT(gNB_id),
    T_INT(frame % 1024),
    T_INT(nr_slot_rx),
    T_INT(nb_rb_pdsch),
    T_INT(fp->N_RB_DL),
    T_INT(fp->symbols_per_slot),
    T_BUFFER(&rxdataF_comp[gNB_id][0], 2 * fp->N_RB_DL * 12 * fp->symbols_per_slot * 2));
#endif

  if (ue->phy_sim_pdsch_rxdataF_comp) {
    for (int a = 0; a < nbRx; a++) {
      memcpy((c16_t *)ue->phy_sim_pdsch_dl_ch_estimates + pdsch_est_size * a, dl_ch_estimates, pdsch_est_size * sizeof(c16_t));
    }
    for (int l = 0; l < nl; l++) {
      int offset = (void *)rxdataF_comp[symbol][l] - (void *)rxdataF_comp[0];
      memcpy(ue->phy_sim_pdsch_rxdataF_comp + offset, rxdataF_comp[symbol][l], sizeof(c16_t) * rx_size_symbol);
    }
  }
  if (ue->phy_sim_pdsch_dl_ch_estimates_ext)
    memcpy(ue->phy_sim_pdsch_dl_ch_estimates_ext + symbol * sizeof(dl_ch_estimates_ext),
           dl_ch_estimates_ext,
           sizeof(dl_ch_estimates_ext));
  return 0;
}

static void nr_mmse_sse(const c16_t *chest,
                        c16_t *rxdataF_comp,
                        c16_t *determin,
                        unsigned int nl,
                        unsigned int nb_rx,
                        unsigned int sse_idx,
                        unsigned int scale,
                        uint32_t noise_var)
{
  // Compute H^*H matrix
  c16_t conjH_H[nb_rx][nl][nl][4] __attribute__((aligned(16)));
  for (unsigned int rtx = 0; rtx < nl; rtx++) {
    for (unsigned int ctx = 0; ctx < nl; ctx++) {
      for (unsigned int aarx = 0; aarx < nb_rx; aarx++) {
        const c16_t *chr = chest + (rtx * nb_rx + aarx) * NR_NB_SC_PER_RB + sse_idx * 4;
        const c16_t *chc = chest + (ctx * nb_rx + aarx) * NR_NB_SC_PER_RB + sse_idx * 4;
        mult_cpx_conj_vector(chr, chc, conjH_H[aarx][rtx][ctx], 4, scale);
        if (aarx != 0)
          *(simde__m128i *)conjH_H[0][rtx][ctx] =
              simde_mm_add_epi16(*(simde__m128i *)conjH_H[0][rtx][ctx], *(simde__m128i *)conjH_H[aarx][rtx][ctx]);
      }
    }
  }

  // Add noise variance
  if (noise_var != 0) {
    simde__m128i nvar = simde_mm_set1_epi32(noise_var >> 3);
    for (unsigned int p = 0; p < nl; p++) {
      *(simde__m128i *)conjH_H[0][p][p] =
          simde_mm_add_epi32(*(simde__m128i *)conjH_H[0][p][p], nvar); // cannot add q15 complex with int32_t. FIXME
    }
  }

  // Inverse
  c16_t inv_H_H[nl][nl][4] __attribute__((aligned(16)));
  c16_t *p_conjH_H[nl][nl];
  c16_t *p_inv_H_H[nl][nl];
  for (unsigned int rtx = 0; rtx < nl; rtx++) {
    for (unsigned int ctx = 0; ctx < nl; ctx++) {
      p_conjH_H[rtx][ctx] = conjH_H[0][rtx][ctx];
      p_inv_H_H[rtx][ctx] = inv_H_H[rtx][ctx];
    }
  }
  nr_matrix_inverse_sse(nl,
                        p_conjH_H, // Input matrix
                        p_inv_H_H, // Inverse
                        determin, // determin
                        scale - 1); // the out put is Q15

  // Multiply inverse H_h_H with rx signal
  simde__m128i rxdataF_zf[nl];
  for (unsigned int rtx = 0; rtx < nl; rtx++) {
    rxdataF_zf[rtx] = simde_mm_setzero_si128();
    for (unsigned int ctx = 0; ctx < nl; ctx++) {
      c16_t outtemp[4] __attribute__((aligned(16)));
      mult_complex_vectors(p_inv_H_H[rtx][ctx], rxdataF_comp + ctx * 4, outtemp, sizeofArray(outtemp), scale - 1);
      ((simde__m128i *)rxdataF_zf)[rtx] = simde_mm_add_epi16(((simde__m128i *)rxdataF_zf)[rtx], *(simde__m128i *)outtemp);
    }
  }

  // Copy back to input
  for (unsigned int rtx = 0; rtx < nl; rtx++) {
    *(simde__m128i *)(rxdataF_comp + rtx * 4) = rxdataF_zf[rtx];
  }
}

#define NUM_AVX2_VECT 3
#define NUM_SSE_VECT NUM_AVX2_VECT
#define NUM_SSE_RE NUM_AVX2_VECT * 4

static inline int compress_c16_inplace(c16_t *arr, uint32_t mask)
{
  int w = 0;
  uint32_t m = mask;
  while (m) {
    int i = __builtin_ctz(m);
    arr[w++] = arr[i];
    m &= m - 1;
  }
  return w;
}

static inline int compress_cf_inplace(cf_t *arr, uint32_t mask)
{
  int w = 0;
  uint32_t m = mask;
  while (m) {
    int i = __builtin_ctz(m);
    arr[w++] = arr[i];
    m &= m - 1;
  }
  return w;
}

// Process one PRB
static int process_symbol_subband_sse(const c16_t *rxdataF,
                                      const c16_t *chest,
                                      c16_t cpe,
                                      int16_t *llr,
                                      uint32_t nvar,
                                      int num_antenna,
                                      uint8_t mod_order,
                                      uint16_t valid_re_mask)
{
  int16_t *llr_start = llr;
  // channel level calc
  const int16_t x = factor2(NUM_SSE_RE);
  const int16_t y = NUM_SSE_RE >> x;
  const int32_t avg = simde_mm_average((simde__m128i *)chest, NUM_SSE_RE, x, y);

  // CPE
  const simde__m128i cpe128 = simde_mm_set_epi16(cpe.i, cpe.r, cpe.i, cpe.r, cpe.i, cpe.r, cpe.i, cpe.r);

  // compensation
  const int32_t log2_maxh = (log2_approx(avg) >> 1) + log2_approx(num_antenna >> 1);
  const simde__m128i *rxF128 = (const simde__m128i *)rxdataF;
  const simde__m128i *ch128 = (const simde__m128i *)chest;
  for (uint_fast8_t i = 0; i < NUM_SSE_VECT; i++) {
    const uint8_t curr_mask = (valid_re_mask >> (i * 4)) & 0xf;
    if (curr_mask == 0) // No valid REs
      continue;

    simde__m128i chmaga = simde_mm_setzero_si128();
    simde__m128i chmagb = simde_mm_setzero_si128();
    simde__m128i chmagc = simde_mm_setzero_si128();
    simde__m128i comp = simde_mm_setzero_si128();

    unsigned int num_valid_re = 0;
    // MRC
    for (int aarx = 0; aarx < num_antenna; aarx++) {
      unsigned int rxoff = aarx * NUM_SSE_VECT + i;
      // chest + cpe
      const simde__m128i ch128_cpe = oai_mm_cpx_mult(ch128[rxoff], cpe128, 15);
      comp = simde_mm_add_epi16(comp, oai_mm_cpx_mult_conj(ch128_cpe, rxF128[rxoff], log2_maxh));

      simde__m128i mag = oai_mm_smadd(ch128[rxoff], ch128[rxoff], log2_maxh);
      mag = simde_mm_packs_epi32(mag, mag);
      mag = simde_mm_unpacklo_epi16(mag, mag);
      if (mod_order == 4)
        chmaga = simde_mm_add_epi16(chmaga, simde_mm_mulhrs_epi16(mag, simde_mm_set1_epi16(QAM16_n1)));
      else if (mod_order == 6) {
        chmaga = simde_mm_add_epi16(chmaga, simde_mm_mulhrs_epi16(mag, simde_mm_set1_epi16(QAM64_n1)));
        chmagb = simde_mm_add_epi16(chmagb, simde_mm_mulhrs_epi16(mag, simde_mm_set1_epi16(QAM64_n2)));
      } else if (mod_order == 8) {
        chmaga = simde_mm_add_epi16(chmaga, simde_mm_mulhrs_epi16(mag, simde_mm_set1_epi16(QAM256_n1)));
        chmagb = simde_mm_add_epi16(chmagb, simde_mm_mulhrs_epi16(mag, simde_mm_set1_epi16(QAM256_n2)));
        chmagc = simde_mm_add_epi16(chmagc, simde_mm_mulhrs_epi16(mag, simde_mm_set1_epi16(QAM256_n3)));
      }
    }

    // Group valid REs
    if (curr_mask == 0xf) {
      num_valid_re += 4;
    } else {
      // Group valid REs to start of buffer
      num_valid_re += compress_c16_inplace((c16_t *)&comp, curr_mask);
    }

    // LLR generation
    switch (mod_order) {
      case 2:
        nr_qpsk_llr((c16_t *)&comp, llr, num_valid_re);
        break;

      case 4:
        nr_16qam_llr((c16_t *)&comp, (c16_t *)&chmaga, llr, num_valid_re);
        break;

      case 6:
        nr_64qam_llr((c16_t *)&comp, (c16_t *)&chmaga, (c16_t *)&chmagb, llr, num_valid_re);
        break;

      case 8:
        nr_256qam_llr((c16_t *)&comp, (c16_t *)&chmaga, (c16_t *)&chmagb, (c16_t *)&chmagc, llr, num_valid_re);
        break;

      default:
        AssertFatal(0, "Unknown mod order!\n");
    }
    llr += (num_valid_re * mod_order);
  }
  return (int)(llr - llr_start);
}

static inline void interleave_complexfloat256(simde__m256 a, simde__m256 b, simde__m256 *lo, simde__m256 *hi)
{
  simde__m256d ad = simde_mm256_castps_pd(a);
  simde__m256d bd = simde_mm256_castps_pd(b);

  simde__m256d lo_lanes = simde_mm256_unpacklo_pd(ad, bd); // a0 b0 | a2 b2
  simde__m256d hi_lanes = simde_mm256_unpackhi_pd(ad, bd); // a1 b1 | a3 b3

  *lo = simde_mm256_castpd_ps(simde_mm256_permute2f128_pd(lo_lanes, hi_lanes, 0x20)); // a0 b0 a1 b1
  *hi = simde_mm256_castpd_ps(simde_mm256_permute2f128_pd(lo_lanes, hi_lanes, 0x31)); // a2 b2 a3 b3
}

static inline void interleave_complexfloat(const cf_t *a, const cf_t *b, const uint len, cf_t *out)
{
  for (uint i = 0; i < len; i++) {
    out[i * 2] = a[i];
    out[i * 2 + 1] = b[i];
  }
}

static inline void nr_qpsk_llr_float(const cf_t *in, const uint len, int16_t *out)
{
  for (uint i = 0; i < len; i++) {
    out[i * 2] = in[i].r * 16;
    out[i * 2 + 1] = in[i].i * 16;
  }
}

// y = conj(a0) * b0 + conj(a1) * b1, componentwise over the 12 REs of one RB. This is the shape of
// every term of H^H * y and of H^H * H for 2 receive antennas. y must not alias any of the inputs.
static inline void sum_conj_mult_2_ant(const simde__m256 *a0,
                                       const simde__m256 *b0,
                                       const simde__m256 *a1,
                                       const simde__m256 *b1,
                                       simde__m256 *y)
{
  simde__m256 t[NUM_AVX2_VECT];
  mult_conj_cf_vector((const cf_t *)a0, (const cf_t *)b0, (cf_t *)y, NR_NB_SC_PER_RB);
  mult_conj_cf_vector((const cf_t *)a1, (const cf_t *)b1, (cf_t *)t, NR_NB_SC_PER_RB);
  for (uint i = 0; i < NUM_AVX2_VECT; i++)
    y[i] = simde_mm256_add_ps(y[i], t[i]);
}


static int process_symbol_subband_sse_2_layer_2_ant(const c16_t *rxdataF0,
                                                    const c16_t *rxdataF1,
                                                    const c16_t *chest0,
                                                    const c16_t *chest1,
                                                    c16_t cpe,
                                                    int16_t *llr,
                                                    uint32_t nvar,
                                                    uint8_t mod_order,
                                                    uint16_t valid_re_mask)
{
  // channel level calc
  const int16_t x = factor2(2 * NR_NB_SC_PER_RB);
  const int16_t y = (2 * NR_NB_SC_PER_RB) >> x;
  const int32_t avg = simde_mm_average((simde__m128i *)chest0, 2 * NR_NB_SC_PER_RB, x, y);
  int16_t *llr_start = llr;

  // Convert c16_t to cf_t
  //
  //           Tx0  Tx1
  //   H = Rx0 h00  h01
  //       Rx1 h10  h11
  //
  //   H^H = conj( h00  h10     <-- chest0
  //               h01  h11 )   <-- chest1
  //
  //   y = y0
  //       y1
  // One RB is 12 REs, i.e. exactly NUM_AVX2_VECT vectors of 4 complex floats, so the cf_t kernels
  // used from here on run over a whole RB, or over both antennas of one RB, in one call
  AssertFatal(4 * NUM_AVX2_VECT == NR_NB_SC_PER_RB, "a RB does not fit in NUM_AVX2_VECT vectors");

  // The fixed point samples are normalized by the RMS amplitude of the channel estimate, which
  // keeps the Gram matrix and its Cholesky factor in a range where float precision is comfortable
  const float cvt_divisor = sqrt((double)avg / 2);
  simde__m256 chest_ps0[2 * NUM_AVX2_VECT];
  simde__m256 chest_ps1[2 * NUM_AVX2_VECT];
  c16_to_cf_vector(chest0, (cf_t *)chest_ps0, 2 * NR_NB_SC_PER_RB, cvt_divisor);
  c16_to_cf_vector(chest1, (cf_t *)chest_ps1, 2 * NR_NB_SC_PER_RB, cvt_divisor);

  simde__m256 rxf_ps0[NUM_AVX2_VECT];
  simde__m256 rxf_ps1[NUM_AVX2_VECT];
  c16_to_cf_vector(rxdataF0, (cf_t *)rxf_ps0, NR_NB_SC_PER_RB, cvt_divisor);
  c16_to_cf_vector(rxdataF1, (cf_t *)rxf_ps1, NR_NB_SC_PER_RB, cvt_divisor);

  //   H^H * y = conj(h00)*y0 + conj(h10)*y1
  //             conj(h01)*y0 + conj(h11)*y1
  simde__m256 comp0[NUM_AVX2_VECT];
  simde__m256 comp1[NUM_AVX2_VECT];
  sum_conj_mult_2_ant(chest_ps0, rxf_ps0, chest_ps0 + NUM_AVX2_VECT, rxf_ps1, comp0);
  sum_conj_mult_2_ant(chest_ps1, rxf_ps0, chest_ps1 + NUM_AVX2_VECT, rxf_ps1, comp1);

  // G = H^H * H = |h00|^2 + |h10|^2   g01
  //               conj(g01)           |h01|^2 + |h11|^2
  // gram0 holds the first row of G, gram1 the second one. G is Hermitian, so the lower off-diagonal
  // is derived from the upper one rather than computed again: that is cheaper, and it makes G
  // exactly Hermitian, which the Cholesky decomposition below assumes.
  simde__m256 gram0[2 * NUM_AVX2_VECT];
  simde__m256 gram1[2 * NUM_AVX2_VECT];
  sum_conj_mult_2_ant(chest_ps0, chest_ps0, chest_ps0 + NUM_AVX2_VECT, chest_ps0 + NUM_AVX2_VECT, gram0);
  sum_conj_mult_2_ant(chest_ps0, chest_ps1, chest_ps0 + NUM_AVX2_VECT, chest_ps1 + NUM_AVX2_VECT, gram0 + NUM_AVX2_VECT);
  sum_conj_mult_2_ant(chest_ps1, chest_ps1, chest_ps1 + NUM_AVX2_VECT, chest_ps1 + NUM_AVX2_VECT, gram1 + NUM_AVX2_VECT);
  const simde__m256 conj_mask = simde_mm256_castsi256_ps(simde_mm256_set_epi32(-1, 0, -1, 0, -1, 0, -1, 0));
  const simde__m256 signflip = simde_mm256_and_ps(conj_mask, simde_mm256_set1_ps(-0.0f));
  for (uint i = 0; i < NUM_AVX2_VECT; i++) {
    gram1[i] = simde_mm256_xor_ps(gram0[i + NUM_AVX2_VECT], signflip); // g10 = conj(g01)
    // |h|^2 is real, but the FMA inside the conjugate product leaves the rounding error of h.r*h.i
    // in the imaginary part; clearing it keeps the diagonal of G, and with it that of L, real
    gram0[i] = simde_mm256_andnot_ps(conj_mask, gram0[i]);
    gram1[i + NUM_AVX2_VECT] = simde_mm256_andnot_ps(conj_mask, gram1[i + NUM_AVX2_VECT]);
  }

  // Cholesky decomposition
  //   G = L * L^H with L = l00   0
  //                        l10  l11
  // l0 holds l00, the two halves of l1 hold l10 and l11. Both diagonal elements are real.
  simde__m256 l0[NUM_AVX2_VECT];
  simde__m256 l1[2 * NUM_AVX2_VECT];
  {
    // l00 = sqrt(g00)
    sqrt_cf_vector((const cf_t *)gram0, (cf_t *)l0, NR_NB_SC_PER_RB);

    // l10 = g10 / l00
    div_cf_vector((const cf_t *)gram1, (const cf_t *)l0, (cf_t *)l1, NR_NB_SC_PER_RB);

    // l11 = sqrt(g11 - |l10|^2), where conj(l10) * l10 gives |l10|^2
    simde__m256 s[NUM_AVX2_VECT];
    mult_conj_cf_vector((const cf_t *)l1, (const cf_t *)l1, (cf_t *)s, NR_NB_SC_PER_RB);
    for (uint i = 0; i < NUM_AVX2_VECT; i++)
      s[i] = simde_mm256_sub_ps(gram1[i + NUM_AVX2_VECT], s[i]); // complex subtraction is componentwise
    sqrt_cf_vector((const cf_t *)s, (cf_t *)(l1 + NUM_AVX2_VECT), NR_NB_SC_PER_RB);
  }

  // Forward substitution
  //   L * z = H^H * y
  simde__m256 z0[NUM_AVX2_VECT];
  simde__m256 z1[NUM_AVX2_VECT];
  {
    // z0 = (H^H * y)0 / l00
    div_cf_vector((const cf_t *)comp0, (const cf_t *)l0, (cf_t *)z0, NR_NB_SC_PER_RB);

    // z1 = ((H^H * y)1 - l10 * z0) / l11
    simde__m256 d[NUM_AVX2_VECT];
    mult_cf_vector((const cf_t *)l1, (const cf_t *)z0, (cf_t *)d, NR_NB_SC_PER_RB);
    for (uint i = 0; i < NUM_AVX2_VECT; i++)
      d[i] = simde_mm256_sub_ps(comp1[i], d[i]);
    div_cf_vector((const cf_t *)d, (const cf_t *)(l1 + NUM_AVX2_VECT), (cf_t *)z1, NR_NB_SC_PER_RB);
  }

  // Backward substitution
  //   L^H * x = z, i.e. conj(l00) * x0 + conj(l10) * x1 = z0
  //                                      conj(l11) * x1 = z1
  // Conjugating the diagonal is a no-op as long as the Gram matrix is exactly Hermitian, it is kept
  // so that a residual imaginary part on the diagonal cannot leak into the result.
  simde__m256 x0[NUM_AVX2_VECT];
  simde__m256 x1[NUM_AVX2_VECT];
  {
    // x1 = z1 / conj(l11)
    simde__m256 diag[NUM_AVX2_VECT];
    for (uint i = 0; i < NUM_AVX2_VECT; i++)
      diag[i] = simde_mm256_xor_ps(l1[i + NUM_AVX2_VECT], signflip);
    div_cf_vector((const cf_t *)z1, (const cf_t *)diag, (cf_t *)x1, NR_NB_SC_PER_RB);

    // x0 = (z0 - conj(l10) * x1) / conj(l00)
    simde__m256 d[NUM_AVX2_VECT];
    mult_conj_cf_vector((const cf_t *)l1, (const cf_t *)x1, (cf_t *)d, NR_NB_SC_PER_RB);
    for (uint i = 0; i < NUM_AVX2_VECT; i++) {
      d[i] = simde_mm256_sub_ps(z0[i], d[i]);
      diag[i] = simde_mm256_xor_ps(l0[i], signflip);
    }
    div_cf_vector((const cf_t *)d, (const cf_t *)diag, (cf_t *)x0, NR_NB_SC_PER_RB);
  }

  // Group valid REs: extraction
  uint num_valid_re = 0;
  if (valid_re_mask == 0xfff) {
    num_valid_re = NR_NB_SC_PER_RB;
  } else {
    // Group valid REs to start of buffer
    num_valid_re = compress_cf_inplace((cf_t *)&x0, valid_re_mask);
    compress_cf_inplace((cf_t *)&x1, valid_re_mask);
  }

  // Layer demapping
  cf_t x_demapped[2 * NR_NB_SC_PER_RB];
  DevAssert(num_valid_re <= sizeofArray(x_demapped));
  interleave_complexfloat((cf_t *)&x0, (cf_t *)&x1, num_valid_re, x_demapped);

  // LLR generation
  const uint num_valid_re_all_layers = 2 * num_valid_re;
  switch (mod_order) {
    case 2:
      nr_qpsk_llr_float(x_demapped, num_valid_re_all_layers, llr);
      break;

    case 4:
      break;

    case 6:
      break;

    case 8:
      break;

    default:
      break;
  }
  llr += (2 * num_valid_re * mod_order);
  return (int)(llr - llr_start);
}

static inline uint16_t get_dmrs_re_bitmap(uint8_t config_type, uint8_t n_dmrs_cdm_groups)
{
  if (config_type == NFAPI_NR_DMRS_TYPE1)
    AssertFatal(n_dmrs_cdm_groups == 1 || n_dmrs_cdm_groups == 2, "n_dmrs_cdm_groups %d is illegal\n", n_dmrs_cdm_groups);
  else
    AssertFatal(n_dmrs_cdm_groups == 1 || n_dmrs_cdm_groups == 2 || n_dmrs_cdm_groups == 3,
                "n_dmrs_cdm_groups %d is illegal\n",
                n_dmrs_cdm_groups);

  uint16_t dmrs_rb_bitmap = 0;
  if (config_type == NFAPI_NR_DMRS_TYPE1 && n_dmrs_cdm_groups == 1)
    dmrs_rb_bitmap = 0x555; // alternating REs starting from 0
  else if (config_type == NFAPI_NR_DMRS_TYPE2 && n_dmrs_cdm_groups == 1)
    dmrs_rb_bitmap = 0xc3; // REs 0,1 and 6,7
  else if (config_type == NFAPI_NR_DMRS_TYPE2 && n_dmrs_cdm_groups == 2)
    dmrs_rb_bitmap = 0x3cf; // REs 0,1,2,3 and 6,7,8,9
  else
    dmrs_rb_bitmap = 0xfff; // all REs taken by dmrs

  return dmrs_rb_bitmap;
}

static inline uint16_t get_ptrs_re_bitmap(uint rnti, uint nb_rb, uint rb, uint k_re_ref, uint k_ptrs, uint k_rb_ref)
{
  if ((rb + k_rb_ref) % k_ptrs)
    return 0;
  else
    return (1U << k_re_ref);
}

// Returns valid RE bitmap starting from LSB.
static inline uint16_t get_combined_valid_re_bitmap(int rb_idx,
                                                    uint16_t ptrs_re_bitmap,
                                                    uint16_t dmrs_re_bitmap,
                                                    uint32_t csi_re_bitmap)
{
  const uint16_t ptrs_dmrs = (~ptrs_re_bitmap & ~dmrs_re_bitmap) & 0xfff;
  if (rb_idx & 1)
    return (~(csi_re_bitmap >> 16) & ptrs_dmrs); // MS 16bits for odd RB
  else
    return (~csi_re_bitmap & ptrs_dmrs);
}

int pdsch_process_symbol(const c16_t *rxdataF,
                         const c16_t *chest,
                         c16_t cpe,
                         int16_t *llr,
                         const freq_alloc_bitmap_t *freq_alloc,
                         uint8_t symbol,
                         uint32_t ofdm_symbol_size,
                         uint8_t mod_order,
                         unsigned int nb_rx,
                         unsigned int nb_layer,
                         uint32_t nvar,
                         uint16_t ptrs_symb_pos,
                         uint16_t rnti,
                         const fapi_nr_dl_config_dlsch_pdu_rel15_t *dlsch_config)
{
  int16_t *llr_start = llr;
  // CSI RE bitmap (LS 16bits even, MS 16bits odd RBs)
  uint32_t csi_res_bitmap = build_csi_overlap_bitmap(dlsch_config, symbol);

  // DMRS RE bitmap
  uint16_t dmrs_res_bitmap = 0;
  if (IS_BIT_SET(dlsch_config->dlDmrsSymbPos, symbol))
    dmrs_res_bitmap = get_dmrs_re_bitmap(dlsch_config->dmrsConfigType, dlsch_config->n_dmrs_cdm_groups);

  // PTRS RE bitmap staic info
  const uint k_re_ref = dlsch_config->PTRSReOffset;
  const uint k_ptrs = dlsch_config->PTRSFreqDensity;
  AssertFatal(k_re_ref < NR_NB_SC_PER_RB, "Invalid k_re_ref\n");
  const uint k_rb_ref = IS_BIT_SET(ptrs_symb_pos, symbol) ? get_ptrs_k_RB(freq_alloc->num_rbs, k_ptrs, rnti) : 0;

  int pos = 0;
  int block_start, block_end;
  while (find_next_rb_block(freq_alloc->bitmap, dlsch_config->BWPSize, &pos, &block_start, &block_end)) {
    int start_rb = block_start + dlsch_config->BWPStart;
    int nb_rb = block_end - block_start + 1;

    // Process one PRB
    for (int rb = start_rb; rb < start_rb + nb_rb; rb++) {
      // PTRS RE bitmap
      uint16_t ptrs_re_bitmap =
          IS_BIT_SET(ptrs_symb_pos, symbol) ? get_ptrs_re_bitmap(rnti, nb_rb, rb, k_re_ref, k_ptrs, k_rb_ref) : 0;

      unsigned int rx_offset = rb * nb_rx * NR_NB_SC_PER_RB;
      unsigned int ch_offset = rb * nb_rx * nb_layer * NR_NB_SC_PER_RB;
      uint16_t valid_re_mask = get_combined_valid_re_bitmap(rb, ptrs_re_bitmap, dmrs_res_bitmap, csi_res_bitmap);

      int num_llr = 0;
      if (nb_layer == 1)
        num_llr =
            process_symbol_subband_sse(rxdataF + rx_offset, chest + ch_offset, cpe, llr, nvar, nb_rx, mod_order, valid_re_mask);
      else if (nb_layer == 2)
        num_llr = process_symbol_subband_sse_2_layer_2_ant(rxdataF + rx_offset,
                                                           rxdataF + rx_offset + NR_NB_SC_PER_RB,
                                                           chest + ch_offset,
                                                           chest + ch_offset + nb_rx * NR_NB_SC_PER_RB,
                                                           cpe,
                                                           llr,
                                                           nvar,
                                                           mod_order,
                                                           valid_re_mask);
      llr += num_llr;
    }
  }
  return (int)(llr - llr_start);
}
