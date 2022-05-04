#include <arm_neon.h>
#include <iostream>
#include <vector>

#include "duration.hpp"

#define ASM_PREFETCH(address)    "PRFM PLDL1KEEP, " address "\n"
#define ASM_PREFETCHU(address)   "PRFUM PLDL1KEEP, " address "\n"
#define ASM_PREFETCHL2(address)  "PRFM PLDL2KEEP, " address "\n"
#define ASM_PREFETCHW(address)   "PRFM PSTL1KEEP, " address "\n"
#define ASM_PREFETCHWL2(address) "PRFM PSTL2KEEP, " address "\n"

void a64_sgemm_asimd_8x12_a53(const float *Apanel, const float *Bpanel, float *Cpanel, int ablocks, int bblocks, int K) {
    const float *a_ptr = Apanel;
    float *c_ptr = Cpanel;

    for (int yb=0; yb<ablocks; yb++) {
        const float *a_ptr0 = a_ptr;
        const float *b_ptr = Bpanel;

        for (int xb=0; xb<bblocks; xb++) {
            a_ptr = a_ptr0;
            // Fix up for odd lengths - set a flag if K is odd, but make
            // sure we round up the iteration count.
            int oddk = (K & 1);
            int k = ((K+1)/2) - 1;

            register float32x4_t a0  asm("v0");
            register float32x4_t a1  asm("v1");
            register float32x4_t b0  asm("v2");
            register float32x4_t b1  asm("v3");
            register float32x4_t b2  asm("v4");
            register float32x4_t a0a asm("v5");
            register float32x4_t a1a asm("v6");

            __asm __volatile (
                // Initialize result registers, load initial operands, prime prefetches.
                "movi    v8.4s, #0x0\n"
                "ldr    %q[a0], [%[a_ptr]]\n"
                "movi    v9.4s, #0x0\n"
                "ldr    %q[b0], [%[b_ptr]]\n"
                "movi    v10.4s, #0x0\n"
                "ldr    %q[a1], [%[a_ptr], #16]\n"
                "movi    v11.4s, #0x0\n"
                "ldr    %q[b1], [%[b_ptr], #16]\n"
                "movi    v12.4s, #0x0\n"
                ASM_PREFETCH("[%[b_ptr], #64]")
                "movi    v13.4s, #0x0\n"
                ASM_PREFETCH("[%[a_ptr], #64]")
                "movi    v14.4s, #0x0\n"
                ASM_PREFETCH("[%[b_ptr], #128]")
                "movi    v15.4s, #0x0\n"
                ASM_PREFETCH("[%[a_ptr], #128]")
                "movi    v16.4s, #0x0\n"
                ASM_PREFETCH("[%[b_ptr], #192]")
                "movi    v17.4s, #0x0\n"
                ASM_PREFETCH("[%[b_ptr], #256]")
                "movi    v18.4s, #0x0\n"
                ASM_PREFETCH("[%[a_ptr], #192]")
                "movi    v19.4s, #0x0\n"
                ASM_PREFETCH("[%[b_ptr], #320]")
                "movi    v20.4s, #0x0\n"
                ASM_PREFETCH("[%[a_ptr], #256]")
                "movi    v21.4s, #0x0\n"
                ASM_PREFETCH("[%[b_ptr], #384]")
                "movi    v22.4s, #0x0\n"
                "movi    v23.4s, #0x0\n"
                "movi    v24.4s, #0x0\n"
                "movi    v25.4s, #0x0\n"
                "movi    v26.4s, #0x0\n"
                "movi    v27.4s, #0x0\n"
                "movi    v28.4s, #0x0\n"
                "movi    v29.4s, #0x0\n"
                "movi    v30.4s, #0x0\n"
                "movi    v31.4s, #0x0\n"

                // Skip loop if we are doing zero iterations of it.
                "cbz    %w[k], 4f\n"

                "1:\n"
                // Unroll 0
                "ldr    %d[b2], [%[b_ptr], #32]\n"
                "nop\n"
                "fmla    v8.4s , %[b0].4s, %[a0].s[0]\n"
                "ldr    x20, [%[b_ptr], #40]\n"
                "fmla    v9.4s , %[b0].4s, %[a0].s[1]\n"
                "subs    %w[k], %w[k], #1\n"
                "fmla    v10.4s, %[b0].4s, %[a0].s[2]\n"

                "ldr    %d[a0a], [%[a_ptr], #32]\n"
                "ins    %[b2].d[1], x20\n"
                "fmla    v11.4s, %[b0].4s, %[a0].s[3]\n"
                "ldr    x20, [%[a_ptr], #40]\n"
                "fmla    v12.4s, %[b0].4s, %[a1].s[0]\n"
                "fmla    v13.4s, %[b0].4s, %[a1].s[1]\n"

                "ldr    %d[a1a], [%[a_ptr], #48]\n"
                "ins    %[a0a].d[1], x20\n"
                "fmla    v14.4s, %[b0].4s, %[a1].s[2]\n"
                "ldr    x20, [%[a_ptr], #56]\n"
                "fmla    v15.4s, %[b0].4s, %[a1].s[3]\n"
                "fmla    v16.4s, %[b1].4s, %[a0].s[0]\n"

                "ldr    %d[b0], [%[b_ptr], #48]\n"
                "ins    %[a1a].d[1], x20\n"
                "fmla    v17.4s, %[b1].4s, %[a0].s[1]\n"
                "ldr    x20, [%[b_ptr], #56]\n"
                "fmla    v18.4s, %[b1].4s, %[a0].s[2]\n"
                "fmla    v19.4s, %[b1].4s, %[a0].s[3]\n"

                ASM_PREFETCH("[%[a_ptr], #320]")
                "ins    %[b0].d[1], x20\n"
                "fmla    v20.4s, %[b1].4s, %[a1].s[0]\n"
                "fmla    v21.4s, %[b1].4s, %[a1].s[1]\n"
                "fmla    v22.4s, %[b1].4s, %[a1].s[2]\n"

                ASM_PREFETCH("[%[b_ptr], #448]")
                "nop\n"
                "fmla    v23.4s, %[b1].4s, %[a1].s[3]\n"
                "fmla    v24.4s, %[b2].4s, %[a0].s[0]\n"
                "fmla    v25.4s, %[b2].4s, %[a0].s[1]\n"

                "ldr    %d[b1], [%[b_ptr], #64]\n"
                "nop\n"
                "fmla    v26.4s, %[b2].4s, %[a0].s[2]\n"
                "ldr    x20, [%[b_ptr], #72]\n"
                "fmla    v27.4s, %[b2].4s, %[a0].s[3]\n"
                "fmla    v28.4s, %[b2].4s, %[a1].s[0]\n"

                ASM_PREFETCH("[%[b_ptr], #512]")
                "ins    %[b1].d[1], x20\n"
                "fmla    v29.4s, %[b2].4s, %[a1].s[1]\n"
                "fmla    v30.4s, %[b2].4s, %[a1].s[2]\n"
                "fmla    v31.4s, %[b2].4s, %[a1].s[3]\n"

                // Unroll 1
                "ldr    %d[b2], [%[b_ptr], #80]\n"
                "nop\n"
                "fmla    v8.4s , %[b0].4s, %[a0a].s[0]\n"
                "ldr    x20, [%[b_ptr], #88]\n"
                "fmla    v9.4s , %[b0].4s, %[a0a].s[1]\n"
                "fmla    v10.4s, %[b0].4s, %[a0a].s[2]\n"

                "ldr    %d[a0], [%[a_ptr], #64]\n"
                "ins    %[b2].d[1], x20\n"
                "fmla    v11.4s, %[b0].4s, %[a0a].s[3]\n"
                "ldr    x20, [%[a_ptr], #72]\n"
                "fmla    v12.4s, %[b0].4s, %[a1a].s[0]\n"
                "fmla    v13.4s, %[b0].4s, %[a1a].s[1]\n"

                "ldr    %d[a1], [%[a_ptr], #80]\n"
                "ins    %[a0].d[1], x20\n"
                "fmla    v14.4s, %[b0].4s, %[a1a].s[2]\n"
                "ldr    x20, [%[a_ptr], #88]\n"
                "fmla    v15.4s, %[b0].4s, %[a1a].s[3]\n"
                "fmla    v16.4s, %[b1].4s, %[a0a].s[0]\n"

                "ldr    %d[b0], [%[b_ptr], #96]\n"
                "ins    %[a1].d[1], x20\n"
                "fmla    v17.4s, %[b1].4s, %[a0a].s[1]\n"
                "ldr    x20, [%[b_ptr], #104]\n"
                "fmla    v18.4s, %[b1].4s, %[a0a].s[2]\n"
                "fmla    v19.4s, %[b1].4s, %[a0a].s[3]\n"

                "nop\n"
                "ins    %[b0].d[1], x20\n"
                "fmla    v20.4s, %[b1].4s, %[a1a].s[0]\n"
                "fmla    v21.4s, %[b1].4s, %[a1a].s[1]\n"
                "fmla    v22.4s, %[b1].4s, %[a1a].s[2]\n"

                "nop\n"
                "nop\n"
                "fmla    v23.4s, %[b1].4s, %[a1a].s[3]\n"
                "fmla    v24.4s, %[b2].4s, %[a0a].s[0]\n"
                "fmla    v25.4s, %[b2].4s, %[a0a].s[1]\n"

                "ldr    %d[b1], [%[b_ptr], #112]\n"
                "nop\n"
                "fmla    v26.4s, %[b2].4s, %[a0a].s[2]\n"
                "ldr    x20, [%[b_ptr], #120]\n"
                "fmla    v27.4s, %[b2].4s, %[a0a].s[3]\n"
                "add    %[a_ptr], %[a_ptr], #64\n"
                "fmla    v28.4s, %[b2].4s, %[a1a].s[0]\n"
                "add    %[b_ptr], %[b_ptr], #96\n"

                "nop\n"
                "ins    %[b1].d[1], x20\n"
                "fmla    v29.4s, %[b2].4s, %[a1a].s[1]\n"
                "fmla    v30.4s, %[b2].4s, %[a1a].s[2]\n"
                "fmla    v31.4s, %[b2].4s, %[a1a].s[3]\n"

                "bne    1b\n"

                // Branch here if K=1 or 2.  Do the right thing for odd/even at the end.
                "4:\n"
                "cbnz    %w[oddk], 2f\n"

                // Detached final iteration. (even K)
                "ldr    %d[b2], [%[b_ptr], #32]\n"
                "nop\n"
                "fmla    v8.4s , %[b0].4s, %[a0].s[0]\n"
                "ldr    x20, [%[b_ptr], #40]\n"
                "fmla    v9.4s , %[b0].4s, %[a0].s[1]\n"
                "subs    %w[k], %w[k], #1\n"
                "fmla    v10.4s, %[b0].4s, %[a0].s[2]\n"

                "ldr    %d[a0a], [%[a_ptr], #32]\n"
                "ins    %[b2].d[1], x20\n"
                "fmla    v11.4s, %[b0].4s, %[a0].s[3]\n"
                "ldr    x20, [%[a_ptr], #40]\n"
                "fmla    v12.4s, %[b0].4s, %[a1].s[0]\n"
                "fmla    v13.4s, %[b0].4s, %[a1].s[1]\n"

                "ldr    %d[a1a], [%[a_ptr], #48]\n"
                "ins    %[a0a].d[1], x20\n"
                "fmla    v14.4s, %[b0].4s, %[a1].s[2]\n"
                "ldr    x20, [%[a_ptr], #56]\n"
                "fmla    v15.4s, %[b0].4s, %[a1].s[3]\n"
                "fmla    v16.4s, %[b1].4s, %[a0].s[0]\n"

                "ldr    %d[b0], [%[b_ptr], #48]\n"
                "ins    %[a1a].d[1], x20\n"
                "fmla    v17.4s, %[b1].4s, %[a0].s[1]\n"
                "ldr    x20, [%[b_ptr], #56]\n"
                "fmla    v18.4s, %[b1].4s, %[a0].s[2]\n"
                "fmla    v19.4s, %[b1].4s, %[a0].s[3]\n"

                "ins    %[b0].d[1], x20\n"
                "fmla    v20.4s, %[b1].4s, %[a1].s[0]\n"
                "fmla    v21.4s, %[b1].4s, %[a1].s[1]\n"
                "fmla    v22.4s, %[b1].4s, %[a1].s[2]\n"

                "nop\n"
                "fmla    v23.4s, %[b1].4s, %[a1].s[3]\n"
                "fmla    v24.4s, %[b2].4s, %[a0].s[0]\n"
                "fmla    v25.4s, %[b2].4s, %[a0].s[1]\n"

                "ldr    %d[b1], [%[b_ptr], #64]\n"
                "nop\n"
                "fmla    v26.4s, %[b2].4s, %[a0].s[2]\n"
                "ldr    x20, [%[b_ptr], #72]\n"
                "fmla    v27.4s, %[b2].4s, %[a0].s[3]\n"
                "fmla    v28.4s, %[b2].4s, %[a1].s[0]\n"

                "ins    %[b1].d[1], x20\n"
                "fmla    v29.4s, %[b2].4s, %[a1].s[1]\n"
                "fmla    v30.4s, %[b2].4s, %[a1].s[2]\n"
                "fmla    v31.4s, %[b2].4s, %[a1].s[3]\n"

                "ldr    %d[b2], [%[b_ptr], #80]\n"
                "nop\n"
                "fmla    v8.4s , %[b0].4s, %[a0a].s[0]\n"
                "ldr    x20, [%[b_ptr], #88]\n"
                "fmla    v9.4s , %[b0].4s, %[a0a].s[1]\n"
                "fmla    v10.4s, %[b0].4s, %[a0a].s[2]\n"

                "ins    %[b2].d[1], x20\n"
                "fmla    v11.4s, %[b0].4s, %[a0a].s[3]\n"
                "fmla    v12.4s, %[b0].4s, %[a1a].s[0]\n"
                "fmla    v13.4s, %[b0].4s, %[a1a].s[1]\n"
                "fmla    v14.4s, %[b0].4s, %[a1a].s[2]\n"
                "fmla    v15.4s, %[b0].4s, %[a1a].s[3]\n"
                "fmla    v16.4s, %[b1].4s, %[a0a].s[0]\n"
                "fmla    v17.4s, %[b1].4s, %[a0a].s[1]\n"
                "fmla    v18.4s, %[b1].4s, %[a0a].s[2]\n"
                "fmla    v19.4s, %[b1].4s, %[a0a].s[3]\n"
                "fmla    v20.4s, %[b1].4s, %[a1a].s[0]\n"
                "fmla    v21.4s, %[b1].4s, %[a1a].s[1]\n"
                "fmla    v22.4s, %[b1].4s, %[a1a].s[2]\n"
                "fmla    v23.4s, %[b1].4s, %[a1a].s[3]\n"
                "fmla    v24.4s, %[b2].4s, %[a0a].s[0]\n"
                "fmla    v25.4s, %[b2].4s, %[a0a].s[1]\n"
                "fmla    v26.4s, %[b2].4s, %[a0a].s[2]\n"
                "fmla    v27.4s, %[b2].4s, %[a0a].s[3]\n"
                "fmla    v28.4s, %[b2].4s, %[a1a].s[0]\n"
                "fmla    v29.4s, %[b2].4s, %[a1a].s[1]\n"
                "add    %[a_ptr], %[a_ptr], #64\n"
                "fmla    v30.4s, %[b2].4s, %[a1a].s[2]\n"
                "add    %[b_ptr], %[b_ptr], #96\n"
                "fmla    v31.4s, %[b2].4s, %[a1a].s[3]\n"
                "b    3f\n"

                // Detached final iteration. (odd K)
                "2:\n"
                "ldr    %d[b2], [%[b_ptr], #32]\n"
                "nop\n"
                "fmla    v8.4s , %[b0].4s, %[a0].s[0]\n"
                "ldr    x20, [%[b_ptr], #40]\n"
                "fmla    v9.4s , %[b0].4s, %[a0].s[1]\n"
                "fmla    v10.4s, %[b0].4s, %[a0].s[2]\n"

                "ins    %[b2].d[1], x20\n"
                "fmla    v11.4s, %[b0].4s, %[a0].s[3]\n"
                "fmla    v12.4s, %[b0].4s, %[a1].s[0]\n"
                "fmla    v13.4s, %[b0].4s, %[a1].s[1]\n"
                "fmla    v14.4s, %[b0].4s, %[a1].s[2]\n"
                "fmla    v15.4s, %[b0].4s, %[a1].s[3]\n"
                "fmla    v16.4s, %[b1].4s, %[a0].s[0]\n"
                "fmla    v17.4s, %[b1].4s, %[a0].s[1]\n"
                "fmla    v18.4s, %[b1].4s, %[a0].s[2]\n"
                "fmla    v19.4s, %[b1].4s, %[a0].s[3]\n"
                "fmla    v20.4s, %[b1].4s, %[a1].s[0]\n"
                "fmla    v21.4s, %[b1].4s, %[a1].s[1]\n"
                "fmla    v22.4s, %[b1].4s, %[a1].s[2]\n"
                "fmla    v23.4s, %[b1].4s, %[a1].s[3]\n"
                "fmla    v24.4s, %[b2].4s, %[a0].s[0]\n"
                "fmla    v25.4s, %[b2].4s, %[a0].s[1]\n"
                "fmla    v26.4s, %[b2].4s, %[a0].s[2]\n"
                "fmla    v27.4s, %[b2].4s, %[a0].s[3]\n"
                "fmla    v28.4s, %[b2].4s, %[a1].s[0]\n"
                "fmla    v29.4s, %[b2].4s, %[a1].s[1]\n"
                "add    %[a_ptr], %[a_ptr], #32\n"
                "fmla    v30.4s, %[b2].4s, %[a1].s[2]\n"
                "add    %[b_ptr], %[b_ptr], #48\n"
                "fmla    v31.4s, %[b2].4s, %[a1].s[3]\n"

                // Common tail
                "3:\n"
                "str    q8,  [%[c_ptr]]\n"
                "str    q16,  [%[c_ptr], #16]\n"
                "str    q24,  [%[c_ptr], #32]\n"
                "str    q9,  [%[c_ptr], #48]\n"
                "str    q17,  [%[c_ptr], #64]\n"
                "str    q25,  [%[c_ptr], #80]\n"
                "str    q10,  [%[c_ptr], #96]\n"
                "str    q18,  [%[c_ptr], #112]\n"
                "str    q26,  [%[c_ptr], #128]\n"
                "str    q11,  [%[c_ptr], #144]\n"
                "str    q19,  [%[c_ptr], #160]\n"
                "str    q27,  [%[c_ptr], #176]\n"
                "str    q12,  [%[c_ptr], #192]\n"
                "str    q20,  [%[c_ptr], #208]\n"
                "str    q28,  [%[c_ptr], #224]\n"
                "str    q13,  [%[c_ptr], #240]\n"
                "str    q21,  [%[c_ptr], #256]\n"
                "str    q29,  [%[c_ptr], #272]\n"
                "str    q14,  [%[c_ptr], #288]\n"
                "str    q22,  [%[c_ptr], #304]\n"
                "str    q30,  [%[c_ptr], #320]\n"
                "str    q15,  [%[c_ptr], #336]\n"
                "str    q23,  [%[c_ptr], #352]\n"
                "str    q31,  [%[c_ptr], #368]\n"
                "add    %[c_ptr], %[c_ptr], #384\n"
            :
              [a_ptr] "+r" (a_ptr), [b_ptr] "+r" (b_ptr), [c_ptr] "+r" (c_ptr),
              [a0] "+w" (a0), [a1] "+w" (a1), [a0a] "+w" (a0a), [a1a] "+w" (a1a),
              [b0] "+w" (b0), [b1] "+w" (b1), [b2] "+w" (b2), [k] "+r" (k)
            : [oddk] "r" (oddk)
            : "x20", "x21", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
              "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "cc"
            );
        }
    }
}


// 12 : nr
// 16 : mr
void a64_quadr_bits_gemm_asimd_12x16_a53(const short *Apanel, const short *Bpanel, short *Cpanel, int ablocks, int bblocks, int K) {
    const short *a_ptr = Apanel;
    short *c_ptr = Cpanel;

    for (int yb=0; yb<ablocks; yb++) {
        const short *a_ptr0 = a_ptr;
        const short *b_ptr = Bpanel;

        for (int xb=0; xb<bblocks; xb++) {
            a_ptr = a_ptr0;
            // Fix up for odd lengths - set a flag if K is odd, but make
            // sure we round up the iteration count.
            int oddk = (K & 1);
            int k = ((K+1)/2) - 1;

            register int16x8_t a0   asm("v0");
            register int16x8_t a1   asm("v1");
            register int16x8_t a0a  asm("v2");
            register int16x8_t a1a  asm("v3");
            register int16x8_t b0   asm("v4");
            register int16x8_t b1   asm("v5");
            register int16x8_t b0a  asm("v6");
            register int16x8_t b1a  asm("v7");

            __asm __volatile (
                // Initialize result registers, load initial operands, prime prefetches.
                "movi    v8.4s, #0x0\n"
                "ldr    %q[a0], [%[a_ptr]]\n"
                "movi    v9.4s, #0x0\n"
                "ldr    %q[b0], [%[b_ptr]]\n"
                "movi    v10.4s, #0x0\n"
                "ldr    %d[a1], [%[a_ptr], #16]\n"
                "movi    v11.4s, #0x0\n"
                "ldr    %q[b1], [%[b_ptr], #16]\n"
                "movi    v12.4s, #0x0\n"
                ASM_PREFETCH("[%[b_ptr], #64]")
                "movi    v13.4s, #0x0\n"
                ASM_PREFETCH("[%[a_ptr], #64]")
                "movi    v14.4s, #0x0\n"
                ASM_PREFETCH("[%[b_ptr], #128]")
                "movi    v15.4s, #0x0\n"
                ASM_PREFETCH("[%[a_ptr], #128]")
                "movi    v16.4s, #0x0\n"
                ASM_PREFETCH("[%[b_ptr], #192]")
                "movi    v17.4s, #0x0\n"
                ASM_PREFETCH("[%[b_ptr], #256]")
                "movi    v18.4s, #0x0\n"
                ASM_PREFETCH("[%[a_ptr], #192]")
                "movi    v19.4s, #0x0\n"
                ASM_PREFETCH("[%[b_ptr], #320]")
                "movi    v20.4s, #0x0\n"
                ASM_PREFETCH("[%[a_ptr], #256]")
                "movi    v21.4s, #0x0\n"
                ASM_PREFETCH("[%[b_ptr], #384]")
                "movi    v22.4s, #0x0\n"
                "movi    v23.4s, #0x0\n"
                "movi    v24.4s, #0x0\n"
                "movi    v25.4s, #0x0\n"
                "movi    v26.4s, #0x0\n"
                "movi    v27.4s, #0x0\n"
                "movi    v28.4s, #0x0\n"
                "movi    v29.4s, #0x0\n"
                "movi    v30.4s, #0x0\n"
                "movi    v31.4s, #0x0\n"

                // Skip loop if we are doing zero iterations of it.
                "cbz    %w[k], 4f\n"

                "1:\n"
                // Unroll 0
                // "ldr    %d[b2], [%[b_ptr], #32]\n"
                "nop\n"
                "mla    v8.8h , %[b0].8h, %[a0].h[0]\n"
                "ldr    x20, [%[b_ptr], #40]\n"
                "mla    v9.8h , %[b0].8h, %[a0].h[1]\n"
                "subs    %w[k], %w[k], #1\n"
                "mla    v10.8h, %[b0].8h, %[a0].h[2]\n"

                "ldr    %d[a0a], [%[a_ptr], #24]\n"
                // "ins    %[b2].d[1], x20\n"
                "mla    v11.8h, %[b0].8h, %[a0].h[3]\n"
                "ldr    x20, [%[a_ptr], #32]\n"
                "mla    v12.8h, %[b0].8h, %[a0].h[4]\n"
                "mla    v13.8h, %[b0].8h, %[a0].h[5]\n"

                "ldr    %d[a1a], [%[a_ptr], #40]\n"
                "ins    %[a0a].d[1], x20\n"
                "mla    v14.8h, %[b0].8h, %[a0].h[6]\n"
                // "ldr    x20, [%[a_ptr], #56]\n"
                "mla    v15.8h, %[b0].8h, %[a0].h[7]\n"
                "mla    v16.8h, %[b0].8h, %[a1].h[0]\n"

                "ldr    %d[b0a], [%[b_ptr], #32]\n"
                // "ins    %[a1a].d[1], x20\n"
                "mla    v17.8h, %[b0].8h, %[a1].h[1]\n"
                "ldr    x20, [%[b_ptr], #40]\n"
                "mla    v18.8h, %[b0].8h, %[a1].h[2]\n"
                "mla    v19.8h, %[b0].8h, %[a1].h[3]\n"

                ASM_PREFETCH("[%[a_ptr], #320]")
                "ins    %[b0a].d[1], x20\n"
                "mla    v20.8h, %[b1].8h, %[a0].h[0]\n"
                "mla    v21.8h, %[b1].8h, %[a0].h[1]\n"
                "mla    v22.8h, %[b1].8h, %[a0].h[2]\n"

                ASM_PREFETCH("[%[b_ptr], #448]")
                "nop\n"
                "mla    v23.8h, %[b1].8h, %[a0].h[3]\n"
                "mla    v24.8h, %[b1].8h, %[a0].h[4]\n"
                "mla    v25.8h, %[b1].8h, %[a0].h[5]\n"

                "ldr    %d[b1a], [%[b_ptr], #48]\n"
                "nop\n"
                "mla    v26.8h, %[b1].8h, %[a0].h[6]\n"
                "ldr    x20, [%[b_ptr], #56]\n"
                "mla    v27.8h, %[b1].8h, %[a0].h[7]\n"
                "mla    v28.8h, %[b1].8h, %[a1].h[0]\n"

                ASM_PREFETCH("[%[b_ptr], #512]")
                "ins    %[b1a].d[1], x20\n"
                "mla    v29.8h, %[b1].8h, %[a1].h[1]\n"
                "mla    v30.8h, %[b1].8h, %[a1].h[2]\n"
                "mla    v31.8h, %[b1].8h, %[a1].h[3]\n"

                // Unroll 1
                "ldr    %d[a0], [%[a_ptr], #48]\n"
                "nop\n"
                "mla    v8.8h , %[b0a].8h, %[a0a].h[0]\n"
                "ldr    x20, [%[a_ptr], #56]\n"
                "mla    v9.8h , %[b0a].8h, %[a0a].h[1]\n"
                "mla    v10.8h, %[b0a].8h, %[a0a].h[2]\n"

                "ldr    %d[b0], [%[b_ptr], #64]\n"
                "ins    %[a0].d[1], x20\n"
                "mla    v11.8h, %[b0a].8h, %[a0a].h[3]\n"
                "ldr    x20, [%[b_ptr], #72]\n"
                "mla    v12.8h, %[b0a].8h, %[a0a].h[4]\n"
                "mla    v13.8h, %[b0a].8h, %[a0a].h[5]\n"

                "ldr    %d[a1], [%[a_ptr], #64]\n"
                "ins    %[b0].d[1], x20\n"
                "mla    v14.8h, %[b0a].8h, %[a0a].h[6]\n"
                "ldr    x20, [%[a_ptr], #72]\n"
                "mla    v15.8h, %[b0a].8h, %[a0a].h[7]\n"
                "mla    v16.8h, %[b0a].8h, %[a1a].h[0]\n"

                "ldr    %d[b1], [%[b_ptr], #80]\n"
                "ins    %[a1].d[1], x20\n"
                "mla    v17.8h, %[b0a].8h, %[a1a].h[1]\n"
                "ldr    x20, [%[b_ptr], #88]\n"
                "mla    v18.8h, %[b0a].8h, %[a1a].h[2]\n"
                "mla    v19.8h, %[b0a].8h, %[a1a].h[3]\n"

                "nop\n"
                "ins    %[b1].d[1], x20\n"
                "mla    v20.8h, %[b1a].8h, %[a0a].h[0]\n"
                "mla    v21.8h, %[b1a].8h, %[a0a].h[1]\n"
                "mla    v22.8h, %[b1a].8h, %[a0a].h[2]\n"

                "nop\n"
                "nop\n"
                "mla    v23.8h, %[b1a].8h, %[a0a].h[3]\n"
                "mla    v24.8h, %[b1a].8h, %[a0a].h[4]\n"
                "mla    v25.8h, %[b1a].8h, %[a0a].h[5]\n"

                "nop\n"
                "mla    v26.8h, %[b1a].8h, %[a0a].h[6]\n"
                "mla    v27.8h, %[b1a].8h, %[a0a].h[7]\n"
                "add    %[a_ptr], %[a_ptr], #48\n"
                "mla    v28.8h, %[b1a].8h, %[a1a].h[0]\n"
                "add    %[b_ptr], %[b_ptr], #64\n"

                "nop\n"
                "mla    v29.8h, %[b1a].8h, %[a1a].h[1]\n"
                "mla    v30.8h, %[b1a].8h, %[a1a].h[2]\n"
                "mla    v31.8h, %[b1a].8h, %[a1a].h[3]\n"

                "bne    1b\n"

                // Branch here if K=1 or 2.  Do the right thing for odd/even at the end.
                "4:\n"
                "cbnz    %w[oddk], 2f\n"

                // Detached final iteration. (even K)
                "ldr    %d[a0a], [%[a_ptr], #24]\n"
                "nop\n"
                "mla    v8.8h , %[b0].8h, %[a0].h[0]\n"
                "ldr    x20, [%[a_ptr], #32]\n"
                "mla    v9.8h , %[b0].8h, %[a0].h[1]\n"
                "subs    %w[k], %w[k], #1\n"
                "mla    v10.8h, %[b0].8h, %[a0].h[2]\n"

                "ldr    %d[b0a], [%[b_ptr], #32]\n"
                "ins    %[a0a].d[1], x20\n"
                "mla    v11.8h, %[b0].8h, %[a0].h[3]\n"
                "ldr    x20, [%[b_ptr], #40]\n"
                "mla    v12.8h, %[b0].8h, %[a0].h[4]\n"
                "mla    v13.8h, %[b0].8h, %[a0].h[5]\n"

                "ldr    %d[a1a], [%[a_ptr], #40]\n"
                "ins    %[b0a].d[1], x20\n"
                "mla    v14.8h, %[b0].8h, %[a0].h[6]\n"
                // "ldr    x20, [%[a_ptr], #56]\n"
                "mla    v15.8h, %[b0].8h, %[a0].h[7]\n"
                "mla    v16.8h, %[b0].8h, %[a1].h[0]\n"

                "ldr    %d[b1a], [%[b_ptr], #48]\n"
                // "ins    %[a1a].d[1], x20\n"
                "mla    v17.8h, %[b1].8h, %[a1].h[1]\n"
                "ldr    x20, [%[b_ptr], #56]\n"
                "mla    v18.8h, %[b1].8h, %[a1].h[2]\n"
                "mla    v19.8h, %[b1].8h, %[a1].h[3]\n"

                "ins    %[b1a].d[1], x20\n"
                "mla    v20.8h, %[b1].8h, %[a0].h[0]\n"
                "mla    v21.8h, %[b1].8h, %[a0].h[1]\n"
                "mla    v22.8h, %[b1].8h, %[a0].h[2]\n"

                "nop\n"
                "mla    v23.8h, %[b1].8h, %[a0].h[3]\n"
                "mla    v24.8h, %[b1].8h, %[a0].h[4]\n"
                "mla    v25.8h, %[b1].8h, %[a0].h[5]\n"

                // "ldr    %d[b1], [%[b_ptr], #64]\n"
                "nop\n"
                "mla    v26.8h, %[b1].8h, %[a0].h[6]\n"
                // "ldr    x20, [%[b_ptr], #72]\n"
                "mla    v27.8h, %[b1].8h, %[a0].h[7]\n"
                "mla    v28.8h, %[b1].8h, %[a1].h[0]\n"

                // "ins    %[b1].d[1], x20\n"
                "mla    v29.8h, %[b1].8h, %[a1].h[1]\n"
                "mla    v30.8h, %[b1].8h, %[a1].h[2]\n"
                "mla    v31.8h, %[b1].8h, %[a1].h[3]\n"

                // "ldr    %d[b2], [%[b_ptr], #80]\n"
                "nop\n"
                "mla    v8.8h , %[b0a].8h, %[a0a].h[0]\n"
                // "ldr    x20, [%[b_ptr], #88]\n"
                "mla    v9.8h , %[b0a].8h, %[a0a].h[1]\n"
                "mla    v10.8h, %[b0a].8h, %[a0a].h[2]\n"

                // "ins    %[b2].d[1], x20\n"
                "mla    v11.8h, %[b0a].8h, %[a0a].h[3]\n"
                "mla    v12.8h, %[b0a].8h, %[a0a].h[4]\n"
                "mla    v13.8h, %[b0a].8h, %[a0a].h[5]\n"
                "mla    v14.8h, %[b0a].8h, %[a0a].h[6]\n"
                "mla    v15.8h, %[b0a].8h, %[a0a].h[7]\n"
                "mla    v16.8h, %[b0a].8h, %[a1a].h[0]\n"
                "mla    v17.8h, %[b0a].8h, %[a1a].h[1]\n"
                "mla    v18.8h, %[b0a].8h, %[a1a].h[2]\n"
                "mla    v19.8h, %[b0a].8h, %[a1a].h[3]\n"
                "mla    v20.8h, %[b1a].8h, %[a0a].h[0]\n"
                "mla    v21.8h, %[b1a].8h, %[a0a].h[1]\n"
                "mla    v22.8h, %[b1a].8h, %[a0a].h[2]\n"
                "mla    v23.8h, %[b1a].8h, %[a0a].h[3]\n"
                "mla    v24.8h, %[b1a].8h, %[a0a].h[4]\n"
                "mla    v25.8h, %[b1a].8h, %[a0a].h[5]\n"
                "mla    v26.8h, %[b1a].8h, %[a0a].h[6]\n"
                "mla    v27.8h, %[b1a].8h, %[a0a].h[7]\n"
                "mla    v28.8h, %[b1a].8h, %[a1a].h[0]\n"
                "mla    v29.8h, %[b1a].8h, %[a1a].h[1]\n"
                "add    %[a_ptr], %[a_ptr], #64\n"
                "mla    v30.8h, %[b1a].8h, %[a1a].h[2]\n"
                "add    %[b_ptr], %[b_ptr], #96\n"
                "mla    v31.8h, %[b1a].8h, %[a1a].h[3]\n"
                "b    3f\n"

                // Detached final iteration. (odd K)
                "2:\n"
                "nop\n"
                "mla    v8.8h , %[b0].8h, %[a0].h[0]\n"
                "mla    v9.8h , %[b0].8h, %[a0].h[1]\n"
                "mla    v10.8h, %[b0].8h, %[a0].h[2]\n"
                "mla    v11.8h, %[b0].8h, %[a0].h[3]\n"
                "mla    v12.8h, %[b0].8h, %[a0].h[4]\n"
                "mla    v13.8h, %[b0].8h, %[a0].h[5]\n"
                "mla    v14.8h, %[b0].8h, %[a0].h[6]\n"
                "mla    v15.8h, %[b0].8h, %[a0].h[7]\n"
                "mla    v16.8h, %[b0].8h, %[a1].h[0]\n"
                "mla    v17.8h, %[b0].8h, %[a1].h[1]\n"
                "mla    v18.8h, %[b0].8h, %[a1].h[2]\n"
                "mla    v19.8h, %[b0].8h, %[a1].h[3]\n"
                "mla    v20.8h, %[b1].8h, %[a0].h[0]\n"
                "mla    v21.8h, %[b1].8h, %[a0].h[1]\n"
                "mla    v22.8h, %[b1].8h, %[a0].h[2]\n"
                "mla    v23.8h, %[b1].8h, %[a0].h[3]\n"
                "mla    v24.8h, %[b1].8h, %[a0].h[4]\n"
                "mla    v25.8h, %[b1].8h, %[a0].h[5]\n"
                "mla    v26.8h, %[b1].8h, %[a0].h[6]\n"
                "mla    v27.8h, %[b1].8h, %[a0].h[7]\n"
                "mla    v28.8h, %[b1].8h, %[a1].h[0]\n"
                "mla    v29.8h, %[b1].8h, %[a1].h[1]\n"
                // "add    %[a_ptr], %[a_ptr], #32\n"
                "mla    v30.8h, %[b1].8h, %[a1].h[2]\n"
                // "add    %[b_ptr], %[b_ptr], #48\n"
                "mla    v31.8h, %[b1].8h, %[a1].h[3]\n"

                // Common tail
                "3:\n"
                "str    q8,  [%[c_ptr]]\n"
                "str    q16,  [%[c_ptr], #16]\n"
                "str    q24,  [%[c_ptr], #32]\n"
                "str    q9,  [%[c_ptr], #48]\n"
                "str    q17,  [%[c_ptr], #64]\n"
                "str    q25,  [%[c_ptr], #80]\n"
                "str    q10,  [%[c_ptr], #96]\n"
                "str    q18,  [%[c_ptr], #112]\n"
                "str    q26,  [%[c_ptr], #128]\n"
                "str    q11,  [%[c_ptr], #144]\n"
                "str    q19,  [%[c_ptr], #160]\n"
                "str    q27,  [%[c_ptr], #176]\n"
                "str    q12,  [%[c_ptr], #192]\n"
                "str    q20,  [%[c_ptr], #208]\n"
                "str    q28,  [%[c_ptr], #224]\n"
                "str    q13,  [%[c_ptr], #240]\n"
                "str    q21,  [%[c_ptr], #256]\n"
                "str    q29,  [%[c_ptr], #272]\n"
                "str    q14,  [%[c_ptr], #288]\n"
                "str    q22,  [%[c_ptr], #304]\n"
                "str    q30,  [%[c_ptr], #320]\n"
                "str    q15,  [%[c_ptr], #336]\n"
                "str    q23,  [%[c_ptr], #352]\n"
                "str    q31,  [%[c_ptr], #368]\n"
                "add    %[c_ptr], %[c_ptr], #384\n"
            :
              [a_ptr] "+r" (a_ptr), [b_ptr] "+r" (b_ptr), [c_ptr] "+r" (c_ptr),
              [a0] "+w" (a0), [a0a] "+w" (a0a), [a1] "+w" (a1), [a1a] "+w" (a1a),
              [b0] "+w" (b0), [b0a] "+w" (b0a), [b1] "+w" (b1), [b1a] "+w" (b1a),
              [k] "+r" (k)
            : [oddk] "r" (oddk)
            : "x20", "x21", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
              "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "cc"
            );
        }
    }
}



int main(int argc, char** argv) {
    std::cout << "QuatrbitsGEMM" << std::endl;

    int K = 512;
    if(argc == 2) {
        K = std::atoi(argv[1]);
    }
    const int M_FLOAT = 8;
    const int N_FLOAT = 12;

    std::vector<float> matrix_fp32_a(M_FLOAT * K);
    std::vector<float> matrix_fp32_b(K * N_FLOAT);
    std::vector<float> matrix_fp32_c(M_FLOAT * N_FLOAT);

    const int M_SHORT = 12;
    const int N_SHORT = 16;
    std::vector<short> matrix_int16_a(M_SHORT * K);
    std::vector<short> matrix_int16_b(K * N_SHORT);
    std::vector<short> matrix_int16_c(M_SHORT * N_SHORT);
    std::cout << "MNK :" << M_SHORT << " " << N_SHORT << " " << K << std::endl;



    {
        //warm up
        std::cout << "Warm up 2000 times" << std::endl;
        for(int i = 0; i < 2000; ++i) {
            a64_sgemm_asimd_8x12_a53(matrix_fp32_a.data(), matrix_fp32_b.data(), matrix_fp32_c.data(), 1, 1, K);
            a64_quadr_bits_gemm_asimd_12x16_a53(matrix_int16_a.data(), matrix_int16_b.data(), matrix_int16_c.data(), 1, 1, K);
        }
    }


    int times = 100000;
    Duration tt;

    tt.start();
    for(int i = 0; i < times; ++i) {
        a64_sgemm_asimd_8x12_a53(matrix_fp32_a.data(), matrix_fp32_b.data(), matrix_fp32_c.data(), 1, 1, K);
    }
    tt.end();

    double tt_float32 = tt.getDuration() / times * 2;


    tt.start();
    for(int i = 0; i < times; ++i) {
        a64_quadr_bits_gemm_asimd_12x16_a53(matrix_int16_a.data(), matrix_int16_b.data(), matrix_int16_c.data(), 1, 1, K);
    }
    tt.end();

    double tt_quadr_bits = tt.getDuration() / times;

    std::cout << "duration fp32 : " << tt_float32  << std::endl;
    std::cout << "duration int16 : " << tt_quadr_bits << std::endl;
    std::cout << "speed up : " << tt_float32 / tt_quadr_bits << std::endl;


    return 0;
}
