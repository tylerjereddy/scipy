// Utility C code for directed_hausdorff
// Try to improve on the performance of the original Cython directed_hausdorff

#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>

struct return_values 
{
    double cmax;
    int    index_1;
    int    index_2;
};

struct return_values ret_vals;

// function for inner loop of directed_hausdorff algorithm
double hausdorff_loop(const int data_dims,
                    double ar1[],
                    double ar2[],
                    int N1,
                    int N2)
{
    double               d, cmin;
    bool                 no_break_happened;
    unsigned int         i_store = 0, j_store = 0;
    struct return_values ret_vals;
    double * const ar1_start_Ptr = ar1;
    double * const ar2_start_Ptr = ar2;
    int size_ar1 = data_dims * N1;
    int size_ar2 = data_dims * N2;
    const double * const ar1_end_Ptr = &ar1[size_ar1 - 1];
    const double * const ar2_end_Ptr = &ar2[size_ar2 - 1];

    ret_vals.cmax = 0;
    
    while (ar1 < ar1_end_Ptr) {
        no_break_happened = 1;
        cmin = INFINITY;
        while (ar2 < ar2_end_Ptr) {
            d = 0;
            for ( int k = 0; k < data_dims; ++k, ++ar1, ++ar2) {
                d += ((*ar1 - *ar2) * (*ar1 - *ar2));
            }
            ar1 -= data_dims;
            if (d < ret_vals.cmax) {
                --no_break_happened;
                ar2 = ar2_start_Ptr;
                break;
            }

            if (d < cmin) {
                cmin = d;
                i_store = ar1 - ar1_start_Ptr;
                j_store = ar2 - ar2_start_Ptr;
            }
        }
        ar2 = ar2_start_Ptr;
        if ( (cmin != INFINITY) && (cmin > ret_vals.cmax) && (no_break_happened)) {
            ret_vals.cmax = cmin;
            ret_vals.index_1 = i_store;
            ret_vals.index_2 = j_store;
        }
    ar1 += data_dims;
    }
    printf("i: %i\n", ret_vals.index_1);
    printf("j: %i\n", ret_vals.index_2);
    return ret_vals.cmax;
}
