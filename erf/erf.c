#include <immintrin.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

#include <mkl.h>

const unsigned long long MM=1024*1024*1024;

// L1 (32K)
#define SIMD_SIZE 16
#define BLOCK_SIZE 128 // K()


void erf_simd(int M, float *x, float *y) {
  int i;
  // constants
  const __m512 a1 = _mm512_set1_ps( 0.254829592);
  const __m512 a2 = _mm512_set1_ps(-0.284496736);
  const __m512 a3 = _mm512_set1_ps( 1.421413741);
  const __m512 a4 = _mm512_set1_ps(-1.453152027);
  const __m512 a5 = _mm512_set1_ps( 1.061405429);
  const __m512 p  = _mm512_set1_ps( 0.3275911);
  const __m512 one= _mm512_set1_ps( 1.);

  #pragma omp parallel for
  for(i=0; i<M; i+=SIMD_SIZE) {
    __m512 xx = _mm512_load_ps(&x[i]);
    __m512 sign = xx/xx;

    __m512 t = one/(one + p*xx);
    __m512 exp_xx = _mm512_exp_ps(-xx*xx);
    __m512 yy = one - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp_xx;
    yy = sign * yy;

    _mm512_store_ps(&y[i], yy);
  }
}


void erf_org(int M, float *x, float *y) {
  int i;
  // constants
  const float a1 =  0.254829592;
  const float a2 = -0.284496736;
  const float a3 =  1.421413741;
  const float a4 = -1.453152027;
  const float a5 =  1.061405429;
  const float p  =  0.3275911;

  #pragma omp parallel for
  for(i=0; i<M; i++) {
    float xx = x[i];
    float sign = xx/xx;

    float t = 1.0/(1.0 + p*xx);
    float yy = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-xx*xx);

    y[i] = yy;
  }
}

int main() {
  int i;
  float *x, *y, *y_std;

  struct timeval begin, end;
  float timeuse;

  x = (float*)_mm_malloc(sizeof(float)*MM, 64);
  y = (float*)_mm_malloc(sizeof(float)*MM, 64);
  y_std = (float*)_mm_malloc(sizeof(float)*MM, 64);

  for (i=0; i<MM; i++) x[i] = i*1./MM;
  for (i=0; i<MM; i++) y[i] = 0.;
  for (i=0; i<MM; i++) y_std[i] = 0.;

  gettimeofday( &begin, NULL );
  #pragma omp parallel for
  for(i=0; i<MM; i++)
    y_std[i] = erf(x[i]);
  gettimeofday( &end, NULL );
  timeuse = (1000000. * ( end.tv_sec - begin.tv_sec ) + end.tv_usec - begin.tv_usec)/1000.;
  printf("std time: %.2f ms\n", timeuse);

  vsErf(MM, x, y);
  gettimeofday( &begin, NULL );
  vsErf(MM, x, y);
  gettimeofday( &end, NULL );
  timeuse = (1000000. * ( end.tv_sec - begin.tv_sec ) + end.tv_usec - begin.tv_usec)/1000.;
  printf("mkl time: %.2f ms\n", timeuse);

  gettimeofday( &begin, NULL );
  erf_org(MM, x, y_std);
  gettimeofday( &end, NULL );
  timeuse = (1000000. * ( end.tv_sec - begin.tv_sec ) + end.tv_usec - begin.tv_usec)/1000.;
  printf("org time: %.2f ms\n", timeuse);

  gettimeofday( &begin, NULL );
  erf_simd(MM, x, y);
  gettimeofday( &end, NULL );
  timeuse = (1000000. * ( end.tv_sec - begin.tv_sec ) + end.tv_usec - begin.tv_usec)/1000.;
  printf("simd time: %.2f ms\n", timeuse);


  for(i=0; i<MM; i++)
    if(abs(y[i]-y_std[i])>0.00000000001)
      printf("%f\t%f\n", y[i], y_std[i]);

  _mm_free(x);
  _mm_free(y);
  _mm_free(y_std);

  return 0;
}
