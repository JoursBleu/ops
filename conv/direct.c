#include <immintrin.h>
#include <sys/time.h>
#include <stdio.h>

/*
#define MM 4096
#define KK 4096
#define NN 4096

#define SIMD_SIZE 16
*/

const unsigned iN=64;
const unsigned iC=256;
const unsigned iH=28;
const unsigned iW=28;

const unsigned wC=256;
const unsigned wH=5;
const unsigned wW=5;
const unsigned wD=32;

const unsigned stride=2;

const unsigned oN=iN;
const unsigned oC=wD;
const unsigned oH=(iH-wH)/stride+1;
const unsigned oW=(iW-wW)/stride+1;

const unsigned SIMD_SIZE=16;

int main() {
  int i, j, l;
  float *A, *B, *C;

  struct timeval begin, end;
  float timeuse;

  In = (float*)_mm_malloc(sizeof(float)*iN*iC*iH*iW, 64);
  We = (float*)_mm_malloc(sizeof(float)*wC*wH*wW*wD, 64);
  Ou = (float*)_mm_malloc(sizeof(float)*oN*oC*oH*oW, 64);



  gettimeofday( &begin, NULL );
  // Conv(A, B, C, MM, KK, NN);
  gettimeofday( &end, NULL );
  timeuse = 1000000 * ( end.tv_sec - begin.tv_sec ) + end.tv_usec - begin.tv_usec;
  printf("org time: %d us\n", timeuse);

  _mm_free(A);
  _mm_free(B);
  _mm_free(C);

  return 0;
}
