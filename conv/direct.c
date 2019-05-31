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

const unsigned wH=5;
const unsigned wW=5;
const unsigned wC=iC;
const unsigned wD=32;

const unsigned stride=1;

const unsigned oN=iN;
const unsigned oC=wD;
const unsigned oH=(iH-wH)/stride+1;
const unsigned oW=(iW-wW)/stride+1;

const unsigned SIMD_SIZE=16;


void Conv2D_opt(float *In, float *We, float *Ou,
          const unsigned iN, const unsigned iC, const unsigned iH, const unsigned iW,
          const unsigned wH, const unsigned wW, const unsigned wC, const unsigned wD,
          const unsigned oN, const unsigned oC, const unsigned oH, const unsigned oW) {
  unsigned i, j, k, l, m, n, o;
  for(i=0; i<iN; i++) { // input N/output N
    float* in_nptr = &In[i*iC*iH*iW]; 
    float* ou_nptr = &Ou[i*oC*oH*oW];
    for(n=0; n<wH; n++) { // weight H
      float *we_hptr = &We[n*wW*wC*wD];
      float *in_hptr = &in_nptr[n*iW];
      for(o=0; o<wW; o++) { // weight W
        float *we_wptr = &we_hptr[o*wC*wD];
        float *in_wptr = &in_hptr[o];
        for(k=0; k<iC; k++) { // input C/weight C
          float *we_cptr = &we_wptr[k*wD];
          float *in_cptr = &in_wptr[k*iH*iW];
          for(j=0; j<wD; j++) { // output C/weight D
            float *ou_cptr = &ou_nptr[j*oH*oW];
            float weight = we_cptr[j];
            for(l=0; l<oH; l++) { // output H
              float *in_line_ptr = &in_cptr[stride*l*iW];
              float *ou_hptr = &ou_cptr[l*oW];
              for(m=0; m<oW; m++) { // output W
                ou_hptr[m] += weight * in_line_ptr[stride*m];
              }
            }
          }
        }
      }
    }
  }
}


void Conv2D_opt2(float *In, float *We, float *Ou,
          const unsigned iN, const unsigned iC, const unsigned iH, const unsigned iW,
          const unsigned wH, const unsigned wW, const unsigned wC, const unsigned wD,
          const unsigned oN, const unsigned oC, const unsigned oH, const unsigned oW) {
  unsigned i, j, k, l, m, n, o;
  //#pragma omp parallel for num_threads(56)
  for(i=0; i<iN; i++) { // input N/output N
    float* in_nptr = &In[i*iC*iH*iW]; 
    float* ou_nptr = &Ou[i*oC*oH*oW];
    for(j=0; j<wD; j++) { // output C/weight D
      float *ou_cptr = &ou_nptr[j*oH*oW];
      float *we_dptr = &We[j];
      for(k=0; k<iC; k++) { // input C/weight C
        float *in_cptr = &in_nptr[k*iH*iW];
        float *we_cptr = &we_dptr[k*wD];
        for(l=0; l<oH; l++) { // output H
          float *ou_hptr = &ou_cptr[l*oW];
          float *in_hptr = &in_cptr[2*l*iW];
          for(m=0; m<oW; m++) { // output W
            float out = 0.;
            float *in_win_ptr = &in_hptr[2*m];
            for(n=0; n<wH; n++) { // weight H
              float *we_hptr = &we_cptr[n*wW*wC*wD];
              float *in_win_line_ptr = &in_win_ptr[n*iW];
              for(o=0; o<wW; o++) { // weight W
                out += we_hptr[o*wC*wD] * in_win_line_ptr[o];
              }
            }
            ou_hptr[m] = out;
          }
        }
      }
    }
  }
}


void Conv2D_org(float *In, float *We, float *Ou,
          const unsigned iN, const unsigned iC, const unsigned iH, const unsigned iW,
          const unsigned wH, const unsigned wW, const unsigned wC, const unsigned wD,
          const unsigned oN, const unsigned oC, const unsigned oH, const unsigned oW) {
  unsigned i, j, k, l, m, n, o;
  //#pragma omp parallel for num_threads(56)
  for(i=0; i<iN; i++) { // input N/output N
    float* in_nptr = &In[i*iC*iH*iW]; 
    float* ou_nptr = &Ou[i*oC*oH*oW];
    for(j=0; j<wD; j++) { // output C/weight D
      float *ou_cptr = &ou_nptr[j*oH*oW];
      float *we_dptr = &We[j];
      for(k=0; k<iC; k++) { // input C/weight C
        float *in_cptr = &in_nptr[k*iH*iW];
        float *we_cptr = &we_dptr[k*wD];
        for(l=0; l<oH; l++) { // output H
          float *ou_hptr = &ou_cptr[l*oW];
          float *in_hptr = &in_cptr[2*l*iW];
          for(m=0; m<oW; m++) { // output W
            float out = 0.;
            float *in_win_ptr = &in_hptr[2*m];
            for(n=0; n<wH; n++) { // weight H
              float *we_hptr = &we_cptr[n*wW*wC*wD];
              float *in_win_line_ptr = &in_win_ptr[n*iW];
              for(o=0; o<wW; o++) { // weight W
                out += we_hptr[o*wC*wD] * in_win_line_ptr[o];
              }
            }
            ou_hptr[m] = out;
          }
        }
      }
    }
  }
}


int main() {
  unsigned i, j, l;
  float *In, *We, *Ou;

  struct timeval begin, end;
  float timeuse;

  In = (float*)_mm_malloc(sizeof(float)*iN*iC*iH*iW, 64);
  We = (float*)_mm_malloc(sizeof(float)*wH*wW*wC*wD, 64);
  Ou = (float*)_mm_malloc(sizeof(float)*oN*oC*oH*oW, 64);

  for(i=0; i<iN*iC*iH*iW; i++) In[i] = i*1.;
  for(i=0; i<wH*wW*wC*wD; i++) In[i] = i*2.;
  for(i=0; i<oN*oC*oH*oW; i++) In[i] = 0.;

  gettimeofday( &begin, NULL );
  Conv2D_org(In, We, Ou, iN, iC, iH, iW, wH, wW, wC, wD, oN, oC, oH, oW);
  gettimeofday( &end, NULL );
  timeuse = (1000000. * ( end.tv_sec - begin.tv_sec ) + end.tv_usec - begin.tv_usec)/1000.;
  printf("org time: %.2f ms\n", timeuse);

  gettimeofday( &begin, NULL );
  Conv2D_opt(In, We, Ou, iN, iC, iH, iW, wH, wW, wC, wD, oN, oC, oH, oW);
  gettimeofday( &end, NULL );
  timeuse = (1000000. * ( end.tv_sec - begin.tv_sec ) + end.tv_usec - begin.tv_usec)/1000.;
  printf("opt time: %.2f ms\n", timeuse);

  _mm_free(In);
  _mm_free(We);
  _mm_free(Ou);

  return 0;
}
