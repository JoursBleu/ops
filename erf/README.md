# erf

compile:
icc erf.c -lmkl_rt -qopenmp -o erf

run:
./erf

result (28 physical cores x 2 sockets):
[]$ export OMP_NUM_THREADS=1
[]$ ./erf
std time: 3278.84 ms
mkl time: 1141.68 ms
org time: 5398.45 ms
simd time: 1155.34 ms
[]$ export OMP_NUM_THREADS=28
[]$ ./erf
std time: 173.57 ms
mkl time: 124.99 ms
org time: 289.07 ms
simd time: 125.32 ms
[]$ export OMP_NUM_THREADS=56
[]$ ./erf
std time: 208.19 ms
mkl time: 158.96 ms
org time: 154.67 ms
simd time: 158.58 ms



