# ops

g++ direct.c -I/PATHTO/OpenBLAS/include -L/PATHTO/OpenBLAS/lib -lopenblas -O2 -o direct

org time: 893.32 ms
opt1 time: 661.40 ms
opt1blas time: 214.68 ms
opt2 time: 645.83 ms
opt2blas time: 231.29 ms
