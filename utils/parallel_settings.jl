using LinearAlgebra
using ThreadPinning
pinthreads(:cores)
BLAS.set_num_threads(1)
MKL_NUM_THREADS = 1
threadinfo(;blas=true, hints=true)