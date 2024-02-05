#include <cuda.h>
#include <math.h>
#include <stdio.h>

__global__ void matvec(const float *A, const float *B, float *C, const int M,
                       const int K) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M) {
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
      acc += A[row * K + k] * B[k];
    }
    C[row] = acc;
  }
}

void matvec_stub(const float *h_A, const float *h_B, float *h_C, const int M,
                 const int K, const int Asize, const int Bsize,
                 const int Csize) {
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, Asize);
  cudaMalloc(&d_B, Bsize);
  cudaMalloc(&d_C, Csize);

  cudaMemcpy(d_A, h_A, Asize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, Bsize, cudaMemcpyHostToDevice);

  printf("gpu begin...\n");

  matvec<<<4, 8>>>(d_A, d_B, d_C, M, K);

  printf("gpu end\n");

  cudaMemcpy(h_C, d_C, Csize, cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

void matvec_naive(const float *A, const float *B, float *C, const int M,
                  const int K) {
  for (int m = 0; m < M; ++m) {
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
      acc += A[m * K + k] * B[k];
    }
    C[m] = acc;
  }
}

int main() {
  int M = 32, N = 1, K = 56;

  size_t Asize = M * K * sizeof(float);
  size_t Bsize = K * N * sizeof(float);
  size_t Csize = M * N * sizeof(float);

  float *h_A = (float *)malloc(Asize);
  float *h_B = (float *)malloc(Bsize);
  float *h_C = (float *)malloc(Csize);
  float *h_C_naive = (float *)malloc(Csize);

  for (int m = 0; m < M; ++m) {
    for (int k = 0; k < K; ++k) {
      h_A[m * K + k] = m * K + k;
    }
  }
  for (int k = 0; k < K; ++k) {
    for (int n = 0; n < N; ++n) {
      h_B[k * N + n] = k * N + n;
    }
  }

  // matmul_stub(h_A, h_B, h_C, M, N, K, Asize, Bsize, Csize);
  // matmul_row_stub(h_A, h_B, h_C, M, N, K, Asize, Bsize, Csize);
  matvec_stub(h_A, h_B, h_C, M, K, Asize, Bsize, Csize);

  printf("cpu begin...\n");

  matvec_naive(h_A, h_B, h_C_naive, M, K);
  printf("cpu end\n");

  free(h_A);
  free(h_B);

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      if (fabsf(h_C[m * N + n] - h_C_naive[m * N + n]) > 1e-5) {
        printf("wrong\n\tm: %d, n: %d, host value: %f, device value: %f\n", m,
               n, h_C_naive[m * N + n], h_C[m * N + n]);
        free(h_C);
        exit(1);
      }
    }
  }

  printf("done\n");
  free(h_C);

  return 0;
}
