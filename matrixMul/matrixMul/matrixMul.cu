#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

const short m = 3;
const short n = 4;
const short k = 2;

template<typename T>
void initalValue(T& a, const short length, const float value)
{
	for (short i = 0; i < length; i++)
	{
		a[i] = value;
	}
}

template<typename T>
void checkValue(T a, const short length)
{
	for (short i = 0; i < length; i++)
	{
		printf("a[%d]: %.3f\n", i, a[i]);
	}
	printf("\n");
}

__global__ void matrixMul(const float* a, const float* b, float*c, const short m, const short n, const short k)
{
	const short col = threadIdx.x + blockIdx.x * blockDim.x;
	const short row = threadIdx.y + blockIdx.y * blockDim.y;

	float c_temp = 0.0;
	if (row < m  && col < k)
	{
		for (short i = 0; i < n; i++)
		{
			c_temp += a[row * n + i] * b[i * k + col];
		}
		c[row * k + col] = c_temp;
	}	
}

int main()
{
	// initial host memory 
	float* h_a = nullptr;		// m * n
	float* h_b = nullptr;		// n * k
	float* h_c = nullptr;		// m * k

	h_a = (float*)malloc(m * n * sizeof(float));
	h_b = (float*)malloc(n * k * sizeof(float));
	h_c = (float*)malloc(m * k * sizeof(float));

	if (h_a == nullptr || h_b == nullptr || h_c == nullptr)
	{
		printf("memroy malloc error with code 001!\n");
		exit(EXIT_FAILURE);
	}

	// initial host matrix value
	initalValue(h_a, m * n, 1.0);
	initalValue(h_b, n * k, 2.0);	

	// value check
	checkValue(h_a, m * n);
	checkValue(h_b, n * k);

	// initial device memory
	float* d_a = nullptr;		// m * n 
	float* d_b = nullptr;		// n * k
	float* d_c = nullptr;		// m * k

	cudaMalloc((void**)&d_a, m * n * sizeof(float));
	cudaMalloc((void**)&d_b, n * k * sizeof(float));
	cudaMalloc((void**)&d_c, m * k * sizeof(float));

	// copy data from host to device
	cudaMemcpy(d_a, h_a, m * n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, n * k * sizeof(float), cudaMemcpyHostToDevice);

	dim3 blockNum(1, 1, 1);
	dim3 threadNum(4, 4, 1);

	matrixMul << <blockNum, threadNum >> > (d_a, d_b, d_c, m, n, k);

	cudaMemcpy(h_c, d_c, m * k * sizeof(float), cudaMemcpyDeviceToHost);
	checkValue(h_c, m * k);

	free(h_a);
	free(h_b);
	free(h_c);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);


	system("pause");
	return 0;
}