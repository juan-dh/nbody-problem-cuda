#include <iostream>
#include <string>
#include <random>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <cstring>

// Constant Memory
__constant__ double d_G;
__constant__ double d_h;
__constant__ double d_epsilon;

// Vector
struct Vec3
{
    float x, y, z;

    // Addition
    __host__ __device__ Vec3 operator+(const Vec3 &other) const
    {
        return {x + other.x, y + other.y, z + other.z};
    }

    // Substraction
    __host__ __device__ Vec3 operator-(const Vec3 &other) const
    {
        return {x - other.x, y - other.y, z - other.z};
    }

    // Scalar multiplication
    __host__ __device__ Vec3 operator*(float scalar) const
    {
        return {x * scalar, y * scalar, z * scalar};
    }
};

__device__ Vec3 computeForce(int i, const Vec3 *q, const float *m, int n_bodies)
{
    Vec3 accum{0.0f, 0.0f, 0.0f};

    Vec3 qi = q[i];
    float mi = m[i];

    for (int j = 0; j < n_bodies; ++j)
    {
        if (j == i)
            continue;

        float dx = qi.x - q[j].x;
        float dy = qi.y - q[j].y;
        float dz = qi.z - q[j].z;

        float r2 = dx * dx + dy * dy + dz * dz + (float)(d_epsilon * d_epsilon);
        float inv_r = rsqrtf(r2);
        float inv_r3 = inv_r * inv_r * inv_r;

        float mag = -(float)(d_G * mi * m[j]) * inv_r3;

        accum.x += dx * mag;
        accum.y += dy * mag;
        accum.z += dz * mag;
    }

    return accum;
}

__global__ void kernelInitHalfStep(const Vec3 *q, const Vec3 *p, Vec3 *p_half, const float *m, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    Vec3 f = computeForce(i, q, m, n);
    p_half[i] = p[i] + f * ((float)d_h * 0.5f);
}

__global__ void kernelLeapFrogPart1(const Vec3 *d_q_n, Vec3 *d_q_n_plus_one, const Vec3 *d_p_n_plus_oneHalf, const float *d_m, int n_bodies)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_bodies)
        return;

    d_q_n_plus_one[i] = d_q_n[i] + d_p_n_plus_oneHalf[i] * ((float)d_h / d_m[i]);
}

__global__ void kernelLeapFrogPart2(const Vec3 *d_q_n_plus_one, const Vec3 *d_p_n_plus_oneHalf, Vec3 *d_p_n_plus_threeHalfs, const float *d_m, int n_bodies)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_bodies)
        return;

    Vec3 f = computeForce(i, d_q_n_plus_one, d_m, n_bodies);
    d_p_n_plus_threeHalfs[i] = d_p_n_plus_oneHalf[i] + f * ((float)d_h);
}

__host__ void initializeBodies(Vec3 *h_q_n, Vec3 *h_p_n, float *h_m, int n_bodies, const std::string method)
{
    // Naive method
    if (method == "naive")
    {
        for (int i = 0; i < n_bodies; ++i)
        {
            static std::mt19937 gen(std::random_device{}());
            std::uniform_int_distribution<> q_dist(-100, 100);
            h_q_n[i].x = static_cast<float>(q_dist(gen));
            h_q_n[i].y = static_cast<float>(q_dist(gen));
            h_q_n[i].z = static_cast<float>(q_dist(gen));

            std::uniform_int_distribution<> p_dist(-10, 10);
            h_p_n[i].x = static_cast<float>(p_dist(gen));
            h_p_n[i].y = static_cast<float>(p_dist(gen));
            h_p_n[i].z = static_cast<float>(p_dist(gen));

            std::uniform_int_distribution<> mass_dist(0, 100);
            h_m[i] = static_cast<float>(mass_dist(gen)) + 0.1f; // Avoid zero mass
        }
    }
}

__host__ void debug_print_history(Vec3 *h_variable_history, int n_bodies, int n_steps)
{
    for (int step = 0; step < n_steps + 1; step++)
    {
        std::cout << "Step: " << step << std::endl;
        for (int i = 0; i < n_bodies; i++)
        {
            int idx = i + step * n_bodies;
            std::cout << "n:" << i << " x_1:" << h_variable_history[idx].x << " x_2:" << h_variable_history[idx].y << " x_3:" << h_variable_history[idx].z << std::endl;
        }
        std::cout << "-------------------------" << std::endl;
    }
}

int main(int argc, char *argv[])
{
    int N_BODIES = 100000;
    int N_STEPS = 100;
    double h_G = 6.67430e-11;
    double h_h = 1e-2;
    double h_epsilon = 1e-9;

    cudaSetDevice(0);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "Max threads per block: "
              << prop.maxThreadsPerBlock << std::endl;

    std::cout << "Max threads dim: "
              << prop.maxThreadsDim[0] << ", "
              << prop.maxThreadsDim[1] << ", "
              << prop.maxThreadsDim[2] << std::endl;

    std::cout << "Max grid size: "
              << prop.maxGridSize[0] << ", "
              << prop.maxGridSize[1] << ", "
              << prop.maxGridSize[2] << std::endl;

    // Allocate host memory

    // Pinned memory
    Vec3 *h_q_n;
    cudaMallocHost((void **)&h_q_n, N_BODIES * sizeof(Vec3));
    Vec3 *h_p_n;
    cudaMallocHost((void **)&h_p_n, N_BODIES * sizeof(Vec3));
    float *h_m;
    cudaMallocHost((void **)&h_m, N_BODIES * sizeof(float));
    // Pageable memory
    Vec3 *h_q_history = (Vec3 *)malloc(N_BODIES * (N_STEPS + 1) * sizeof(Vec3));
    Vec3 *h_p_history = (Vec3 *)malloc(N_BODIES * (N_STEPS + 1) * sizeof(Vec3));

    // Allocate device memory

    // Global memory
    Vec3 *d_q_n;
    cudaMalloc((void **)&d_q_n, N_BODIES * sizeof(Vec3));
    Vec3 *d_q_n_plus_one;
    cudaMalloc((void **)&d_q_n_plus_one, N_BODIES * sizeof(Vec3));
    Vec3 *d_p_n;
    cudaMalloc((void **)&d_p_n, N_BODIES * sizeof(Vec3));
    Vec3 *d_p_n_plus_oneHalf;
    cudaMalloc((void **)&d_p_n_plus_oneHalf, N_BODIES * sizeof(Vec3));
    Vec3 *d_p_n_plus_threeHalfs;
    cudaMalloc((void **)&d_p_n_plus_threeHalfs, N_BODIES * sizeof(Vec3));
    float *d_m;
    cudaMalloc((void **)&d_m, N_BODIES * sizeof(float));

    // Initialize bodies qs and ps

    initializeBodies(h_q_n, h_p_n, h_m, N_BODIES, "naive");

    // Copy h_q and h_p to h_q_history and h_p_history
    memcpy(h_q_history, h_q_n, N_BODIES * sizeof(Vec3));
    memcpy(h_p_history, h_p_n, N_BODIES * sizeof(Vec3));

    // Copy from host to device

    // Copy h_q_n, h_q_m, h_m to d_q_n, d_q_m, d_m respectively
    cudaMemcpy(d_q_n, h_q_n, N_BODIES * sizeof(Vec3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p_n, h_p_n, N_BODIES * sizeof(Vec3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, h_m, N_BODIES * sizeof(float), cudaMemcpyHostToDevice);
    // Copy to constant memory
    cudaMemcpyToSymbol(d_G, &h_G, sizeof(double));
    cudaMemcpyToSymbol(d_h, &h_h, sizeof(double));
    cudaMemcpyToSymbol(d_epsilon, &h_epsilon, sizeof(double));

    // Print

    // debug_print_history(h_q_history, N_BODIES, N_STEPS);

    // Run the simulation for N_STEPS steps
    int threadsPerBlock = 256;
    int blocksBodies = (N_BODIES + threadsPerBlock - 1) / threadsPerBlock;

    // Initial step of Leapfrog
    kernelInitHalfStep<<<blocksBodies, threadsPerBlock>>>(
        d_q_n, d_p_n, d_p_n_plus_oneHalf, d_m, N_BODIES);

    for (int step = 0; step < N_STEPS; ++step)
    {
        kernelLeapFrogPart1<<<blocksBodies, threadsPerBlock>>>(
            d_q_n, d_q_n_plus_one, d_p_n_plus_oneHalf, d_m, N_BODIES);

        kernelLeapFrogPart2<<<blocksBodies, threadsPerBlock>>>(
            d_q_n_plus_one, d_p_n_plus_oneHalf, d_p_n_plus_threeHalfs, d_m, N_BODIES);

        // Copy results back to host for history
        cudaMemcpy(h_q_history + N_BODIES * (step + 1),
                   d_q_n_plus_one, N_BODIES * sizeof(Vec3), cudaMemcpyDeviceToHost);

        cudaMemcpy(h_p_history + N_BODIES * (step + 1),
                   d_p_n_plus_threeHalfs, N_BODIES * sizeof(Vec3), cudaMemcpyDeviceToHost);

        // Swap pointers for next iteration
        Vec3 *tmp_q = d_q_n;
        d_q_n = d_q_n_plus_one;
        d_q_n_plus_one = tmp_q;

        Vec3 *tmp_p = d_p_n_plus_oneHalf;
        d_p_n_plus_oneHalf = d_p_n_plus_threeHalfs;
        d_p_n_plus_threeHalfs = tmp_p;
    }
    // debug_print_history(h_q_history, N_BODIES, N_STEPS);

    // Free device memory
    cudaFree(d_q_n);
    cudaFree(d_q_n_plus_one);
    cudaFree(d_p_n);
    cudaFree(d_p_n_plus_oneHalf);
    cudaFree(d_p_n_plus_threeHalfs);
    cudaFree(d_m);

    // Free host memory
    cudaFreeHost(h_q_n);
    cudaFreeHost(h_p_n);
    cudaFreeHost(h_m);
    free(h_q_history);
    free(h_p_history);

    return 0;
}