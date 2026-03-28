#include <iostream>
#include <string>
#include <random>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <cstring>

struct Vec3
{
    float x, y, z;
};

struct Body
{
    Vec3 q;
    Vec3 p;
    float mass;
};

__global__ void kernelcomputeDistances(float *d_distances, Vec3 *d_q_n, int n_bodies)
{
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;

    if (threadId < n_bodies * n_bodies)
    {
        int n2 = (int)threadId / n_bodies;
        int n1 = (int)threadId - n2 * n_bodies;

        if (n1 != n2)
        {
            Vec3 q1 = d_q_n[n1];
            Vec3 q2 = d_q_n[n2];

            float dqx = q1.x - q2.x;
            float dqy = q1.y - q2.y;
            float dqz = q1.z - q2.z;

            int distance = sqrtf(dqx * dqx + dqy * dqy + dqz * dqz);

            d_distances[n1 + n2 * n_bodies] = distance;
        }
        else
        {
            d_distances[n1 + n2 * n_bodies] = 0.0f;
        }
    }
}

__global__ void kernelLeapFrogPart1(Vec3 *d_q_n, Vec3 *d_p_n, Vec3 *d_p_n_plus_oneHalf, float *d_m, float *d_distances, int n_bodies)
{
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;

    if (threadId <= n_bodies)
    {
        // Register memory
        Vec3 q_n = d_q_n[threadId];
        Vec3 p_n = d_q_n[threadId];
        float m = d_m[threadId];

        // Leapfrog algorithm
    }
}

__global__ void kernelLeapFrogPart2(Vec3 *d_q_n, Vec3 *d_p_n, Vec3 *d_p_n_plus_oneHalf, float *d_m, float *d_distances, int n_bodies)
{
    // TODO
}

__host__ void
initializeBodies(Vec3 *h_q_n, Vec3 *h_p_n, float *h_m, int n_bodies, const std::string method)
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

int main(int argc, char *argv[])
{
    int N_BODIES = 1000;
    int N_STEPS = 1000;

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
    Vec3 *h_p_n_plus_oneHalf;
    cudaMallocHost((void **)&h_p_n_plus_oneHalf, N_BODIES * sizeof(Vec3));
    float *h_m;
    cudaMallocHost((void **)&h_m, N_BODIES * sizeof(float));
    // Pageable memory
    Vec3 *h_q_history = (Vec3 *)malloc(N_BODIES * N_STEPS * sizeof(Vec3));
    Vec3 *h_p_history = (Vec3 *)malloc(N_BODIES * N_STEPS * sizeof(Vec3));

    // Allocate device memory

    // Global memory
    Vec3 *d_q_n;
    cudaMalloc((void **)&d_q_n, N_BODIES * sizeof(Vec3));
    Vec3 *d_p_n;
    cudaMalloc((void **)&d_p_n, N_BODIES * sizeof(Vec3));
    Vec3 *d_p_n_plus_oneHalf;
    cudaMalloc((void **)&d_p_n_plus_oneHalf, N_BODIES * sizeof(Vec3));
    float *d_m;
    cudaMalloc((void **)&d_m, N_BODIES * sizeof(float));
    float *d_distances;
    cudaMalloc((void **)&d_distances, N_BODIES * N_BODIES * sizeof(float));

    // Initialize bodies qs and ps

    initializeBodies(h_q_n, h_p_n, h_m, N_BODIES, "naive");

    // Copy h_bodies_current to d_bodies_current

    cudaMemcpy(d_q_n, h_q_n, N_BODIES * sizeof(Vec3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p_n, h_p_n, N_BODIES * sizeof(Vec3), cudaMemcpyHostToDevice);

    // Copy h_bodies_current to h_history for the first step

    memcpy(h_q_history, h_q_n, N_BODIES * sizeof(Vec3));
    memcpy(h_p_history, h_p_n, N_BODIES * sizeof(Vec3));

    // Run the simulation for N_STEPS steps

    for (int step = 0; step < N_STEPS; ++step)
    {
        // Launch kernel to compute the next state of the bodies

        int threadsPerBlock = 512;
        int blocksPerGrid = (N_BODIES + threadsPerBlock - 1) / threadsPerBlock;

        kernelcomputeDistances<<<blocksPerGrid, threadsPerBlock>>>(d_distances, d_q_n, N_BODIES);
        kernelLeapFrogPart1<<<blocksPerGrid, threadsPerBlock>>>(d_q_n, d_p_n, d_p_n_plus_oneHalf, d_m, d_distances, N_BODIES);
    }

    return 0;
}