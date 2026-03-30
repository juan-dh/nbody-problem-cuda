#include <iostream>
#include <string>
#include <tuple>
#include <random>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <cstring>

#define BLOCK_SIZE 512

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

__global__ void computeForceTiled(Vec3 *forces, const Vec3 *q, const float *m, int n_bodies)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    bool active = (i < n_bodies);

    __shared__ Vec3 s_q[BLOCK_SIZE];
    __shared__ float s_m[BLOCK_SIZE];

    Vec3 accum{0.0f, 0.0f, 0.0f};

    Vec3 qi = active ? q[i] : Vec3{0, 0, 0};
    float mi = active ? m[i] : 0.0f;

    int n_tiles = (n_bodies + blockDim.x - 1) / blockDim.x;

    for (int tile = 0; tile < n_tiles; tile++)
    {
        int global_load = tile * blockDim.x + threadIdx.x;

        if (global_load < n_bodies)
        {
            s_q[threadIdx.x] = q[global_load];
            s_m[threadIdx.x] = m[global_load];
        }
        else
        {
            s_q[threadIdx.x] = {0.0f, 0.0f, 0.0f};
            s_m[threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int j = 0; j < blockDim.x; ++j)
        {
            int global_j = tile * blockDim.x + j;

            if (global_j >= n_bodies)
                continue;
            if (global_j == i)
                continue;

            float dx = qi.x - s_q[j].x;
            float dy = qi.y - s_q[j].y;
            float dz = qi.z - s_q[j].z;

            float r2 = dx * dx + dy * dy + dz * dz + (float)(d_epsilon * d_epsilon);
            float inv_r = rsqrtf(r2);
            float inv_r3 = inv_r * inv_r * inv_r;

            float mag = -(float)(d_G * mi * s_m[j]) * inv_r3;

            accum.x += dx * mag;
            accum.y += dy * mag;
            accum.z += dz * mag;
        }

        __syncthreads();
    }

    if (active)
        forces[i] = accum;
}

__global__ void kernelInitHalfStep(const Vec3 *q, const Vec3 *p, Vec3 *p_half, const float *m, const Vec3 *forces, int n_bodies)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_bodies)
        return;

    p_half[i] = p[i] + forces[i] * ((float)d_h * 0.5f);
}

__global__ void kernelLeapFrogPart1(const Vec3 *d_q_n, Vec3 *d_q_n_plus_one, const Vec3 *d_p_n_plus_oneHalf, const float *d_m, int n_bodies)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_bodies)
        return;

    d_q_n_plus_one[i] = d_q_n[i] + d_p_n_plus_oneHalf[i] * ((float)d_h / d_m[i]);
}

__global__ void kernelLeapFrogPart2(const Vec3 *d_q_n_plus_one, const Vec3 *d_p_n_plus_oneHalf, Vec3 *d_p_n_plus_threeHalfs, const float *d_m, const Vec3 *forces, int n_bodies)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_bodies)
        return;

    d_p_n_plus_threeHalfs[i] = d_p_n_plus_oneHalf[i] + forces[i] * ((float)d_h);
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
    else if (method == "multiple gaussians")
    {
        // Multiple Gaussians method
        int n_clusters = 5;
        int cluster_size = n_bodies / n_clusters;

        std::mt19937 gen(std::random_device{}());
        std::normal_distribution<float> cluster_dist(0.0f, 40.0f);
        std::normal_distribution<float> q_dist(0.0f, 15.0f);
        std::normal_distribution<float> p_dist(0.0f, 20.0f);
        std::uniform_int_distribution<> mass_dist(0, 100);

        for (int c = 0; c < n_clusters; ++c)
        {
            float cluster_qx = cluster_dist(gen);
            float cluster_qy = cluster_dist(gen);
            float cluster_qz = cluster_dist(gen);

            for (int i = 0; i < cluster_size; ++i)
            {
                int idx = c * cluster_size + i;
                if (idx >= n_bodies)
                    break;

                h_q_n[idx].x = cluster_qx + q_dist(gen); // Cluster around center
                h_q_n[idx].y = cluster_qy + q_dist(gen);
                h_q_n[idx].z = cluster_qz + q_dist(gen);

                h_p_n[idx].x = p_dist(gen);
                h_p_n[idx].y = p_dist(gen);
                h_p_n[idx].z = p_dist(gen);

                h_m[idx] = static_cast<float>(mass_dist(gen)) + 0.1f; // Avoid zero mass
            }
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

template <typename T>
__host__ T *allocatePinnedHostMemory(int n_elements)
{
    T *ptr = NULL;
    cudaError_t err;

    // Pinned memory allocation
    err = cudaMallocHost((void **)&ptr, n_elements * sizeof(T));
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to allocate pinned host memory: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    return ptr;
}

template <typename T>
__host__ T *allocateDeviceMemory(int n_elements)
{
    T *ptr = NULL;
    cudaError_t err;

    // Device memory allocation
    err = cudaMalloc((void **)&ptr, n_elements * sizeof(T));
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to allocate device memory: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    return ptr;
}

template <typename T>
__host__ void copyFromHostToDeviceGlobal(T *d_ptr, const T *h_ptr, int n_elements)
{
    cudaError_t err = cudaMemcpy(d_ptr, h_ptr, n_elements * sizeof(T), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to copy from host to device: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

template <typename T>
__host__ void copyFromHostToDeviceConst(const T &d_symbol, const T &h_value)
{
    cudaError_t err = cudaMemcpyToSymbol(d_symbol, &h_value, sizeof(T));
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to copy to constant memory: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void saveToCSV(const char *filename, const Vec3 *h_q_history, int N_BODIES, int N_STEPS)
{
    FILE *file = fopen(filename, "w");
    if (!file)
    {
        std::cerr << "Error opening CSV file\n";
        return;
    }

    // Header
    fprintf(file, "step,particle,x,y,z\n");

    for (int step = 0; step < N_STEPS + 1; step++)
    {
        for (int i = 0; i < N_BODIES; i++)
        {
            int idx = i + step * N_BODIES;

            fprintf(file, "%d,%d,%f,%f,%f\n", step, i, h_q_history[idx].x, h_q_history[idx].y, h_q_history[idx].z);
        }
    }

    fclose(file);
}

void saveToBinary(const char *filename, const Vec3 *h_q_history, int N_BODIES, int N_STEPS)
{
    FILE *file = fopen(filename, "wb");
    if (!file)
    {
        std::cerr << "Error opening binary file\n";
        return;
    }

    // Save metadata
    fwrite(&N_BODIES, sizeof(int), 1, file);
    fwrite(&N_STEPS, sizeof(int), 1, file);

    // Save data
    size_t total = (size_t)N_BODIES * (N_STEPS + 1);
    fwrite(h_q_history, sizeof(Vec3), total, file);

    fclose(file);
}

#include <tuple>

std::tuple<int, int, std::string, double> parseArguments(int argc, char *argv[])
{

    // Default values
    int n_bodies = 100000;
    int n_steps = 1000;
    std::string method = "multiple gaussians";
    double step_size = 1e-3;

    try
    {
        if (argc > 1)
            n_bodies = std::stoi(argv[1]);
        if (argc > 2)
            n_steps = std::stoi(argv[2]);
        if (argc > 3)
            method = argv[3];
        if (argc > 4)
            step_size = std::stod(argv[4]);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }

    // Validation
    if (n_bodies <= 0 || n_steps <= 0)
    {
        std::cerr << "n_bodies and n_steps must be positive.\n";
        exit(EXIT_FAILURE);
    }

    if (step_size <= 0)
    {
        std::cerr << "step_size must be positive.\n";
        exit(EXIT_FAILURE);
    }

    return std::make_tuple(n_bodies, n_steps, method, step_size);
}

int main(int argc, char *argv[])
{

    // Simulation parameters

    // double h_G = 6.67430e-11;
    double h_G = 1;
    double h_epsilon = 1e-2;

    // Parse command line arguments

    auto [N_BODIES, N_STEPS, init_method, h_h] = parseArguments(argc, argv);

    // Print simulation parameters

    std::cout << "Simulation parameters:\n";
    std::cout << "Bodies: " << N_BODIES << "\n";
    std::cout << "Steps: " << N_STEPS << "\n";
    std::cout << "Method: " << init_method << "\n";
    std::cout << "Step size: " << h_h << "\n";

    // Set device

    cudaSetDevice(0);

    // Print device properties

    std::cout << "Device properties:" << std::endl;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "Name: " << prop.name << std::endl;
    std::cout << "Total global memory: " << prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;
    std::cout << "Shared memory per block: " << prop.sharedMemPerBlock / 1024.0 << " KB" << std::endl;

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

    std::cout << "Allocating host memory..." << std::endl;

    Vec3 *h_q_n = allocatePinnedHostMemory<Vec3>(N_BODIES);
    Vec3 *h_p_n = allocatePinnedHostMemory<Vec3>(N_BODIES);
    float *h_m = allocatePinnedHostMemory<float>(N_BODIES);
    Vec3 *h_q_history = allocatePinnedHostMemory<Vec3>(N_BODIES * (N_STEPS + 1));
    Vec3 *h_p_history = allocatePinnedHostMemory<Vec3>(N_BODIES * (N_STEPS + 1));

    // Allocate device memory

    std::cout << "Allocating device memory..." << std::endl;

    // Global memory
    Vec3 *d_q_n = allocateDeviceMemory<Vec3>(N_BODIES);
    Vec3 *d_q_n_plus_one = allocateDeviceMemory<Vec3>(N_BODIES);
    Vec3 *d_p_n = allocateDeviceMemory<Vec3>(N_BODIES);
    Vec3 *d_p_n_plus_oneHalf = allocateDeviceMemory<Vec3>(N_BODIES);
    Vec3 *d_p_n_plus_threeHalfs = allocateDeviceMemory<Vec3>(N_BODIES);
    float *d_m = allocateDeviceMemory<float>(N_BODIES);
    Vec3 *d_forces = allocateDeviceMemory<Vec3>(N_BODIES);

    // Initialize bodies qs and ps

    std::cout << "Initializing bodies..." << std::endl;

    initializeBodies(h_q_n, h_p_n, h_m, N_BODIES, "multiple gaussians");

    std::cout << "Copying memory..." << std::endl;

    // Copy h_q and h_p to h_q_history and h_p_history

    memcpy(h_q_history, h_q_n, N_BODIES * sizeof(Vec3));
    memcpy(h_p_history, h_p_n, N_BODIES * sizeof(Vec3));

    // Copy from host to device

    // Copy h_q_n, h_q_m, h_m to d_q_n, d_q_m, d_m respectively
    copyFromHostToDeviceGlobal(d_q_n, h_q_n, N_BODIES);
    copyFromHostToDeviceGlobal(d_p_n, h_p_n, N_BODIES);
    copyFromHostToDeviceGlobal(d_m, h_m, N_BODIES);
    // Copy to constant memory
    copyFromHostToDeviceConst<double>(d_G, h_G);
    copyFromHostToDeviceConst<double>(d_h, h_h);
    copyFromHostToDeviceConst<double>(d_epsilon, h_epsilon);

    // Print

    // debug_print_history(h_q_history, 10, 1);

    // Streams and events to copy data back to host asynchronously

    cudaEvent_t computeDone;
    cudaStream_t streamCompute;
    cudaStream_t streamCopy;

    cudaStreamCreate(&streamCompute);
    cudaStreamCreate(&streamCopy);

    cudaEventCreate(&computeDone);

    std::cout << "Running simulation..." << std::endl;

    // Run the simulation for N_STEPS steps
    int threadsPerBlock = BLOCK_SIZE;
    int blocksBodies = (N_BODIES + threadsPerBlock - 1) / threadsPerBlock;

    // Initial step of Leapfrog
    computeForceTiled<<<blocksBodies, threadsPerBlock>>>(d_forces, d_q_n, d_m, N_BODIES);
    kernelInitHalfStep<<<blocksBodies, threadsPerBlock>>>(
        d_q_n, d_p_n, d_p_n_plus_oneHalf, d_m, d_forces, N_BODIES);

    for (int step = 0; step < N_STEPS; ++step)
    {

        std::cout << "Step: " << step + 1 << "/" << N_STEPS << std::endl;

        kernelLeapFrogPart1<<<blocksBodies, threadsPerBlock, 0, streamCompute>>>(
            d_q_n, d_q_n_plus_one, d_p_n_plus_oneHalf, d_m, N_BODIES);

        computeForceTiled<<<blocksBodies, threadsPerBlock, 0, streamCompute>>>(d_forces, d_q_n_plus_one, d_m, N_BODIES);

        kernelLeapFrogPart2<<<blocksBodies, threadsPerBlock, 0, streamCompute>>>(
            d_q_n_plus_one, d_p_n_plus_oneHalf, d_p_n_plus_threeHalfs, d_m, d_forces, N_BODIES);

        // Copy results back to host for history
        cudaEventRecord(computeDone, streamCompute);
        cudaStreamWaitEvent(streamCopy, computeDone, 0);

        cudaMemcpyAsync(h_q_history + N_BODIES * (step + 1),
                        d_q_n_plus_one, N_BODIES * sizeof(Vec3), cudaMemcpyDeviceToHost, streamCopy);

        cudaMemcpyAsync(h_p_history + N_BODIES * (step + 1),
                        d_p_n_plus_threeHalfs, N_BODIES * sizeof(Vec3), cudaMemcpyDeviceToHost, streamCopy);

        // Swap pointers for next iteration
        Vec3 *tmp_q = d_q_n;
        d_q_n = d_q_n_plus_one;
        d_q_n_plus_one = tmp_q;

        Vec3 *tmp_p = d_p_n_plus_oneHalf;
        d_p_n_plus_oneHalf = d_p_n_plus_threeHalfs;
        d_p_n_plus_threeHalfs = tmp_p;
    }

    // Wait for all operations to finish before accessing results

    cudaDeviceSynchronize();

    // Debug print first 10 bodies and 10 steps

    std::cout << "Debug print of first 10 bodies and 3 steps:" << std::endl;

    debug_print_history(h_q_history, 10, 3);

    // Write results to file

    std::cout << "Writing results to file..." << std::endl;

    // For the CSV file, just as a proof of execution, we will only save first 10 bodies and 10 steps to avoid creating a huge file. For the binary file, we will save all data.
    saveToCSV("nbody_proof_of_execution.csv", h_q_history, 10, 50);
    saveToBinary("nbody_data.bin", h_q_history, N_BODIES, N_STEPS);

    // Free device memory
    cudaFree(d_q_n);
    cudaFree(d_q_n_plus_one);
    cudaFree(d_p_n);
    cudaFree(d_p_n_plus_oneHalf);
    cudaFree(d_p_n_plus_threeHalfs);
    cudaFree(d_m);
    cudaFree(d_forces);

    // Free host memory
    cudaFreeHost(h_q_n);
    cudaFreeHost(h_p_n);
    cudaFreeHost(h_m);
    cudaFreeHost(h_q_history);
    cudaFreeHost(h_p_history);

    // Destroy streams and events
    cudaEventDestroy(computeDone);
    cudaStreamDestroy(streamCompute);
    cudaStreamDestroy(streamCopy);

    std::cout << "Simulation completed successfully!" << std::endl;

    return 0;
}