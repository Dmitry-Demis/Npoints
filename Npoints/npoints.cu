
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iomanip>

using namespace std;
#define N 1000 // number of points
#define tau 1e-3 // τ
#define block_size 32 // a size of a block
#define tmax 1.0 // maximum time
#define xmax 5.0 // x coordinate - max
#define xmin -5.0 // x coordinate - min
#define ymax  5.0 // the same
#define ymin -5.0 

__device__ double Pow(const double& value, const double& degree)
{
    return pow(value, degree);
}


__global__ void Nbody(double* u, double *unew)
{

    double 
        A1 = 1.0, A2 = 2.0, p1 = 3.0, 
        p2 = 0.0, vx = 0.0, vy = 0.0, m = 1.0;
    int index = 4 * (blockIdx.x * blockDim.x + threadIdx.x); // every thread has a unique value
    if (index < N*4)
    {
        for (int j = 0; j < N*4; j+=4)
        {
            if (index!=j)
            {
                double z =
                    sqrt
                    (
                        Pow(u[j + 2] - u[index + 2], 2) 
                        +
                        Pow(u[j + 3] - u[index + 3], 2)
                    );
                vx += (A1 / Pow(z, p1) - A2 / Pow(z, p2)) * (u[j + 2] - u[index + 2]) / m;
                vy += (A1 / Pow(z, p1) - A2 / Pow(z, p2)) * (u[j + 3] - u[index + 3]) / m;
            }
        }
        unew[index] = u[index] + tau * vx;
        unew[index + 1] = u[index + 1] + tau * vy;
        double t = u[index + 2] + tau * unew[index];
        if (t > xmax || t < xmin)
        {
            unew[index + 2] = u[index + 2] + tau * (-unew[index]);
        }
        else
        {
            unew[index + 2] = t;
        }
        t = u[index + 3] + tau * unew[index+1];
        if (t > ymax || t < ymin)
        {
            unew[index + 3] = u[index + 3] + tau * (-unew[index+1]);
        }
        else
        {
            unew[index + 3] = t;
        }
    }
}

int main()
{
    //srand((unsigned)(time(NULL)));
    int sizeForKernel = N * 4 * sizeof(double);
    double
        *uDev= NULL,
        *unewDev = NULL,
        *u = new double[N * 4], 
        *unew = new double[N * 4];
    int ERROR = 0;
    float
        timeS = 0.0f;
    ofstream output("result.txt");
    output.close();
    cudaEvent_t tn, tk;
    cudaEventCreate(&tn);
    cudaEventCreate(&tk);
    cudaEventRecord(tn, 0);
    // определим значения исходной функции на границах
    for (int i = 0; i < N; i++)
    {
        u[i * 4] = 0.0; //vx - velocity
        u[i * 4 + 1] = 0.0; //vy - velocity
        u[i * 4 + 2] = (double)(rand()) / RAND_MAX * (xmax - xmin) + xmin; // x - coordinate
        u[i * 4 + 3] = (double)(rand()) / RAND_MAX * (xmax - xmin) + xmin; // x - coordinate
    }
    if (cudaMalloc((void**)&uDev, sizeForKernel)!=cudaSuccess)
    {
        cout << "Невозможно выделить память на uDev\n";
        ERROR = 1;
    }
    if (cudaMalloc((void**)&unewDev, sizeForKernel)!=cudaSuccess)
    {
        cout << "Невозможно выделить память на unewDev\n";
        ERROR = 1;
    }
    if (cudaMemcpy(uDev, u, sizeForKernel, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        cout << ("Произошла ошибка при копировании из \"u\" в \"uDev\"\n");
        ERROR = 1;
    }

    for (double t = 0.0; t < tmax; t+=tau)
    {
        Nbody<< < (int)(N / block_size) + 1, block_size  >> >  (uDev, unewDev);
        cudaThreadSynchronize();
    }


    //Скопируем значения из массива uDev в массив u размера size
    if (cudaMemcpy(u, uDev, sizeForKernel, cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        printf("Произошла ошибка при копировании из \"uDev\" в \"u\"\n");
        ERROR = 1;
    }
   
    output.open("result.txt", ios::app);
    
    output << "[";
    bool t = false;
    for (int i = 2; i < N*4; i+=4)
    {
       
        if (t)
        {
            output << ", ";
        }
        output << "(" << setw(10) << setprecision(6) << fixed << u[i] << ", " <<
            setw(10) << setprecision(6) << fixed << u[i + 1] << ")";
        t = true;
        //
        //cout << setw(10) << setprecision(6) << fixed << u[i] << "  ";
    }
    output << "]";
    
    cudaEventRecord(tk, 0);
    cudaEventSynchronize(tk);
    cudaEventElapsedTime(&timeS, tn, tk);
    output << "timePARALLEL = " << timeS / 1000 << endl;
    output.close();
    //Удалим события
    cudaEventDestroy(tn); //очистка
    cudaEventDestroy(tk); //очистка
    //Освободим память на device
    cudaFree(uDev);
    cudaFree(unewDev);
    delete[] u, unew;
    if (ERROR)
    {
        return -1;
    }
    else
    {
        return 0;
    }
}

