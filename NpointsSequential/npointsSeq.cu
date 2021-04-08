
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

double Pow(const double& value, const double& degree)
{
    return pow(value, degree);
}

void Nbody(double* u, double* unew)
{
    double
        A1 = 1.0, A2 = 2.0, p1 = 3.0,
        p2 = 0.0, vx = 0.0, vy = 0.0, m = 1.0;
    for (int index = 0; index < N * 4; index += 4)
    {
        for (int j = 0; j < N * 4; j += 4)
        {
            if (index != j)
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
        t = u[index + 3] + tau * unew[index + 1];
        if (t > ymax || t < ymin)
        {
            unew[index + 3] = u[index + 3] + tau * (-unew[index + 1]);
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
    double
        * u = new double[N * 4],
        * unew = new double[N * 4];
    float
        timeS = 0.0f;
    ofstream output("result.txt");
    output.close();
    cudaEvent_t tn, tk;
    cudaEventCreate(&tn);
    cudaEventCreate(&tk);
    cudaEventRecord(tn, 0);
    for (int i = 0; i < N; i++)
    {
        u[i * 4] = 0.0; //vx - velocity
        u[i * 4 + 1] = 0.0; //vy - velocity
        u[i * 4 + 2] = (double)(rand()) / RAND_MAX * (xmax - xmin) + xmin; // x - coordinate
        u[i * 4 + 3] = (double)(rand()) / RAND_MAX * (xmax - xmin) + xmin; // x - coordinate
    }
    for (double t = 0.0; t < tmax; t += tau)
    {
        Nbody(u, unew);
    }
    output.open("result.txt", ios::app);
    output << "[";
    bool t = false;
    for (int i = 2; i < N * 4; i += 4)
    {
        if (t)
        {
            output << ", ";
        }
        output << "(" << setw(10) << setprecision(6) << fixed << u[i] << ", " <<
            setw(10) << setprecision(6) << fixed << u[i + 1] << ")";
        t = true;
    }
    output << "]";
    cudaEventRecord(tk, 0);
    cudaEventSynchronize(tk);
    cudaEventElapsedTime(&timeS, tn, tk);
    cout << "timeSEQ = " << timeS / 1000 << endl;
    output << "timeSEQ = " << timeS / 1000 << endl;
    cout << "Programm has completed its implementation";
    output.close();
    //Удалим события
    cudaEventDestroy(tn); //очистка
    cudaEventDestroy(tk); //очистка
    delete[] u, unew;
    return 0;
}

