%%cuda

#include <iostream>
#include <time.h>
#include <cuda_runtime.h>

void Read(float** R, float** G, float** B, int *N, int *S, int **Orden, const char *filename) {    
    FILE *fp;
    fp = fopen(filename, "r");
    fscanf(fp, "%d %d\n", N, S);
	int P = (*N)/(*S);
    int imsize = (*N) * (*N);
	int orsize = P * P;
    float* R1 = new float[imsize];
    float* G1 = new float[imsize];
    float* B1 = new float[imsize];
	int *O = new int[orsize];
	for (int i = 0; i < orsize; i++)
		fscanf(fp, "%d ", &(O[i]));
	for(int i = 0; i < imsize; i++)
	    fscanf(fp, "%f ", &(R1[i]));
	for(int i = 0; i < imsize; i++)
	    fscanf(fp, "%f ", &(G1[i]));
	for(int i = 0; i < imsize; i++)
	    fscanf(fp, "%f ", &(B1[i]));
    fclose(fp);
    *R = R1; *G = G1; *B = B1, *Orden = O;
}
void Write(float* R, float* G, float* B, int N, const char *filename) {
    FILE *fp;
    fp = fopen(filename, "w");
    fprintf(fp, "%d\n", N);
    for(int i = 0; i < N*N-1; i++)
        fprintf(fp, "%f ", R[i]);
    fprintf(fp, "%f\n", R[N*N-1]);
    for(int i = 0; i < N*N-1; i++)
        fprintf(fp, "%f ", G[i]);
    fprintf(fp, "%f\n", G[N*N-1]);
    for(int i = 0; i < N*N-1; i++)
        fprintf(fp, "%f ", B[i]);
    fprintf(fp, "%f\n", B[N*N-1]);
    fclose(fp);
}


void funcionCPU(float *R, float *G, float *B, float *Rout, float *Gout, float *Bout, int N, int S, int* Orden){
	
	int P = N/S;
	for (int Idx = 0; Idx < N*N; Idx++){
		// (i,j) representa la posicion de un pixel
		int i = Idx % N;
		int j = Idx / N;

		// Piezas
		// (pi, pj) representa la posicion de la pieza
		int pi = i / S;
		int pj = j / S;
		int pieza = pj * P + pi;
		int nueva_pieza = Orden[pieza];
		
		// Posicion final del pixel
		int fi = ( nueva_pieza % P ) * S + pi;
		int fj = ( nueva_pieza / P ) * S + pj;

		// Indexar
		int Idx_nueva = fj * N + fi;

		Rout[Idx_nueva] = R[Idx];	
		Gout[Idx_nueva] = G[Idx];	
		Bout[Idx_nueva] = B[Idx];	
	}
}

__global__ void kernelGPU(float *R, float *G, float *B, float *Rout, float *Gout, float *Bout,  int N, int S, int* Orden){
	int P = N/S;
	int Idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (Idx < N*N){
		// (i,j) representa la posicion de un pixel
		int i = Idx % N;
		int j = Idx / N;

		// Piezas
		// (pi, pj) representa la posicion de la pieza
		int pi = i / S;
		int pj = j / S;
		int pieza = pj * P + pi;
		int nueva_pieza = Orden[pieza];
		
		// Posicion final del pixel
		int fi = (nueva_pieza % P ) * S + i % S;
		int fj = (nueva_pieza / P ) * S + j % S;

		// Indexar
		int Idx_nueva = fj * N + fi;

		Rout[Idx_nueva] = R[Idx];	
		Gout[Idx_nueva] = G[Idx];	
		Bout[Idx_nueva] = B[Idx];	
	}
	
}

int main(int argc, char **argv){
	int N, S;
	int *Ohost, *Odev;
  float *Rhost, *Ghost, *Bhost;
  float *Rhostout, *Ghostout, *Bhostout;
	float *Rdev, *Gdev, *Bdev;
  float *Rdevout, *Gdevout, *Bdevout;
  
  char names[5][3][20] = {
        {"img100x100.txt", "img100x100CPU.txt", "img100x100GPU.txt"},
        {"img200x200.txt", "img200x200CPU.txt", "img200x200GPU.txt"},
        {"img400x400.txt", "img400x400CPU.txt", "img400x400GPU.txt"},
        {"img800x800.txt", "img800x800CPU.txt", "img800x800GPU.txt"},
        {"img1600x1600.txt", "img1600x1600CPU.txt", "img1600x1600GPU.txt"}};
 for(int i = 0; i < 5; i++) {
    Read(&Rhost, &Ghost, &Bhost, &N, &S, &Ohost, names[i][0]);
    // CPU
    Rhostout = (float*)malloc(N*N*sizeof(float));
    Ghostout = (float*)malloc(N*N*sizeof(float));
    Bhostout = (float*)malloc(N*N*sizeof(float));
    
    clock_t t1, t2;
    t1 = clock();

    // Procesar imagen
    funcionCPU(Rhost, Ghost, Bhost, Rhostout, Ghostout, Bhostout, N, S, Ohost); 

    t2 = clock();
    double dif_cpu = 1000.0 * (double) (t2 - t1) / CLOCKS_PER_SEC;
    printf("Tiempo CPU %s: %f [ms]\n", names[i][0], dif_cpu);

    // Guardar imagen salida y liberar memoria
    Write(Rhostout, Ghostout, Bhostout, N, names[i][1]);
    delete[] Rhostout; delete[] Ghostout; delete[] Bhostout;
    
    // GPU
    int grid_size, block_size = 256;
    grid_size = (int)ceil((float) N * N / block_size);
    

    // Reservar memoria imagen GPU
    cudaMalloc((void**)&Rdev, N * N * sizeof(float));
    cudaMalloc((void**)&Gdev, N * N * sizeof(float));
    cudaMalloc((void**)&Bdev, N * N * sizeof(float));

    // Copiar imagen CPU a imagen GPU
    cudaMemcpy(Rdev, Rhost, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Gdev, Ghost, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Bdev, Bhost, N * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Reservar memoria imagen salida GPU
    cudaMalloc((void**)&Rdevout, N * N * sizeof(float));
    cudaMalloc((void**)&Gdevout, N * N * sizeof(float));
    cudaMalloc((void**)&Bdevout, N * N * sizeof(float));

    // Reservar memoria y copiar orden CPU a orden GPU
    cudaMalloc((void**)&Odev, (N/S)*(N/S)*sizeof(int));
    cudaMemcpy(Odev, Ohost, (N/S)*(N/S)*sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t ct1, ct2;
    float dif_gpu;
    cudaEventCreate(&ct1);
    cudaEventCreate(&ct2);
    cudaEventRecord(ct1);

    // Procesar imagen
    kernelGPU<<<grid_size, block_size>>>(Rdev, Gdev, Bdev, Rdevout, Gdevout, Bdevout, N, S, Odev);
    cudaDeviceSynchronize();

    cudaEventRecord(ct2);
    cudaEventSynchronize(ct2);
    cudaEventElapsedTime(&dif_gpu, ct1, ct2);
    printf("Tiempo GPU %s: %f [ms]\n", names[i][0], dif_gpu);

    // Reservar memoria imagen salida CPU
    Rhostout = (float*)malloc(N*N*sizeof(float));
    Ghostout = (float*)malloc(N*N*sizeof(float));
    Bhostout = (float*)malloc(N*N*sizeof(float));

    // Copiar imagen salida GPU a imagen salida CPU
    cudaMemcpy(Rhostout, Rdevout, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Ghostout, Gdevout, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Bhostout, Bdevout, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Guardar imagen salida y liberar memoria
    Write(Rhostout, Ghostout, Bhostout, N, names[i][2]);
    free(Rhost); free(Ghost); free(Bhost); free(Ohost);
    free(Rhostout); free(Ghostout); free(Bhostout);
    cudaFree(Rdev); cudaFree(Gdev); cudaFree(Bdev);  cudaFree(Odev);
    cudaFree(Rdevout); cudaFree(Gdevout); cudaFree(Bdevout);
  }
	return 0;
}
