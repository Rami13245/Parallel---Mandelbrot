#include <stdio.h>
#include <mpi.h>

#define WIDTH 640
#define HEIGHT 480
#define MAX_ITER 255

struct complex {
    double real;
    double imag;
};

int cal_pixel(struct complex c) {
    double z_real = 0;
    double z_imag = 0;
    double z_real2, z_imag2, lengthsq;
    int iter = 0;
    do {
        z_real2 = z_real * z_real;
        z_imag2 = z_imag * z_imag;
        z_imag = 2 * z_real * z_imag + c.imag;
        z_real = z_real2 - z_imag2 + c.real;
        lengthsq = z_real * z_real + z_imag * z_imag;
        iter++;
    } while ((iter < MAX_ITER) && (lengthsq < 4.0));
    return iter;
}

void save_pgm(const char *filename, int image[HEIGHT][WIDTH]) {
    FILE* pgmimg;
    pgmimg = fopen(filename, "wb");
    fprintf(pgmimg, "P2\n");
    fprintf(pgmimg, "%d %d\n", WIDTH, HEIGHT);
    fprintf(pgmimg, "255\n");
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            fprintf(pgmimg, "%d ", image[i][j]);
        }
        fprintf(pgmimg, "\n");
    }
    fclose(pgmimg);
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int image[HEIGHT][WIDTH];
    struct complex c;

    int rows_per_proc = HEIGHT / size;
    int remainder = HEIGHT % size;
    int start_row = rank * rows_per_proc + (rank < remainder ? rank : remainder);
    int end_row = start_row + rows_per_proc + (rank < remainder ? 1 : 0);

    double start_time = MPI_Wtime();

    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < WIDTH; j++) {
            c.real = (j - WIDTH / 2.0) * 4.0 / WIDTH;
            c.imag = (i - HEIGHT / 2.0) * 4.0 / HEIGHT;
            image[i][j] = cal_pixel(c);
        }
    }

    if (rank != 0) {
        for (int i = start_row; i < end_row; i++) {
            MPI_Send(image[i], WIDTH, MPI_INT, 0, i, MPI_COMM_WORLD);
        }
    } else {
        for (int p = 1; p < size; p++) {
            int p_start = p * rows_per_proc + (p < remainder ? p : remainder);
            int p_end = p_start + rows_per_proc + (p < remainder ? 1 : 0);
            for (int i = p_start; i < p_end; i++) {
                MPI_Recv(image[i], WIDTH, MPI_INT, p, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }

    double end_time = MPI_Wtime();
    if (rank == 0) {
        printf("Static execution time: %f seconds\n", end_time - start_time);
        save_pgm("mandelbrot_static.pgm", image);
    }

    MPI_Finalize();
    return 0;
}