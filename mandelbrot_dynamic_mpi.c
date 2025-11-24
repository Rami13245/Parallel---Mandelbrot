#include <stdio.h>
#include <mpi.h>

#define WIDTH 640
#define HEIGHT 480
#define MAX_ITER 255
#define TAG_TASK 1
#define TAG_RESULT 2
#define TAG_TERMINATE 0

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

    double start_time = MPI_Wtime();

    if (rank == 0) {
        int image[HEIGHT][WIDTH];
        int next_row = 0;
        int active_workers = size - 1;

        for (int worker = 1; worker < size; worker++) {
            MPI_Send(&next_row, 1, MPI_INT, worker, TAG_TASK, MPI_COMM_WORLD);
            next_row++;
        }

        while (active_workers > 0) {
            int row_data[WIDTH + 1];
            MPI_Status status;
            MPI_Recv(row_data, WIDTH + 1, MPI_INT, MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &status);
            int src = status.MPI_SOURCE;
            int recv_row = row_data[0];
            for (int j = 0; j < WIDTH; j++) {
                image[recv_row][j] = row_data[j + 1];
            }

            if (next_row < HEIGHT) {
                MPI_Send(&next_row, 1, MPI_INT, src, TAG_TASK, MPI_COMM_WORLD);
                next_row++;
            } else {
                int terminate = -1;
                MPI_Send(&terminate, 1, MPI_INT, src, TAG_TERMINATE, MPI_COMM_WORLD);
                active_workers--;
            }
        }

        double end_time = MPI_Wtime();
        printf("Dynamic execution time: %f seconds\n", end_time - start_time);
        save_pgm("mandelbrot_dynamic.pgm", image);
    } else {
        while (1) {
            int row;
            MPI_Recv(&row, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (row < 0) break;

            int row_data[WIDTH + 1];
            row_data[0] = row;
            struct complex c;
            for (int j = 0; j < WIDTH; j++) {
                c.real = (j - WIDTH / 2.0) * 4.0 / WIDTH;
                c.imag = (row - HEIGHT / 2.0) * 4.0 / HEIGHT;
                row_data[j + 1] = cal_pixel(c);
            }
            MPI_Send(row_data, WIDTH + 1, MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}