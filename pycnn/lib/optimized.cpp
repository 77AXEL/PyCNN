#include <math.h>

extern "C" __declspec(dllexport)
void softmax(const double* __restrict__ x, double* __restrict__ out, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        const double* x_row = &x[i * cols];
        double* out_row = &out[i * cols];

        double row_max = x_row[0];
        for (int j = 1; j < cols; j++) {
            if (x_row[j] > row_max) row_max = x_row[j];
        }

        double sum = 0.0;
        for (int j = 0; j < cols; j++) {
            double e = exp(x_row[j] - row_max);
            out_row[j] = e;
            sum += e;
        }

        double inv_sum = 1.0 / sum;
        for (int j = 0; j < cols; j++) {
            out_row[j] *= inv_sum;
        }
    }
}

extern "C" __declspec(dllexport)
void cross_entropy(const double* __restrict__ preds, const double* __restrict__ labels, double* __restrict__ out, int rows, int cols) {
    const double epsilon = 1e-9;

    for (int i = 0; i < rows; i++) {
        double row_sum = 0.0;
        int row_offset = i * cols;

        for (int j = 0; j < cols; j++) {
            row_sum += labels[row_offset + j] * log(preds[row_offset + j] + epsilon);
        }
        out[i] = -row_sum;
    }
}