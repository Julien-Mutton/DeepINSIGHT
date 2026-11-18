/* LINSIGHT training inner loop: a line-by-line port of L-INSIGHT-fit.cpp
 * (updateGradient + updateParameters, GOMPERTZ_SIGMOID build, eta = 0).
 *
 * Inputs x = [F (nf raw columns) | (Z - mu) / sd (nz standardised columns)], never materialised.
 * para / grad / hist have length 2 * (1 + nf + nz): [b, w(nx), c, v(nx)].
 * The gradient of an unfinished mini-batch stays in `grad` (the C++ resets only the counter
 * at the start of an epoch), so it carries into the next call.
 */
#include <math.h>
#include <stdint.h>

static inline double xval(int64_t s, int j, const float *F, int nf, const float *Z, int nz,
                          const double *mu, const double *sd) {
    if (j < nf) return (double)F[s * nf + j];
    int k = j - nf;
    return ((double)Z[s * nz + k] - mu[k]) / sd[k];
}

int64_t fit_epoch(int64_t n, const int64_t *order, const float *F, int nf, const float *Z, int nz,
                  const double *mu, const double *sd, const double *A, const double *B,
                  const double *D, const double *Fv, double beta0, double shift, int64_t mb,
                  double lr_rho, double lr_gamma, double eps, double *para, double *grad,
                  double *hist) {
    int nx = nf + nz;
    int64_t H = 1 + nx, P = 2 * H, counter = 0, updates = 0;
    double x[8192];
    for (int64_t t = 0; t < n; t++) {
        int64_t s = order[t];
        double lr = para[0], lg = para[H];
        for (int j = 0; j < nx; j++) {
            x[j] = xval(s, j, F, nf, Z, nz, mu, sd);
            lr += para[1 + j] * x[j];
            lg += para[H + 1 + j] * x[j];
        }
        double e = exp(-lr);
        double rho = exp(shift * e);
        double gp = 1. / (1. + exp(-lg));
        double gamma = gp * beta0;
        double dR = A[s] * (B[s] + D[s] * gamma) - Fv[s];
        double dG = A[s] * D[s];
        double valP = dR * rho + Fv[s];
        dR = -dR / valP;
        dG = -dG / valP;
        double dGp = dG / beta0;
        double rI = dR * -1 * shift * exp(shift * exp(-lr)) * exp(-lr);
        double gI = dGp * gp * gp * exp(-lg);
        grad[0] += rI;
        grad[H] += gI;
        for (int j = 0; j < nx; j++) {
            grad[1 + j] += rI * x[j];
            grad[H + 1 + j] += gI * x[j];
        }
        counter++;
        if (counter == mb) {
            for (int64_t i = 0; i < P; i++) {
                double d = grad[i];
                hist[i] += d * d;
                para[i] -= d / (sqrt(hist[i]) + eps) * (i < P / 2 ? lr_rho : lr_gamma);
                grad[i] = 0;
            }
            counter = 0;
            updates++;
        }
    }
    return updates;
}

void score(int64_t n, const int64_t *idx, const float *F, int nf, const float *Z, int nz,
           const double *mu, const double *sd, const double *para, double shift,
           double *rho_out, double *gp_out, double *lat_extra_out) {
    int nx = nf + nz;
    int64_t H = 1 + nx;
    for (int64_t t = 0; t < n; t++) {
        int64_t s = idx[t];
        double lr = para[0], lg = para[H], le = 0;
        for (int j = 0; j < nx; j++) {
            double xv = xval(s, j, F, nf, Z, nz, mu, sd);
            lr += para[1 + j] * xv;
            lg += para[H + 1 + j] * xv;
            if (j >= nf) le += para[1 + j] * xv;
        }
        rho_out[t] = exp(shift * exp(-lr));
        gp_out[t] = 1. / (1. + exp(-lg));
        lat_extra_out[t] = le;
    }
}
