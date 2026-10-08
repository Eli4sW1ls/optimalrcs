"""Polynomial basis in (r, y) that is never materialized.

The NPNEq/NPNEt iterations only need two things from the basis f_k, k < M:

    akj = sum_t fa_k(t) g_j(t),   b_k = sum_t fa_k(t) v(t),   delta_r = sum_k al_k f_k

with fa = f(t), fb = f(t+1) and g = fb*w - fa*(w+gamma). Holding f as an
(M, N) array costs M*8 bytes per frame and makes the akj product a skinny GEMM
that runs far below the GPU's FP64 peak. Instead, `_MOMENTS_SRC` evaluates the
monomials of each frame in shared memory and accumulates akj and b directly,
and `_UPDATE_SRC` evaluates sum_k al_k f_k frame by frame, so the iteration
needs only a few length-N vectors.
"""
import cupy as cp

_MOMENTS_SRC = r"""
extern "C" __global__ void npneq_moments(
        const double* __restrict__ rh, const double* __restrict__ yh, const double* __restrict__ env,
        const double* __restrict__ w, const double* __restrict__ v, const double gamma,
        const long long n_trans, double* __restrict__ out)
{
    // rows 0..M-1 of sg hold g_j, row M holds v (for b); +1 padding avoids bank conflicts
    __shared__ double sfa[M * LDS];
    __shared__ double sg[(M + 1) * LDS];
    double acc[NACC];
    #pragma unroll
    for (int q = 0; q < NACC; ++q) acc[q] = 0.0;

    for (long long t0 = (long long)blockIdx.x * TILE; t0 < n_trans; t0 += (long long)gridDim.x * TILE) {
        __syncthreads();
        const int j = threadIdx.x;
        if (j < TILE) {
            const long long t = t0 + j;
            if (t < n_trans) {
                // monomials r^ir y^iy env, in the row order of basis_poly_ry
                const double wt = w[t], ra = rh[t], rb = rh[t + 1], ya = yh[t], yb = yh[t + 1];
                double fya = env[t], fyb = env[t + 1];
                int k = 0;
                for (int iy = 0; iy <= NDEG; ++iy) {
                    double fa = fya, fb = fyb;
                    for (int ir = 0; ir <= NDEG - iy; ++ir) {
                        sfa[k * LDS + j] = fa;
#if STABLE
                        sg[k * LDS + j] = -(fa * wt);
#else
                        sg[k * LDS + j] = fb * wt - fa * (wt + gamma);
#endif
                        fa *= ra; fb *= rb; ++k;
                    }
                    fya *= ya; fyb *= yb;
                }
                sg[M * LDS + j] = v[t];
            } else {
                for (int k = 0; k < M; ++k) { sfa[k * LDS + j] = 0.0; sg[k * LDS + j] = 0.0; }
                sg[M * LDS + j] = 0.0;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int q = 0; q < NACC; ++q) {
            const int o = threadIdx.x + q * BLOCK;
            if (o < M * (M + 1)) {
                const double* pa = sfa + (o / (M + 1)) * LDS;
                const double* pg = sg + (o % (M + 1)) * LDS;
                double s = acc[q];
                for (int jj = 0; jj < TILE; ++jj) s = fma(pa[jj], pg[jj], s);
                acc[q] = s;
            }
        }
    }
    #pragma unroll
    for (int q = 0; q < NACC; ++q) {
        const int o = threadIdx.x + q * BLOCK;
        if (o < M * (M + 1)) out[(long long)blockIdx.x * (M * (M + 1)) + o] = acc[q];
    }
}
"""

_UPDATE_SRC = r"""
double s = 0.0, fy = env;
int k = 0;
for (int iy = 0; iy <= ndeg; ++iy) {
    double fr = fy;
    for (int ir = 0; ir <= ndeg - iy; ++ir) { s += a[k] * fr; fr *= rh; ++k; }
    fy *= yh;
}
const double x = r + s;
rn = x < lo ? lo : (x > hi ? hi : x);
"""

_BLOCK = 256
_moment_kernels = {}
_update_kernel = cp.ElementwiseKernel(
    "float64 r, float64 rh, float64 yh, float64 env, raw float64 a, int32 ndeg, float64 lo, float64 hi",
    "float64 rn", _UPDATE_SRC, "polybasis_update")


def _moments_kernel(ndeg, stable):
    key = (ndeg, bool(stable))
    if key not in _moment_kernels:
        m = (ndeg + 1) * (ndeg + 2) // 2
        # largest tile of frames whose basis values fit in 46 KB of shared memory
        tile = next(t for t in (128, 64, 32, 16, 8) if (2 * m + 1) * (t + 1) * 8 <= 46 * 1024)
        nacc = -(-m * (m + 1) // _BLOCK)
        opts = (f"-DM={m}", f"-DNDEG={ndeg}", f"-DTILE={tile}", f"-DLDS={tile + 1}",
                f"-DNACC={nacc}", f"-DBLOCK={_BLOCK}", f"-DSTABLE={int(bool(stable))}")
        _moment_kernels[key] = (cp.RawKernel(_MOMENTS_SRC, "npneq_moments", options=opts), tile)
    return _moment_kernels[key]


class PolyBasisRY:
    """The basis of `basis_poly_ry(r, y, n, fenv)`, evaluated on the fly.

    Calling it as fk(s, e) returns the columns s..e-1 as an (M, e-s) array, so
    it can stand in wherever a lazily evaluated basis is accepted.
    """

    def __init__(self, r, y, n, fenv=None):
        self.r_hat = cp.ascontiguousarray(r / cp.max(cp.abs(r)), dtype=cp.float64)
        self.y_hat = cp.ascontiguousarray(y / cp.max(cp.abs(y)), dtype=cp.float64)
        self.env = cp.ones_like(self.r_hat) if fenv is None else cp.ascontiguousarray(fenv, dtype=cp.float64)
        self.n = int(n)
        self.n_basis = (self.n + 1) * (self.n + 2) // 2

    def __call__(self, s, e):
        rh, yh, f = self.r_hat[s:e], self.y_hat[s:e], self.env[s:e]
        fk = cp.empty((self.n_basis, e - s), dtype=cp.float64)
        k = 0
        for iy in range(self.n + 1):
            fk[k] = f
            for _ in range(self.n - iy):
                cp.multiply(fk[k], rh, out=fk[k + 1])
                k += 1
            k += 1
            if iy < self.n:
                f = f * yh
        return fk

    def transition_moments(self, w, v, gamma=0.0, stable=False):
        """akj = sum_t fa_k (fb_j w - fa_j (w + gamma)) and b_k = sum_t fa_k v, over
        the transitions t -> t+1 (w and v have length N-1).
        With stable=True, akj = -sum_t fa_k fa_j w instead."""
        kernel, tile = _moments_kernel(self.n, stable)
        n_trans = self.r_hat.shape[0] - 1
        n_sm = cp.cuda.Device().attributes["MultiProcessorCount"]
        blocks = max(1, min(-(-n_trans // tile), 8 * n_sm))
        m = self.n_basis
        out = cp.empty((blocks, m * (m + 1)), dtype=cp.float64)
        kernel((blocks,), (_BLOCK,),
               (self.r_hat, self.y_hat, self.env, cp.ascontiguousarray(w, dtype=cp.float64),
                cp.ascontiguousarray(v, dtype=cp.float64), cp.float64(gamma), cp.int64(n_trans), out))
        moments = out.sum(0).reshape(m, m + 1)
        return moments[:, :m], moments[:, m]

    def apply(self, r, al_j, lo=0.0, hi=1.0):
        """clip(r + sum_k al_k f_k, lo, hi), one pass over the frames."""
        return _update_kernel(r, self.r_hat, self.y_hat, self.env, cp.ascontiguousarray(al_j, dtype=cp.float64),
                              cp.int32(self.n), cp.float64(lo), cp.float64(hi))
