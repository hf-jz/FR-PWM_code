// FR-PWM 2.1.3 Safety Kernel — C++ 实现（spec: 纯确定性 C/C++ 模块，P99<1ms）
// 离散时间 CBF-QP（CBF_mode=Discrete-time 冻结）：
//   min ½‖u−u_cand‖²  s.t.  G u ≤ h,  lo ≤ u ≤ hi
// 约束来自逐约束离散 CBF 一阶线性化：a_j·B·u ≤ −a_j·(Ax+c)+b_j−m_j−(1−γ)h_j(x_t)。
// 求解器：原始 active-set（H=I 最小距离规划），固定迭代上限 4×n_cons；
// 不可解/超上限 → ok=false（Python 侧回退 Safe Lookup O(1) 快速路径）。
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <cstring>

namespace py = pybind11;

static bool solve_dense(std::vector<double>& A, int n, std::vector<double>& b) {
    // 部分主元高斯消元（n≤64 小稠密），A 列主序被销毁
    for (int col = 0; col < n; ++col) {
        int piv = col;
        for (int r = col + 1; r < n; ++r)
            if (std::fabs(A[r * n + col]) > std::fabs(A[piv * n + col])) piv = r;
        if (std::fabs(A[piv * n + col]) < 1e-14) return false;
        if (piv != col) {
            for (int k = col; k < n; ++k) std::swap(A[piv * n + k], A[col * n + k]);
            std::swap(b[piv], b[col]);
        }
        for (int r = col + 1; r < n; ++r) {
            double f = A[r * n + col] / A[col * n + col];
            for (int k = col; k < n; ++k) A[r * n + k] -= f * A[col * n + k];
            b[r] -= f * b[col];
        }
    }
    for (int r = n - 1; r >= 0; --r) {
        for (int k = r + 1; k < n; ++k) b[r] -= A[r * n + k] * b[k];
        b[r] /= A[r * n + r];
    }
    return true;
}

// active-set QP：min ½uᵀu − cᵀu  s.t.  C u ≤ d（box 已并入 C,d）
// 原始可行法：每迭代解等式子问题得试步，朝试步走最长可行步（阻塞约束入集）；
// 试步为零时按 λ 符号出集；λ 全 ≥0 且可行 → 最优。
static bool qp_active_set(const std::vector<double>& c, int n,
                          const std::vector<double>& C, const std::vector<double>& d,
                          int m, std::vector<double>& u, int maxiter) {
    u = c;                                              // box 可行初值（box 行在 C 内）
    std::vector<int> W;
    for (int it = 0; it < maxiter; ++it) {
        std::vector<double> un = c, lam;
        int p = (int)W.size();
        if (p > 0) {
            // 解 KKT：u = c − C_Wᵀλ,  C_W u = d_W  →  C_W C_Wᵀ λ = C_W c − d_W
            std::vector<double> M(p * p, 0.0), rhs(p);
            for (int i = 0; i < p; ++i) {
                double s = -d[W[i]];
                for (int k = 0; k < n; ++k) s += C[W[i] * n + k] * c[k];
                rhs[i] = s;
                for (int j = 0; j < p; ++j) {
                    double q = 0.0;
                    for (int k = 0; k < n; ++k) q += C[W[i] * n + k] * C[W[j] * n + k];
                    M[i * p + j] = q + (i == j ? 1e-12 : 0.0);   // 微脊正则（重复行）
                }
            }
            lam = rhs;
            if (!solve_dense(M, p, lam)) return false;
            for (int i = 0; i < p; ++i)
                for (int k = 0; k < n; ++k) un[k] -= C[W[i] * n + k] * lam[i];
        }
        // 试步方向与零步判定
        double mv2 = 0.0;
        for (int k = 0; k < n; ++k) mv2 += (un[k] - u[k]) * (un[k] - u[k]);
        if (mv2 < 1e-20) {
            // 零步：先查全部约束；有违反 → 最违反者入集继续（u=c 不天然可行）
            int worst = -1; double worst_v = 1e-9;
            for (int j = 0; j < m; ++j) {
                double s = -d[j];
                for (int k = 0; k < n; ++k) s += C[j * n + k] * u[k];
                if (s > worst_v) { worst_v = s; worst = j; }
            }
            if (worst >= 0) {
                bool dup = false;
                for (int w : W) if (w == worst) { dup = true; break; }
                if (dup) return false;
                W.push_back(worst);
                continue;
            }
            if (p == 0) return true;                     // u=c 全可行 → 最优
            int drop = -1; double min_l = -1e-10;
            for (int i = 0; i < p; ++i)
                if (lam[i] < min_l) { min_l = lam[i]; drop = i; }
            if (drop < 0) {
                for (int j = 0; j < m; ++j) {            // 终检：全部约束满足才收
                    double s = -d[j];
                    for (int k = 0; k < n; ++k) s += C[j * n + k] * u[k];
                    if (s > 1e-7) return false;
                }
                return true;
            }
            W.erase(W.begin() + drop);
            continue;
        }
        // 最长可行步：α = min(1, min_j (d_j − C_j u)/(C_j·dir))，仅对非工作集约束
        // （工作集约束由 EQP 保证，脊正则的 1e-12 残差不参与阻塞判定）
        double alpha = 1.0; int block = -1;
        for (int j = 0; j < m; ++j) {
            bool in_w = false;
            for (int w : W) if (w == j) { in_w = true; break; }
            if (in_w) continue;
            double cd = 0.0, cu = -d[j];
            for (int k = 0; k < n; ++k) {
                cd += C[j * n + k] * (un[k] - u[k]);
                cu += C[j * n + k] * u[k];
            }
            if (cd > 1e-14) {
                double a = -cu / cd;
                if (a < alpha - 1e-14) { alpha = a; block = j; }
            }
        }
        if (alpha < -1e-9) return false;                 // 数值异常
        if (alpha < 0.0) alpha = 0.0;
        for (int k = 0; k < n; ++k) u[k] += alpha * (un[k] - u[k]);
        if (block >= 0) {
            bool dup = false;
            for (int w : W) if (w == block) { dup = true; break; }
            if (dup) return false;                       // 循环防御
            W.push_back(block);
        }
    }
    return false;
}

static py::tuple cbf_qp(py::array_t<double> u_cand, py::array_t<double> G,
                        py::array_t<double> h, py::array_t<double> lo,
                        py::array_t<double> hi, int maxiter) {
    auto uc = u_cand.unchecked<1>();
    int n = (int)uc.shape(0);
    auto Gm = G.unchecked<2>();
    auto hv = h.unchecked<1>();
    auto lo_v = lo.unchecked<1>(), hi_v = hi.unchecked<1>();
    int mg = (int)Gm.shape(0);
    int m = mg + 2 * n;                                  // box 并入约束矩阵
    std::vector<double> C(m * n), d(m), c(n);
    for (int j = 0; j < mg; ++j) {
        for (int k = 0; k < n; ++k) C[j * n + k] = Gm(j, k);
        d[j] = hv(j);
    }
    for (int i = 0; i < n; ++i) {
        c[i] = uc(i);
        for (int k = 0; k < n; ++k) {
            C[(mg + i) * n + k] = (k == i) ? 1.0 : 0.0;         // u ≤ hi
            C[(mg + n + i) * n + k] = (k == i) ? -1.0 : 0.0;    // −u ≤ −lo
        }
        d[mg + i] = hi_v(i);
        d[mg + n + i] = -lo_v(i);
        // u_cand 投影到 box（与 Python 侧 np.clip 一致）
        if (c[i] > hi_v(i)) c[i] = hi_v(i);
        if (c[i] < lo_v(i)) c[i] = lo_v(i);
    }
    std::vector<double> u;
    int cap = maxiter > 0 ? maxiter : 4 * m;
    bool ok = qp_active_set(c, n, C, d, m, u, cap);
    auto out = py::array_t<double>(n);
    auto o = out.mutable_unchecked<1>();
    if (ok) for (int i = 0; i < n; ++i) o(i) = u[i];
    else for (int i = 0; i < n; ++i) o(i) = c[i];
    return py::make_tuple(out, ok);
}

PYBIND11_MODULE(frpwm_cpp, m) {
    m.def("cbf_qp", &cbf_qp,
          "离散时间 CBF-QP（active-set，定迭代上限）→ (u, ok)；ok=false 回退 Safe Lookup");
}
