#include <iostream>
#include <vector>
#include <cmath>

static constexpr int H = 2;
static constexpr int D = 4;

void gdn_forward(
    const float* s_prev,
    const float* k,
    const float* v,
    const float* gate,
    const float* beta,
    float* s_new
) {
    for (int h = 0; h < H; h++) {
        float alpha = std::exp(gate[h]);
        float b = beta[h];

        const float* s = s_prev + h * D * D;
        float* out = s_new + h * D * D;
        const float* kk = k + h * D;
        const float* vv = v + h * D;

        for (int i = 0; i < D * D; i++) {
            out[i] = alpha * s[i];
        }

        std::vector<float> proj(D, 0);

        for (int i = 0; i < D; i++) {
            for (int j = 0; j < D; j++) {
                proj[i] += out[i * D + j] * kk[j];
            }
        }

        for (int i = 0; i < D; i++) {
            float factor = b * proj[i];
            for (int j = 0; j < D; j++) {
                out[i * D + j] -= factor * kk[j];
                out[i * D + j] += b * vv[i] * kk[j];
            }
        }
    }
}

int main() {
    float s_prev[H * D * D] = {0};
    float s_new[H * D * D] = {0};
    float k[H * D] = {1};
    float v[H * D] = {1};
    float gate[H] = {0.1, 0.2};
    float beta[H] = {0.5, 0.8};

    gdn_forward(s_prev, k, v, gate, beta, s_new);

    std::cout << "GDN kernel OK\n";
}
