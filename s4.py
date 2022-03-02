import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pykeops.torch import Genred


class S4(nn.Module):
    def __init__(self, hidden_size, state_size, max_len):
        super(S4, self).__init__()
        self.max_len = max_len
        self.s4_kernel = HippoKernel(hidden_size, state_size, max_len)
        self.s4_kernel_cache = None
        self.s4_kernel_cache_training = False
        self.D = nn.Parameter(torch.randn(hidden_size))

    @staticmethod
    def conv(k, x):
        L = k.shape[-1] * 2
        k_f = torch.fft.rfft(k, n=L)
        x_f = torch.fft.rfft(x, n=L)
        y_f = k_f * x_f
        y = torch.fft.irfft(y_f, n=L)
        y = y[..., :x.shape[-1]]
        return y

    def forward(self, x):
        assert x.shape[1] <= self.max_len
        x = x.transpose(1, 2) # [B, H, L]
        if self.s4_kernel_cache is None or self.training != self.s4_kernel_cache_training:
            self.s4_kernel_cache = self.s4_kernel() # [H, L]
            self.s4_kernel_cache_training = self.training
        k = self.s4_kernel_cache
        y = S4.conv(k.unsqueeze(0), x) # [B, H, L]
        d = x * self.D.unsqueeze(0).unsqueeze(-1)
        y = y + d
        y = y.transpose(1, 2) # [B, L, H]
        return y

    def reset_s4(self):
        self.s4_kernel_cache = None


class BiS4(nn.Module):
    def __init__(self, hidden_size, state_size, max_len, drop_prob):
        super(BiS4, self).__init__()
        self.ltr_in = nn.Linear(hidden_size, hidden_size)
        self.rtl_in = nn.Linear(hidden_size, hidden_size)
        self.ltr = S4(hidden_size, state_size, max_len)
        self.rtl = S4(hidden_size, state_size, max_len)
        self.dropout = nn.Dropout(drop_prob)
        self.out = nn.Linear(2*hidden_size, hidden_size)

    def forward(self, x, mask):
        x = x * mask.unsqueeze(-1)
        ya = self.ltr(self.ltr_in(x))
        yb = self.rtl((self.rtl_in(x)).flip(1)).flip(1)
        y = torch.cat((ya, yb), dim=-1)
        y = F.gelu(y)
        y = self.dropout(y)
        y = self.out(y)
        return y

    def reset_s4(self):
        self.ltr.reset_s4()
        self.rtl.reset_s4()


class HippoKernel(nn.Module):
    def __init__(self, hidden_size, state_size, max_len):
        super(HippoKernel, self).__init__()
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.max_len = max_len

        w, p, q, V, B, A = HippoKernel.nplr(state_size)
        B = B.unsqueeze(0).repeat((hidden_size, 1))
        C = torch.randn(hidden_size, state_size).to(torch.cfloat)
        C = torch.einsum("hn,nm->hm", C, V.conj())
        log_dt = math.log(0.001) + (math.log(0.1) - math.log(0.001)) * torch.rand(hidden_size)

        dt = torch.exp(log_dt).unsqueeze(-1).unsqueeze(-1)
        I = torch.eye(state_size, dtype=torch.cfloat).unsqueeze(0)
        dA = torch.linalg.inv(I - dt * A / 2) @ (I + dt * A / 2)
        dA_power = I - torch.linalg.matrix_power(dA, max_len)
        C = torch.cat((C, C.conj()), dim=-1)
        C = torch.einsum("hmn,hm->hn", dA_power.conj(), C)
        C = C[..., :C.shape[-1] // 2]

        B = B.unsqueeze(1)
        C = C.unsqueeze(1)
        p = p.unsqueeze(0).unsqueeze(1).repeat((hidden_size, 1, 1))
        q = q.unsqueeze(0).unsqueeze(1).repeat((hidden_size, 1, 1))
        B = torch.cat((B, p), dim=-2)
        C = torch.cat((C, q), dim=-2)

        self.log_dt = nn.Parameter(log_dt)
        # self.register_buffer("log_dt", log_dt)
        self.w = nn.Parameter(torch.view_as_real(w)) # [N]
        self.B = nn.Parameter(torch.view_as_real(B)) # [H, 2, N]
        self.C = nn.Parameter(torch.view_as_real(C)) # [H, 2, N]
        self.register_buffer("freq", torch.exp(-2j*torch.pi/max_len*torch.arange(max_len//2+1)))
        self.register_buffer("z", 2*(1-self.freq)/(1+self.freq))

    @staticmethod
    def nplr(state_size):
        i = torch.arange(state_size)
        j = (2*i+1).sqrt()
        A = torch.tril(j.unsqueeze(0) * j.unsqueeze(1)) - torch.diag(i)
        A = -A
        B = j
        p = j / math.sqrt(2)
        w, V = torch.linalg.eig(A + p.unsqueeze(0)*p.unsqueeze(1))
        w, V = w[::2].contiguous(), V[:, ::2].contiguous()
        B = V.conj().T @ B.to(V.dtype)
        p = V.conj().T @ p.to(V.dtype)
        V_full = torch.cat((V, V.conj()), dim=-1)
        VtAV = V_full.conj().T @ A.to(torch.cfloat) @ V_full
        return w, p, p, V, B, VtAV

    @staticmethod
    def cauchy_mult_slow(v, z, w):
        assert v.shape == w.shape
        r = (v.unsqueeze(-1) / (z - w.unsqueeze(-1))).sum(dim=-2)
        r += (v.conj().unsqueeze(-1) / (z - w.conj().unsqueeze(-1))).sum(dim=-2)
        return r

    @staticmethod
    def cauchy_mult(v, z, w):
        assert v.shape == w.shape
        v = v.contiguous()
        w = w.contiguous()
        z = z.view(1, 1, 1, -1).contiguous()
        operation = Genred(
            "ComplexDivide(z * ComplexReal(v) - Real2Complex(Sum(v * w)), ComplexMult(z-w, z-Conj(w)))",
            [
                "v = Vj(2)",
                "z = Vi(2)",
                "w = Vj(2)",
            ],
            reduction_op="Sum",
            axis=1,
            dtype="float32"
        )
        r = operation(
            torch.view_as_real(v),
            torch.view_as_real(z),
            torch.view_as_real(w),
            backend="auto",
        )
        r = torch.view_as_complex(2*r)
        return r

    def forward(self):
        dt = torch.exp(self.log_dt)
        w = torch.view_as_complex(self.w)
        B = torch.view_as_complex(self.B)
        C = torch.view_as_complex(self.C)

        v = C.unsqueeze(-2).conj() * B.unsqueeze(-3) # [H, 2, 2, N]
        w = (w.unsqueeze(0) * dt.unsqueeze(1)).unsqueeze(-2).unsqueeze(-2).expand(v.shape) # [H, 2, 2, N]
        r = HippoKernel.cauchy_mult(v, self.z, w) # [H, 2, 2, L]
        r *= dt.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        k_f = r[:, 0, 0, :] - r[:, 0, 1, :] * r[:, 1, 0, :] / (1 + r[:, 1, 1, :])
        k_f = k_f * 2 / (1 + self.freq)
        k = torch.fft.irfft(k_f)
        return k

    def forward_slow(self):
        dt = torch.exp(self.log_dt).unsqueeze(-1).unsqueeze(-1) # [H, 1, 1]
        w = torch.view_as_complex(self.w)
        B = torch.view_as_complex(self.B)
        C = torch.view_as_complex(self.C)
        w = torch.cat((w, w.conj()), dim=-1) # [N]
        B = torch.cat((B, B.conj()), dim=-1)
        C = torch.cat((C, C.conj()), dim=-1)
        B, p = B[:, 0], B[:, 1] # [H, N]
        C, q = C[:, 0], C[:, 1] # [H, N]
        A = torch.diag(w).unsqueeze(0) - p.unsqueeze(2) * q.conj().unsqueeze(1)
        I = torch.eye(self.state_size, dtype=torch.cfloat).unsqueeze(0)
        dA = torch.linalg.inv(I - dt * A / 2) @ (I + dt * A / 2)
        dB = (torch.linalg.inv(I - dt * A / 2) @ B.unsqueeze(-1)).squeeze(-1)
        dB *= dt.squeeze(-1)
        dA_power = I - torch.linalg.matrix_power(dA, self.max_len)
        C = torch.einsum("hmn,hm->hn", torch.linalg.inv(dA_power.conj()), C)
        k = []
        power = torch.eye(self.state_size, dtype=torch.cfloat).unsqueeze(0)
        for i in range(self.max_len):
            k_i = torch.einsum("hm,hmn,hn->h", C.conj(), power, dB)
            k.append(k_i)
            power = power @ dA
        k = torch.stack(k, dim=-1)
        return k


def main():
    L = 500
    k = torch.rand(L)
    x = torch.rand(L)
    y = torch.zeros(L)
    for i in range(L):
        for j in range(i+1):
            y[i] += x[j] * k[i-j]
    v = S4.conv(k, x)
    print((v-y).square().mean(), y.square().mean())

    w, p, q, V, B, VtAV = HippoKernel.nplr(4)
    w = torch.cat((w, w.conj()), dim=-1)
    p = torch.cat((p, p.conj()), dim=-1)
    q = torch.cat((q, q.conj()), dim=-1)
    V = torch.cat((V, V.conj()), dim=-1)
    B = torch.cat((B, B.conj()), dim=-1)
    A = torch.diag(w) - p.unsqueeze(1) * q.conj().unsqueeze(0)
    print((A - VtAV).norm())
    A = V @ A @ V.conj().T
    B = V @ B
    print(A.real)
    print(B.real)

    hippo = HippoKernel(128, 64, 500)
    k = hippo()
    k_r = hippo.forward_slow()
    print(k)
    print("error", (k_r - k).norm() / k_r.norm())
    print(torch.std_mean(k))
    s4 = S4(128, 64, 500)
    x = torch.randn(4, 500, 128)
    y = s4(x)
    print(torch.std_mean(x))
    print(torch.std_mean(y))

if __name__ == "__main__":
    main()
