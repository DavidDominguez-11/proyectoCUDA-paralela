"""K01-K19, DIS-CUDA-001: FP32 NHWC, gather backward, explicit reductions.

Flat buffers have explicit active dimensions; capacity never defines batch size.
No floating point atomics or cross-block barriers. Partial gradients are unique.
"""
import math
from numba import float32
from .runtime import cuda


@cuda.jit(device=True, inline=True)
def _left(a, mode, m, k, H, W, Ci, Co, I, O):
    if mode == 0:  # virtual im2col
        n = m // (H * W)
        h = (m // W) % H + k // (3 * Ci) - 1
        w = m % W + (k // Ci) % 3 - 1
        if 0 <= h < H and 0 <= w < W:
            return a[((n * H + h) * W + w) * Ci + k % Ci]
        return float32(0)
    if mode == 1:  # gather dX, spatially reversed offsets
        n = m // (H * W)
        h = (m // W) % H - k // (3 * Co) + 1
        w = m % W - (k // Co) % 3 + 1
        if 0 <= h < H and 0 <= w < W:
            return a[((n * H + h) * W + w) * Co + k % Co]
        return float32(0)
    if mode == 2:  # im2col transpose without materializing it
        n = k // (H * W)
        h = (k // W) % H + m // (3 * Ci) - 1
        w = k % W + (m // Ci) % 3 - 1
        if 0 <= h < H and 0 <= w < W:
            return a[((n * H + h) * W + w) * Ci + m % Ci]
        return float32(0)
    if mode == 3:
        return a[m * I + k]
    if mode == 4:
        return a[m * O + k]
    return a[k * I + m]


@cuda.jit(device=True, inline=True)
def _right(v, mode, k, n, Ci, Co, I, O):
    if mode == 0 or mode == 2:
        return v[k * Co + n]
    if mode == 1:
        return v[((k // Co) * Ci + n) * Co + k % Co]
    if mode == 3 or mode == 5:
        return v[k * O + n]
    return v[n * O + k]


def _make_gemm(mode):
    @cuda.jit(cache=True, fastmath=False)
    def kernel(a, v, bias, out, M, N, K, H, W, Ci, Co, I, O, relu, m0):
        sa = cuda.shared.array((16, 16), float32)
        sb = cuda.shared.array((16, 16), float32)
        tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
        m, n = m0 + cuda.blockIdx.y * 16 + ty, cuda.blockIdx.x * 16 + tx
        begin, end = 0, K
        if mode == 2 or mode == 5:
            begin = cuda.blockIdx.z * 2048
            end = min(K, begin + 2048)
        acc = float32(0)
        for k0 in range(begin, end, 16):
            sa[ty, tx] = float32(0)
            sb[ty, tx] = float32(0)
            if m < M and k0 + tx < end:
                sa[ty, tx] = _left(a, mode, m, k0 + tx, H, W, Ci, Co, I, O)
            if n < N and k0 + ty < end:
                sb[ty, tx] = _right(v, mode, k0 + ty, n, Ci, Co, I, O)
            cuda.syncthreads()
            for kk in range(16):
                acc += sa[ty, kk] * sb[kk, tx]
            cuda.syncthreads()
        if m < M and n < N:
            index = m * N + n
            if mode == 2 or mode == 5:
                index += cuda.blockIdx.z * M * N
            if mode == 0 or mode == 3:
                acc += bias[n]
                if relu and acc < 0:
                    acc = float32(0)
            out[index] = acc
    return kernel


GEMM = tuple(_make_gemm(i) for i in range(6))


@cuda.jit(cache=True)
def merge_partials(parts, out, E, P):
    for i in range(cuda.grid(1), E, cuda.gridsize(1)):
        value = float32(0)
        for p in range(P):
            value += parts[p * E + i]
        out[i] = value


@cuda.jit(cache=True)
def copy_flat(x, y, E):
    for i in range(cuda.grid(1), E, cuda.gridsize(1)):
        y[i] = x[i]


@cuda.jit(cache=True)
def relu_backward(grad, activation, E):
    for i in range(cuda.grid(1), E, cuda.gridsize(1)):
        if activation[i] <= 0:
            grad[i] = float32(0)


@cuda.jit(cache=True)
def dropout(x, mask, out, factor, E):
    for i in range(cuda.grid(1), E, cuda.gridsize(1)):
        out[i] = x[i] * float32(mask[i]) * factor


@cuda.jit(cache=True)
def pool_forward(x, y, arg, b, H, W, C):
    E = b * (H // 2) * (W // 2) * C
    for i in range(cuda.grid(1), E, cuda.gridsize(1)):
        c = i % C
        w = (i // C) % (W // 2)
        h = (i // (C * (W // 2))) % (H // 2)
        n = i // (C * (W // 2) * (H // 2))
        base = ((n * H + 2 * h) * W + 2 * w) * C + c
        value, pos = x[base], 0
        for p in range(1, 4):
            candidate = x[base + (p // 2) * W * C + (p % 2) * C]
            if candidate > value:
                value, pos = candidate, p
        y[i], arg[i] = value, pos


@cuda.jit(cache=True)
def pool_backward(dy, arg, dx, b, H, W, C):
    for i in range(cuda.grid(1), b * H * W * C, cuda.gridsize(1)):
        c, w = i % C, (i // C) % W
        h, n = (i // (C * W)) % H, i // (C * W * H)
        out_i = ((n * (H // 2) + h // 2) * (W // 2) + w // 2) * C + c
        dx[i] = float32(0)
        if arg[out_i] == 2 * (h % 2) + w % 2:
            dx[i] = dy[out_i]


@cuda.jit(cache=True, fastmath=False)
def reduce_channels(x, dy, mean, inv, parts1, parts2, L, C, mode):
    # mode0 sum; mode1 centered variance; mode2 BN backward pair.
    s1 = cuda.shared.array((8, 32), float32)
    s2 = cuda.shared.array((8, 32), float32)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    c = cuda.blockIdx.y * 32 + tx
    v1, v2 = float32(0), float32(0)
    if c < C:
        for j in range(256):
            l = cuda.blockIdx.x * 2048 + ty + 8 * j
            if l < L:
                value = x[l * C + c]
                if mode == 0:
                    v1 += value
                elif mode == 1:
                    centered = value - mean[c]
                    v1 += centered * centered
                else:
                    g = dy[l * C + c]
                    v1 += g
                    v2 += g * ((value - mean[c]) * inv[c])
    s1[ty, tx], s2[ty, tx] = v1, v2
    cuda.syncthreads()
    stride = 4
    while stride > 0:
        if ty < stride:
            s1[ty, tx] += s1[ty + stride, tx]
            s2[ty, tx] += s2[ty + stride, tx]
        cuda.syncthreads()
        stride //= 2
    if ty == 0 and c < C:
        parts1[cuda.blockIdx.x * C + c] = s1[0, tx]
        parts2[cuda.blockIdx.x * C + c] = s2[0, tx]


@cuda.jit(cache=True, fastmath=False)
def finish_channels(p1, p2, out1, out2, movmean, movvar, mean, inv, P, C, L, mode):
    # mode0 mean; mode1 variance+moving state; mode2 dbeta/dgamma; mode3 bias.
    s1 = cuda.shared.array(256, float32)
    s2 = cuda.shared.array(256, float32)
    t, c = cuda.threadIdx.x, cuda.blockIdx.x
    s1[t], s2[t] = float32(0), float32(0)
    if t < P:
        s1[t], s2[t] = p1[t * C + c], p2[t * C + c]
    cuda.syncthreads()
    stride = 128
    while stride > 0:
        if t < stride:
            s1[t] += s1[t + stride]
            s2[t] += s2[t + stride]
        cuda.syncthreads()
        stride //= 2
    if t == 0:
        if mode == 0:
            out1[c] = s1[0] / float32(L)
        elif mode == 1:
            var = s1[0] / float32(L)
            inv[c] = float32(1) / math.sqrt(var + float32(.001))
            movmean[c] = float32(.9) * movmean[c] + float32(.1) * mean[c]
            movvar[c] = float32(.9) * movvar[c] + float32(.1) * var
        elif mode == 2:
            out1[c], out2[c] = s1[0], s2[0]
        else:
            out1[c] = s1[0]


@cuda.jit(cache=True, fastmath=False)
def bn_forward(x, y, gamma, beta, mean, inv, E, C):
    for i in range(cuda.grid(1), E, cuda.gridsize(1)):
        c = i % C
        y[i] = gamma[c] * ((x[i] - mean[c]) * inv[c]) + beta[c]


@cuda.jit(cache=True, fastmath=False)
def bn_backward(x, dy, dx, gamma, mean, inv, dbeta, dgamma, L, C):
    for i in range(cuda.grid(1), L * C, cuda.gridsize(1)):
        c = i % C
        xhat = (x[i] - mean[c]) * inv[c]
        dx[i] = (gamma[c] * inv[c] / float32(L)) * (
            float32(L) * dy[i] - dbeta[c] - xhat * dgamma[c])


@cuda.jit(cache=True, fastmath=False)
def softmax_loss(logits, labels, probs, dlogits, loss, correct, b):
    s = cuda.shared.array(32, float32)
    t, n = cuda.threadIdx.x, cuda.blockIdx.x
    z = float32(-math.inf)
    if t < 29:
        z = logits[n * 29 + t]
    s[t] = z
    cuda.syncthreads()
    stride = 16
    while stride > 0:
        if t < stride:
            s[t] = max(s[t], s[t + stride])
        cuda.syncthreads()
        stride //= 2
    maximum = s[0]
    cuda.syncthreads()
    value = float32(0)
    if t < 29:
        value = math.exp(z - maximum)
    s[t] = value
    cuda.syncthreads()
    stride = 16
    while stride > 0:
        if t < stride:
            s[t] += s[t + stride]
        cuda.syncthreads()
        stride //= 2
    denominator = s[0]
    label = labels[n]
    if t < 29:
        probability = value / denominator
        probs[n * 29 + t] = probability
        dlogits[n * 29 + t] = (probability - float32(t == label)) / float32(b)
    if t == 0:
        loss[n] = (maximum - logits[n * 29 + label]) + math.log(denominator)
        winner = 0
        for c in range(1, 29):
            if logits[n * 29 + c] > logits[n * 29 + winner]:
                winner = c
        correct[n] = winner == label


@cuda.jit(cache=True, fastmath=False)
def adam(theta, grad, moment, velocity, alpha, E, error):
    for i in range(cuda.grid(1), E, cuda.gridsize(1)):
        g = grad[i]
        if not math.isfinite(g):
            cuda.atomic.max(error, 0, 1)
        if error[0] == 0:
            m = moment[i] + (g - moment[i]) * float32(.1)
            v = velocity[i] + (g * g - velocity[i]) * float32(.001)
            value = theta[i] - (m * alpha) / (math.sqrt(v) + float32(1e-7))
            if not math.isfinite(value):
                cuda.atomic.max(error, 0, 2)
            theta[i], moment[i], velocity[i] = value, m, v


@cuda.jit(cache=True)
def finite_check(x, E, error):
    for i in range(cuda.grid(1), E, cuda.gridsize(1)):
        if not math.isfinite(x[i]):
            cuda.atomic.max(error, 0, 3)


@cuda.jit(cache=True)
def metrics(loss, correct, b, total):
    s = cuda.shared.array(256, float32)
    q = cuda.shared.array(256, float32)
    t = cuda.threadIdx.x
    s[t], q[t] = float32(0), float32(0)
    if t < b:
        s[t], q[t] = loss[t], float32(correct[t])
    cuda.syncthreads()
    stride = 128
    while stride > 0:
        if t < stride:
            s[t] += s[t + stride]
            q[t] += q[t + stride]
        cuda.syncthreads()
        stride //= 2
    if t == 0:
        total[0] += s[0]
        total[1] += q[0]
        total[2] += float32(b)


@cuda.jit(cache=True)
def advance_step(counter):
    if cuda.grid(1) == 0:
        counter[0] += 1


@cuda.jit(cache=True)
def im2col(x, cols, R, H, W, Ci):
    K = 9 * Ci
    for i in range(cuda.grid(1), R * K, cuda.gridsize(1)):
        r, kk = i // K, i % K
        n = r // (H * W)
        h = (r // W) % H + kk // (3 * Ci) - 1
        w = r % W + (kk // Ci) % 3 - 1
        value = float32(0)
        if 0 <= h < H and 0 <= w < W:
            value = x[((n * H + h) * W + w) * Ci + kk % Ci]
        cols[i] = value


@cuda.jit(cache=True)
def col2im_gather(cols, dx, R, H, W, Ci):
    for i in range(cuda.grid(1), R * Ci, cuda.gridsize(1)):
        c = i % Ci
        w, h = (i // Ci) % W, (i // (Ci * W)) % H
        n = i // (Ci * W * H)
        value = float32(0)
        for kh in range(3):
            for kw in range(3):
                ho, wo = h - kh + 1, w - kw + 1
                if 0 <= ho < H and 0 <= wo < W:
                    row = (n * H + ho) * W + wo
                    value += cols[row * 9 * Ci + (kh * 3 + kw) * Ci + c]
        dx[i] = value


@cuda.jit(cache=True)
def bias_activation(y, bias, E, C, relu):
    for i in range(cuda.grid(1), E, cuda.gridsize(1)):
        value = y[i] + bias[i % C]
        if relu and value < 0:
            value = float32(0)
        y[i] = value
