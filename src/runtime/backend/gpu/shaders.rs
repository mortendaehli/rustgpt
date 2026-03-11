// The two shaders below intentionally mirror the CPU helpers in tensor.rs.
// `matvec` computes y = x @ W^T and `matvec_t` computes y = x @ W.
pub(super) const MATVEC_SHADER: &str = r#"
struct Params {
    rows: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0)
var<storage, read> x: array<f32>;

@group(0) @binding(1)
var<storage, read> w: array<f32>;

@group(0) @binding(2)
var<storage, read_write> y: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= params.rows) {
        return;
    }

    var acc = 0.0;
    let base = row * params.cols;
    for (var col = 0u; col < params.cols; col = col + 1u) {
        acc = acc + x[col] * w[base + col];
    }
    y[row] = acc;
}
"#;

pub(super) const MATVEC_TRANSPOSED_SHADER: &str = r#"
struct Params {
    rows: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0)
var<storage, read> x: array<f32>;

@group(0) @binding(1)
var<storage, read> w: array<f32>;

@group(0) @binding(2)
var<storage, read_write> y: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    if (col >= params.cols) {
        return;
    }

    var acc = 0.0;
    for (var row = 0u; row < params.rows; row = row + 1u) {
        acc = acc + x[row] * w[row * params.cols + col];
    }
    y[col] = acc;
}
"#;

pub(super) const MATMUL_ROWS_SHADER: &str = r#"
struct Params {
    batch: u32,
    out_cols: u32,
    in_cols: u32,
    _pad0: u32,
}

@group(0) @binding(0)
var<storage, read> x: array<f32>;

@group(0) @binding(1)
var<storage, read> w: array<f32>;

@group(0) @binding(2)
var<storage, read_write> y: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_col = gid.x;
    let batch_row = gid.y;
    if (batch_row >= params.batch || out_col >= params.out_cols) {
        return;
    }

    var acc = 0.0;
    let x_base = batch_row * params.in_cols;
    let w_base = out_col * params.in_cols;
    for (var i = 0u; i < params.in_cols; i = i + 1u) {
        acc = acc + x[x_base + i] * w[w_base + i];
    }
    y[batch_row * params.out_cols + out_col] = acc;
}
"#;

pub(super) const MATMUL_ROWS_TRANSPOSED_SHADER: &str = r#"
struct Params {
    batch: u32,
    w_rows: u32,
    w_cols: u32,
    _pad0: u32,
}

@group(0) @binding(0)
var<storage, read> x: array<f32>;

@group(0) @binding(1)
var<storage, read> w: array<f32>;

@group(0) @binding(2)
var<storage, read_write> y: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_col = gid.x;
    let batch_row = gid.y;
    if (batch_row >= params.batch || out_col >= params.w_cols) {
        return;
    }

    var acc = 0.0;
    let x_base = batch_row * params.w_rows;
    for (var row = 0u; row < params.w_rows; row = row + 1u) {
        acc = acc + x[x_base + row] * w[row * params.w_cols + out_col];
    }
    y[batch_row * params.w_cols + out_col] = acc;
}
"#;

pub(super) const RMSNORM_ROWS_SHADER: &str = r#"
struct Params {
    rows: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0)
var<storage, read> x: array<f32>;

@group(0) @binding(1)
var<storage, read_write> y: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= params.rows) {
        return;
    }

    let base = row * params.cols;
    var mean_square = 0.0;
    for (var col = 0u; col < params.cols; col = col + 1u) {
        let value = x[base + col];
        mean_square = mean_square + value * value;
    }
    mean_square = mean_square / f32(params.cols);
    let rms_inv = inverseSqrt(mean_square + 1e-5);
    for (var col = 0u; col < params.cols; col = col + 1u) {
        y[base + col] = x[base + col] * rms_inv;
    }
}
"#;

pub(super) const RMSNORM_BACKWARD_ROWS_SHADER: &str = r#"
struct Params {
    rows: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0)
var<storage, read> x: array<f32>;

@group(0) @binding(1)
var<storage, read> dy: array<f32>;

@group(0) @binding(2)
var<storage, read_write> dx: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= params.rows) {
        return;
    }

    let base = row * params.cols;
    var mean_square = 0.0;
    for (var col = 0u; col < params.cols; col = col + 1u) {
        let value = x[base + col];
        mean_square = mean_square + value * value;
    }
    mean_square = mean_square / f32(params.cols);
    let rms_inv = inverseSqrt(mean_square + 1e-5);

    var dot = 0.0;
    for (var col = 0u; col < params.cols; col = col + 1u) {
        dot = dot + dy[base + col] * x[base + col];
    }
    let scale = dot * rms_inv * rms_inv * rms_inv / f32(params.cols);
    for (var col = 0u; col < params.cols; col = col + 1u) {
        dx[base + col] = dy[base + col] * rms_inv - x[base + col] * scale;
    }
}
"#;

pub(super) const ADAM_SHADER: &str = r#"
struct AdamParams {
    len: u32,
    step: u32,
    offset: u32,
    _pad1: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    _pad2: vec3<f32>,
}

@group(0) @binding(0)
var<storage, read_write> weight: array<f32>;

@group(0) @binding(1)
var<storage, read_write> m: array<f32>;

@group(0) @binding(2)
var<storage, read_write> v: array<f32>;

@group(0) @binding(3)
var<storage, read_write> grad: array<f32>;

@group(0) @binding(4)
var<uniform> params: AdamParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = params.offset + gid.x;
    if (idx >= params.len) {
        return;
    }

    let g = grad[idx];
    let beta1 = params.beta1;
    let beta2 = params.beta2;
    let m_new = beta1 * m[idx] + (1.0 - beta1) * g;
    let v_new = beta2 * v[idx] + (1.0 - beta2) * g * g;
    m[idx] = m_new;
    v[idx] = v_new;

    let step = f32(params.step);
    let m_hat = m_new / (1.0 - pow(beta1, step));
    let v_hat = v_new / (1.0 - pow(beta2, step));
    if (params.weight_decay > 0.0) {
        weight[idx] = weight[idx] - params.lr * params.weight_decay * weight[idx];
    }
    weight[idx] = weight[idx] - params.lr * m_hat / (sqrt(v_hat) + params.eps);
    grad[idx] = 0.0;
}
"#;

pub(super) const OUTER_PRODUCT_ACCUM_SHADER: &str = r#"
struct Params {
    rows: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0)
var<storage, read> x: array<f32>;

@group(0) @binding(1)
var<storage, read> dout: array<f32>;

@group(0) @binding(2)
var<storage, read_write> grad: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if (row >= params.rows || col >= params.cols) {
        return;
    }

    let idx = row * params.cols + col;
    grad[idx] = grad[idx] + dout[row] * x[col];
}
"#;

pub(super) const OUTER_PRODUCT_ROWS_ACCUM_SHADER: &str = r#"
struct Params {
    batch: u32,
    out_rows: u32,
    in_cols: u32,
    _pad0: u32,
}

@group(0) @binding(0)
var<storage, read> x: array<f32>;

@group(0) @binding(1)
var<storage, read> dout: array<f32>;

@group(0) @binding(2)
var<storage, read_write> grad: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let in_col = gid.x;
    let out_row = gid.y;
    if (out_row >= params.out_rows || in_col >= params.in_cols) {
        return;
    }

    var acc = 0.0;
    for (var batch_row = 0u; batch_row < params.batch; batch_row = batch_row + 1u) {
        acc = acc + dout[batch_row * params.out_rows + out_row]
            * x[batch_row * params.in_cols + in_col];
    }
    let idx = out_row * params.in_cols + in_col;
    grad[idx] = grad[idx] + acc;
}
"#;

pub(super) const ROW_ADD_ACCUM_SHADER: &str = r#"
struct Params {
    row: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0)
var<storage, read> x: array<f32>;

@group(0) @binding(1)
var<storage, read_write> grad: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    if (col >= params.cols) {
        return;
    }
    let idx = params.row * params.cols + col;
    grad[idx] = grad[idx] + x[col];
}
"#;

pub(super) const GATHER_ROW_SHADER: &str = r#"
struct Params {
    row: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0)
var<storage, read> matrix: array<f32>;

@group(0) @binding(1)
var<storage, read_write> y: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    if (col >= params.cols) {
        return;
    }
    y[col] = matrix[params.row * params.cols + col];
}
"#;

pub(super) const ADD_SHADER: &str = r#"
struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0)
var<storage, read> a: array<f32>;

@group(0) @binding(1)
var<storage, read> b: array<f32>;

@group(0) @binding(2)
var<storage, read_write> y: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    y[idx] = a[idx] + b[idx];
}
"#;

pub(super) const RELU_SHADER: &str = r#"
struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0)
var<storage, read> x: array<f32>;

@group(0) @binding(1)
var<storage, read_write> y: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    y[idx] = max(x[idx], 0.0);
}
"#;

pub(super) const RELU_BACKWARD_SHADER: &str = r#"
struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0)
var<storage, read> pre: array<f32>;

@group(0) @binding(1)
var<storage, read> dout: array<f32>;

@group(0) @binding(2)
var<storage, read_write> y: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    y[idx] = select(0.0, dout[idx], pre[idx] > 0.0);
}
"#;

pub(super) const GELU_SHADER: &str = r#"
struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0)
var<storage, read> x: array<f32>;

@group(0) @binding(1)
var<storage, read_write> y: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

fn gelu_approx(value: f32) -> f32 {
    let cubic = value * value * value;
    let inner = 0.7978846 * (value + 0.044715 * cubic);
    return 0.5 * value * (1.0 + tanh(inner));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    y[idx] = gelu_approx(x[idx]);
}
"#;

pub(super) const GELU_BACKWARD_SHADER: &str = r#"
struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0)
var<storage, read> pre: array<f32>;

@group(0) @binding(1)
var<storage, read> dout: array<f32>;

@group(0) @binding(2)
var<storage, read_write> y: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

fn gelu_derivative(value: f32) -> f32 {
    let cubic = value * value * value;
    let inner = 0.7978846 * (value + 0.044715 * cubic);
    let tanh_inner = tanh(inner);
    let sech2 = 1.0 - tanh_inner * tanh_inner;
    let inner_prime = 0.7978846 * (1.0 + 3.0 * 0.044715 * value * value);
    return 0.5 * (1.0 + tanh_inner) + 0.5 * value * sech2 * inner_prime;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    y[idx] = dout[idx] * gelu_derivative(pre[idx]);
}
"#;

pub(super) const SILU_SHADER: &str = r#"
struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0)
var<storage, read> x: array<f32>;

@group(0) @binding(1)
var<storage, read_write> y: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

fn silu(value: f32) -> f32 {
    return value / (1.0 + exp(-value));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    y[idx] = silu(x[idx]);
}
"#;

pub(super) const SILU_BACKWARD_SHADER: &str = r#"
struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0)
var<storage, read> pre: array<f32>;

@group(0) @binding(1)
var<storage, read> dout: array<f32>;

@group(0) @binding(2)
var<storage, read_write> y: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

fn silu_derivative(value: f32) -> f32 {
    let sigmoid = 1.0 / (1.0 + exp(-value));
    return sigmoid * (1.0 + value * (1.0 - sigmoid));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    y[idx] = dout[idx] * silu_derivative(pre[idx]);
}
"#;

pub(super) const MUL_SHADER: &str = r#"
struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0)
var<storage, read> left: array<f32>;

@group(0) @binding(1)
var<storage, read> right: array<f32>;

@group(0) @binding(2)
var<storage, read_write> y: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    y[idx] = left[idx] * right[idx];
}
"#;

pub(super) const EXPAND_GQA_ROWS_SHADER: &str = r#"
struct Params {
    rows: u32,
    n_head: u32,
    n_kv_head: u32,
    head_dim: u32,
}

@group(0) @binding(0)
var<storage, read> x: array<f32>;

@group(0) @binding(1)
var<storage, read_write> y: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let expanded_cols = params.n_head * params.head_dim;
    let total = params.rows * expanded_cols;
    let idx = gid.x;
    if (idx >= total) {
        return;
    }

    let row = idx / expanded_cols;
    let col = idx % expanded_cols;
    let head = col / params.head_dim;
    let feature = col % params.head_dim;
    let query_heads_per_kv = params.n_head / params.n_kv_head;
    let kv_head = head / query_heads_per_kv;
    let compact_cols = params.n_kv_head * params.head_dim;
    let src_idx = row * compact_cols + kv_head * params.head_dim + feature;
    y[idx] = x[src_idx];
}
"#;

pub(super) const COLLAPSE_GQA_ROWS_SHADER: &str = r#"
struct Params {
    rows: u32,
    n_head: u32,
    n_kv_head: u32,
    head_dim: u32,
}

@group(0) @binding(0)
var<storage, read> x: array<f32>;

@group(0) @binding(1)
var<storage, read_write> y: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let compact_cols = params.n_kv_head * params.head_dim;
    let total = params.rows * compact_cols;
    let idx = gid.x;
    if (idx >= total) {
        return;
    }

    let row = idx / compact_cols;
    let col = idx % compact_cols;
    let kv_head = col / params.head_dim;
    let feature = col % params.head_dim;
    let query_heads_per_kv = params.n_head / params.n_kv_head;
    let expanded_cols = params.n_head * params.head_dim;

    var acc = 0.0;
    for (var offset = 0u; offset < query_heads_per_kv; offset = offset + 1u) {
        let head = kv_head * query_heads_per_kv + offset;
        let src_idx = row * expanded_cols + head * params.head_dim + feature;
        acc = acc + x[src_idx];
    }
    y[idx] = acc;
}
"#;

pub(super) const ROPE_SHADER: &str = r#"
struct Params {
    len: u32,
    pos: u32,
    head_dim: u32,
    inverse: u32,
}

@group(0) @binding(0)
var<storage, read> x: array<f32>;

@group(0) @binding(1)
var<storage, read_write> y: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

fn rope_value(current: f32, partner: f32, pair_idx: u32, half: u32, first_half: bool) -> f32 {
    let theta = f32(params.pos) / pow(10000.0, f32(pair_idx) / f32(half));
    let cos_theta = cos(theta);
    let sin_theta = sin(theta);
    if (params.inverse == 0u) {
        if (first_half) {
            return current * cos_theta - partner * sin_theta;
        }
        return current * cos_theta + partner * sin_theta;
    }
    if (first_half) {
        return current * cos_theta + partner * sin_theta;
    }
    return current * cos_theta - partner * sin_theta;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.len) {
        return;
    }
    let head_dim = params.head_dim;
    let half = head_dim / 2u;
    let local = idx % head_dim;
    let first_half = local < half;
    let pair_idx = select(local - half, local, first_half);
    let base = idx - local;
    let partner_idx = select(base + pair_idx, base + half + pair_idx, first_half);
    y[idx] = rope_value(x[idx], x[partner_idx], pair_idx, half, first_half);
}
"#;

pub(super) const ROPE_ROWS_SHADER: &str = r#"
struct Params {
    rows: u32,
    cols: u32,
    seq_len: u32,
    head_dim: u32,
}

@group(0) @binding(0)
var<storage, read> x: array<f32>;

@group(0) @binding(1)
var<storage, read_write> y: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

fn rope_value(current: f32, partner: f32, pos: u32, pair_idx: u32, half: u32) -> f32 {
    let theta = f32(pos) / pow(10000.0, f32(pair_idx) / f32(half));
    let cos_theta = cos(theta);
    let sin_theta = sin(theta);
    return current * cos_theta - partner * sin_theta;
}

fn rope_value_second(current: f32, partner: f32, pos: u32, pair_idx: u32, half: u32) -> f32 {
    let theta = f32(pos) / pow(10000.0, f32(pair_idx) / f32(half));
    let cos_theta = cos(theta);
    let sin_theta = sin(theta);
    return current * cos_theta + partner * sin_theta;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.rows * params.cols;
    if (idx >= total) {
        return;
    }
    let row = idx / params.cols;
    let col = idx % params.cols;
    let pos = row % params.seq_len;
    let head_dim = params.head_dim;
    let half = head_dim / 2u;
    let local = col % head_dim;
    let first_half = local < half;
    let pair_idx = select(local - half, local, first_half);
    let head_base = idx - local;
    let partner_idx = select(head_base + pair_idx, head_base + half + pair_idx, first_half);
    if (first_half) {
        y[idx] = rope_value(x[idx], x[partner_idx], pos, pair_idx, half);
    } else {
        y[idx] = rope_value_second(x[idx], x[partner_idx], pos, pair_idx, half);
    }
}
"#;

pub(super) const ROPE_ROWS_BACKWARD_SHADER: &str = r#"
struct Params {
    rows: u32,
    cols: u32,
    seq_len: u32,
    head_dim: u32,
}

@group(0) @binding(0)
var<storage, read> x: array<f32>;

@group(0) @binding(1)
var<storage, read_write> y: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

fn rope_value(current: f32, partner: f32, pos: u32, pair_idx: u32, half: u32) -> f32 {
    let theta = f32(pos) / pow(10000.0, f32(pair_idx) / f32(half));
    let cos_theta = cos(theta);
    let sin_theta = sin(theta);
    return current * cos_theta + partner * sin_theta;
}

fn rope_value_second(current: f32, partner: f32, pos: u32, pair_idx: u32, half: u32) -> f32 {
    let theta = f32(pos) / pow(10000.0, f32(pair_idx) / f32(half));
    let cos_theta = cos(theta);
    let sin_theta = sin(theta);
    return current * cos_theta - partner * sin_theta;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.rows * params.cols;
    if (idx >= total) {
        return;
    }
    let row = idx / params.cols;
    let col = idx % params.cols;
    let pos = row % params.seq_len;
    let head_dim = params.head_dim;
    let half = head_dim / 2u;
    let local = col % head_dim;
    let first_half = local < half;
    let pair_idx = select(local - half, local, first_half);
    let head_base = idx - local;
    let partner_idx = select(head_base + pair_idx, head_base + half + pair_idx, first_half);
    if (first_half) {
        y[idx] = rope_value(x[idx], x[partner_idx], pos, pair_idx, half);
    } else {
        y[idx] = rope_value_second(x[idx], x[partner_idx], pos, pair_idx, half);
    }
}
"#;

pub(super) const RMSNORM_SHADER: &str = r#"
struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0)
var<storage, read> x: array<f32>;

@group(0) @binding(1)
var<storage, read_write> y: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x > 0u) {
        return;
    }

    var mean_square = 0.0;
    for (var idx = 0u; idx < params.len; idx = idx + 1u) {
        mean_square = mean_square + x[idx] * x[idx];
    }
    mean_square = mean_square / f32(params.len);
    let rms_inv = inverseSqrt(mean_square + 1e-5);
    for (var idx = 0u; idx < params.len; idx = idx + 1u) {
        y[idx] = x[idx] * rms_inv;
    }
}
"#;

pub(super) const RMSNORM_BACKWARD_SHADER: &str = r#"
struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    rms_inv: f32,
    _pad4: f32,
    _pad5: f32,
    _pad6: f32,
}

@group(0) @binding(0)
var<storage, read> x: array<f32>;

@group(0) @binding(1)
var<storage, read> dy: array<f32>;

@group(0) @binding(2)
var<storage, read_write> dx: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x > 0u) {
        return;
    }

    var mean_square = 0.0;
    for (var idx = 0u; idx < params.len; idx = idx + 1u) {
        mean_square = mean_square + x[idx] * x[idx];
    }
    mean_square = mean_square / f32(params.len);
    let rms_inv = inverseSqrt(mean_square + 1e-5);

    var dot = 0.0;
    for (var idx = 0u; idx < params.len; idx = idx + 1u) {
        dot = dot + dy[idx] * x[idx];
    }
    let scale = dot * rms_inv * rms_inv * rms_inv / f32(params.len);
    for (var idx = 0u; idx < params.len; idx = idx + 1u) {
        dx[idx] = dy[idx] * rms_inv - x[idx] * scale;
    }
}
"#;

pub(super) const SOFTMAX_ROWS_SHADER: &str = r#"
struct Params {
    rows: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0)
var<storage, read> x: array<f32>;

@group(0) @binding(1)
var<storage, read_write> y: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= params.rows) {
        return;
    }

    let base = row * params.cols;
    var max_value = x[base];
    for (var col = 1u; col < params.cols; col = col + 1u) {
        max_value = max(max_value, x[base + col]);
    }

    var total = 0.0;
    for (var col = 0u; col < params.cols; col = col + 1u) {
        let value = exp(x[base + col] - max_value);
        y[base + col] = value;
        total = total + value;
    }

    for (var col = 0u; col < params.cols; col = col + 1u) {
        y[base + col] = y[base + col] / total;
    }
}
"#;

pub(super) const ATTN_SCORES_SHADER: &str = r#"
struct Params {
    time_len: u32,
    n_head: u32,
    head_dim: u32,
    _pad0: u32,
}

@group(0) @binding(0)
var<storage, read> q: array<f32>;

@group(0) @binding(1)
var<storage, read> keys: array<f32>;

@group(0) @binding(2)
var<storage, read_write> scores: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let head = gid.x;
    if (head >= params.n_head) {
        return;
    }

    let head_offset = head * params.head_dim;
    let scale = sqrt(f32(params.head_dim));
    for (var t = 0u; t < params.time_len; t = t + 1u) {
        let key_base = t * params.n_head * params.head_dim + head_offset;
        var dot = 0.0;
        for (var i = 0u; i < params.head_dim; i = i + 1u) {
            dot = dot + q[head_offset + i] * keys[key_base + i];
        }
        scores[head * params.time_len + t] = dot / scale;
    }
}
"#;

pub(super) const ATTN_VALUES_SHADER: &str = r#"
struct Params {
    time_len: u32,
    n_head: u32,
    head_dim: u32,
    _pad0: u32,
}

@group(0) @binding(0)
var<storage, read> weights: array<f32>;

@group(0) @binding(1)
var<storage, read> values: array<f32>;

@group(0) @binding(2)
var<storage, read_write> y: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let head = gid.x;
    if (head >= params.n_head) {
        return;
    }

    let head_offset = head * params.head_dim;
    for (var i = 0u; i < params.head_dim; i = i + 1u) {
        var acc = 0.0;
        for (var t = 0u; t < params.time_len; t = t + 1u) {
            let weight = weights[head * params.time_len + t];
            let value_base = t * params.n_head * params.head_dim + head_offset;
            acc = acc + weight * values[value_base + i];
        }
        y[head_offset + i] = acc;
    }
}
"#;

pub(super) const ATTN_SCORES_SEQ_SHADER: &str = r#"
struct Params {
    seq_len: u32,
    n_head: u32,
    head_dim: u32,
    n_embd: u32,
}

@group(0) @binding(0)
var<storage, read> q: array<f32>;

@group(0) @binding(1)
var<storage, read> k: array<f32>;

@group(0) @binding(2)
var<storage, read_write> scores: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let query_head = gid.x;
    if (query_head >= params.seq_len * params.n_head) {
        return;
    }

    let query = query_head / params.n_head;
    let head = query_head % params.n_head;
    let head_offset = head * params.head_dim;
    let q_base = query * params.n_embd + head_offset;
    let row_base = query_head * params.seq_len;
    let scale = sqrt(f32(params.head_dim));

    for (var key_pos = 0u; key_pos < params.seq_len; key_pos = key_pos + 1u) {
        let out_idx = row_base + key_pos;
        if (key_pos > query) {
            scores[out_idx] = -1e9;
            continue;
        }
        let k_base = key_pos * params.n_embd + head_offset;
        var dot = 0.0;
        for (var i = 0u; i < params.head_dim; i = i + 1u) {
            dot = dot + q[q_base + i] * k[k_base + i];
        }
        scores[out_idx] = dot / scale;
    }
}
"#;

pub(super) const ATTN_VALUES_SEQ_SHADER: &str = r#"
struct Params {
    seq_len: u32,
    n_head: u32,
    head_dim: u32,
    n_embd: u32,
}

@group(0) @binding(0)
var<storage, read> weights: array<f32>;

@group(0) @binding(1)
var<storage, read> values: array<f32>;

@group(0) @binding(2)
var<storage, read_write> y: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let query_head = gid.x;
    if (query_head >= params.seq_len * params.n_head) {
        return;
    }

    let query = query_head / params.n_head;
    let head = query_head % params.n_head;
    let head_offset = head * params.head_dim;
    let out_base = query * params.n_embd + head_offset;
    let weight_base = query_head * params.seq_len;

    for (var i = 0u; i < params.head_dim; i = i + 1u) {
        var acc = 0.0;
        for (var key_pos = 0u; key_pos < params.seq_len; key_pos = key_pos + 1u) {
            let value_base = key_pos * params.n_embd + head_offset;
            acc = acc + weights[weight_base + key_pos] * values[value_base + i];
        }
        y[out_base + i] = acc;
    }
}
"#;

pub(super) const ATTN_SCORES_BATCH_SHADER: &str = r#"
struct Params {
    batch_size: u32,
    seq_len: u32,
    n_head: u32,
    head_dim: u32,
    n_embd: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0)
var<storage, read> q: array<f32>;

@group(0) @binding(1)
var<storage, read> k: array<f32>;

@group(0) @binding(2)
var<storage, read_write> scores: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let query_head = gid.x;
    let total_query_heads = params.batch_size * params.seq_len * params.n_head;
    if (query_head >= total_query_heads) {
        return;
    }

    let query_row = query_head / params.n_head;
    let head = query_head % params.n_head;
    let batch_idx = query_row / params.seq_len;
    let query = query_row % params.seq_len;
    let head_offset = head * params.head_dim;
    let q_base = query_row * params.n_embd + head_offset;
    let row_base = query_head * params.seq_len;
    let batch_row_base = batch_idx * params.seq_len;
    let scale = sqrt(f32(params.head_dim));

    for (var key_pos = 0u; key_pos < params.seq_len; key_pos = key_pos + 1u) {
        let out_idx = row_base + key_pos;
        if (key_pos > query) {
            scores[out_idx] = -1e9;
            continue;
        }
        let k_row = batch_row_base + key_pos;
        let k_base = k_row * params.n_embd + head_offset;
        var dot = 0.0;
        for (var i = 0u; i < params.head_dim; i = i + 1u) {
            dot = dot + q[q_base + i] * k[k_base + i];
        }
        scores[out_idx] = dot / scale;
    }
}
"#;

pub(super) const ATTN_VALUES_BATCH_SHADER: &str = r#"
struct Params {
    batch_size: u32,
    seq_len: u32,
    n_head: u32,
    head_dim: u32,
    n_embd: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0)
var<storage, read> weights: array<f32>;

@group(0) @binding(1)
var<storage, read> values: array<f32>;

@group(0) @binding(2)
var<storage, read_write> y: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let query_head = gid.x;
    let total_query_heads = params.batch_size * params.seq_len * params.n_head;
    if (query_head >= total_query_heads) {
        return;
    }

    let query_row = query_head / params.n_head;
    let head = query_head % params.n_head;
    let batch_idx = query_row / params.seq_len;
    let head_offset = head * params.head_dim;
    let out_base = query_row * params.n_embd + head_offset;
    let weight_base = query_head * params.seq_len;
    let batch_row_base = batch_idx * params.seq_len;

    for (var i = 0u; i < params.head_dim; i = i + 1u) {
        var acc = 0.0;
        for (var key_pos = 0u; key_pos < params.seq_len; key_pos = key_pos + 1u) {
            let value_row = batch_row_base + key_pos;
            let value_base = value_row * params.n_embd + head_offset;
            acc = acc + weights[weight_base + key_pos] * values[value_base + i];
        }
        y[out_base + i] = acc;
    }
}
"#;

pub(super) const ATTN_BACKWARD_SHADER: &str = r#"
const ATTN_BWD_WG: u32 = 64u;
const ATTN_BWD_MAX_SEQ: u32 = 1024u;

struct Params {
    query: u32,
    time_len: u32,
    seq_len: u32,
    n_head: u32,
    head_dim: u32,
    n_embd: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0)
var<storage, read> d_attn: array<f32>;

@group(0) @binding(1)
var<storage, read> q: array<f32>;

@group(0) @binding(2)
var<storage, read> weights: array<f32>;

@group(0) @binding(3)
var<storage, read> keys: array<f32>;

@group(0) @binding(4)
var<storage, read> values: array<f32>;

@group(0) @binding(5)
var<storage, read_write> d_q: array<f32>;

@group(0) @binding(6)
var<storage, read_write> d_keys: array<f32>;

@group(0) @binding(7)
var<storage, read_write> d_values: array<f32>;

@group(0) @binding(8)
var<uniform> params: Params;

var<workgroup> partials: array<f32, ATTN_BWD_WG>;
var<workgroup> raw_terms: array<f32, ATTN_BWD_MAX_SEQ>;
var<workgroup> dot_term_shared: f32;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let head = gid.x;
    if (head >= params.n_head) {
        return;
    }
    let lane = lid.x;
    if (lane == 0u) {
        dot_term_shared = 0.0;
    }
    workgroupBarrier();

    let scale = sqrt(f32(params.head_dim));
    let head_offset = head * params.head_dim;
    let q_base = params.query * params.n_embd + head_offset;
    let d_attn_base = params.query * params.n_embd + head_offset;
    let d_q_base = params.query * params.n_embd + head_offset;

    for (var t = 0u; t < params.time_len; t = t + 1u) {
        let weight_idx = (params.query * params.n_head + head) * params.seq_len + t;
        let base = t * params.n_embd + head_offset;
        var raw_partial = 0.0;
        for (var i = lane; i < params.head_dim; i = i + ATTN_BWD_WG) {
            let grad = d_attn[d_attn_base + i];
            d_values[base + i] = d_values[base + i] + grad * weights[weight_idx];
            raw_partial = raw_partial + grad * values[base + i];
        }
        partials[lane] = raw_partial;
        workgroupBarrier();

        var stride = ATTN_BWD_WG / 2u;
        loop {
            if (stride == 0u) {
                break;
            }
            if (lane < stride) {
                partials[lane] = partials[lane] + partials[lane + stride];
            }
            workgroupBarrier();
            stride = stride / 2u;
        }
        if (lane == 0u) {
            raw_terms[t] = partials[0];
            dot_term_shared = dot_term_shared + weights[weight_idx] * partials[0];
        }
        workgroupBarrier();
    }

    let dot_term = dot_term_shared;
    for (var i = lane; i < params.head_dim; i = i + ATTN_BWD_WG) {
        let q_i = q[q_base + i];
        var d_q_acc = 0.0;
        for (var t = 0u; t < params.time_len; t = t + 1u) {
            let weight_idx = (params.query * params.n_head + head) * params.seq_len + t;
            let base = t * params.n_embd + head_offset;
            let d_logit = weights[weight_idx] * (raw_terms[t] - dot_term);
            d_q_acc = d_q_acc + d_logit * keys[base + i] / scale;
            d_keys[base + i] = d_keys[base + i] + d_logit * q_i / scale;
        }
        d_q[d_q_base + i] = d_q[d_q_base + i] + d_q_acc;
    }
}
"#;

pub(super) const SCATTER_EMBED_GRADS_SHADER: &str = r#"
struct Params {
    seq_len: u32,
    n_embd: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0)
var<storage, read> token_ids: array<u32>;

@group(0) @binding(1)
var<storage, read> d_embed: array<f32>;

@group(0) @binding(2)
var<storage, read_write> wte_grad: array<f32>;

@group(0) @binding(3)
var<storage, read_write> wpe_grad: array<f32>;

@group(0) @binding(4)
var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x > 0u) {
        return;
    }

    for (var pos = 0u; pos < params.seq_len; pos = pos + 1u) {
        let token = token_ids[pos];
        let src_base = pos * params.n_embd;
        let token_base = token * params.n_embd;
        let pos_base = pos * params.n_embd;
        for (var i = 0u; i < params.n_embd; i = i + 1u) {
            let grad = d_embed[src_base + i];
            wte_grad[token_base + i] = wte_grad[token_base + i] + grad;
            wpe_grad[pos_base + i] = wpe_grad[pos_base + i] + grad;
        }
    }
}
"#;

pub(super) const CROSS_ENTROPY_ROWS_SHADER: &str = r#"
struct Params {
    rows: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
    norm: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
}

@group(0) @binding(0)
var<storage, read> probs: array<f32>;

@group(0) @binding(1)
var<storage, read> targets: array<u32>;

@group(0) @binding(2)
var<storage, read> loss_weights: array<f32>;

@group(0) @binding(3)
var<storage, read_write> d_logits: array<f32>;

@group(0) @binding(4)
var<storage, read_write> losses: array<f32>;

@group(0) @binding(5)
var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= params.rows) {
        return;
    }

    let target_idx = targets[row];
    let row_base = row * params.cols;
    let target_prob = max(probs[row_base + target_idx], 1e-9);
    let weight = loss_weights[row];
    losses[row] = -log(target_prob) * weight;

    for (var col = 0u; col < params.cols; col = col + 1u) {
        let idx = row_base + col;
        var grad = probs[idx];
        if (col == target_idx) {
            grad = grad - 1.0;
        }
        d_logits[idx] = grad * params.norm * weight;
    }
}
"#;

#[derive(Clone, Debug, PartialEq)]
pub struct AdapterSummary {
    pub name: String,
    pub backend: String,
    pub device_type: String,
    pub driver: String,
    pub driver_info: String,
}
