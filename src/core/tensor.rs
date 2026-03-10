use crate::core::error::{Result, RustGptError};
use crate::core::rng::Rng;

#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
    pub grad: Vec<f32>,
    pub m: Vec<f32>,
    pub v: Vec<f32>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        let len = rows * cols;
        Self {
            rows,
            cols,
            data: vec![0.0; len],
            grad: vec![0.0; len],
            m: vec![0.0; len],
            v: vec![0.0; len],
        }
    }

    pub fn from_gaussian(rows: usize, cols: usize, std: f32, rng: &mut Rng) -> Self {
        let mut matrix = Self::zeros(rows, cols);
        for value in &mut matrix.data {
            *value = rng.gauss(std);
        }
        matrix
    }

    pub fn from_parts(
        rows: usize,
        cols: usize,
        data: Vec<f32>,
        m: Vec<f32>,
        v: Vec<f32>,
    ) -> Result<Self> {
        let len = rows * cols;
        if data.len() != len || m.len() != len || v.len() != len {
            return Err(RustGptError::Tensor(format!(
                "matrix parts length mismatch for shape ({rows}, {cols})"
            )));
        }
        Ok(Self {
            rows,
            cols,
            data,
            grad: vec![0.0; len],
            m,
            v,
        })
    }

    pub fn parameter_count(&self) -> usize {
        self.data.len()
    }

    pub fn row(&self, row_idx: usize) -> &[f32] {
        let start = row_idx * self.cols;
        &self.data[start..start + self.cols]
    }

    pub fn row_mut(&mut self, row_idx: usize) -> &mut [f32] {
        let start = row_idx * self.cols;
        &mut self.data[start..start + self.cols]
    }

    pub fn grad_row_mut(&mut self, row_idx: usize) -> &mut [f32] {
        let start = row_idx * self.cols;
        &mut self.grad[start..start + self.cols]
    }

    pub fn zero_grad(&mut self) {
        self.grad.fill(0.0);
    }

    pub fn adam_step(&mut self, lr_t: f32, beta1: f32, beta2: f32, eps: f32, step_num: usize) {
        let step_num = step_num as i32;
        for idx in 0..self.data.len() {
            let grad = self.grad[idx];
            self.m[idx] = beta1 * self.m[idx] + (1.0 - beta1) * grad;
            self.v[idx] = beta2 * self.v[idx] + (1.0 - beta2) * grad * grad;
            let m_hat = self.m[idx] / (1.0 - beta1.powi(step_num));
            let v_hat = self.v[idx] / (1.0 - beta2.powi(step_num));
            self.data[idx] -= lr_t * m_hat / (v_hat.sqrt() + eps);
            self.grad[idx] = 0.0;
        }
    }
}

pub fn linear(x: &[f32], w: &Matrix) -> Result<Vec<f32>> {
    if x.len() != w.cols {
        return Err(RustGptError::Tensor(format!(
            "linear shape mismatch: input has {} elements, matrix expects {}",
            x.len(),
            w.cols
        )));
    }

    let mut out = Vec::with_capacity(w.rows);
    for row_idx in 0..w.rows {
        let row = w.row(row_idx);
        out.push(x.iter().zip(row).map(|(a, b)| a * b).sum());
    }
    Ok(out)
}

pub fn linear_transposed(x: &[f32], w: &Matrix) -> Result<Vec<f32>> {
    if x.len() != w.rows {
        return Err(RustGptError::Tensor(format!(
            "linear_transposed shape mismatch: input has {} elements, matrix expects {} rows",
            x.len(),
            w.rows
        )));
    }

    let mut out = vec![0.0; w.cols];
    for (row_idx, value) in x.iter().copied().enumerate() {
        let row = w.row(row_idx);
        for col_idx in 0..w.cols {
            out[col_idx] += value * row[col_idx];
        }
    }
    Ok(out)
}

pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |acc, value| acc.max(value));
    let exps = logits
        .iter()
        .map(|value| (value - max_logit).exp())
        .collect::<Vec<_>>();
    let total = exps.iter().sum::<f32>();
    exps.into_iter().map(|value| value / total).collect()
}

pub fn rmsnorm(x: &[f32]) -> (Vec<f32>, f32) {
    let mean_square = x.iter().map(|value| value * value).sum::<f32>() / x.len() as f32;
    let rms_inv = (mean_square + 1e-5).sqrt().recip();
    (x.iter().map(|value| value * rms_inv).collect(), rms_inv)
}

pub fn relu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|value| value.max(0.0)).collect()
}

pub fn accumulate_linear_grad(x: &[f32], dout: &[f32], w: &mut Matrix) -> Result<()> {
    let cols = w.cols;
    let rows = w.rows;
    if x.len() != w.cols {
        return Err(RustGptError::Tensor(format!(
            "accumulate_linear_grad input shape mismatch: {} vs {}",
            x.len(),
            w.cols
        )));
    }
    if dout.len() != rows {
        return Err(RustGptError::Tensor(format!(
            "accumulate_linear_grad output shape mismatch: {} vs {}",
            dout.len(),
            rows
        )));
    }

    for (row_idx, d_out) in dout.iter().copied().enumerate() {
        let grad_row = w.grad_row_mut(row_idx);
        for col_idx in 0..cols {
            grad_row[col_idx] += d_out * x[col_idx];
        }
    }
    Ok(())
}

pub fn linear_backward(x: &[f32], dout: &[f32], w: &mut Matrix) -> Result<Vec<f32>> {
    accumulate_linear_grad(x, dout, w)?;
    linear_transposed(dout, w)
}

pub fn rmsnorm_backward(x: &[f32], rms_inv: f32, dy: &[f32]) -> Vec<f32> {
    let dot = dy
        .iter()
        .zip(x)
        .map(|(grad, value)| grad * value)
        .sum::<f32>();
    let scale = dot * rms_inv.powi(3) / x.len() as f32;
    x.iter()
        .zip(dy)
        .map(|(value, grad)| grad * rms_inv - value * scale)
        .collect()
}

pub fn relu_backward(pre_activation: &[f32], dout: &[f32]) -> Vec<f32> {
    pre_activation
        .iter()
        .zip(dout)
        .map(|(pre, grad)| if *pre > 0.0 { *grad } else { 0.0 })
        .collect()
}

pub fn softmax_backward(probs: &[f32], dprobs: &[f32]) -> Vec<f32> {
    let dot = probs.iter().zip(dprobs).map(|(p, d)| p * d).sum::<f32>();
    probs
        .iter()
        .zip(dprobs)
        .map(|(p, d)| p * (d - dot))
        .collect()
}

pub fn add_in_place(target: &mut [f32], source: &[f32]) {
    for (target_value, source_value) in target.iter_mut().zip(source) {
        *target_value += source_value;
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Matrix, accumulate_linear_grad, add_in_place, linear, linear_backward, linear_transposed,
        relu, relu_backward, rmsnorm, rmsnorm_backward, softmax, softmax_backward,
    };

    #[test]
    fn linear_multiplies_row_major_matrix() {
        let matrix = Matrix {
            rows: 2,
            cols: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            grad: vec![0.0; 6],
            m: vec![0.0; 6],
            v: vec![0.0; 6],
        };
        let out = linear(&[1.0, 0.5, -1.0], &matrix).unwrap();
        assert_eq!(out.len(), 2);
        assert!((out[0] - -1.0).abs() < 1e-6);
        assert!((out[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn softmax_returns_probabilities() {
        let probs = softmax(&[1.0, 2.0, 3.0]);
        let total = probs.iter().sum::<f32>();
        assert!((total - 1.0).abs() < 1e-6);
        assert!(probs[2] > probs[1] && probs[1] > probs[0]);
    }

    #[test]
    fn rmsnorm_returns_scale_factor() {
        let (normed, rms_inv) = rmsnorm(&[3.0, 4.0]);
        assert_eq!(normed.len(), 2);
        assert!(rms_inv.is_finite());
        assert!(normed.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn relu_zeroes_negative_values() {
        assert_eq!(relu(&[-2.0, 0.5, 0.0]), vec![0.0, 0.5, 0.0]);
    }

    #[test]
    fn linear_backward_returns_input_gradient() {
        let mut matrix = Matrix {
            rows: 2,
            cols: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            grad: vec![0.0; 6],
            m: vec![0.0; 6],
            v: vec![0.0; 6],
        };
        let dx = linear_backward(&[1.0, -1.0, 0.5], &[0.25, -0.5], &mut matrix).unwrap();
        assert_eq!(dx.len(), 3);
        assert!(dx.iter().all(|value| value.is_finite()));
        assert!(matrix.grad.iter().any(|value| *value != 0.0));
    }

    #[test]
    fn linear_transposed_multiplies_against_matrix_columns() {
        let matrix = Matrix {
            rows: 2,
            cols: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            grad: vec![0.0; 6],
            m: vec![0.0; 6],
            v: vec![0.0; 6],
        };
        let out = linear_transposed(&[0.25, -0.5], &matrix).unwrap();
        assert_eq!(out, vec![-1.75, -2.0, -2.25]);
    }

    #[test]
    fn accumulate_linear_grad_updates_only_weight_gradient() {
        let mut matrix = Matrix {
            rows: 2,
            cols: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            grad: vec![0.0; 6],
            m: vec![0.0; 6],
            v: vec![0.0; 6],
        };
        accumulate_linear_grad(&[1.0, -1.0, 0.5], &[0.25, -0.5], &mut matrix).unwrap();
        assert_eq!(matrix.grad, vec![0.25, -0.25, 0.125, -0.5, 0.5, -0.25]);
    }

    #[test]
    fn softmax_backward_is_finite() {
        let probs = softmax(&[0.5, 1.0, -0.25]);
        let dlogits = softmax_backward(&probs, &[1.0, -0.5, 0.25]);
        assert!(dlogits.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn rmsnorm_backward_is_finite() {
        let x = vec![1.0, 2.0, -1.0];
        let (_, rms_inv) = rmsnorm(&x);
        let dx = rmsnorm_backward(&x, rms_inv, &[0.25, -0.5, 0.75]);
        assert!(dx.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn relu_backward_masks_negative_inputs() {
        assert_eq!(
            relu_backward(&[-1.0, 0.25, 2.0], &[0.5, 0.5, 0.5]),
            vec![0.0, 0.5, 0.5]
        );
    }

    #[test]
    fn add_in_place_accumulates_vectors() {
        let mut target = vec![1.0, 2.0];
        add_in_place(&mut target, &[0.5, -1.0]);
        assert_eq!(target, vec![1.5, 1.0]);
    }
}
