use crate::core::config::PositionEncodingKind;

const ROPE_BASE: f32 = 10_000.0;

pub fn apply_qk_position_encoding_in_place(
    values: &mut [f32],
    position_encoding: PositionEncodingKind,
    pos_id: usize,
    n_head: usize,
    head_dim: usize,
) {
    if position_encoding == PositionEncodingKind::Rope {
        apply_rope_in_place(values, pos_id, n_head, head_dim);
    }
}

pub fn apply_qk_position_gradient_in_place(
    grads: &mut [f32],
    position_encoding: PositionEncodingKind,
    pos_id: usize,
    n_head: usize,
    head_dim: usize,
) {
    if position_encoding == PositionEncodingKind::Rope {
        apply_rope_inverse_in_place(grads, pos_id, n_head, head_dim);
    }
}

pub fn apply_rope_in_place(values: &mut [f32], pos_id: usize, n_head: usize, head_dim: usize) {
    rotate_rope_in_place(values, pos_id, n_head, head_dim, false);
}

pub fn apply_rope_inverse_in_place(
    values: &mut [f32],
    pos_id: usize,
    n_head: usize,
    head_dim: usize,
) {
    rotate_rope_in_place(values, pos_id, n_head, head_dim, true);
}

fn rotate_rope_in_place(
    values: &mut [f32],
    pos_id: usize,
    n_head: usize,
    head_dim: usize,
    inverse: bool,
) {
    if values.is_empty() || head_dim == 0 {
        return;
    }
    let half = head_dim / 2;
    if half == 0 {
        return;
    }
    debug_assert_eq!(values.len(), n_head * head_dim);
    let pos = pos_id as f32;
    for head_idx in 0..n_head {
        let base = head_idx * head_dim;
        for pair_idx in 0..half {
            let theta = pos / ROPE_BASE.powf(pair_idx as f32 / half as f32);
            let (sin_theta, cos_theta) = theta.sin_cos();
            let left_idx = base + pair_idx;
            let right_idx = base + half + pair_idx;
            let left = values[left_idx];
            let right = values[right_idx];
            if inverse {
                values[left_idx] = left * cos_theta + right * sin_theta;
                values[right_idx] = right * cos_theta - left * sin_theta;
            } else {
                values[left_idx] = left * cos_theta - right * sin_theta;
                values[right_idx] = left * sin_theta + right * cos_theta;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::core::config::PositionEncodingKind;

    use super::{
        apply_qk_position_encoding_in_place, apply_qk_position_gradient_in_place,
        apply_rope_in_place, apply_rope_inverse_in_place,
    };

    #[test]
    fn rope_is_identity_at_position_zero() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0];
        let before = values.clone();
        apply_rope_in_place(&mut values, 0, 1, 4);
        assert_eq!(values, before);
    }

    #[test]
    fn rope_inverse_recovers_original_vector() {
        let mut values = vec![0.5, -0.25, 1.25, 0.75, 0.2, -0.1, 0.4, 0.8];
        let original = values.clone();
        apply_rope_in_place(&mut values, 7, 2, 4);
        apply_rope_inverse_in_place(&mut values, 7, 2, 4);
        for (left, right) in values.iter().zip(&original) {
            assert!((left - right).abs() < 1e-5);
        }
    }

    #[test]
    fn rope_helpers_only_activate_in_rope_mode() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0];
        let original = values.clone();
        apply_qk_position_encoding_in_place(
            &mut values,
            PositionEncodingKind::LearnedAbsolute,
            3,
            1,
            4,
        );
        apply_qk_position_gradient_in_place(
            &mut values,
            PositionEncodingKind::LearnedAbsolute,
            3,
            1,
            4,
        );
        assert_eq!(values, original);
    }
}
