//! Attention-shape helpers shared by the CPU reference path and the GPU path.
//! Grouped-query attention stores fewer K/V heads than query heads, but the attention
//! computation itself still expects a full per-query-head layout.

use crate::core::error::{Result, RustGptError};

pub fn expand_grouped_heads(
    compact: &[f32],
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    let compact_dim = n_kv_head * head_dim;
    if compact.len() != compact_dim {
        return Err(RustGptError::Tensor(format!(
            "expand_grouped_heads expected {} elements, got {}",
            compact_dim,
            compact.len()
        )));
    }
    let query_heads_per_kv = n_head / n_kv_head;
    let mut expanded = vec![0.0; n_head * head_dim];
    for kv_head in 0..n_kv_head {
        let kv_start = kv_head * head_dim;
        let kv_end = kv_start + head_dim;
        let kv_slice = &compact[kv_start..kv_end];
        for offset in 0..query_heads_per_kv {
            let head_idx = kv_head * query_heads_per_kv + offset;
            let head_start = head_idx * head_dim;
            let head_end = head_start + head_dim;
            expanded[head_start..head_end].copy_from_slice(kv_slice);
        }
    }
    Ok(expanded)
}

pub fn collapse_grouped_head_grads(
    expanded: &[f32],
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    let expanded_dim = n_head * head_dim;
    if expanded.len() != expanded_dim {
        return Err(RustGptError::Tensor(format!(
            "collapse_grouped_head_grads expected {} elements, got {}",
            expanded_dim,
            expanded.len()
        )));
    }
    let query_heads_per_kv = n_head / n_kv_head;
    let mut compact = vec![0.0; n_kv_head * head_dim];
    for kv_head in 0..n_kv_head {
        let compact_start = kv_head * head_dim;
        let compact_end = compact_start + head_dim;
        let compact_slice = &mut compact[compact_start..compact_end];
        for offset in 0..query_heads_per_kv {
            let head_idx = kv_head * query_heads_per_kv + offset;
            let head_start = head_idx * head_dim;
            let head_end = head_start + head_dim;
            for feature_idx in 0..head_dim {
                compact_slice[feature_idx] += expanded[head_start..head_end][feature_idx];
            }
        }
    }
    Ok(compact)
}

#[cfg(test)]
mod tests {
    use super::{collapse_grouped_head_grads, expand_grouped_heads};

    #[test]
    fn grouped_heads_expand_by_repeating_kv_slices() {
        let expanded = expand_grouped_heads(&[1.0, 2.0, 3.0, 4.0], 4, 2, 2).unwrap();
        assert_eq!(expanded, vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
    }

    #[test]
    fn grouped_head_grads_collapse_by_summing_query_groups() {
        let collapsed =
            collapse_grouped_head_grads(&[1.0, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0], 4, 2, 2)
                .unwrap();
        assert_eq!(collapsed, vec![11.0, 22.0, 33.0, 44.0]);
    }
}
