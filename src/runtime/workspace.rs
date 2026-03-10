use crate::core::error::{Result, RustGptError};

#[derive(Clone, Debug, PartialEq)]
pub struct AttentionWeights {
    n_head: usize,
    time_len: usize,
    data: Vec<f32>,
}

impl AttentionWeights {
    pub fn zeros(n_head: usize, time_len: usize) -> Self {
        Self {
            n_head,
            time_len,
            data: vec![0.0; n_head * time_len],
        }
    }

    pub fn head(&self, head_idx: usize) -> &[f32] {
        let start = head_idx * self.time_len;
        &self.data[start..start + self.time_len]
    }

    pub fn head_mut(&mut self, head_idx: usize) -> &mut [f32] {
        let start = head_idx * self.time_len;
        &mut self.data[start..start + self.time_len]
    }

    pub fn time_len(&self) -> usize {
        self.time_len
    }

    pub fn head_count(&self) -> usize {
        self.n_head
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LayerKvCache {
    n_embd: usize,
    keys: Vec<f32>,
    values: Vec<f32>,
}

impl LayerKvCache {
    pub fn new(n_embd: usize) -> Self {
        Self {
            n_embd,
            keys: Vec::new(),
            values: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.keys.len() / self.n_embd
    }

    pub fn push(&mut self, key: &[f32], value: &[f32]) -> Result<()> {
        if key.len() != self.n_embd || value.len() != self.n_embd {
            return Err(RustGptError::Tensor(format!(
                "kv cache push shape mismatch: expected {} elements, got key={} value={}",
                self.n_embd,
                key.len(),
                value.len()
            )));
        }
        self.keys.extend_from_slice(key);
        self.values.extend_from_slice(value);
        Ok(())
    }

    pub fn key(&self, time_idx: usize) -> &[f32] {
        let start = time_idx * self.n_embd;
        &self.keys[start..start + self.n_embd]
    }

    pub fn value(&self, time_idx: usize) -> &[f32] {
        let start = time_idx * self.n_embd;
        &self.values[start..start + self.n_embd]
    }
}

pub type KvCache = Vec<LayerKvCache>;

pub fn new_kv_cache(n_layer: usize, n_embd: usize) -> KvCache {
    (0..n_layer).map(|_| LayerKvCache::new(n_embd)).collect()
}

#[derive(Clone, Debug, PartialEq)]
pub struct LayerKvGrad {
    n_embd: usize,
    d_keys: Vec<f32>,
    d_values: Vec<f32>,
}

impl LayerKvGrad {
    pub fn new(seq_len: usize, n_embd: usize) -> Self {
        Self {
            n_embd,
            d_keys: vec![0.0; seq_len * n_embd],
            d_values: vec![0.0; seq_len * n_embd],
        }
    }

    pub fn key(&self, time_idx: usize) -> &[f32] {
        let start = time_idx * self.n_embd;
        &self.d_keys[start..start + self.n_embd]
    }

    pub fn key_mut(&mut self, time_idx: usize) -> &mut [f32] {
        let start = time_idx * self.n_embd;
        &mut self.d_keys[start..start + self.n_embd]
    }

    pub fn value(&self, time_idx: usize) -> &[f32] {
        let start = time_idx * self.n_embd;
        &self.d_values[start..start + self.n_embd]
    }

    pub fn value_mut(&mut self, time_idx: usize) -> &mut [f32] {
        let start = time_idx * self.n_embd;
        &mut self.d_values[start..start + self.n_embd]
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct KvGrad {
    pub layers: Vec<LayerKvGrad>,
}

impl KvGrad {
    pub fn new(n_layer: usize, seq_len: usize, n_embd: usize) -> Self {
        Self {
            layers: (0..n_layer)
                .map(|_| LayerKvGrad::new(seq_len, n_embd))
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{AttentionWeights, KvGrad, LayerKvCache};

    #[test]
    fn attention_weights_are_contiguous_per_head() {
        let mut weights = AttentionWeights::zeros(2, 3);
        weights.head_mut(1).copy_from_slice(&[0.1, 0.2, 0.3]);
        assert_eq!(weights.head(1), &[0.1, 0.2, 0.3]);
        assert_eq!(weights.time_len(), 3);
        assert_eq!(weights.head_count(), 2);
    }

    #[test]
    fn kv_cache_stores_time_major_rows() {
        let mut cache = LayerKvCache::new(4);
        cache
            .push(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0])
            .unwrap();
        cache
            .push(&[9.0, 10.0, 11.0, 12.0], &[13.0, 14.0, 15.0, 16.0])
            .unwrap();
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.key(1), &[9.0, 10.0, 11.0, 12.0]);
        assert_eq!(cache.value(0), &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn kv_grad_exposes_mutable_time_slices() {
        let mut grad = KvGrad::new(1, 3, 4);
        grad.layers[0]
            .key_mut(2)
            .copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        grad.layers[0]
            .value_mut(1)
            .copy_from_slice(&[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(grad.layers[0].key(2), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(grad.layers[0].value(1), &[5.0, 6.0, 7.0, 8.0]);
    }
}
