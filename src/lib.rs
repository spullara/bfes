#![feature(test)]
use lazy_static::lazy_static;

use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Mutex;
use itertools::Itertools;

struct Index {
    dim: usize,
    index: Vec<Vec<f32>>,
}

struct Score {
    id: usize,
    score: f32,
}

impl PartialEq<Self> for Score {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.id == other.id
    }
}

impl PartialOrd<Self> for Score {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match other.score.partial_cmp(&self.score) {
            Some(Ordering::Equal) => Some(self.id.cmp(&other.id)),
            Some(ordering) => Some(ordering),
            None => None,
        }
    }
}

impl Eq for Score {}

impl Ord for Score {
    // Reverse the order so that the highest score is first.
    fn cmp(&self, other: &Self) -> Ordering {
        match other.score.total_cmp(&self.score) {
            Ordering::Equal => other.id.cmp(&self.id),
            other => other,
        }
    }
}

// This currently only supports vectors that have a length with a
// multiple of 16.
impl Index {
    fn new(dim: usize) -> Index {
        assert_eq!(dim % 16, 0);
        Index { index: vec![], dim }
    }
    fn add(&mut self, data: Vec<f32>) {
        assert_eq!(data.len(), self.dim);
        // Precompute the unit vector and store it
        let unit_factor = mag_squared(&data).sqrt();
        self.index
            .push(data.into_iter().map(|x| x / unit_factor).collect());
    }
    // Use cosine similarity to search index
    fn search(&self, query: &Vec<f32>, topk: usize) -> Vec<(usize, f32)> {
        assert!(topk > 0);
        assert_eq!(query.len(), self.dim);

        // Precompute the unit coefficient for the search vector.
        let query_unit = 1.0 / mag_squared(query).sqrt();

        // Get the top k highest scoring embeddings
        self.index
            .iter()
            .enumerate()
            .map(|(id, vec)| Score {
                id,
                score: dot_product(query, vec),
            })
            .k_smallest(topk)
            .map(|s| (s.id, s.score * query_unit))
            .collect()
    }
    fn len(&self) -> usize {
        self.index.len()
    }
}

// Short hand
fn mag_squared(a: &Vec<f32>) -> f32 {
    dot_product(a, a)
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        panic!("Vectors must have the same length");
    }

    let mut dot = 0.0;
    for i in 0..a.len() {
        dot += a[i] * b[i];
    }

    dot
}

#[cfg(test)]
mod tests {
    use test::Bencher;

    use rand::{Rng, SeedableRng};
    use rand::distributions::Standard;
    use rand::rngs::StdRng;

    use crate::Index;

    extern crate test;

    #[test]
    fn search_test() {
        let (index, v) = prepare();
        let result = index.search(&v, 10);
        assert_eq!(result.len(), 10);
        let mut last = f32::MAX;
        for (_i, score) in result.iter().enumerate() {
            assert!(score.1 < last);
            last = score.1;
        }
        println!("{:?}", result);
        assert_eq!(result[0].0, 77918);
    }

    #[bench]
    fn bench_search(b: &mut Bencher) {
        let (index, v) = prepare();
        // Search the index
        b.iter(|| {
            index.search(&v, 10);
        });
    }

    fn prepare() -> (Index, Vec<f32>) {
        // Make a new index
        let dim = 512;
        let mut index = Index::new(dim);
        // Generate 100000 random 512 dimension vectors
        for i in 0..100000 {
            let rng = StdRng::seed_from_u64(1337 + i + 1);
            let v: Vec<f32> = rng.sample_iter(Standard).take(dim).collect();
            index.add(v);
        }
        // Thread rng
        let rng = StdRng::seed_from_u64(1337);
        // Generate a random 512 dimension vector
        let v: Vec<f32> = rng.sample_iter(Standard).take(dim).collect();
        (index, v)
    }
}