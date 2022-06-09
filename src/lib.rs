#![feature(test)]
#![feature(portable_simd)]

use std::simd::f32x16;
use std::simd::StdFloat;

struct Index {
    index: Vec<Vec<f32>>,
}

impl Index {
    fn new() -> Index {
        Index {
            index: vec![]
        }
    }
    fn add(&mut self, data: Vec<f32>) {
        self.index.push(data);
    }
    // Use cosine similarity to search index
    fn search_simd(&self, data: &Vec<f32>, topk: usize) -> Vec<(usize, f32)> {
        let mut result: Vec<(usize, f32)> = vec![];
        for (i, v) in self.index.iter().enumerate() {
            let score = cosine_similarity_simd(data, v);
            result.push((i, score));
        }
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        result.truncate(topk);
        result
    }
    // Use cosine similarity to search index
    fn search_cpu(&self, data: &Vec<f32>, topk: usize) -> Vec<(usize, f32)> {
        let mut result: Vec<(usize, f32)> = vec![];
        for (i, v) in self.index.iter().enumerate() {
            let score = cosine_similarity_cpu(data, v);
            result.push((i, score));
        }
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        result.truncate(topk);
        result
    }
}

fn cosine_similarity_cpu(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    let mut sum: f32 = 0.0;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    let a_norm = a.iter().fold(0.0, |acc, x| acc + x.powi(2));
    let b_norm = b.iter().fold(0.0, |acc, x| acc + x.powi(2));
    sum / (a_norm * b_norm).sqrt()
}

fn cosine_similarity_simd(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    let lanes = 16;
    let partitions = a.len() / lanes;
    // Use simd to calculate cosine similarity
    let mut dot: f32 = 0.0;
    let mut norm_a: f32 = 0.0;
    let mut norm_b: f32 = 0.0;
    // Loop over partitions
    for i in 0..partitions {
        let a_simd = f32x16::from_slice(a.as_slice().split_at(i * lanes).1);
        let b_simd = f32x16::from_slice(b.as_slice().split_at(i * lanes).1);
        dot += (a_simd * b_simd).reduce_sum();
        norm_a += (a_simd * a_simd).reduce_sum();
        norm_b += (b_simd * b_simd).reduce_sum();
    }
    dot / (norm_a * norm_b).sqrt()
}

extern crate test;

#[cfg(test)]
mod tests {
    use crate::{Index, cosine_similarity_cpu, cosine_similarity_simd};

    #[test]
    // Compare that search_simd and search_cpu are the same
    fn test_search_simd_cpu() {
        println!("preparing");
        let (index, data) = prepare();
        println!("simd");
        let result_simd = index.search_simd(&data, 10);
        println!("cpu");
        let result_cpu = index.search_cpu(&data, 10);
        assert_eq!(result_simd, result_cpu);
    }

    use test::Bencher;
    use rand::Rng;

    #[bench]
    fn bench_cosine_similarity_simd(b: &mut Bencher) {
        let (index, v) = prepare();
        // Search the index
        b.iter(|| {
            index.search_simd(&v, 10);
        });
    }

    #[bench]
    fn bench_cosine_similarity_cpu(b: &mut Bencher) {
        let (index, v) = prepare();
        // Search the index
        b.iter(|| {
            index.search_simd(&v, 10);
        });
    }

    fn prepare() -> (Index, Vec<f32>) {
        // Thread rng
        let mut rng = rand::thread_rng();
        // Make a new index
        let mut index = super::Index::new();
        // Generate 100000 random 512 dimension vectors
        for _ in 0..100000 {
            let mut v: Vec<f32> = vec![];
            v.reserve(512);
            for _ in 0..512 {
                v.push(rng.gen());
            }
            index.add(v);
        }
        // Generate a random 512 dimension vector
        let mut v: Vec<f32> = vec![];
        for _ in 0..512 {
            v.reserve(512);
            v.push(rand::random::<f32>());
        }
        (index, v)
    }
}
