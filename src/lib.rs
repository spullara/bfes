#![feature(test)]
#![feature(portable_simd)]
#[macro_use]
extern crate lazy_static;

use std::simd::f32x16;

struct Index {
    index: Vec<Vec<f32>>,
    squared: Vec<f32>
}

impl Index {
    fn new() -> Index {
        Index {
            index: vec![],
            squared: vec![]
        }
    }
    fn add(&mut self, data: Vec<f32>) {
        self.index.push(data.clone());
        self.squared.push(square(&data));
    }
    // Use cosine similarity to search index
    fn search(&self, data: &Vec<f32>, topk: usize) -> Vec<(usize, f32)> {
        let mut result: Vec<(usize, f32)> = vec![];
        let a2 = square(&data);
        for (i, v) in self.index.iter().enumerate() {
            let score = cosine_similarity(data, v, a2, self.squared[i]);
            result.push((i, score));
        }
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        result.truncate(topk);
        result
    }
    fn len(&self) -> usize {
        self.index.len()
    }
}

fn square(a: &Vec<f32>) -> f32 {
    let mut result = 0.0;
    for i in 0..a.len() {
        result += a[i] * a[i];
    }
    result
}


fn cosine_similarity(a: &Vec<f32>, b: &Vec<f32>, a2: f32, b2: f32) -> f32 {
    let lanes = 16;
    let partitions = a.len() / lanes;
    // Use simd to calculate cosine similarity
    let mut dot: f32 = 0.0;
    // Loop over partitions
    for i in 0..partitions {
        let i1 = i * lanes;
        let i2 = (i + 1) * lanes;
        let a_simd = f32x16::from_slice(&a.as_slice()[i1..i2]);
        let b_simd = f32x16::from_slice(&b.as_slice()[i1..i2]);
        dot += (a_simd * b_simd).reduce_sum();
    }
    dot / (a2 * b2).sqrt()
}

#[cfg(test)]
mod tests {
    use crate::Index;

    extern crate test;
    use test::Bencher;
    use rand::distributions::Standard;
    use rand::Rng;

    /// The following test function is necessary for the header generation.
    #[::safer_ffi::cfg_headers]
    #[test]
    fn generate_headers() -> ::std::io::Result<()> {
        ::safer_ffi::headers::builder()
            .to_file("include/bfes.h")?
            .generate()
    }

    #[bench]
    fn bench_cosine_similarity(b: &mut Bencher) {
        let (index, v) = prepare();
        // Search the index
        b.iter(|| {
            index.search(&v, 10);
        });
    }

    fn prepare() -> (Index, Vec<f32>) {
        // Thread rng
        let rng = rand::thread_rng();
        // Make a new index
        let mut index = super::Index::new();
        // Generate 100000 random 512 dimension vectors
        for _ in 0..100000 {
            let v: Vec<f32> = rng.clone().sample_iter(Standard).take(512).collect();
            index.add(v);
        }
        // Generate a random 512 dimension vector
        let v: Vec<f32> = rng.sample_iter(Standard).take(512).collect();
        (index, v)
    }
}

// Here is the C API for Index
use std::ffi::CStr;
use std::os::raw::c_char;
use ::safer_ffi::prelude::*;
use std::collections::HashMap;
use std::sync::Mutex;
use std::slice;

lazy_static! {
    static ref INDEX_MANAGER: Mutex<HashMap<String, Box<Index>>> = Mutex::new(HashMap::new());
}

fn cchar_to_string(name: *const c_char) -> String {
    let idx_name;
    unsafe {
        idx_name = CStr::from_ptr(name).to_string_lossy().into_owned();
    }
    idx_name
}

#[ffi_export]
pub extern fn bfes_new_index(
    name: *const c_char,
) {
    let idx_name = cchar_to_string(name);

    INDEX_MANAGER.lock().unwrap().insert(
        idx_name,
        Box::new(Index::new()),
    );
}

#[ffi_export]
pub extern fn bfes_add(
    name: *const c_char,
    features: *const f32,
    dimension: usize,
) -> usize {
    let idx_name: String = cchar_to_string(name);
    let data_slice = unsafe { slice::from_raw_parts(features as *const f32, dimension) };
    let buf = data_slice.to_vec();

    match &mut INDEX_MANAGER.lock().unwrap().get_mut(&idx_name) {
        Some(index) => {
            index.add(Vec::from(buf));
            index.len()
        }
        None => 0
    }
}

#[derive_ReprC]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SearchResult {
    index: usize,
    score: f32
}

#[ffi_export]
pub extern fn bfes_search(
    name: *const c_char,
    k: usize,
    features: *const f32,
    dimension: usize,
) -> repr_c::Vec<SearchResult> {
    let idx_name: String = cchar_to_string(name);
    let data_slice = unsafe { slice::from_raw_parts(features, dimension) };
    let buf = data_slice.to_vec();
    let topk = k;

    let mut result: Vec<SearchResult> = vec![];
    if let Some(index) = INDEX_MANAGER.lock().unwrap().get(&idx_name) {
        index.search(&Vec::from(buf), topk).iter().for_each(|x| {
            result.push(SearchResult { index: x.0, score: x.1})
        })
    }
    result.into()
}

#[ffi_export]
pub extern fn bfes_parse_embedding(
    embedding_string: *const c_char
) -> repr_c::Vec<f32> {
    let embedding_string = cchar_to_string(embedding_string);
    embedding_string.split(",").filter_map(|s| s.parse::<f32>().ok()).collect::<Vec<_>>().into()
}

#[ffi_export]
pub extern fn bfes_parse_embeddings(
    embedding_strings: *const *const c_char,
    num: isize
) -> repr_c::Vec<repr_c::Vec<f32>> {
    let mut result: Vec<repr_c::Vec<f32>> = Vec::new();
    for i in 0..num {
        unsafe {
            let vec = bfes_parse_embedding(*(embedding_strings.offset(i)));
            result.push(vec)
        }
    }
    result.into()
}