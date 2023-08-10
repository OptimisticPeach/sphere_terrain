use std::simd::{LaneCount, Simd, SimdFloat, SupportedLaneCount};

use clatter::{Sample, Simplex3d};
use glam::Vec3;
use rand::SeedableRng;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};

// Credited to clatter example: https://github.com/Ralith/clatter/blob/main/examples/demo.rs

/// Compute a patch of fractal brownian motion noise
#[derive(Serialize, Deserialize, Copy, Clone, Debug)]
pub struct Opts {
    /// Number of times to sample the noise
    pub octaves: usize,
    /// How smooth the noise should be (sensible values are around 0.5-1)
    pub hurst_exponent: f32,
    /// Frequency ratio between successive octaves
    pub lacunarity: f32,
    /// Maximum value
    pub max: f32,
    /// Minumum value
    pub min: f32,
    /// Scales points before sampling
    pub sample_scale: f32,
    /// Seeds the rng
    pub seed: u32,
    #[serde(default)]
    /// Offsets the noise sampling
    pub offset: Vec3,
}

pub fn sample_all_noise(points: &[Vec3], opts: Opts) -> Vec<f32> {
    let mut output = Vec::with_capacity(points.len());
    generate(opts, points, &mut output);
    output
}

#[cfg(target_arch = "wasm32")]
fn generate(opts: Opts, samples: &[Vec3], pixels: &mut Vec<f32>) {
    generate_inner::<4>(opts, samples, pixels);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn generate(opts: Opts, samples: &[Vec3], pixels: &mut Vec<f32>) {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe {
            generate_avx2(opts, samples, pixels);
        }
    } else if is_x86_feature_detected!("sse4.2") {
        unsafe {
            generate_sse(opts, samples, pixels);
        }
    } else {
        generate_inner::<4>(opts, samples, pixels);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn generate_avx2(opts: Opts, samples: &[Vec3], pixels: &mut Vec<f32>) {
    generate_inner::<8>(opts, samples, pixels);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse4.2")]
unsafe fn generate_sse(opts: Opts, samples: &[Vec3], pixels: &mut Vec<f32>) {
    generate_inner::<4>(opts, samples, pixels);
}

#[inline(always)]
fn generate_inner<const LANES: usize>(opts: Opts, samples: &[Vec3], pixels: &mut Vec<f32>)
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let rest_len = samples.len() % LANES;
    pixels.extend(std::iter::repeat(0.0).take(samples.len() - rest_len));
    let user_seed = opts.seed.to_be_bytes();
    let mut seed = [0u8; 32];

    seed[0..4].copy_from_slice(&user_seed);

    let mut rng = rand::rngs::StdRng::from_seed(seed);

    let noise = Simplex3d::random(&mut rng);

    let offset_x = Simd::splat(opts.offset.x);
    let offset_y = Simd::splat(opts.offset.y);
    let offset_z = Simd::splat(opts.offset.z);

    let sample_scale = Simd::splat(opts.sample_scale);
    samples
        .par_chunks_exact(LANES)
        .zip(pixels.par_chunks_exact_mut(LANES))
        .for_each(|(chunk, result)| {
            let mut px = Simd::splat(0.0f32);
            let mut py = Simd::splat(0.0f32);
            let mut pz = Simd::splat(0.0f32);
            for i in 0..LANES {
                px[i] = chunk[i].x;
                py[i] = chunk[i].y;
                pz[i] = chunk[i].z;
            }
            px *= sample_scale;
            py *= sample_scale;
            pz *= sample_scale;

            px += offset_x;
            py += offset_y;
            pz += offset_z;

            let sample = fbm::<LANES>(
                opts.octaves,
                (-opts.hurst_exponent).exp2(),
                opts.lacunarity,
                [px, py, pz],
                &noise,
            );
            let value = sample.value;
            result.copy_from_slice(&*value.as_array());
        });

    let rest = &samples[samples.len() - rest_len..];
    let mut px = Simd::splat(0.0f32);
    let mut py = Simd::splat(0.0f32);
    let mut pz = Simd::splat(0.0f32);
    for i in 0..rest_len {
        px[i] = rest[i].x;
        py[i] = rest[i].y;
        pz[i] = rest[i].z;
    }
    px *= sample_scale;
    py *= sample_scale;
    pz *= sample_scale;

    px += offset_x;
    py += offset_y;
    pz += offset_z;

    let sample = fbm::<LANES>(
        opts.octaves,
        (-opts.hurst_exponent).exp2(),
        opts.lacunarity,
        [px, py, pz],
        &noise,
    );
    let value = sample.value;
    pixels.extend_from_slice(&value.as_array()[..rest_len]);

    let max = Simd::splat(f32::NEG_INFINITY);
    let min = Simd::splat(f32::INFINITY);
    let (max, min) = pixels
        .par_chunks_exact(LANES)
        .map(|x| Simd::from_slice(x))
        .fold(
            || (max, min),
            |(max, min), y| (max.simd_max(y), min.simd_min(y)),
        )
        .reduce_with(|(max, min), (omax, omin)| (max.simd_max(omax), min.simd_min(omin)))
        .unwrap();

    let mut max = max.reduce_max();
    let mut min = min.reduce_min();

    pixels[pixels.len() - rest_len..].iter().for_each(|x| {
        max = max.max(*x);
        min = min.min(*x);
    });

    let slope = (opts.min - opts.max) / (min - max);
    let offset = opts.min - slope * min;

    for item in pixels.iter_mut() {
        *item = *item * slope + offset;
    }
}

#[inline(always)]
fn fbm<const LANES: usize>(
    octaves: usize,
    gain: f32,
    lacunarity: f32,
    point: [Simd<f32, LANES>; 3],
    noise: &Simplex3d,
) -> Sample<LANES, 3>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let mut result = Sample::default();

    let mut frequency = 1.0;
    let mut amplitude = 1.0;
    let mut scale = 0.0;
    for _ in 0..octaves {
        result += noise.sample(point.map(|x| x * Simd::splat(frequency))) * Simd::splat(amplitude);
        scale += amplitude;
        frequency *= lacunarity;
        amplitude *= gain;
    }
    result / Simd::splat(scale)
}
