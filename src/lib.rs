#![feature(portable_simd)]

// use std::collections::HashSet;
// use fxhash::FxBuildHasher;
use glam::Vec3;
use hexasphere::shapes::IcoSphere;
use serde::{Deserialize, Serialize};
use tinyvec::ArrayVec;
use crate::noisegen::Opts;

pub mod noisegen;
mod rehex;

#[derive(Serialize, Deserialize, Copy, Clone, Debug, PartialEq)]
pub struct DropSettings {
    // [0, 1]
    pub inertia: f32,
    // [0, 32]
    pub capacity: f32,
    // [0, 1]
    pub deposition: f32,
    // [0, 1]
    pub erosion: f32,
    // [0, 0.5]
    pub evaporation: f32,
    // [0, 10]
    pub radius: f32,
    // [0, 0.06]
    pub min_slope: f32,
    // 10?
    pub gravity: f32,
    // 64?
    pub max_steps: usize,
}

impl Default for DropSettings {
    fn default() -> Self {
        Self {
            inertia: 0.3,
            capacity: 8.0,
            deposition: 0.2,
            erosion: 0.7,
            evaporation: 0.02,
            radius: 4.0,
            min_slope: 0.01,
            gravity: 10.0,
            max_steps: 64,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct Drop {
    pos: Vec3,
    dir: Vec3,
    vel: f32,
    water: f32,
    sediment: f32,
    height: f32,
    triangle: [usize; 3],
    bary: [f32; 3],
}

pub struct World {
    pub heights: Vec<f32>,
    pub hardness: Vec<f32>,
    pub wetness: Vec<f32>,
    pub adjacent: Vec<ArrayVec<[usize; 6]>>,
    pub positions: Vec<Vec3>,
    pub subdivisions: usize,
    pub min_dist: f32,
    pub settings: DropSettings,
}

impl World {
    pub fn new(subdivisions: usize, settings: DropSettings) -> Self {
        let icosphere = IcoSphere::new(subdivisions, |_| {});
        let points = icosphere.raw_points();
        let indices = icosphere.get_all_indices();
        let adjacent = rehex::rehexed(&indices, points.len());

        let heights = Vec::new();
        let hardness = Vec::new();
        let wetness = vec![0.0; points.len()];
        let positions = points.iter().copied().map(Vec3::from).collect::<Vec<_>>();
        let min_dist = calculate_min_dist(&adjacent, &positions);
        println!("Max distance: {}", calculate_max_dist(&adjacent, &positions));
        println!("Min proj distance: {}", min_dist);

        Self {
            heights,
            hardness,
            wetness,
            adjacent,
            positions,
            subdivisions,
            min_dist,
            settings,
        }
    }

    pub fn fill_noise_heights(&mut self, opts: Opts) {
        self.heights = noisegen::sample_all_noise(&self.positions, opts);
    }

    pub fn fill_hardness(&mut self, opts: Opts) {
        self.hardness = noisegen::sample_all_noise(&self.positions, opts);
    }

    pub fn fill_wetness(&mut self) {
        self.wetness
            .iter_mut()
            .for_each(|x| *x = 0.0);

        let mut old_pos = 0;
        for i in 0..self.positions.len() {
            let mut pos = self.positions[i];
            let mut dir = Vec3::ZERO;
            for _ in 0..self.settings.max_steps {
                self.simulate_noneroding_step(&mut pos, &mut dir, &mut old_pos);
            }
        }
    }

    pub fn simulate_water_drop(&mut self, start: Vec3) {
        let start = start.normalize();
        assert!(start.is_normalized());

        let start_triangle = self.find_triangle(0, start);
        let start_bary = self.barycentric_coords(start, start_triangle);
        let start_height = self.height_at(start_triangle, start_bary);

        let mut drop = Drop {
            pos: start,
            dir: Vec3::ZERO,
            vel: 0.0,
            water: 1.0,
            sediment: 0.0,
            height: start_height,
            triangle: start_triangle,
            bary: start_bary,
        };

        for _ in 0..self.settings.max_steps {
            let gradient = self.gradient_at(drop.triangle, drop.bary);

            let new_dir = (drop.dir * self.settings.inertia - gradient * (1.0 - self.settings.inertia)).normalize_or_zero();

            let new_pos = (drop.pos + new_dir * self.min_dist).normalize();
            let new_triangle = self.find_triangle(drop.triangle[0], new_pos);
            let new_bary = self.barycentric_coords(new_pos, new_triangle);
            let new_height = self.height_at(new_triangle, new_bary);

            let height_diff = new_height - drop.height;

            let mut new_sediment = drop.sediment;

            if height_diff > 0.0 {
                let to_deposit = drop.sediment.min(height_diff);
                self.deposit(drop.triangle, drop.bary, to_deposit);
                new_sediment -= to_deposit;
            } else {
                let capacity = (-height_diff).max(self.settings.min_slope) * drop.vel * drop.water * self.settings.capacity;
                if drop.sediment > capacity {
                    new_sediment *= 1.0 - self.settings.deposition;
                    let to_deposit = drop.sediment * self.settings.deposition;
                    self.deposit(drop.triangle, drop.bary, to_deposit);
                } else {
                    let hardness = self.hardness_at(drop.triangle, drop.bary);
                    let to_erode = (capacity - drop.sediment) * hardness;
                    let to_erode = to_erode.min(-height_diff);
                    self.deposit(drop.triangle, drop.bary, -to_erode);
                    new_sediment += to_erode;
                }
            }

            let new_vel = (drop.vel * drop.vel + height_diff * self.settings.gravity).abs().sqrt();
            let new_water = drop.water * (1.0 - self.settings.evaporation);

            drop = Drop {
                pos: new_pos,
                dir: new_dir,
                vel: new_vel,
                water: new_water,
                sediment: new_sediment,
                height: new_height,
                triangle: new_triangle,
                bary: new_bary,
            };
        }
    }

    fn simulate_noneroding_step(&mut self, pos: &mut Vec3, dir: &mut Vec3, old_pos: &mut usize) {
        let triangle = self.find_triangle(*old_pos, *pos);
        let bary = self.barycentric_coords(*pos, triangle);
        self.add_wetness(triangle, bary);
        let gradient = self.gradient_at(triangle, bary);

        let new_dir = *dir * self.settings.inertia - gradient * (1.0 - self.settings.inertia);
        let new_pos = (*pos + new_dir.normalize_or_zero() * self.min_dist).normalize();

        *dir = new_dir;
        *pos = new_pos;
        *old_pos = triangle[0];
    }

    fn deposit(&mut self, triangle: [usize; 3], bary: [f32; 3], amount: f32) {
        for i in 0..3 {
            self.heights[triangle[i]] += bary[i] * amount;
        }
    }

    pub fn find_triangle(&self, mut guess: usize, point: Vec3) -> [usize; 3] {
        let mut guess_dot = self.positions[guess].dot(point);

        let mut adjacent = self.adjacent[guess];

        let mut max_idx = 0;
        let mut max_val = f32::NEG_INFINITY;

        for i in adjacent {
            let factor = self.positions[i].dot(point);
            if factor > max_val {
                max_idx = i;
                max_val = factor;
            }
        }

        while max_val > guess_dot {
            guess = max_idx;
            guess_dot = max_val;
            adjacent = self.adjacent[guess];

            match adjacent.len() {
                5 => for i in adjacent {
                    let factor = self.positions[i].dot(point);
                    if factor > max_val {
                        max_idx = i;
                        max_val = factor;
                    }
                },
                6 => for i in adjacent {
                    let factor = self.positions[i].dot(point);
                    if factor > max_val {
                        max_idx = i;
                        max_val = factor;
                    }
                },
                _ => unreachable!(),
            }
        }

        let mut around_dots = ArrayVec::<[f32; 6]>::new();
        around_dots.extend(
            adjacent
                .iter()
                .copied()
                .map(|x| self.positions[x].dot(point))
        );

        let mut max_idx = 0;
        for (idx, val) in around_dots.into_iter().enumerate().skip(1) {
            if val > around_dots[max_idx] {
                max_idx = idx;
            }
        }

        let next_idx = (max_idx + 1) % around_dots.len();
        let prev_idx = (max_idx + around_dots.len() - 1) % around_dots.len();

        if around_dots[next_idx] > around_dots[prev_idx] {
            [guess, self.adjacent[guess][max_idx], self.adjacent[guess][next_idx]]
        } else {
            [guess, self.adjacent[guess][prev_idx], self.adjacent[guess][max_idx]]
        }
    }

    pub fn barycentric_coords(&self, at: Vec3, triangle: [usize; 3]) -> [f32; 3] {
        calculate_barycentric_sphere(at, triangle.map(|x| self.positions[x]))
    }

    pub fn normalized_gradient(&self, at: usize) -> Vec3 {
        let me = self.positions[at];
        let height_me = self.heights[at];
        let around = self.adjacent[at];
        let mut sum = Vec3::ZERO;
        for i in around {
            let point = self.positions[i];
            let projected = point * point.dot(me).recip();
            let normalized = projected - me;
            let height = self.heights[i];
            let delta = height - height_me;
            sum += normalized * delta;
        }
        sum.normalize()
    }

    pub fn add_wetness(&mut self, triangle: [usize; 3], bary: [f32; 3]) {
        triangle
            .into_iter()
            .zip(bary.into_iter())
            .for_each(|(idx, val)| self.wetness[idx] += val);
    }

    pub fn gradient_at(&self, triangle: [usize; 3], bary: [f32; 3]) -> Vec3 {
        let sum = bary[0] * self.normalized_gradient(triangle[0]) +
            bary[1] * self.normalized_gradient(triangle[1]) +
            bary[2] * self.normalized_gradient(triangle[2]);

        sum.normalize_or_zero()
    }

    pub fn height_at(&self, triangle: [usize; 3], bary: [f32; 3]) -> f32 {
        bary[0] * self.heights[triangle[0]] +
            bary[1] * self.heights[triangle[1]] +
            bary[2] * self.heights[triangle[2]]
    }

    pub fn hardness_at(&self, triangle: [usize; 3], bary: [f32; 3]) -> f32 {
        bary[0] * self.hardness[triangle[0]] +
            bary[1] * self.hardness[triangle[1]] +
            bary[2] * self.hardness[triangle[2]]
    }
}

#[cfg(feature = "ij")]
mod ij {
    impl Copy for usize {}
    impl Copy for f32 {}
}

fn calculate_min_dist(adjacent: &[ArrayVec<[usize; 6]>], points: &[Vec3]) -> f32 {
    adjacent
        .iter()
        .enumerate()
        .map(|(idx_this, others)| {
            let pt = points[idx_this];
            others
                .iter()
                .map(|&neighbour| (points[neighbour] * points[neighbour].dot(pt) - pt).length())
                .min_by(|x, y| x.total_cmp(y))
                .unwrap()
        })
        .min_by(|x, y| x.total_cmp(y))
        .unwrap()
}

fn calculate_max_dist(adjacent: &[ArrayVec<[usize; 6]>], points: &[Vec3]) -> f32 {
    adjacent
        .iter()
        .enumerate()
        .map(|(idx_this, others)| {
            let pt = points[idx_this];
            others
                .iter()
                .map(|&neighbour| points[neighbour].distance(pt))
                .max_by(|x, y| x.total_cmp(y))
                .unwrap()
        })
        .max_by(|x, y| x.total_cmp(y))
        .unwrap()
}

fn calculate_barycentric_sphere(v: Vec3, p: [Vec3; 3]) -> [f32; 3] {
    // planar coordinates
    let [a, b, c] = p.map(|x| x * x.dot(v).recip());
    let ab = b - a;
    let ac = c - a;
    let cb = b - c;

    let va = v - a;
    let vb = v - b;

    let c = va.cross(ab);
    let b = va.cross(ac);
    let a = vb.cross(cb);

    let lengths = [a, b, c].map(Vec3::length);
    let total = lengths[0] + lengths[1] + lengths[2];
    lengths.map(|x| x / total)
}
