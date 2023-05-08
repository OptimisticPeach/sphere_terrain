#![feature(portable_simd)]

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
    lifetime: usize,
}

pub struct World {
    pub heights: Vec<f32>,
    pub hardness: Vec<f32>,
    pub wetness: Vec<f32>,
    pub slope: Vec<f32>,
    pub adjacent: Vec<ArrayVec<[usize; 6]>>,
    pub positions: Vec<Vec3>,
    pub subdivisions: usize,
    pub min_dist: f32,
    pub settings: DropSettings,

    pub debug_drop: Vec3,
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
        let slope = vec![0.0; points.len()];
        let positions = points.iter().copied().map(Vec3::from).collect::<Vec<_>>();
        let min_dist = calculate_min_dist(&adjacent, &positions);
        println!("Max distance: {}", calculate_max_dist(&adjacent, &positions));
        println!("Min proj distance: {}", min_dist);

        Self {
            heights,
            hardness,
            wetness,
            slope,
            adjacent,
            positions,
            subdivisions,
            min_dist,
            settings,
            debug_drop: Vec3::ZERO,
        }
    }

    pub fn fill_noise_heights(&mut self, opts: Opts) {
        self.heights = noisegen::sample_all_noise(&self.positions, opts);
    }

    pub fn fill_hardness(&mut self, opts: Opts) {
        self.hardness = noisegen::sample_all_noise(&self.positions, opts);
    }

    pub fn simulate_water_drop(&mut self, start: Vec3, points: &mut Vec<Vec3>) {
        // let start = self.debug_drop.normalize();
        let start = start.normalize();
        assert!(start.is_normalized());

        let start_triangle = self.find_triangle(0, start);
        let start_bary = self.barycentric_coords(start, start_triangle);
        let start_height = self.height_at(start_triangle, start_bary);

        let mut alloc = Vec::new();
        let mut drop = Drop {
            pos: start,
            dir: Vec3::ZERO,
            vel: 0.0,
            water: 1.0,
            sediment: 0.0,
            height: start_height,
            triangle: start_triangle,
            bary: start_bary,
            lifetime: 0,
        };

        while self.simulate_water_step(&mut drop, &mut alloc, points) {}

        // println!("{:?}", drop.sediment);
    }

    fn simulate_water_step(&mut self, drop: &mut Drop, alloc: &mut Vec<(usize, f32)>, points: &mut Vec<Vec3>) -> bool {
        let gradient = self.gradient_at(drop.triangle, drop.bary);

        let new_dir = drop.dir * self.settings.inertia - gradient * (1.0 - self.settings.inertia);

        let new_pos = (drop.pos + new_dir.normalize_or_zero() * self.min_dist).normalize();
        let new_triangle = self.find_triangle(drop.triangle[0], new_pos);
        let new_bary = self.barycentric_coords(new_pos, new_triangle);
        let new_height = self.height_at(new_triangle, new_bary);

        // println!("gradient: {gradient:?}, pos: {new_pos:?}, dir: {new_dir:?}, tri: {new_triangle:?}, bary: {new_bary:?}, height: {new_height:?}");

        // println!("dir: {:?}, pos: {:?}, triangle: {:?}, bary: {:?}, height: {:?}", new_dir, new_pos, new_triangle, new_bary, new_height);

        let height_diff = new_height - drop.height;

        let mut new_sediment = drop.sediment;

        // println!("height diff: {:?}", height_diff);
        if height_diff > 0.0  {
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
                // println!("erosion: {}, to_erode: {}, height_diff: {}", self.settings.erosion, to_erode, height_diff);
                let to_erode = to_erode.min(-height_diff);
                self.erode(drop.pos, drop.triangle[0],to_erode, alloc);
                new_sediment += to_erode;
            }
        }

        let new_vel = (drop.vel * drop.vel + height_diff * self.settings.gravity).abs().sqrt();
        let new_water = drop.water * (1.0 - self.settings.evaporation);
        let new_lifetime = drop.lifetime + 1;

        *drop = Drop {
            pos: new_pos,
            dir: new_dir,
            vel: new_vel,
            water: new_water,
            sediment: new_sediment,
            height: new_height,
            triangle: new_triangle,
            bary: new_bary,
            lifetime: new_lifetime,
        };

        drop.lifetime < self.settings.max_steps
    }

    fn deposit(&mut self, triangle: [usize; 3], bary: [f32; 3], amount: f32) {
        // println!("Depositing {:?} amount", amount);
        for i in 0..3 {
            self.heights[triangle[i]] += bary[i] * amount;
        }
    }

    fn erode(&mut self, at: Vec3, start: usize, amount: f32, alloc: &mut Vec<(usize, f32)>) {
        // println!("eroding {}", amount);
        let threshold = self.settings.radius * self.min_dist;
        let mut acc = threshold - self.positions[start].dot(at).recip();
        alloc.push((start, acc));

        let mut begin_recurse = 0;
        let mut end_recurse = alloc.len();

        while begin_recurse != end_recurse {
            for i in begin_recurse..end_recurse {
                'outer: for neighbour in self.adjacent[alloc[i].0] {
                    for &(found, _) in alloc.iter().rev() {
                        if neighbour == found {
                            continue 'outer;
                        }
                    }
                    let scale = self.positions[neighbour].dot(at).recip();
                    let dist = (scale * scale - 1.0).sqrt();
                    let factor = threshold - dist;
                    // let actual_dist = at.distance(self.positions[neighbour] * scale);
                    // println!("factor {} dist {} scale {} actual {} diff_dist: {}", factor, dist, scale, actual_dist, dist - actual_dist);
                    if factor > 0.0 {
                        alloc.push((neighbour, factor));
                        acc += factor;
                    }
                }
            }
            begin_recurse = end_recurse;
            end_recurse = alloc.len();
        }

        let norm_factor = acc.recip() * amount;
        // println!("alloc: {:?}", alloc.len());
        alloc
            .drain(..)
            .for_each(|(idx, factor)| {
                self.heights[idx] -= factor * norm_factor;
            });
    }

    pub fn find_triangle(&self, mut guess: usize, point: Vec3) -> [usize; 3] {
        let mut guess_dot = self.positions[guess].dot(point);

        let mut adjacent = self.adjacent[guess];

        let mut around_dots = Vec::with_capacity(6);
        around_dots.extend(
            adjacent
                .iter()
                .copied()
                .map(|x| self.positions[x].dot(point))
        );

        let (mut max_idx, mut max_val) = calculate_max(&around_dots);

        while max_val > guess_dot {
            // println!("guess dot: {:?}, idx: {}", guess_dot, guess);
            guess = adjacent[max_idx];
            guess_dot = max_val;

            adjacent = self.adjacent[guess];

            around_dots.clear();
            around_dots.extend(
                adjacent
                    .iter()
                    .copied()
                    .map(|x| self.positions[x].dot(point))
            );
            (max_idx, max_val) = calculate_max(&around_dots);
        }

        // }

        // println!("Values: {:?}, guess_dot: {}, max_val: {}, idx: {}\nIndices {:?}", around_dots, guess_dot, max_val, self.adjacent[guess][max_idx], self.adjacent[guess]);

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
        let around = self.adjacent[at];
        let around_points = around.into_iter().map(|x| self.positions[x]);
        let projected_points = around_points.map(|x| x * x.dot(me).recip());
        let normalized_points = projected_points.map(|x| x - me);
        let heights = around.into_iter().map(|x| self.heights[x]);
        let height_me = self.heights[at];
        let delta_heights = heights.map(|x| x - height_me);
        std::iter::zip(
            normalized_points,
            delta_heights,
        )
            .map(|(x, y)| x * y)
            .sum::<Vec3>()
            .normalize()
    }

    pub fn gradient_at(&self, triangle: [usize; 3], bary: [f32; 3]) -> Vec3 {
        let sum = triangle.into_iter()
            .zip(bary)
            .map(|(x, y)| self.normalized_gradient(x) * y)
            .sum::<Vec3>();

        sum.normalize_or_zero()
    }

    pub fn height_at(&self, triangle: [usize; 3], bary: [f32; 3]) -> f32 {
        triangle.iter()
            .zip(bary)
            .map(|(x, y)| self.heights[*x] * y)
            .sum()
    }

    pub fn hardness_at(&self, triangle: [usize; 3], bary: [f32; 3]) -> f32 {
        triangle.iter()
            .zip(bary)
            .map(|(x, y)| self.hardness[*x] * y)
            .sum()
    }
}

// impl Copy for usize {}
// impl Copy for f32 {}

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

fn calculate_max(items: &[f32]) -> (usize, f32) {
    items
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, x), (_, y)| x.total_cmp(y))
        .unwrap()
}

// #[inline]
// fn calculate_max(items: &[f32]) -> (usize, f32) {
//     let mut max_val = items[0];
//     let mut max_idx = 0;
//
//     items[1..]
//         .iter()
//         .enumerate()
//         .for_each(|(idx, &val)| {
//             if max_val < val {
//                 max_idx = idx;
//                 max_val = val;
//             }
//         });
//
//     (max_idx, max_val)
// }

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
    // println!("area_triangle: {}, divided_area: {}", area_triangle, divided_area);
    // println!("lengths: {:?}", lengths);
    // println!("total: {lengths:?}");
    lengths.map(|x| x / total)
}
