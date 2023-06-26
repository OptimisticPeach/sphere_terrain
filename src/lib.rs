#![feature(portable_simd)]

use crate::noisegen::Opts;
use atomic_float::AtomicF32;
use glam::Vec3;
use hexasphere::shapes::IcoSphere;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::atomic::Ordering;
use tinyvec::ArrayVec;

pub mod noisegen;
mod rehex;

/// Settings for each simulated drop.
///
/// If you're not sure where to begin, `Default::default` for this
/// structure has some nice defaults.
#[derive(Serialize, Deserialize, Copy, Clone, Debug, PartialEq)]
pub struct DropSettings {
    /// The inertia a drop feels when it is prompted (by way of terrain gradient)
    /// to change direction. 0 means the drop will always go with the flow of the
    /// terrain, and 1 means the drop will never change direction.
    ///
    /// Reasonable values lie within `[0, 1]`.
    pub inertia: f32,
    /// Sediment capacity factor of the drop. Small changes have drastic effects,
    /// and I'd recommend you keep it at 1.0.
    pub capacity: f32,
    /// Controls how much sediment a drop will deposit when its reached more than
    /// what it can carry.
    ///
    /// Reasonable values lie within `[0, 1]`.
    pub deposition: f32,
    /// Dictates much erosion will happen as a drop moves through the terrain.
    ///
    /// Reasonable values lie within `[0, 1]`.
    pub erosion: f32,
    /// As drops move, they evaporate exponentially (and hence their capacity
    /// decreases). At each step, their water content is multiplied by `1.0 - evaporation`.
    ///
    /// Reasonable values lie within `[0, 0.5]`.
    pub evaporation: f32,
    /// If a drop reaches a plateau, it will simply not erode. This value will
    /// set a minimum amount of erosion to occur.
    ///
    /// Reasonable values lie within `[0, 0.06]`.
    pub min_slope: f32,
    /// Constant of gravity.
    ///
    /// Reasonable value is around 10.
    pub gravity: f32,
    /// Determines the number of steps each drop is simulated for.
    ///
    /// Reasonable values lie around 20 or so.
    pub max_steps: usize,
}

impl Default for DropSettings {
    fn default() -> Self {
        Self {
            inertia: 0.8,
            capacity: 1.0,
            deposition: 0.23,
            erosion: 0.001,
            evaporation: 0.27,
            min_slope: 0.01,
            gravity: 9.81,
            max_steps: 30,
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

/// All the data associated with the world.
///
/// All `Vec` fields are indexed by indices provided by
/// the `adjacent` field.
///
/// Floating point values are stored as [`AF32`](AF32) so that
/// atomic operations may be performed and multithreading may
/// occur.
pub struct World {
    /// The height of each tile.
    pub heights: Vec<AF32>,

    pub delta_height: Vec<AF32>,

    /// The hardness (local "erosion" factor) of each tile.
    pub hardness: Vec<f32>,
    /// A measure of how wet a tile is (how much water goes over it)
    pub wetness: Vec<AF32>,
    /// A measure of how likely a tile is to be a river.
    pub river_likeliness: Vec<AF32>,
    /// Which tiles lie adjacent to each tile.
    pub adjacent: Vec<ArrayVec<[usize; 6]>>,
    /// Positions of the base sphere (unit vectors).
    pub positions: Vec<Vec3>,
    /// Number of subdivisions.
    pub subdivisions: usize,
    /// Minimum "distance" between two vectors in their respective projected
    /// planes tangent to the sphere.
    pub min_dist: f32,
    /// Settings for each drop.
    pub settings: DropSettings,
}

impl World {
    /// Creates a new world with the requested drop settings and subdivision count.
    pub fn new(subdivisions: usize, settings: DropSettings) -> Self {
        let icosphere = IcoSphere::new(subdivisions, |_| {});
        let points = icosphere.raw_points();
        let indices = icosphere.get_all_indices();
        let adjacent = rehex::rehexed(&indices, points.len());

        let heights = Vec::new();
        let hardness = Vec::new();
        let wetness = (0..points.len()).map(|_| AF32::new(0.0)).collect();
        let rivers = (0..points.len()).map(|_| AF32::new(0.0)).collect();
        let positions = points.iter().copied().map(Vec3::from).collect::<Vec<_>>();
        let min_dist = calculate_min_dist(&adjacent, &positions);

        Self {
            heights,
            delta_height: vec![AF32::new(0.0); points.len()],
            hardness,
            wetness,
            river_likeliness: rivers,
            adjacent,
            positions,
            subdivisions,
            min_dist,
            settings,
        }
    }

    /// Fills the world's height data with FBM noise generated from the options.
    pub fn fill_noise_heights(&mut self, opts: Opts) {
        self.heights = noisegen::sample_all_noise(&self.positions, opts)
            .into_iter()
            .map(|x| AF32::new(x))
            .collect();
    }

    /// Fills the world's hardness data with FBM noise generated from the options.
    ///
    /// Reasonable values for this should lie between 0 and 1 (with 1 preventing
    /// any erosion from occurring).
    pub fn fill_hardness(&mut self, opts: Opts) {
        self.hardness = noisegen::sample_all_noise(&self.positions, opts);
    }

    /// Fills the world's wetness and river values by running a non-eroding simulation
    /// from every tile on the world.
    ///
    /// River values are only added if a point can reach a source of water (height <
    /// 1.0) within `steps` steps.
    ///
    /// - Use the evaporation factor to determine how long runoffs produce wetness for.
    /// - Use the steps parameter to tweak how long steps will be taken to determine if
    /// a path is a viable river.
    /// - The inertia factor here is identical in function to the one in `DropSettings`,
    /// and is used instead of the `DropSettings` inertia. (This allows the look and
    /// feel of the terrain to be tweaked).
    pub fn fill_wetness(&mut self, evaporation: f32, inertia: f32, steps: usize) {
        self.wetness.iter_mut().for_each(|x| *x = AF32::new(0.0));
        self.river_likeliness.iter_mut().for_each(|x| *x = AF32::new(0.0));

        (0..self.positions.len())
            .into_par_iter()
            .for_each(|i: usize| {
                let mut old_pos = 0;
                let mut river_pos = Vec::with_capacity(self.settings.max_steps);

                let mut is_river = false;
                let mut pos = self.positions[i];
                let mut dir = Vec3::ZERO;
                let mut water = 1.0;
                for _ in 0..steps {
                    let triangle = self.find_triangle(old_pos, pos);
                    let bary = self.barycentric_coords(pos, triangle);
                    is_river |= self.heights[triangle[0]].load() < 1.0;
                    river_pos.push((triangle, bary));
                    self.add_wetness(triangle, bary, water);
                    let gradient = self.gradient_at(triangle, bary);

                    let new_dir =
                        (dir * inertia - gradient * (1.0 - inertia))
                            .normalize_or_zero();

                    // Ensures that dir remains on the plane orthogonal to pos.
                    let (new_dir, new_pos) = apply_rotation_by(new_dir * self.min_dist, pos);

                    dir = new_dir;
                    pos = new_pos;
                    old_pos = triangle[0];
                    water *= 1.0 - evaporation;
                }
                if is_river {
                    for (triangle, bary) in river_pos.iter().copied() {
                        self.add_river(triangle, bary);
                    }
                }
                river_pos.clear();
            });
    }

    /// Simulates `n` tile/node-centered eroding drops on the world.
    ///
    /// This does one drop on each tile in "order", so that each tile will get the same
    /// number of drops landing on it originally.
    ///
    /// `offset` dictates at what index to begin to simulate drops. This is used to avoid
    /// repeatedly simulating drops at the same area (say you simulate 100 drops 20 times
    /// with offset 0; you'd get all 2000 drops spawning on the same 100 tiles).
    pub fn simulate_node_centered_drops(&self, drop_showers: usize, blur_counts: usize) {
        self.delta_height.iter().for_each(|x| x.store(0.0));

        if blur_counts == 0 {
            for _ in 0..drop_showers {
                self.positions
                    .par_iter()
                    .for_each(|&x| {
                        // self.simulate_water_drop(x, &self.heights);
                        self.simulate_water_drop(x, &self.delta_height);
                    });

                self.delta_height
                    .iter()
                    .map(|x| x.load())
                    .zip(self.heights.iter())
                    .for_each(|(x, y)| { y.fetch_add(x); });
            }

            return;
        }

        let mut data_delta = vec![AF32::new(0.0); self.adjacent.len()];
        let mut blurred = vec![AF32::new(0.0); self.adjacent.len()];
        for _ in 0..drop_showers {
            self.positions
                .par_iter()
                .for_each(|&x| {
                    self.simulate_water_drop(x, &data_delta);
                });

            for _ in 0..blur_counts - 1 {
                self.blur_apply(&data_delta, &blurred);
                std::mem::swap(&mut data_delta, &mut blurred);
                blurred.fill(AF32::new(0.0));
            }
            self.blur_apply(&data_delta, &self.heights);
            self.blur_apply(&data_delta, &self.delta_height);
        }
    }

    pub fn blur_apply(&self, from: &[AF32], to: &[AF32]) {
        for (index, (adjacent, into)) in self.adjacent.iter().zip(to.iter()).enumerate() {
            let my_pos = self.positions[index];
            let mut total = 1.0;
            let coeffs = adjacent
                .iter()
                .map(|x| (self.positions[*x].dot(my_pos) - 1.0).exp())
                .collect::<ArrayVec<[f32; 6]>>();
            total += coeffs.into_iter().sum::<f32>();
            let mut result = from[index].load();
            adjacent
                .iter()
                .zip(coeffs.into_iter())
                .for_each(|(x, coeff)| {
                    result += from[*x].load() * coeff;
                });
            result /= total;
            into.fetch_add(result);
        }
    }

    /// Simulates an eroding water drop starting in the direction of `start`.
    ///
    /// `start` must not be zero nor infinite.
    pub fn simulate_water_drop(&self, start: Vec3, data_delta: &[AF32]) {
        let start = start.normalize();
        assert!(!start.is_nan());

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

            let new_dir = (drop.dir * self.settings.inertia
                - gradient * (1.0 - self.settings.inertia))
                .normalize_or_zero();

            // Ensures that dir remains on the plane orthogonal to pos.
            let (new_dir, new_pos) = apply_rotation_by(new_dir * self.min_dist, drop.pos);
            let new_triangle = self.find_triangle(drop.triangle[0], new_pos);
            let new_bary = self.barycentric_coords(new_pos, new_triangle);
            let new_height = self.height_at(new_triangle, new_bary);

            let height_diff = new_height - drop.height;

            let mut new_sediment = drop.sediment;

            if height_diff > 0.0 {
                let to_deposit = drop.sediment.min(height_diff);
                self.deposit(drop.triangle, drop.bary, to_deposit, data_delta);
                new_sediment -= to_deposit;
            } else {
                let capacity = (-height_diff).max(self.settings.min_slope)
                    * drop.vel
                    * drop.water
                    * self.settings.capacity;
                if drop.sediment > capacity {
                    new_sediment *= 1.0 - self.settings.deposition;
                    let to_deposit = drop.sediment * self.settings.deposition;
                    self.deposit(drop.triangle, drop.bary, to_deposit, data_delta);
                } else {
                    let hardness = self.hardness_at(drop.triangle, drop.bary);
                    let to_erode = (capacity - drop.sediment) * hardness * self.settings.erosion;
                    let to_erode = to_erode.min(-height_diff);
                    self.deposit(drop.triangle, drop.bary, -to_erode, data_delta);
                    new_sediment += to_erode;
                }
            }

            let new_vel = (drop.vel * drop.vel + height_diff * self.settings.gravity)
                .abs()
                .sqrt();
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

    /// Determines the triangle (trio of nodes) which contains `point`.
    pub fn find_triangle(&self, mut guess: usize, point: Vec3) -> [usize; 3] {
        if guess >= self.positions.len() {
            guess = 0;
        }
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
                5 => {
                    for i in adjacent {
                        let factor = self.positions[i].dot(point);
                        if factor > max_val {
                            max_idx = i;
                            max_val = factor;
                        }
                    }
                }
                6 => {
                    for i in adjacent {
                        let factor = self.positions[i].dot(point);
                        if factor > max_val {
                            max_idx = i;
                            max_val = factor;
                        }
                    }
                }
                _ => unreachable!(),
            }
        }

        let mut around_dots = ArrayVec::<[f32; 6]>::new();
        around_dots.extend(
            adjacent
                .iter()
                .copied()
                .map(|x| self.positions[x].dot(point)),
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
            [
                guess,
                self.adjacent[guess][max_idx],
                self.adjacent[guess][next_idx],
            ]
        } else {
            [
                guess,
                self.adjacent[guess][prev_idx],
                self.adjacent[guess][max_idx],
            ]
        }
    }

    /// Computes the spherical barycentric coordinates for a point `at` relative to a triangle.
    ///
    /// The results are only sensible if `at` is normalized.
    pub fn barycentric_coords(&self, at: Vec3, triangle: [usize; 3]) -> [f32; 3] {
        calculate_barycentric_sphere(at, triangle.map(|x| self.positions[x]))
    }

    /// Finds the normalized gradient at a node.
    pub fn normalized_gradient(&self, at: usize) -> Vec3 {
        let me = self.positions[at];
        let height_me = self.heights[at].load();
        let around = self.adjacent[at];
        let mut sum = Vec3::ZERO;
        for i in around {
            let point = self.positions[i];
            let projected = point * point.dot(me).recip();
            let normalized = projected - me;
            let height = self.heights[i].load();
            let delta = height - height_me;
            sum += normalized * delta;
        }
        sum.normalize()
    }

    /// Deposits `amount` of sediment in the `triangle` at the point specified
    /// by `bary`.
    pub fn deposit(&self, triangle: [usize; 3], bary: [f32; 3], amount: f32, into: &[AF32]) {
        for i in 0..3 {
            into[triangle[i]].fetch_add(bary[i] * amount);
        }
    }

    /// Notes that the position depicted by `triangle` and `bary` is a river.
    pub fn add_river(&self, triangle: [usize; 3], bary: [f32; 3]) {
        for i in 0..3 {
            self.river_likeliness[triangle[i]].fetch_add(bary[i]);
        }
    }

    /// Deposits `amount` of wetness in the `triangle` at the point specified
    /// by `bary`.
    pub fn add_wetness(&self, triangle: [usize; 3], bary: [f32; 3], amount: f32) {
        for i in 0..3 {
            self.wetness[triangle[i]].fetch_add(bary[i] * amount);
        }
    }

    /// Finds the gradient at a point.
    pub fn gradient_at(&self, triangle: [usize; 3], bary: [f32; 3]) -> Vec3 {
        let sum = bary[0] * self.normalized_gradient(triangle[0])
            + bary[1] * self.normalized_gradient(triangle[1])
            + bary[2] * self.normalized_gradient(triangle[2]);

        sum.normalize_or_zero()
    }

    /// Finds the height at a point.
    pub fn height_at(&self, triangle: [usize; 3], bary: [f32; 3]) -> f32 {
        bary[0] * self.heights[triangle[0]].load()
            + bary[1] * self.heights[triangle[1]].load()
            + bary[2] * self.heights[triangle[2]].load()
    }

    /// Finds the hardness at a point.
    pub fn hardness_at(&self, triangle: [usize; 3], bary: [f32; 3]) -> f32 {
        bary[0] * self.hardness[triangle[0]]
            + bary[1] * self.hardness[triangle[1]]
            + bary[2] * self.hardness[triangle[2]]
    }
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

fn apply_rotation_by(vel: Vec3, pos: Vec3) -> (Vec3, Vec3) {
    let sum = vel + pos;
    let alpha = sum.length();
    let resulting_pos = sum / alpha;
    let resulting_vel = resulting_pos - alpha * pos;
    (resulting_vel, resulting_pos)
}

pub struct AF32(pub AtomicF32);

impl Clone for AF32 {
    fn clone(&self) -> Self {
        Self::new(self.load())
    }
}

impl AF32 {
    #[inline]
    pub const fn new(float: f32) -> Self {
        Self(AtomicF32::new(float))
    }

    #[inline]
    pub fn get_mut(&mut self) -> &mut f32 {
        self.0.get_mut()
    }

    #[inline]
    pub fn into_inner(self) -> f32 {
        self.0.into_inner()
    }

    #[inline]
    pub fn load(&self) -> f32 {
        self.0.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn store(&self, value: f32) {
        self.0.store(value, Ordering::Relaxed)
    }

    #[inline]
    pub fn swap(&self, new_value: f32) -> f32 {
        self.0.swap(new_value, Ordering::Relaxed)
    }

    #[inline]
    pub fn compare_and_swap(&self, current: f32, new: f32) -> f32 {
        self.0.compare_and_swap(current, new, Ordering::Relaxed)
    }

    #[inline]
    pub fn compare_exchange(&self, current: f32, new: f32) -> Result<f32, f32> {
        self.0
            .compare_exchange(current, new, Ordering::Relaxed, Ordering::Relaxed)
    }

    #[inline]
    pub fn compare_exchange_weak(&self, current: f32, new: f32) -> Result<f32, f32> {
        self.0
            .compare_exchange_weak(current, new, Ordering::Relaxed, Ordering::Relaxed)
    }

    #[inline]
    pub fn fetch_update<F>(&self, update: F) -> Result<f32, f32>
    where
        F: FnMut(f32) -> Option<f32>,
    {
        self.0
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, update)
    }

    #[inline]
    pub fn fetch_add(&self, val: f32) -> f32 {
        self.0.fetch_add(val, Ordering::Relaxed)
    }

    #[inline]
    pub fn fetch_sub(&self, val: f32) -> f32 {
        self.0.fetch_sub(val, Ordering::Relaxed)
    }

    #[inline]
    pub fn fetch_abs(&self) -> f32 {
        self.0.fetch_abs(Ordering::Relaxed)
    }

    #[inline]
    pub fn fetch_neg(&self) -> f32 {
        self.0.fetch_neg(Ordering::Relaxed)
    }

    #[inline]
    pub fn fetch_min(&self, value: f32) -> f32 {
        self.0.fetch_min(value, Ordering::Relaxed)
    }

    #[inline]
    pub fn fetch_max(&self, value: f32) -> f32 {
        self.0.fetch_max(value, Ordering::Relaxed)
    }
}
