use std::hint::black_box;
use sphere_terrain::*;
use sphere_terrain::noisegen::Opts;

fn main() {
    let mut world = World::new(80, DropSettings {
        inertia: 0.2,
        capacity: 1.0,
        deposition: 0.85,
        erosion: 0.78,
        evaporation: 0.27,
        radius: 10.0,
        min_slope: 0.01,
        gravity: 10.0,
        max_steps: 32,
    });
    
    world.fill_hardness(Opts {
        octaves: 10,
        hurst_exponent: 1.1,
        lacunarity: 2.299,
        max: 0.65,
        min: 0.0,
        sample_scale: 0.8,
        seed: 3,
        offset: Default::default(),
    });

    world.fill_noise_heights(Opts {
        octaves: 9,
        hurst_exponent: 1.1,
        lacunarity: 2.5794,
        max: 1.2,
        min: 0.7664,
        sample_scale: 0.8,
        seed: 0,
        offset: Default::default(),
    });

    for i in 0..720000 {
        let idx = i % world.heights.len();
        let pt = world.positions[idx];
        world.simulate_water_drop(pt);
    }

    black_box((&world.heights, &world.positions, &world.adjacent, &world.wetness));
}
