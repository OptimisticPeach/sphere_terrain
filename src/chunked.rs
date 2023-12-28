use std::collections::VecDeque;
use glam::Vec3;
use rand::{Rng, thread_rng};
use tinyvec::ArrayVec;

/// Chunks the world into chunks of at most size `size`.
/// Returns the boundaries
pub fn make_chunks(adjacent: &mut [ArrayVec<[usize; 6]>], points: &mut [Vec3], size: usize) -> Vec<usize> {
    // points in queue to be added to the chunk
    let mut queue = VecDeque::new();
    // how many points have we processed total?
    let mut processed = 0;

    // how many points in this chunk?
    let mut in_chunk = 0;

    let mut rng = thread_rng();

    let mut chunks = Vec::new();

    while processed != adjacent.len() {
        queue.clear();

        queue.push_back(rng.gen_range(processed..adjacent.len()));

        while !queue.is_empty() && in_chunk < size {
            let next = queue.pop_front().unwrap();

            queue.extend(
                adjacent[next].iter().copied().filter(|&x| x > processed)
            );

            swap(adjacent, processed, next);
            points.swap(processed, next);
            processed += 1;
            in_chunk += 1;
        }

        chunks.push(processed);
    }

    chunks
}

pub fn swap(adjacent: &mut [ArrayVec<[usize; 6]>], i: usize, j: usize) {
    if i == j {
        return;
    }

    // Collect all the neighbours which will be affected
    let mut surroundings_to_process = adjacent[i]
        .iter()
        .copied()
        .chain(adjacent[j].iter().copied())
        .collect::<ArrayVec<[usize; 12]>>();

    let first_surroundings = &surroundings_to_process[..adjacent[i].len()];

    // Only dedup neighbours if we need to
    if adjacent[i].contains(&j) ||
        first_surroundings
            .iter()
            .any(|&x| adjacent[x].contains(&j)) {

        // Dedup neighbours in case some are shared
        // A linear search is likely the fastest.
        for i in (0..surroundings_to_process.len()).rev() {
            for j in 0..i {
                if surroundings_to_process[j] == surroundings_to_process[i] {
                    surroundings_to_process.remove(i);
                    break;
                }
            }
        }
    }

    // Actually swap what i and j's neighbours are pointing to
    // j's neighbours will now point to i instead of j and similarly for i
    for surrounding in surroundings_to_process {
        for elem in &mut adjacent[surrounding] {
            if *elem == i {
                *elem = j;
            } else if *elem == j {
                *elem = i;
            }
        }
    }

    // Swap what neighbours i and j think they have
    adjacent.swap(i, j);
}
