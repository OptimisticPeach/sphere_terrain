use tinyvec::ArrayVec;

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
enum RehexState {
    Empty,
    Clear,
    TwoTwo,
    ThreeTwo,
    TwoTwoTwo,
    Complete,
}

pub fn rehexed(indices: &[u32], len: usize) -> Vec<ArrayVec<[usize; 6]>> {
    let mut state = std::iter::repeat(RehexState::Empty).take(len).collect::<Vec<_>>();
    let mut result = std::iter::repeat(ArrayVec::new()).take(len).collect::<Vec<_>>();

    let mut insert = |a: u32, b: u32, c: u32| {
        let (a, b, c) = (a as usize, b as usize, c as usize);
        let state = &mut state[a];
        if let RehexState::Complete = state {
            return;
        }

        let result = &mut result[a];

        match state {
            RehexState::Empty => {
                result.extend([b, c]);
                *state = RehexState::Clear;
            }
            RehexState::Clear => {
                if result[result.len() - 1] == b {
                    if result[0] == c {
                        *state = RehexState::Complete;
                    } else {
                        result.push(c);
                        if result.len() == 6 {
                            *state = RehexState::Complete;
                        }
                    }
                } else if result[0] == c {
                    result.insert(0, b);
                    if result.len() == 6 {
                        *state = RehexState::Complete;
                    }
                } else {
                    *state = match result.len() {
                        2 => RehexState::TwoTwo,
                        3 => RehexState::ThreeTwo,
                        4 => RehexState::Complete,
                        _ => unreachable!(),
                    };
                    result.extend([b, c]);
                }
            }
            RehexState::TwoTwo => {
                if result[1] == b {
                    if result[2] == c {
                        *state = RehexState::Clear;
                    } else {
                        result.insert(2, c);
                        *state = RehexState::ThreeTwo;
                    }
                } else if result[0] == c {
                    if result[3] == b {
                        let temp = result[2];
                        result.pop();
                        result.pop();
                        result.insert(0, temp);
                        result.insert(1, b);
                        *state = RehexState::Clear;
                    } else {
                        result.insert(0, b);
                        *state = RehexState::ThreeTwo;
                    }
                } else if result[2] == c {
                    result.insert(0, b);
                    let t2 = result.swap_remove(2);
                    let t1 = result.swap_remove(1);
                    result.push(t1);
                    result.push(t2);
                    *state = RehexState::ThreeTwo;
                } else {
                    result.extend([b, c]);
                    *state = RehexState::TwoTwoTwo;
                }
            }
            RehexState::ThreeTwo => {
                if result[2] == b {
                    if result[3] == c {
                        *state = RehexState::Clear;
                    } else {
                        result.insert(3, c);
                        *state = RehexState::Complete;
                    }
                } else {
                    //if result[0] == c {
                    if result[4] == b {
                        result.pop();
                        let temp = result.pop().unwrap();
                        result.insert(0, b);
                        result.insert(0, temp);
                        *state = RehexState::Clear;
                    } else {
                        result.insert(0, b);
                        *state = RehexState::Complete;
                    }
                }
            }
            RehexState::TwoTwoTwo => {
                if (result[1] != b || result[2] != c)
                    && (result[3] != b || result[4] != c)
                    && (result[5] != b || result[0] != c)
                {
                    let t2 = result.swap_remove(3);
                    let t1 = result.swap_remove(2);
                    result.extend([t1, t2]);
                }
                *state = RehexState::Complete;
            }
            RehexState::Complete => unreachable!(),
        }
    };

    for chunk in indices.chunks_exact(3) {
        let &[a, b, c] = chunk else { unreachable!() };

        insert(a, b, c);
        insert(c, a, b);
        insert(b, c, a);
    }

    drop(insert);

    for (idx, &around) in result.iter().enumerate() {
        if around.contains(&idx) {
            panic!("idx {} contains itself: {:?}", idx, around);
        }
    }

    result
}
