use std::cmp::min;

/// Find a fft length larger that is at least `minimum_size` in length that is made up of a limited set of factors.
/// The `once_factor` will be used exactly once, the multi factors may be used any number of times to reach the minimum.
fn fast_fft_len(
    minimum_len: usize,
    once_factor: usize,
    multi_factor1: usize,
    multi_factor2: usize,
) -> usize {
    assert!(once_factor >= 1);
    assert!(multi_factor1 > 1);
    assert!(multi_factor2 > 1);

    // Apply once factor
    let mut product = once_factor;

    // apply second factor until at or above minimum
    while product < minimum_len {
        product *= multi_factor2
    }

    // remove second factor one at a time then add enough first factor to reach minimum_size
    // repeat while tracking lowest viable product
    let mut best = product;
    loop {
        match product.cmp(&minimum_len) {
            std::cmp::Ordering::Less => product *= multi_factor1,
            std::cmp::Ordering::Equal => return product,
            std::cmp::Ordering::Greater => {
                best = min(best, product);
                if product % multi_factor2 != 0 {
                    return best;
                }
                product /= multi_factor2;
            }
        }
    }
}

/// Returns a new length greater than the lower limit that mostly consists of factors of 2 and 3, with up to one other factor less than 12.
///
/// While factors of 2 and 3 are optimal, factors below 12 are still reasonably fast.
pub fn fastish_fft_len(lower_limit: usize) -> usize {
    let mut x = lower_limit.next_power_of_two();
    for &alt in &[1, 5, 7, 11] {
        x = min(x, fast_fft_len(lower_limit, alt, 2, 3));
    }
    x
}

#[test]
fn factors23() {
    use std::fs::File;
    use std::io::prelude::*;
    use std::io::LineWriter;
    let file = File::create("factors23.txt").unwrap();
    let mut log = LineWriter::new(file);
    for i in 1..=1_000_000 {
        let x = fast_fft_len(i, 1, 2, 3);
        writeln!(log, "{} {}", i, x).unwrap();
    }
}

#[test]
fn factors23x() {
    use std::fs::File;
    use std::io::prelude::*;
    use std::io::LineWriter;
    let file = File::create("factors23x.txt").unwrap();
    let mut log = LineWriter::new(file);
    for i in 1..=1_000_000usize {
        let mut x = i.next_power_of_two();
        for &alt in &[1, 5, 7, 11] {
            x = min(x, fast_fft_len(i, alt, 2, 3));
        }
        writeln!(log, "{} {}", i, x).unwrap();
    }
}

#[test]
fn factors23xx() {
    use std::fs::File;
    use std::io::prelude::*;
    use std::io::LineWriter;
    let file = File::create("factors23xx.txt").unwrap();
    let mut log = LineWriter::new(file);
    for i in 1..=1_000_000usize {
        let mut x = i.next_power_of_two();
        for alt in [1, 5, 7, 11].iter().flat_map(|&e1| {
            [1, 5, 7, 11]
                .iter()
                .filter_map(move |&e2| if e2 >= e1 { Some(e1 * e2) } else { None })
        }) {
            x = min(x, fast_fft_len(i, alt, 2, 3));
        }
        writeln!(log, "{} {}", i, x).unwrap();
    }
}
