use std::cmp::min;

fn fast_fft_len(lower_limit: usize, single_factor: usize, multi_factor1: usize, multi_factor2: usize) -> usize {
    let mut product = single_factor;
    while product < lower_limit {
        product *= multi_factor2
    }
    let mut min = product;
    loop {
        if product > lower_limit {
            if product % multi_factor2 != 0 {
                return min
            }
            product /= multi_factor2;
        } else if product < lower_limit {
            product *= multi_factor1;
        } else {
            return product
        }
        if product > lower_limit && product < min {
            min = product;
        }
    }
}

/// Returns a new length greater than the lower limit that mostly consists of factors of 2 and 3, with up to one other factor less than 12.
///
/// While factors of 2 and 3 are optimal, factors below 12 are still reasonably fast.
pub fn fastish_fft_len(lower_limit: usize) -> usize {
    let mut x = lower_limit.next_power_of_two();
    for &alt in &[1, 5, 7, 11]{
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
        for &alt in &[1, 5, 7, 11]{
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
        for alt in [1, 5, 7, 11].iter().flat_map(|&e1| [1, 5, 7, 11].iter().filter_map(move |&e2| if e2 >= e1 {Some(e1*e2)} else {None})) {
            x = min(x, fast_fft_len(i, alt, 2, 3));
        }
        writeln!(log, "{} {}", i, x).unwrap();
    }
}