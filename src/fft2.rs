use ndarray::parallel::prelude::{IntoParallelIterator, ParallelIterator};
use ndarray::{s, Array2, ArrayView2, ArrayViewMut1, ArrayViewMut2, Axis, Zip};
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::{FftDirection, FftPlanner};

pub fn fft2(mut input: Array2<Complex<f64>>) -> Array2<Complex<f64>> {
    _fft2(input.view_mut(), FftDirection::Forward);
    input
}
pub fn ifft2(mut input: Array2<Complex<f64>>) -> Array2<Complex<f64>> {
    _fft2(input.view_mut(), FftDirection::Inverse);
    input
}
pub fn _fft2(mut input: ArrayViewMut2<Complex<f64>>, direction: FftDirection) {
    let mut planner = FftPlanner::new();
    let fft_row = planner.plan_fft(input.shape()[1], direction);
    let fft_col = planner.plan_fft(input.shape()[0], direction);
    let normalisation = 1.0 / ((input.shape()[0] * input.shape()[1]) as f64).sqrt();

    Zip::from(input.rows_mut()).into_par_iter().for_each_init(
        || vec![Zero::zero(); fft_row.get_inplace_scratch_len()],
        |scratch, mut row| {
            fft_row.process_with_scratch(row.0.as_slice_mut().unwrap(), scratch);
        },
    );

    Zip::from(input.columns_mut())
        .into_par_iter()
        .for_each_init(
            || {
                (
                    vec![Zero::zero(); fft_col.len()],
                    vec![Zero::zero(); fft_col.get_inplace_scratch_len()],
                )
            },
            |(temp, scratch), mut col| {
                debug_assert_eq!(col.0.len(), temp.len());
                unsafe {
                    for (i, col) in col.0.iter_mut().enumerate() {
                        *temp.get_unchecked_mut(i) = *col;
                    }
                    fft_col.process_with_scratch(temp, scratch);
                    for (i, col) in col.0.iter_mut().enumerate() {
                        *col = *temp.get_unchecked(i) * normalisation;
                    }
                }
            },
        );
}

/// performs a 2D fft where the 0th component is at the center rather than the normal right
/// removes the need for ifft_shift before and fft_shift after.
pub fn fft2c(mut input: Array2<Complex<f64>>) -> Array2<Complex<f64>> {
    _fft2c(input.view_mut(), FftDirection::Forward);
    input
}
/// performs a 2D ifft where the 0th component is at the center rather than the normal right
/// removes the need for ifft_shift before and fft_shift after.
pub fn ifft2c(mut input: Array2<Complex<f64>>) -> Array2<Complex<f64>> {
    _fft2c(input.view_mut(), FftDirection::Inverse);
    input
}

pub fn _fft2c(mut input: ArrayViewMut2<Complex<f64>>, direction: FftDirection) {
    let i0 = input.shape()[0];
    let i1 = input.shape()[1];

    let normalisation = 1.0 / ((input.shape()[0] * input.shape()[1]) as f64).sqrt();

    let mut planner = FftPlanner::new();
    let fft0 = planner.plan_fft(i0, direction);
    let fft1 = planner.plan_fft(i1, direction);
    // fft along axis1, iteration over axis0
    Zip::from(input.axis_iter_mut(Axis(0)))
        .into_par_iter()
        .for_each_init(
            || vec![Zero::zero(); fft1.get_inplace_scratch_len()],
            |scratch, input_row| {
                let mut input_row = input_row.0;
                ifft_shift_inplace(input_row.view_mut());
                fft1.process_with_scratch(input_row.as_slice_mut().unwrap(), scratch);
                fft_shift_inplace(input_row);
            },
        );

    // fft along axis0, iteration over axis1
    Zip::from(input.axis_iter_mut(Axis(1)))
        .into_par_iter()
        .for_each_init(
            || {
                (
                    vec![Zero::zero(); fft0.len()],
                    vec![Zero::zero(); fft0.get_inplace_scratch_len()],
                )
            },
            |(fft_buffer, scratch), input_col| {
                let mut input_col = input_col.0;
                let fft_buffer = fft_buffer.as_mut_slice();
                let half = i0 / 2;

                // construct input equivalent to ifft_shift followed by padding to resample size
                // the halves of the input are reverse compared to the hillenbrand tiling because the input has not yet been fft_shifted
                // this is also why half rounds down, as in an ifft
                unsafe {
                    let mut k = 0;
                    for &e in input_col.slice(s![half..]) {
                        *fft_buffer.get_unchecked_mut(k) = e;
                        k += 1;
                    }
                    for &e in input_col.slice(s![..half]) {
                        *fft_buffer.get_unchecked_mut(k) = e;
                        k += 1;
                    }
                }

                fft0.process_with_scratch(fft_buffer, scratch);

                // fft_shift and depad then multiply by write back
                unsafe {
                    let mut k = 0;
                    for e in input_col.slice_mut(s![half..]) {
                        *e = *fft_buffer.get_unchecked_mut(k) * normalisation;
                        k += 1;
                    }
                    for e in input_col.slice_mut(s![..half]) {
                        *e = *fft_buffer.get_unchecked_mut(k) * normalisation;
                        k += 1;
                    }
                }
            },
        );
}

/// Moves the origin (0, 0) to the "center" of the array (H/2, W/2)
///
/// For even array lengths, which have no center value, this moves the value to the next value after the center
pub fn fft2_shift_inplace(mut input: ArrayViewMut2<Complex<f64>>) {
    Zip::from(input.lanes_mut(Axis(1))).par_for_each(|row| {
        fft_shift_inplace(row);
    });

    Zip::from(input.lanes_mut(Axis(0))).par_for_each(|col| {
        fft_shift_inplace(col);
    });
}

/// Moves the "center" of the array (H/2, W/2) to the origin (0, 0)
///
/// Inverts fft_shift exactly, accounting for the asymmetry of even arrays
pub fn ifft2_shift_inplace(mut input: ArrayViewMut2<Complex<f64>>) {
    Zip::from(input.lanes_mut(Axis(1))).par_for_each(|row| {
        ifft_shift_inplace(row);
    });

    Zip::from(input.lanes_mut(Axis(0))).par_for_each(|col| {
        ifft_shift_inplace(col);
    });
}

/// Moves the origin (0) to the "center" of the array (N/2)
///
/// For even array lengths, which have no center value, this moves the value to the next value after the center
pub fn fft_shift_inplace(mut input: ArrayViewMut1<Complex<f64>>) {
    if input.len() % 2 == 0 {
        return fft_shift_even(input);
    }

    let len = input.len();
    let half = len / 2;

    let mut i = input.len();
    let mut j = half;
    let mut temp1 = input[half];
    for _ in 0..half {
        i -= 1;
        j -= 1;
        std::mem::swap(&mut temp1, &mut input[i]);

        std::mem::swap(&mut temp1, &mut input[j]);
    }
    input[half] = temp1;
}

/// Moves the "center" of the array (N/2) to the origin (0)
///
/// Inverts fft_shift exactly, accounting for the asymmetry of even arrays
pub fn ifft_shift_inplace(mut input: ArrayViewMut1<Complex<f64>>) {
    if input.len() % 2 == 0 {
        return fft_shift_even(input);
    }

    let len = input.len();
    let half = len / 2;

    let mut j = half + 1;
    let mut temp1 = input[half];
    for i in 0..half {
        std::mem::swap(&mut temp1, &mut input[i]);

        std::mem::swap(&mut temp1, &mut input[j]);

        //i += 1;
        j += 1;
    }
    input[half] = temp1;
}

fn fft_shift_even(mut input: ArrayViewMut1<Complex<f64>>) {
    let half = input.len() / 2;
    for i in 0..half {
        let temp = input[i];
        input[i] = input[i + half];
        input[i + half] = temp;
    }
}

// Double the size of each axis, adding zeros to the start and end
pub fn pad_zero_2D(input: ArrayView2<Complex<f64>>) -> Array2<Complex<f64>> {
    let m0 = input.shape()[0];
    let m1 = input.shape()[1];
    let mut out = Array2::zeros([m0 * 2, m1 * 2]);
    let slice = s![
        (m0 + 1) / 2..m0 + (m0 + 1) / 2,
        (m1 + 1) / 2..m1 + (m1 + 1) / 2
    ];
    let mut view = out.slice_mut(slice);
    view.assign(&input);
    out
}

// Halve the size of each axis, removing a quater of the values from the start and end
pub fn depad_2D(input: ArrayView2<Complex<f64>>) -> ArrayView2<Complex<f64>> {
    let m0 = input.shape()[0] / 2;
    let m1 = input.shape()[1] / 2;
    let slice = s![
        (m0 + 1) / 2..m0 + (m0 + 1) / 2,
        (m1 + 1) / 2..m1 + (m1 + 1) / 2
    ];
    input.slice_move(slice)
}

// Double the size of each axis, adding zeros to the start and end
pub fn pad_2D_to(input: ArrayView2<Complex<f64>>, out_shape: [usize; 2]) -> Array2<Complex<f64>> {
    let m0 = input.shape()[0];
    let m1 = input.shape()[1];
    let mut out = Array2::zeros(out_shape);
    let slice = s![
        out_shape[0] / 2 - m0 / 2..m0 + out_shape[0] / 2 - m0 / 2,
        out_shape[1] / 2 - m1 / 2..m1 + out_shape[1] / 2 - m1 / 2
    ];
    let mut view = out.slice_mut(slice);
    view.assign(&input);
    out
}

// // Halve the size of each axis, removing a quater of the values from the start and end
// pub fn depad_2D_to(
//     input: ArrayView2<Complex<f64>>,
//     out_shape: [usize; 2],
// ) -> ArrayView2<Complex<f64>> {
//     let m0 = input.shape()[0] / 2;
//     let m1 = input.shape()[1] / 2;
//     let slice = s![
//         out_shape[0] / 2 - m0 / 2..m0 + out_shape[0] / 2 - m0 / 2,
//         out_shape[1] / 2 - m1 / 2..m1 + out_shape[1] / 2 - m1 / 2
//     ];
//     input.slice_move(slice)
// }

// // Double the size of each axis, adding zeros to the start and end
// pub fn pad_to(input: ArrayView1<Complex<f64>>, out_shape: usize) -> Array1<Complex<f64>> {
//     let m0 = input.shape()[0];
//     let mut out = Array1::zeros(out_shape);
//     let slice = s![out_shape / 2 - m0 / 2..m0 + out_shape / 2 - m0 / 2,];
//     let mut view = out.slice_mut(slice);
//     view.assign(&input);
//     out
// }

// // Halve the size of each axis, removing a quater of the values from the start and end
// pub fn depad_1D(input: ArrayView1<Complex<f64>>) -> ArrayView1<Complex<f64>> {
//     let m0 = input.shape()[0] / 2;
//     let slice = s![
//         (m0 + 1) / 2..m0 + (m0 + 1) / 2,
//     ];
//     input.slice_move(slice)
// }

// // Halve the size of each axis, removing a quater of the values from the start and end
// pub fn depad_to(input: ArrayView1<Complex<f64>>, out_shape: usize) -> ArrayView1<Complex<f64>> {
//     let m0 = input.shape()[0] / 2;
//     let slice = s![out_shape / 2 - m0 / 2..m0 + out_shape / 2 - m0 / 2,];
//     input.slice_move(slice)
// }

#[cfg(test)]
mod tests {
    use super::{fft2, fft_shift_inplace, ifft2, ifft_shift_inplace};
    use ndarray::ArrayViewMut;
    use rustfft::num_complex::Complex;

    fn assert_eq_vecs(a: &[Complex<f64>], b: &[Complex<f64>]) {
        for (a, b) in a.iter().zip(b) {
            assert!((a - b).norm() < 1e-7, "{}", (a - b).norm());
        }
    }

    #[test]
    fn test_fft_shift_odd() {
        let mut input: Vec<Complex<f64>> = vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]
            .into_iter()
            .map(|x| Complex::new(x, 0.))
            .collect();
        let expected: Vec<Complex<f64>> = vec![6., 7., 8., 9., 1., 2., 3., 4., 5.]
            .into_iter()
            .map(|x| Complex::new(x, 0.))
            .collect();

        let input_view = ArrayViewMut::from_shape(9, &mut input).unwrap();
        fft_shift_inplace(input_view);

        assert_eq!(input, expected);
    }

    #[test]
    fn test_fft_shift_even() {
        let mut input: Vec<Complex<f64>> = vec![1., 2., 3., 4., 5., 6., 7., 8.]
            .into_iter()
            .map(|x| Complex::new(x, 0.))
            .collect();
        let expected: Vec<Complex<f64>> = vec![5., 6., 7., 8., 1., 2., 3., 4.]
            .into_iter()
            .map(|x| Complex::new(x, 0.))
            .collect();

        let input_view = ArrayViewMut::from_shape(8, &mut input).unwrap();
        fft_shift_inplace(input_view);

        assert_eq!(input, expected);
    }

    #[test]
    fn test_ifft_shift_odd() {
        let mut input: Vec<Complex<f64>> = vec![6., 7., 8., 9., 1., 2., 3., 4., 5.]
            .into_iter()
            .map(|x| Complex::new(x, 0.))
            .collect();
        let expected: Vec<Complex<f64>> = vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]
            .into_iter()
            .map(|x| Complex::new(x, 0.))
            .collect();

        let input_view = ArrayViewMut::from_shape(9, &mut input).unwrap();
        ifft_shift_inplace(input_view);

        assert_eq!(input, expected);
    }

    #[test]
    fn test_ifft_shift_even() {
        let mut input: Vec<Complex<f64>> = vec![5., 6., 7., 8., 1., 2., 3., 4.]
            .into_iter()
            .map(|x| Complex::new(x, 0.))
            .collect();
        let expected: Vec<Complex<f64>> = vec![1., 2., 3., 4., 5., 6., 7., 8.]
            .into_iter()
            .map(|x| Complex::new(x, 0.))
            .collect();

        let input_view = ArrayViewMut::from_shape(8, &mut input).unwrap();
        ifft_shift_inplace(input_view);

        assert_eq!(input, expected);
    }

    #[test]
    fn test_fft2() {
        let mut input: Vec<Complex<f64>> = vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]
            .into_iter()
            .map(|x| Complex::new(x, 0.))
            .collect();
        let input_view = ArrayViewMut::from_shape((3, 3), &mut input).unwrap();

        let output = fft2(input_view.to_owned());

        let expected = [
            Complex::new(15.0, 0.),
            Complex::new(-1.5, 0.866_025_403_333_333_3),
            Complex::new(-1.5, -0.866_025_403_333_333_3),
            Complex::new(-4.5, 2.59807621),
            Complex::new(0.0, 0.),
            Complex::new(0.0, 0.),
            Complex::new(-4.5, -2.59807621),
            Complex::new(0.0, 0.),
            Complex::new(0.0, 0.),
        ];
        assert_eq_vecs(&expected, output.as_slice().unwrap());
    }

    #[test]
    fn test_inverse_fft2() {
        let mut input: Vec<Complex<f64>> = vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]
            .into_iter()
            .map(|x| Complex::new(x, 0.))
            .collect();
        let input_view = ArrayViewMut::from_shape((3, 3), &mut input).unwrap();

        let output = fft2(input_view.to_owned());

        let output2 = ifft2(output);

        let expected: Vec<Complex<f64>> = vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]
            .into_iter()
            .map(|x| Complex::new(x, 0.))
            .collect();
        assert_eq_vecs(&expected, output2.as_slice().unwrap());
    }
}
