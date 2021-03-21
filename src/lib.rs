#![allow(non_snake_case)]

use crate::fft2::{
    depad_2D, fft2, fft2_shift_inplace, fft2c, fft_shift_inplace, ifft2, ifft2_shift_inplace,
    ifft2c, ifft_shift_inplace, pad_2D_to, pad_zero_2D,
};
use find_fast_number::fastish_fft_len;
use ndarray::parallel::prelude::{IntoParallelIterator, ParallelIterator};
use ndarray::{aview_mut1, s, Array2, Array3, ArrayView2, Axis, Zip};
use num_complex::Complex;
use num_integer::gcd;
use rustfft::num_traits::Zero;
use rustfft::{FftDirection, FftPlanner};
use std::cmp::{max, min};
use std::f64::consts::PI;
use unchecked_index::get_unchecked_mut;
mod fft2;
mod find_fast_number;
pub mod mask;

/// Returns the minimum resample size that meets ASM sampling critera as well as the resampling factor for each axis.
/// Resample shape is never smaller than the input array shape
///
/// * `array_shape` - The shape of the input array
/// * `pitch` - The sample pitch of the input array
/// * `lambda` - The (minimum) wavelength of interest in the focal system
/// * `fl` - focal length
/// * `oversample_factor` - A multiplier applied the the minimum nyquist sampling. A typical value would be `1.1`.
pub fn resample_shape_min(
    array_shape: &[usize],
    pitch: (f64, f64),
    lambda: f64,
    fl: f64,
    oversample_factor: f64,
) -> ([usize; 2], [f64; 2]) {
    let na0 = na(pitch.0 * array_shape[0] as f64, fl, 1.0);
    let na1 = na(pitch.1 * array_shape[1] as f64, fl, 1.0);
    let required_pitch0 = 0.5 * lambda / na0;
    let required_pitch1 = 0.5 * lambda / na1;

    let upscaling_factor0 = ((pitch.0 / required_pitch0) * oversample_factor).max(1.0);

    let upscaling_factor1 = ((pitch.1 / required_pitch1) * oversample_factor).max(1.0);

    // minimum required resample
    let r0 = (array_shape[0] as f64 * upscaling_factor0).ceil() as usize;
    let r1 = (array_shape[1] as f64 * upscaling_factor1).ceil() as usize;

    let f0 = r0 as f64 / array_shape[0] as f64;
    let f1 = r1 as f64 / array_shape[1] as f64;

    ([r0, r1], [f0, f1])
}

/// Returns a resample size that meets ASM sampling critera as well as the resampling factor, which is kept equal for both axes.
///
/// If both input axis are equal in length and pitch then this will produce the same value as the
/// This effectively extends each axis by steps of axis length divided by the greatest common divisor of both axes.
/// If the axes are coprime or have a large GCD this can result in a large increase over `resample_shape_min(..)`.
///
/// * `array_shape` - The shape of the input array
/// * `pitch` - The sample pitch of the input array
/// * `lambda` - The (minimum) wavelength of interest in the focal system
/// * `fl` - focal length
/// * `oversample_factor` - A multiplier applied the the minimum nyquist sampling. A typical value would be `1.1`.
pub fn resample_shape_equal(
    array_shape: &[usize],
    pitch: (f64, f64),
    lambda: f64,
    fl: f64,
    oversample_factor: f64,
) -> ([usize; 2], f64) {
    let na0 = na(pitch.0 * array_shape[0] as f64, fl, 1.0);
    let na1 = na(pitch.1 * array_shape[1] as f64, fl, 1.0);
    let required_pitch0 = 0.5 * lambda / na0;
    let required_pitch1 = 0.5 * lambda / na1;

    let upscaling_factor =
        ((pitch.0 / required_pitch0).max(pitch.1 / required_pitch1) * oversample_factor).max(1.0);

    let gcd = gcd(array_shape[0], array_shape[1]);

    let n = ((upscaling_factor - 1.0) / (1.0 / gcd as f64)).ceil() as usize;

    // minimum required resample
    let r0 = array_shape[0] + (array_shape[0] * n) / gcd;
    let r1 = array_shape[1] + (array_shape[1] * n) / gcd;

    let factor = r0 as f64 / array_shape[0] as f64;

    ([r0, r1], factor)
}

/// Returns the updated resample shape, having been extended by 1 if required to match tile shape parity, and a resampling factor for each axis.
///
/// This may result in the resample factor of each axis differing slightly if only one is extended.
pub fn match_parity(
    resample_shape: [usize; 2],
    array_shape: &[usize],
    tile_shape: &[usize],
) -> ([usize; 2], [f64; 2]) {
    let r0 = resample_shape[0] + (resample_shape[0] + tile_shape[0]) % 2;
    let r1 = resample_shape[1] + (resample_shape[1] + tile_shape[1]) % 2;

    let f0 = r0 as f64 / array_shape[0] as f64;
    let f1 = r1 as f64 / array_shape[1] as f64;

    ([r0, r1], [f0, f1])
}

/// Returns the updated resample shape, padded to a shape that will allow for fast FFT, and a resampling factor for each axis.
///
/// Typically this produces axis lengths that predominantly have factors of 2 and 3, and optionally one other integer less that 12.
///
/// This may result in the resample factor of each axis differing moderately if each axis is extended by different amounts.
pub fn find_fast(resample_shape: [usize; 2], array_shape: &[usize]) -> ([usize; 2], [f64; 2]) {
    let r0 = fastish_fft_len(resample_shape[0]);

    let r1 = fastish_fft_len(resample_shape[1]);

    let f0 = r0 as f64 / array_shape[0] as f64;
    let f1 = r1 as f64 / array_shape[1] as f64;

    ([r0, r1], [f0, f1])
}

fn sum_to_tile(array: ArrayView2<Complex<f64>>, tile_shape: [usize; 2]) -> Array2<Complex<f64>> {
    assert_eq!(array.shape()[0] % 2, tile_shape[0] % 2);
    assert_eq!(array.shape()[1] % 2, tile_shape[1] % 2);

    let offset0 = array.shape()[0] / 2 - tile_shape[0] / 2;
    let offset1 = array.shape()[1] / 2 - tile_shape[1] / 2;

    let mut out = Array2::zeros(tile_shape);

    let tile_shift0 = div_up(offset0, tile_shape[0]);
    let tile_shift1 = div_up(offset1, tile_shape[1]);

    for i in 0..array.shape()[0] {
        let row = array.slice_axis(Axis(0), (i..=i).into());
        let out_i = (tile_shift0 * tile_shape[0] + i - offset0) % tile_shape[0];
        let mut out_row = out.slice_axis_mut(Axis(0), (out_i..=out_i).into());

        let row_slice = row.as_slice().unwrap();
        let out_row_slice = out_row.as_slice_mut().unwrap();

        for j in 0..array.shape()[1] {
            let out_j = (tile_shift1 * tile_shape[1] + j - offset1) % tile_shape[1];
            out_row_slice[out_j] += row_slice[j];
        }
    }

    out
}

/// Naive implementation of the summed field method. Input is fully resampled, has focal phase applied, and then is split into tiles which are then summed.
///
/// This approach is extremely memory intensive for large apertures, and is implemented only to validate the more complex low memory variants.
#[allow(dead_code)]
fn input_tile_high_memory(
    mut A_xi: Field,
    fl: f64,
    wavelengths: &[f64],
    tile_shape: [usize; 2],
    resample_shape: [usize; 2],
) -> Vec<Field> {
    let f_step = freq_res(A_xi.values.shape(), A_xi.pitch);
    ifft2_shift_inplace(A_xi.values.view_mut());
    let mut a_u = fft2(A_xi.values);
    fft2_shift_inplace(a_u.view_mut());
    let sp_step = spatial_res(&resample_shape, f_step);

    wavelengths
        .iter()
        .map(|lambda| {
            let a_u_star = pad_2D_to(a_u.view(), resample_shape);

            let mut A_xi_star = ifft2c(a_u_star);

            // Apply perfect lens phase
            // A(ξ) * exp(-2πi(sqrt(f^2+ξ^2)-f)/λ)
            centered_par_iter(&mut A_xi_star, sp_step, |(y, x), e| {
                //let theta = -2.0 * PI / lambda * ((y * y + x * x + fl * fl).sqrt() - fl); // numerically unstable - cancellation
                let theta = -2.0 * PI / lambda
                    * ((y * y + x * x) / ((y * y + x * x + fl * fl).sqrt() + fl)); // stable
                *e = *e * Complex::new(0.0, theta).exp()
            });

            // sum to tile
            // sp_step stays the same
            let A_xi_tile = sum_to_tile(A_xi_star.view(), tile_shape);
            Field {
                values: A_xi_tile,
                pitch: sp_step,
            }
        })
        .collect()
}

fn input_tile_low_memory(
    mut A_xi: Field,
    fl: f64,
    wavelengths: &[f64],
    tile_shape: [usize; 2],
    resample_shape: [usize; 2],
    extra_low_memory: bool,
) -> Vec<Field> {
    assert!(resample_shape[0] > A_xi.values.shape()[0]);
    assert!(resample_shape[1] > A_xi.values.shape()[1]);

    let offset0 = resample_shape[0] / 2 - tile_shape[0] / 2;
    let offset1 = resample_shape[1] / 2 - tile_shape[1] / 2;

    let border_tiles0 = div_up(offset0, tile_shape[0]);
    let border_tiles1 = div_up(offset1, tile_shape[1]);

    let mut planner = FftPlanner::new();
    let fft0 = planner.plan_fft(resample_shape[0], FftDirection::Inverse);
    let fft1 = planner.plan_fft(resample_shape[1], FftDirection::Inverse);

    let f_step = freq_res(A_xi.values.shape(), A_xi.pitch);
    ifft2_shift_inplace(A_xi.values.view_mut());
    let a_u = fft2(A_xi.values);
    let sp_step = spatial_res(&resample_shape, f_step);

    let mut tile_sums: Array3<Complex<f64>> =
        Array3::zeros([wavelengths.len(), tile_shape[0], tile_shape[1]]);
    let mut start0 = 0;
    while start0 < resample_shape[0] {
        let end0 = if extra_low_memory {
            min(resample_shape[0], start0 + a_u.shape()[0])
        } else {
            resample_shape[0]
        };

        // store a panel of the 1d fft along axis 0
        let mut axis0_ifft_panel: Array2<Complex<f64>> =
            Array2::zeros([end0 - start0, a_u.shape()[1]]);

        // fill temp with axis0 ifft
        Zip::from(axis0_ifft_panel.axis_iter_mut(Axis(1)))
            .and(a_u.axis_iter(Axis(1)))
            .into_par_iter()
            .for_each_init(
                || {
                    (
                        vec![Zero::zero(); fft0.len()],
                        vec![Zero::zero(); fft0.get_inplace_scratch_len()],
                    )
                },
                |(fft_input, scratch), (mut axis0_ifft_panel, a_u0)| {
                    let pad = resample_shape[0].checked_sub(a_u0.len()).unwrap();
                    let half = (a_u0.len() + 1) / 2;

                    // input is already ifft_shift'd just need padding in center to bring to resample size
                    unsafe {
                        let mut k = 0;
                        for &e in a_u0.slice(s![..half]) {
                            *get_unchecked_mut(fft_input.as_mut_slice(), k) = e;
                            //*fft_input.get_unchecked_mut(k) = e;
                            k += 1;
                        }
                        for _ in 0..pad {
                            *get_unchecked_mut(fft_input.as_mut_slice(), k) = Zero::zero();
                            //*fft_input.get_unchecked_mut(k) = Zero::zero();
                            k += 1;
                        }
                        for &e in a_u0.slice(s![half..]) {
                            *get_unchecked_mut(fft_input.as_mut_slice(), k) = e;
                            //*fft_input.get_unchecked_mut(k) = e;
                            k += 1;
                        }
                    }

                    fft0.process_with_scratch(fft_input, scratch);
                    let fft_out = fft_input;

                    let half = (fft_out.len() + 1) / 2;

                    // this needs to put the end of a_u at the end of input, padding goes in the middle
                    for (i, t) in axis0_ifft_panel.iter_mut().enumerate() {
                        *t += fft_out[(i + half + start0) % fft_out.len()];
                    }
                },
            );

        // Once temp is filled, perform fft acros axis 1, and sum tile contributions
        let mut temp0_slice_start = 0;
        while temp0_slice_start < axis0_ifft_panel.shape()[0] {
            let tile0_slice_start = ((border_tiles0 * tile_shape[0] + temp0_slice_start + start0)
                .checked_sub(offset0)
                .unwrap())
                % tile_shape[0];
            let temp0_slice_end = min(
                axis0_ifft_panel.shape()[0],
                (temp0_slice_start + tile_shape[0])
                    .checked_sub(tile0_slice_start)
                    .unwrap(),
            ); // end slice at either end of temp or end of
            let tile0_slice_end = (tile0_slice_start + temp0_slice_end)
                .checked_sub(temp0_slice_start)
                .unwrap();

            let temp_slice = axis0_ifft_panel.slice(s![temp0_slice_start..temp0_slice_end, ..]);
            let mut tile_slices =
                tile_sums.slice_mut(s![.., tile0_slice_start..tile0_slice_end, ..]);

            Zip::indexed(tile_slices.axis_iter_mut(Axis(1)))
                .and(temp_slice.axis_iter(Axis(0)))
                .into_par_iter()
                .for_each_init(
                    || {
                        (
                            vec![Zero::zero(); fft1.len()],
                            vec![Zero::zero(); fft1.get_inplace_scratch_len()],
                        )
                    },
                    |(fft_input, scratch), (i, mut tiles, temp)| {
                        let pad = (resample_shape[1]).checked_sub(temp.len()).unwrap();
                        let half = (temp.len() + 1) / 2;

                        // input is already fft_shift'd just need padding in center to bring to resample size
                        unsafe {
                            let mut k = 0;
                            for &e in temp.slice(s![..half]) {
                                *get_unchecked_mut(fft_input.as_mut_slice(), k) = e;
                                //*fft_input.get_unchecked_mut(k) = e;
                                k += 1;
                            }
                            for _ in 0..pad {
                                *get_unchecked_mut(fft_input.as_mut_slice(), k) = Zero::zero();
                                //*fft_input.get_unchecked_mut(k) = Zero::zero();
                                k += 1;
                            }
                            for &e in temp.slice(s![half..]) {
                                *get_unchecked_mut(fft_input.as_mut_slice(), k) = e;
                                //*fft_input.get_unchecked_mut(k) = e;
                                k += 1;
                            }
                        }

                        fft1.process_with_scratch(fft_input, scratch);
                        let fft_out = fft_input;

                        // write from ifft to tile - linear chunks (fast)
                        let half = (fft_out.len() + 1) / 2;
                        let mut j_start = 0;
                        let i = i + start0 + temp0_slice_start;
                        let y = (i as f64 - (resample_shape[0] / 2) as f64) * sp_step.0;
                        while j_start < resample_shape[1] {
                            let fft_j_start = (j_start + half) % fft_out.len(); //fftshift
                            let out_j_start =
                                (border_tiles1 * tile_shape[1] + j_start - offset1) % tile_shape[1];

                            let len = min(
                                if fft_j_start < half {
                                    half - fft_j_start
                                } else {
                                    fft_out.len() - fft_j_start
                                },
                                tile_shape[1] - out_j_start,
                            );

                            for (i, mut tile) in tiles.axis_iter_mut(Axis(0)).enumerate() {
                                let lambda = wavelengths[i];
                                let tile_slice = tile.as_slice_mut().unwrap();

                                debug_assert!(out_j_start + len <= tile_slice.len());
                                debug_assert!(fft_j_start + len <= fft_out.len());

                                for n in 0..len {
                                    let x = ((j_start + n) as f64 - (resample_shape[1] / 2) as f64)
                                        * sp_step.1;

                                    //let theta = -2.0 * PI / lambda * ((y * y + x * x + fl * fl).sqrt() - fl); // numerically unstable due to cancellation
                                    let theta = -2.0 * PI / lambda
                                        * ((y * y + x * x)
                                            / ((y * y + x * x + fl * fl).sqrt() + fl)); // stable
                                    let focal_phase = Complex::new(0.0, theta).exp();
                                    unsafe {
                                        *get_unchecked_mut(tile_slice, out_j_start + n) +=
                                            fft_out.get_unchecked(fft_j_start + n) * focal_phase;
                                    }
                                }
                            }

                            j_start += len;
                        }
                    },
                );

            temp0_slice_start = temp0_slice_end;
        }

        start0 = end0;
    }

    let fft_normalisation = 1.0 / (resample_shape[0] as f64 * resample_shape[1] as f64).sqrt();
    tile_sums
        .axis_iter_mut(Axis(0))
        .map(|mut tile_sum| {
            tile_sum.par_map_inplace(|e| *e *= fft_normalisation);
            Field {
                values: tile_sum.to_owned(),
                pitch: sp_step,
            }
        })
        .collect()
}

pub(crate) fn div_up(num: usize, denom: usize) -> usize {
    (num + denom - 1) / denom
}

/// Represents a field sampled at a given pitch.
///
/// This represents a complex scalar field (e.g. transverse electrical), the square of which is the Irradience.
#[derive(Clone, Debug)]
pub struct Field {
    pub values: Array2<Complex<f64>>,
    pub pitch: (f64, f64),
}

impl Field {
    /// Calculates the area weighted sum of the squared norm of the field.
    ///
    /// This results in a conserved value, Radiant flux.
    pub fn intensity_integral(&self) -> f64 {
        self.values.iter().fold(0.0, |sum, &v| sum + v.norm_sqr()) * (self.pitch.0 * self.pitch.1)
    }
}

/// Represents a spectrum at a given frequency resolution.
///
/// DC value is centered. That is it is at len/2 on each axis.
#[derive(Clone, Debug)]
pub struct Spectrum {
    pub values: Array2<Complex<f64>>,
    pub freq_res: (f64, f64),
}

/// From the input Field, calculate the input Spectrum
pub fn sfm_asm_part_1(
    A_xi: Field,
    //z: f64,
    fl: f64,
    wavelengths: &[f64],
    tile_shape: [usize; 2],
    resample_shape: [usize; 2],
    extra_low_memory: bool,
) -> Vec<Spectrum> {
    let area_scaling = ((resample_shape[0] as f64 / A_xi.values.shape()[0] as f64)
        * (resample_shape[1] as f64 / A_xi.values.shape()[1] as f64))
        .sqrt();

    let A_xi_tiles = input_tile_low_memory(
        A_xi,
        fl,
        wavelengths,
        tile_shape,
        resample_shape,
        extra_low_memory,
    );

    A_xi_tiles
        .into_iter()
        .map(|mut A_xi_tile| {
            Zip::from(&mut A_xi_tile.values).par_apply(|e| *e *= area_scaling);

            // a(u)
            let f_res = freq_res(A_xi_tile.values.shape(), A_xi_tile.pitch);
            let a_u_tile = fft2c(A_xi_tile.values);

            Spectrum {
                values: a_u_tile,
                freq_res: f_res,
            }
        })
        .collect()
}

/// From the input Spectrum, calculate the Spectrum at distance z
pub fn sfm_asm_part_2(mut a_u_tile: Spectrum, z: f64, lambda: f64) -> Spectrum {
    centered_par_iter(&mut a_u_tile.values, a_u_tile.freq_res, |(y, x), e| {
        *e = *e
            * (Complex::new(
                0.0,
                2.0 * PI * z * (1.0 / (lambda * lambda) - (y * y + x * x)).sqrt(),
            ))
            .exp()
    });

    a_u_tile
}

/// Convert the output field from frequency spectrum to spatial field
///
/// * `gamma` - Gamma is the zoom factor for each axis. Values greater than 1.0 decrease the sample pitch proportionally. Must not be less than 1.0.
/// * `shift` - Shift values move the image on their respective axes. The values are interpreted as the value of the output pitch. This should be left at (0.0, 0.0) unless used for supersampling when it should typically be kept small (0 to 1.0). On a given axis, at a shift of 0.0, the image center will be at len/2, and at a shift of 1 will be at len/2-1.
pub fn sfm_asm_part_3(a_u_tile: &Spectrum, gamma: (f64, f64), shift: (f64, f64)) -> Field {
    let area_scaling = (gamma.0 * gamma.1).sqrt();

    let f_step = a_u_tile.freq_res;
    let sp_step = spatial_res(a_u_tile.values.shape(), f_step);
    let sp_step = (sp_step.0 / gamma.0, sp_step.1 / gamma.1);

    let mut a_u_tile_shift = a_u_tile.values.clone();
    centered_par_iter(&mut a_u_tile_shift, f_step, |(y, x), e| {
        *e = *e
            * (Complex::new(
                0.0,
                2.0 * PI * ((sp_step.0 * shift.0) * y + (sp_step.1 * shift.1) * x),
            ))
            .exp()
            * area_scaling
    });

    let A_x_tile = scaling_czt3(a_u_tile_shift, gamma);
    //let A_x_tile = ifft2c(a_u_tile_shift);
    Field {
        values: A_x_tile,
        pitch: sp_step,
    }
}

// to replace the fixed scale final ifft, a CZT is performed to allow scaling
// input must not be padded, and must be centered and not fft_shifted
#[allow(dead_code)]
fn scaling_czt(mut B_vw: Array2<Complex<f64>>, gamma: f64) -> Array2<Complex<f64>> {
    let M0 = B_vw.shape()[0];
    let M1 = B_vw.shape()[1];

    let a0 = gamma * M0 as f64;
    let a1 = gamma * M1 as f64;

    // B_vw * E_vw / (a_x * a_y)
    B_vw.indexed_iter_mut().for_each(|((p0, p1), e)| {
        let omega0 = p0 as f64 - (M0 / 2) as f64;
        let omega1 = p1 as f64 - (M1 / 2) as f64;
        *e = *e * Complex::new(0.0, PI * (omega0 * omega0 / a0 + omega1 * omega1 / a1)).exp();
    });

    let mut B_vw = pad_zero_2D(B_vw.view());

    let mut D_vw = Array2::from_shape_fn([M0 * 2, M1 * 2], |(p0, p1)| {
        let omega0 = p0 as f64 - (M0) as f64;
        let omega1 = p1 as f64 - (M1) as f64;
        Complex::new(0.0, -PI * (omega0 * omega0 / a0 + omega1 * omega1 / a1)).exp()
    });

    ifft2_shift_inplace(B_vw.view_mut());
    ifft2_shift_inplace(D_vw.view_mut());

    let mut out = ifft2(fft2(D_vw) * fft2(B_vw));

    fft2_shift_inplace(out.view_mut());

    let mut out = depad_2D(out.view()).to_owned();

    out.indexed_iter_mut().for_each(|((m0, m1), e)| {
        let x0 = m0 as f64 - (M0 / 2) as f64;
        let x1 = m1 as f64 - (M1 / 2) as f64;

        *e = *e * Complex::new(0.0, PI * (x0 * x0 / a0 + x1 * x1 / a1)).exp() * 2.0 / gamma;
    });

    out
}

/// If gamma == 1.0 an ifft is performed. If gamma > 1.0 then a CZT is performed, with field intensity preserved.
/// input must not be padded, and must be centered at len/2.
fn scaling_czt3(mut B_vw: Array2<Complex<f64>>, gamma: (f64, f64)) -> Array2<Complex<f64>> {
    let mut planner = FftPlanner::new();

    let M0 = B_vw.shape()[0];
    if gamma.0 == 1.0 {
        let ifft0 = planner.plan_fft(M0, FftDirection::Inverse);
        let normalisation = 1.0 / (M0 as f64).sqrt();

        // fft along axis0, iteration over axis1
        Zip::from(B_vw.axis_iter_mut(Axis(1)))
            .into_par_iter()
            .for_each_init(
                || {
                    (
                        vec![Zero::zero(); ifft0.len()],
                        vec![Zero::zero(); ifft0.get_inplace_scratch_len()],
                    )
                },
                |(fft_buffer, scratch), input_col| {
                    let mut input_col = input_col.0;
                    let fft_buffer = fft_buffer.as_mut_slice();
                    let half = M0 / 2;

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

                    ifft0.process_with_scratch(fft_buffer, scratch);

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
    } else if gamma.0 > 1.0 {
        let a0 = gamma.0 * M0 as f64;
        let fft0 = planner.plan_fft(M0 * 2, FftDirection::Forward);
        let ifft0 = planner.plan_fft(M0 * 2, FftDirection::Inverse);
        let axis0_factor =
            2.0 / ((M0 * 2) as f64 * (M0 * 2) as f64 * (M0 * 2) as f64 * gamma.0).sqrt();
        let scratch_len0 = max(
            fft0.get_inplace_scratch_len(),
            ifft0.get_inplace_scratch_len(),
        );

        let mut d_uv_col: Vec<_> = (0..M0 * 2)
            .map(|p0| {
                let omega0 = p0 as f64 - (M0) as f64;
                Complex::new(0.0, -PI * (omega0 * omega0 / a0)).exp()
            })
            .collect();
        ifft_shift_inplace(aview_mut1(&mut d_uv_col));
        fft0.process(&mut d_uv_col);
        let axis0_start_phases: Vec<_> = (0..M0)
            .map(|p0| {
                let omega0 = p0 as f64 - (M0 / 2) as f64;
                Complex::new(0.0, PI * (omega0 * omega0 / a0)).exp()
            })
            .collect();
        let axis0_end_phases: Vec<_> = (0..M0)
            .map(|m0| {
                let x0 = m0 as f64 - (M0 / 2) as f64;
                Complex::new(0.0, PI * (x0 * x0 / a0)).exp()
            })
            .collect();

        // CZT along axis0, iteration over axis1
        Zip::from(B_vw.axis_iter_mut(Axis(1)))
            .into_par_iter()
            .for_each_init(
                || {
                    (
                        vec![Zero::zero(); fft0.len()],
                        vec![Zero::zero(); scratch_len0],
                    )
                },
                |(fft_buffer, scratch), B_vw_col| {
                    let mut B_vw_col = B_vw_col.0;
                    let fft_buffer = fft_buffer.as_mut_slice();
                    let pad = B_vw_col.len();
                    let half = M0 / 2;

                    // construct input equivalent to fft_shift followed by padding to resample size
                    // the halves of the input are reverse compared to the hillenbrand tiling because the input has not yet been fft_shifted
                    // this is also why half rounds up, as in an ifft
                    unsafe {
                        let mut k = 0;
                        for (p0, &e) in B_vw_col.slice(s![half..]).iter().enumerate() {
                            *fft_buffer.get_unchecked_mut(k) =
                                e * axis0_start_phases.get_unchecked(p0 + half);
                            k += 1;
                        }
                        for _ in 0..pad {
                            *fft_buffer.get_unchecked_mut(k) = Zero::zero();
                            k += 1;
                        }
                        for (p0, &e) in B_vw_col.slice(s![..half]).iter().enumerate() {
                            *fft_buffer.get_unchecked_mut(k) =
                                e * axis0_start_phases.get_unchecked(p0);
                            k += 1;
                        }
                    }

                    fft0.process_with_scratch(fft_buffer, scratch);

                    // multiply by d_uv_col
                    unsafe {
                        for (k, e) in fft_buffer.iter_mut().enumerate() {
                            *e *= d_uv_col.get_unchecked(k);
                        }
                    }

                    ifft0.process_with_scratch(fft_buffer, scratch);

                    // fft_shift and depad then multiply and write back
                    unsafe {
                        let mut k = 0;
                        for (m0, e) in B_vw_col.slice_mut(s![half..]).iter_mut().enumerate() {
                            *e = *fft_buffer.get_unchecked_mut(k)
                                * axis0_end_phases.get_unchecked(m0 + half)
                                * axis0_factor;
                            k += 1;
                        }
                        k += pad;
                        for (m0, e) in B_vw_col.slice_mut(s![..half]).iter_mut().enumerate() {
                            *e = *fft_buffer.get_unchecked_mut(k)
                                * axis0_end_phases.get_unchecked(m0)
                                * axis0_factor;
                            k += 1;
                        }
                    }
                },
            );
    } else {
        panic!("Gamma must not be less than 1.0: {:?}", gamma);
    }

    let M1 = B_vw.shape()[1];
    if gamma.1 == 1.0 {
        let ifft1 = planner.plan_fft(M1, FftDirection::Inverse);
        let normalisation = 1.0 / (M1 as f64).sqrt();

        // IFFT along axis1, iteration over axis0
        Zip::from(B_vw.axis_iter_mut(Axis(0)))
            .into_par_iter()
            .for_each_init(
                || vec![Zero::zero(); ifft1.get_inplace_scratch_len()],
                |scratch, mut B_vw_row| {
                    for e in &mut B_vw_row.0 {
                        *e *= normalisation;
                    }
                    ifft_shift_inplace(B_vw_row.0.view_mut());
                    ifft1.process_with_scratch(B_vw_row.0.as_slice_mut().unwrap(), scratch);
                    fft_shift_inplace(B_vw_row.0.view_mut());
                },
            );
    } else if gamma.1 > 1.0 {
        let a1 = gamma.1 * M1 as f64;
        let fft1 = planner.plan_fft(M1 * 2, FftDirection::Forward);
        let ifft1 = planner.plan_fft(M1 * 2, FftDirection::Inverse);
        let axis1_factor =
            1.0 / ((M1 * 2) as f64 * (M1 * 2) as f64 * (M1 * 2) as f64 * gamma.1).sqrt();
        let scratch_len1 = max(
            fft1.get_inplace_scratch_len(),
            ifft1.get_inplace_scratch_len(),
        );

        let mut D_vw_row: Vec<_> = (0..M1 * 2)
            .map(|p1| {
                let omega1 = p1 as f64 - (M1) as f64;
                Complex::new(0.0, -PI * (omega1 * omega1 / a1)).exp()
            })
            .collect();
        ifft_shift_inplace(aview_mut1(&mut D_vw_row));
        fft1.process(&mut D_vw_row);
        let d_uv_row = D_vw_row;
        let axis1_start_phases: Vec<_> = (0..M1)
            .map(|p1| {
                let omega1 = p1 as f64 - (M1 / 2) as f64;
                Complex::new(0.0, PI * (omega1 * omega1 / a1)).exp()
            })
            .collect();
        let axis1_end_phases: Vec<_> = (0..M1)
            .map(|m1| {
                let x1 = m1 as f64 - (M1 / 2) as f64;
                Complex::new(0.0, PI * (x1 * x1 / a1)).exp()
            })
            .collect();

        // CZT along axis1, iteration over axis0
        Zip::from(B_vw.axis_iter_mut(Axis(0)))
            .into_par_iter()
            .for_each_init(
                || {
                    (
                        vec![Zero::zero(); fft1.len()],
                        vec![Zero::zero(); scratch_len1],
                    )
                },
                |(fft_buffer, scratch), B_vw_row| {
                    let mut B_vw_row = B_vw_row.0;
                    let fft_buffer = fft_buffer.as_mut_slice();
                    let pad = M1;
                    let half = M1 / 2;

                    // construct input equivalent to ifft_shift followed by padding to resample size
                    // the halves of the input are reverse compared to the hillenbrand tiling because the input has not yet been fft_shifted
                    // this is also why half rounds down, as in an ifft
                    unsafe {
                        let mut k = 0;
                        for (p1, &e) in B_vw_row.slice(s![half..]).iter().enumerate() {
                            *fft_buffer.get_unchecked_mut(k) =
                                e * axis1_start_phases.get_unchecked(p1 + half);
                            k += 1;
                        }
                        for _ in 0..pad {
                            *fft_buffer.get_unchecked_mut(k) = Zero::zero();
                            k += 1;
                        }
                        for (p1, &e) in B_vw_row.slice(s![..half]).iter().enumerate() {
                            *fft_buffer.get_unchecked_mut(k) =
                                e * axis1_start_phases.get_unchecked(p1);
                            k += 1;
                        }
                    }

                    fft1.process_with_scratch(fft_buffer, scratch);

                    // multiply by d_uv_row
                    unsafe {
                        for (k, e) in fft_buffer.iter_mut().enumerate() {
                            *e *= d_uv_row.get_unchecked(k);
                        }
                    }

                    ifft1.process_with_scratch(fft_buffer, scratch);

                    // fft_shift and depad then multiply by write back
                    unsafe {
                        let mut k = 0;
                        for (m1, e) in B_vw_row.slice_mut(s![half..]).iter_mut().enumerate() {
                            *e = *fft_buffer.get_unchecked_mut(k)
                                * axis1_end_phases.get_unchecked(m1 + half)
                                * axis1_factor;
                            k += 1;
                        }
                        k += pad;
                        for (m1, e) in B_vw_row.slice_mut(s![..half]).iter_mut().enumerate() {
                            *e = *fft_buffer.get_unchecked_mut(k)
                                * axis1_end_phases.get_unchecked(m1)
                                * axis1_factor;
                            k += 1;
                        }
                    }
                },
            );
    } else {
        panic!("Gamma must not be less than 1.0: {:?}", gamma);
    }

    B_vw
}

/// Calculate the aperture diameter from numerical aperture and focal length
///
/// * na - numerical aperture
/// * fl - focal length
/// * n - optical density of adjacent medium (1.0 for air)
pub fn diameter(na: f64, fl: f64, n: f64) -> f64 {
    let na = na / n;
    2.0 * fl * na / (1.0 - na * na).sqrt()
}

/// Calculate the numerical aperture from diameter and focal length
///
/// * d - aperture diameter
/// * fl - focal length
/// * n - optical density of adjacent medium (1.0 for air)
pub fn na(d: f64, fl: f64, n: f64) -> f64 {
    // simplify NA = n sin(atan(D/(2f)))
    n * d / (4.0 * fl * fl + d * d).sqrt()
}

/// Radius of Airy pattern from the central peak to the first minimum
///
/// * na - numerical aperture
/// * lambda - wavelength of light
pub fn airy_radius(na: f64, lambda: f64) -> f64 {
    1.22 * 0.5 * lambda / na
}

fn freq_res(array_shape: &[usize], spatial_res: (f64, f64)) -> (f64, f64) {
    (
        1.0 / (spatial_res.0 * array_shape[0] as f64),
        1.0 / (spatial_res.1 * array_shape[1] as f64),
    )
}

fn spatial_res(array_shape: &[usize], freq_res: (f64, f64)) -> (f64, f64) {
    (
        1.0 / (freq_res.0 * array_shape[0] as f64),
        1.0 / (freq_res.1 * array_shape[1] as f64),
    )
}

fn centered_par_iter<F: Fn((f64, f64), &mut Complex<f64>) + Sync>(
    array: &mut Array2<Complex<f64>>,
    (dh, dw): (f64, f64),
    f: F,
) {
    let h = array.shape()[0];
    let w = array.shape()[1];
    Zip::indexed(array).par_apply(|(y, x), e| {
        let y = (y as f64 - (h / 2) as f64) * dh;
        let x = (x as f64 - (w / 2) as f64) * dw;
        f((y, x), e)
    });
}
