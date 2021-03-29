use image::{Rgb, RgbImage};
use ndarray::{s, ArrayView2};
use ndarray::{Array2, Zip};
use num_complex::Complex;
use palette::{Lch, LinSrgb, Srgb};
use summed_field_method::Field;
use summed_field_method::{resample_shape_min, sfm_asm_part_1, sfm_asm_part_2, sfm_asm_part_3};

pub fn main() {
    let od = 0.0254 * 6.0;

    let mask_shape = 4096;
    let n_vanes = 4;
    let vane_width = 0.002;
    let vane_offset = 0.1;
    let r_outer = od / 2.0;
    let r_co = od / 2.0 * 0.47;
    let fl = od * 8.0;

    let input_field = generate_mask(mask_shape, n_vanes, vane_width, vane_offset, r_outer, r_co);
    let input_intensity = input_field.values.map(|e| e.norm_sqr());
    save_real_image("test_input.png", input_intensity.view(), 1.0, true).unwrap();

    let lambda = 600e-9;
    let z = fl;

    let oversample = 1.1;

    let (resample_shape, _factor) = resample_shape_min(
        input_field.values.shape(),
        input_field.pitch,
        lambda,
        fl,
        oversample,
    );

    let gamma = (4.0, 4.0);
    let tile_shape = [
        (mask_shape as f64 * oversample).ceil() as usize,
        (mask_shape as f64 * oversample).ceil() as usize,
    ];

    let mut input_spectrum = sfm_asm_part_1(
        input_field,
        fl,
        &[lambda, lambda * 1.1],
        tile_shape,
        resample_shape,
        false,
    );
    let output_spectrum = sfm_asm_part_2(input_spectrum.swap_remove(0), z, lambda);
    save_complex_image("test_spectrum.png", output_spectrum.values.view()).unwrap();

    let super_sample = 1;
    let mut super_sample_output =
        Array2::zeros([tile_shape[0] * super_sample, tile_shape[1] * super_sample]);

    for x in 0..super_sample {
        for y in 0..super_sample {
            let output = sfm_asm_part_3(
                &output_spectrum,
                gamma,
                (
                    y as f64 / super_sample as f64,
                    x as f64 / super_sample as f64,
                ),
            );
            super_sample_output
                .slice_mut(s![y..;super_sample, x..;super_sample])
                .assign(&output.values);
        }
    }

    let output_intensity = super_sample_output.map(|e| e.norm_sqr());
    let output_log_intensity = log_intensity(output_intensity.view(), 1e-10);

    save_real_image("test_output.png", output_log_intensity.view(), 1.0, true).unwrap();
    save_complex_image("test_outputc.png", super_sample_output.view()).unwrap();
}

pub(crate) fn div_up(num: usize, denom: usize) -> usize {
    (num + denom - 1) / denom
}

pub fn save_grayscale_real_image<T: AsRef<std::path::Path> + std::fmt::Debug>(
    file_name: T,
    arr: ArrayView2<f64>,
    amp: f64,
    normalise: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if let &[h, w, ..] = arr.shape() {
        let mut max: f64 = arr.iter().fold(0.0, |max, val| val.max(max));
        let sum = arr.iter().fold(0.0, |sum, val| val + sum);
        println!("h:{} w:{} max:{} sum:{} - {:?}", h, w, max, sum, file_name);

        let mut img = RgbImage::new(w as u32, h as u32);
        if !normalise {
            max = 1.0;
        }

        for (x, y, p) in img.enumerate_pixels_mut() {
            let value = arr[[y as usize, x as usize]] / max;
            let value = (value * amp).min(1.0).max(0.0);

            let para = (value - value * value) * 0.1;

            //let colour = Srgb::from(Hsl::new(360.0*(-value*0.65+0.65), 1.0, 0.01 + 0.99*value));
            let colour = Srgb::from_linear(LinSrgb::new(
                value + para * ((value + 2.0 / 3.0) * std::f64::consts::PI * 2.0).sin(),
                value + para * ((value + 1.0 / 3.0) * std::f64::consts::PI * 2.0).sin(),
                value + para * (value * std::f64::consts::PI * 2.0).sin(),
            ));
            *p = Rgb([
                (colour.red * 255.0) as u8,
                (colour.green * 255.0) as u8,
                (colour.blue * 255.0) as u8,
            ]);
        }

        img.save(file_name).unwrap();
    }
    Ok(())
}

pub fn save_real_image<T: AsRef<std::path::Path> + std::fmt::Debug>(
    file_name: T,
    arr: ArrayView2<f64>,
    amp: f64,
    normalise: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if let &[h, w, ..] = arr.shape() {
        let mut max: f64 = arr.iter().fold(0.0, |max, val| val.max(max));
        let sum = arr.iter().fold(0.0, |sum, val| val + sum);
        println!("h:{} w:{} max:{} sum:{} - {:?}", h, w, max, sum, file_name);

        let mut img = RgbImage::new(w as u32, h as u32);
        if !normalise {
            max = 1.0;
        }

        for (x, y, p) in img.enumerate_pixels_mut() {
            let value = arr[[y as usize, x as usize]] / max;
            let value = (value * amp).min(1.0);

            //let colour = Srgb::from(Hsl::new(360.0*(-value*0.65+0.65), 1.0, 0.01 + 0.99*value));
            let colour = Srgb::from(Lch::new(value * 70.0, value * 128.0, 280.0 - 245.0 * value));
            *p = Rgb([
                (colour.red * 255.0) as u8,
                (colour.green * 255.0) as u8,
                (colour.blue * 255.0) as u8,
            ]);
        }

        img.save(file_name).unwrap();
    }
    Ok(())
}

// returns 1.0 if greater than nominal, with a soft transition of a distance of 1.0 straddling the nominal transition.
fn soft_greater_than(x: f64, x_nominal: f64, pitch: f64) -> f64 {
    if x < x_nominal - 0.5 * pitch {
        0.0
    } else if x > x_nominal + 0.5 * pitch {
        1.0
    } else {
        (x - (x_nominal - 0.5 * pitch)) / pitch
    }
}

// generate an aperture obstructured by a secondary mirror and support vanes
pub fn generate_mask(
    shape: usize,
    n_vanes: usize,
    vane_width: f64,
    vane_offset: f64,
    r_outer: f64,
    r_co: f64,
) -> Field {
    let pitch = r_outer / (div_up(shape, 2) - 2) as f64;

    let c = (shape / 2) as f64;

    let mut mask = Array2::zeros([shape, shape]);

    Zip::indexed(&mut mask).par_for_each(|(y, x), e| {
        let mut value = 1.0;

        let y0 = (y as f64 - c) * pitch;
        let x0 = (x as f64 - c) * pitch;
        let r = (x0 * x0 + y0 * y0).sqrt();

        value *= soft_greater_than(r, r_co, pitch);
        value *= 1.0 - soft_greater_than(r, r_outer, pitch);

        for i in 0..n_vanes {
            let theta = (i as f64 / n_vanes as f64) * ::std::f64::consts::PI * 2.0 + vane_offset;
            let y1 = theta.cos();
            let x1 = theta.sin();

            // distance from line
            let d = (-x1 * (y1 - y0) - (x1 - x0) * -y1).abs() / (x1 * x1 + y1 * y1).sqrt();

            let projected_r = x1 * x0 + y1 * y0;
            if projected_r > 0.0 {
                value *= soft_greater_than(d, vane_width * 0.5, pitch);
            }
        }
        *e = Complex::new(value, 0.0);
    });

    Field {
        values: mask,
        pitch: (pitch, pitch),
    }
}

pub fn log_intensity(arr: ArrayView2<f64>, min: f64) -> Array2<f64> {
    let log_min = -min.ln();
    let max = arr.iter().fold(0.0, |max, e| e.max(max));
    arr.map(|e| ((e / max).ln() / log_min + 1.0).max(0.0).min(1.0))
}

pub fn save_complex_image<T: AsRef<std::path::Path> + std::fmt::Debug>(
    file_name: T,
    arr: ArrayView2<Complex<f64>>,
) -> Result<(), Box<dyn std::error::Error>> {
    if let &[h, w, ..] = arr.shape() {
        let max_sqr: f64 = arr.iter().fold(0.0, |max, val| val.norm_sqr().max(max));
        let sum_sqr: f64 = arr.iter().fold(0.0, |sum, val| val.norm_sqr() + sum);
        println!(
            "h:{} w:{} max_sqr:{} sum_sqr:{} - {:?}",
            h, w, max_sqr, sum_sqr, file_name
        );

        let max = max_sqr.sqrt();

        let mut img = RgbImage::new(w as u32, h as u32);

        for (x, y, p) in img.enumerate_pixels_mut() {
            let (r, theta) = arr[[y as usize, x as usize]].to_polar();
            let r = r / max;

            //let colour = Srgb::from(Hsv::new(360.0*(theta/std::fxx::consts::TAU + 0.5), 1.0, r*0.9));
            let colour = Srgb::from(Lch::new(
                r * 100.0,
                r * 128.0,
                360.0 * (theta / ::std::f64::consts::PI + 1.0) * 0.5,
            ));
            *p = Rgb([
                (colour.red * 255.0) as u8,
                (colour.green * 255.0) as u8,
                (colour.blue * 255.0) as u8,
            ]);
        }

        img.save(file_name).unwrap();
    }
    Ok(())
}
