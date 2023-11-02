use image::{Rgb, RgbImage};
use ndarray::ArrayView2;
use ndarray::{Array2, Zip};
use num_complex::Complex;
use palette::{FromColor, Lch, LinSrgb, Srgb};
use summed_field_method::fresnel::fresnel;
use std::cmp::min;
use summed_field_method::Field;

pub fn main() {
    let od = 0.0254 * 8.0;

    let mask_shape = 4096;
    let r_outer = od / 2.0;
    let r_co = od / 2.0 * 0.33;

    let input_field = generate_bahtinov_mask(
        mask_shape,
        0.005,
        0.005,
        0.11111 * ::std::f64::consts::PI,
        r_outer,
        r_co,
    );
    println!("field_sum: {}", field_sum(&input_field));
    let input_intensity = input_field.values.map(|e| e.norm_sqr());
    save_grayscale_real_image("bahtinov0.png", input_intensity.view(), 1.0, true).unwrap();

    let input_field = generate_bahtinov1_mask(
        mask_shape,
        0.005,
        0.005,
        0.11111 * ::std::f64::consts::PI,
        r_outer,
        r_co,
    );
    println!("field_sum: {}", field_sum(&input_field));
    let input_intensity = input_field.values.map(|e| e.norm_sqr());
    save_grayscale_real_image("bahtinov1.png", input_intensity.view(), 1.0, true).unwrap();

    let input_field = generate_lin_chirp_bahtinov_mask(
        mask_shape,
        0.005,
        0.005,
        20.0 / 180.0 * ::std::f64::consts::PI,
        r_outer,
        r_co,
    );
    println!("field_sum: {}", field_sum(&input_field));
    let input_intensity = input_field.values.map(|e| e.norm_sqr());
    save_grayscale_real_image("bahtinov5.png", input_intensity.view(), 1.0, true).unwrap();

    let input_field = generate_exp_chirp_bahtinov_mask(
        mask_shape,
        0.005,
        0.005,
        5.0,
        20.0 / 180.0 * ::std::f64::consts::PI,
        r_outer,
        r_co,
    );
    println!("field_sum: {}", field_sum(&input_field));
    let input_intensity = input_field.values.map(|e| e.norm_sqr());
    save_grayscale_real_image("bahtinov6.png", input_intensity.view(), 1.0, true).unwrap();

    let input_field = generate_exp_chirp_bahtinov_mask(
        mask_shape,
        0.005,
        0.005,
        3.0,
        20.0 / 180.0 * ::std::f64::consts::PI,
        r_outer,
        r_co,
    );
    println!("field_sum: {}", field_sum(&input_field));
    let input_intensity = input_field.values.map(|e| e.norm_sqr());
    save_grayscale_real_image("bahtinov7.png", input_intensity.view(), 1.0, true).unwrap();
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
            let colour = Srgb::<f64>::from_linear(LinSrgb::new(
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
            let colour =
                Srgb::from_color(Lch::new(value * 70.0, value * 128.0, 280.0 - 245.0 * value));
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


/// Standard circle Mask
pub fn generate_circle_mask(
    shape: usize,
    padding_factor: f64,
    r_outer: f64,
    r_co: f64,
) -> Field {
    let pitch = padding_factor * r_outer / (div_up(shape, 2) - 2) as f64;
    let c = (shape / 2) as f64;

    let mut mask = Array2::zeros([shape, shape]);

    Zip::indexed(&mut mask).par_for_each(|(y, x), e| {
        let mut value = 1.0;

        let y0 = (y as f64 - c) * pitch;
        let x0 = (x as f64 - c) * pitch;


        let r = (x0 * x0 + y0 * y0).sqrt();

        value *= soft_greater_than(r, r_co, pitch);
        value *= 1.0 - soft_greater_than(r, r_outer, pitch);

        *e = Complex::new(value, 0.0);
    });

    Field {
        values: mask,
        pitch: (pitch, pitch),
    }
}

/// Standard Bahtinov Mask
pub fn generate_bahtinov_mask(
    shape: usize,
    support_width: f64,
    grating_width: f64,
    grating_angle: f64,
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

        // input supports
        value *= soft_greater_than(x0.abs(), support_width * 0.5, pitch);
        if x0 > 0.0 {
            value *= soft_greater_than(y0.abs(), support_width * 0.5, pitch);
        }

        let r = (x0 * x0 + y0 * y0).sqrt();

        value *= soft_greater_than(r, r_co, pitch);
        value *= 1.0 - soft_greater_than(r, r_outer, pitch);

        if x0 > 0.0 {
            // angled
            if y0 > 0.0 {
                let theta = grating_angle;
                let x1 = theta.cos();
                let y1 = theta.sin();

                // distance from line
                let d = (-x1 * (y1 - y0) - (x1 - x0) * -y1).abs() / (x1 * x1 + y1 * y1).sqrt();

                value *= (-(d / grating_width * ::std::f64::consts::PI).cos().signum() + 1.0) * 0.5
            } else {
                let theta = -grating_angle;
                let x1 = theta.cos();
                let y1 = theta.sin();

                // distance from line
                let d = (-x1 * (y1 - y0) - (x1 - x0) * -y1).abs() / (x1 * x1 + y1 * y1).sqrt();

                value *= (-(d / grating_width * ::std::f64::consts::PI).cos().signum() + 1.0) * 0.5
            }
        } else {
            //vertical
            value *= (-(y0 / grating_width * ::std::f64::consts::PI).cos().signum() + 1.0) * 0.5
        }

        *e = Complex::new(value, 0.0);
    });

    Field {
        values: mask,
        pitch: (pitch, pitch),
    }
}

/// generate clipped bahtinov mask
pub fn generate_bahtinov1_mask(
    shape: usize,
    support_width: f64,
    grating_width: f64,
    grating_angle: f64,
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

        // input supports
        value *= soft_greater_than(x0.abs(), support_width * 0.5, pitch);
        if x0 > 0.0 {
            value *= soft_greater_than(y0.abs(), support_width * 0.5, pitch);
        }

        let r = (x0 * x0 + y0 * y0).sqrt();

        value *= soft_greater_than(r, r_co, pitch);
        value *= 1.0 - soft_greater_than(r, r_outer, pitch);

        if x0 > 0.0 {
            // angled
            if y0 > 0.0 {
                let theta = grating_angle;
                let x1 = theta.cos();
                let y1 = theta.sin();

                // distance from line
                let d = (-x1 * (y1 - y0) - (x1 - x0) * -y1).abs() / (x1 * x1 + y1 * y1).sqrt();

                value *= (-(d / grating_width * ::std::f64::consts::PI).cos().signum() + 1.0) * 0.5;

                let x2 = r_outer * 1.5 * x1;
                let y2 = r_outer * 1.5 * y1;
                let rr = ((x0 - x2) * (x0 - x2) + (y0 - y2) * (y0 - y2)).sqrt();
                value *= 1.0 - soft_greater_than(rr, r_outer, pitch);
            } else {
                let theta = -grating_angle;
                let x1 = theta.cos();
                let y1 = theta.sin();

                // distance from line
                let d = (-x1 * (y1 - y0) - (x1 - x0) * -y1).abs() / (x1 * x1 + y1 * y1).sqrt();

                value *= (-(d / grating_width * ::std::f64::consts::PI).cos().signum() + 1.0) * 0.5;

                let x2 = r_outer * 1.5 * x1;
                let y2 = r_outer * 1.5 * y1;
                let rr = ((x0 - x2) * (x0 - x2) + (y0 - y2) * (y0 - y2)).sqrt();
                value *= 1.0 - soft_greater_than(rr, r_outer, pitch);
            }
        } else {
            //vertical
            value *= (-(y0 / grating_width * ::std::f64::consts::PI).cos().signum() + 1.0) * 0.5;

            let x2 = -r_outer * 1.5;
            let rr = ((x0 - x2) * (x0 - x2) + y0 * y0).sqrt();
            value *= 1.0 - soft_greater_than(rr, r_outer, pitch);
        }

        *e = Complex::new(value, 0.0);
    });

    Field {
        values: mask,
        pitch: (pitch, pitch),
    }
}

/// generate a clipped bahtinov with a linear chirp
pub fn generate_lin_chirp_bahtinov_mask(
    shape: usize,
    support_width: f64,
    grating_width: f64,
    grating_angle: f64,
    r_outer: f64,
    r_co: f64,
) -> Field {
    let end = r_outer / 2.0;

    let pitch = r_outer / (div_up(shape, 2) - 2) as f64;
    let c = (shape / 2) as f64;

    let mut mask = Array2::zeros([shape, shape]);

    Zip::indexed(&mut mask).par_for_each(|(y, x), e| {
        let mut value = 1.0;

        let y0 = (y as f64 - c) * pitch;
        let x0 = (x as f64 - c) * pitch;

        // input supports
        value *= soft_greater_than(x0.abs(), support_width * 0.5, pitch);
        if x0 > 0.0 {
            value *= soft_greater_than(y0.abs(), support_width * 0.5, pitch);
        }

        let r = (x0 * x0 + y0 * y0).sqrt();

        value *= soft_greater_than(r, r_co, pitch);
        value *= 1.0 - soft_greater_than(r, r_outer, pitch);

        if x0 > 0.0 {
            // angled
            if y0 > 0.0 {
                let theta = grating_angle;
                let x1 = theta.cos();
                let y1 = theta.sin();

                // distance from line
                let d = (-x1 * (y1 - y0) - (x1 - x0) * -y1).abs() / (x1 * x1 + y1 * y1).sqrt();

                let x = (d + 2.0 * d * d / end) / grating_width;
                value *= 1.0 - soft_greater_than(d, end, pitch);
                value *= (-(x * ::std::f64::consts::PI).cos().signum() + 1.0) * 0.5;

                let x2 = r_outer * 1.5 * x1;
                let y2 = r_outer * 1.5 * y1;
                let rr = ((x0 - x2) * (x0 - x2) + (y0 - y2) * (y0 - y2)).sqrt();
                value *= 1.0 - soft_greater_than(rr, r_outer, pitch);
            } else {
                let theta = -grating_angle;
                let x1 = theta.cos();
                let y1 = theta.sin();

                // distance from line
                let d = (-x1 * (y1 - y0) - (x1 - x0) * -y1).abs() / (x1 * x1 + y1 * y1).sqrt();

                let x = (d + 2.0 * d * d / end) / grating_width;
                value *= 1.0 - soft_greater_than(d, end, pitch);
                value *= (-(x * ::std::f64::consts::PI).cos().signum() + 1.0) * 0.5;

                let x2 = r_outer * 1.5 * x1;
                let y2 = r_outer * 1.5 * y1;
                let rr = ((x0 - x2) * (x0 - x2) + (y0 - y2) * (y0 - y2)).sqrt();
                value *= 1.0 - soft_greater_than(rr, r_outer, pitch);
            }
        } else {
            //vertical
            let x = (y0.abs() + 2.0 * y0 * y0 / end) / grating_width;
            value *= 1.0 - soft_greater_than(y0.abs(), end, pitch);
            value *= (-(x * ::std::f64::consts::PI).cos().signum() + 1.0) * 0.5;

            let x2 = -r_outer * 1.5;
            let rr = ((x0 - x2) * (x0 - x2) + y0 * y0).sqrt();
            value *= 1.0 - soft_greater_than(rr, r_outer, pitch);
        }

        *e = Complex::new(value, 0.0);
    });

    Field {
        values: mask,
        pitch: (pitch, pitch),
    }
}

/// generate a clipped bahtinov with an exponential chirp
pub fn generate_exp_chirp_bahtinov_mask(
    shape: usize,
    support_width: f64,
    grating_width: f64,
    grating_ratio: f64,
    grating_angle: f64,
    r_outer: f64,
    r_co: f64,
) -> Field {
    let end = r_outer / 2.0;

    let pitch = r_outer / (div_up(shape, 2) - 2) as f64;
    let c = (shape / 2) as f64;

    let mut mask = Array2::zeros([shape, shape]);

    Zip::indexed(&mut mask).par_for_each(|(y, x), e| {
        let mut value = 1.0;

        let y0 = (y as f64 - c) * pitch;
        let x0 = (x as f64 - c) * pitch;

        // input supports
        value *= soft_greater_than(x0.abs(), support_width * 0.5, pitch);
        if x0 > 0.0 {
            value *= soft_greater_than(y0.abs(), support_width * 0.5, pitch);
        }

        let r = (x0 * x0 + y0 * y0).sqrt();

        value *= soft_greater_than(r, r_co, pitch);
        value *= 1.0 - soft_greater_than(r, r_outer, pitch);

        if x0 > 0.0 {
            // angled
            if y0 > 0.0 {
                let theta = grating_angle;
                let x1 = theta.cos();
                let y1 = theta.sin();

                // distance from line
                let d = (-x1 * (y1 - y0) - (x1 - x0) * -y1).abs() / (x1 * x1 + y1 * y1).sqrt();

                let a = grating_ratio.ln() / end;
                let x = (d * a).exp_m1() / (a * grating_width);

                value *= 1.0 - soft_greater_than(d, end, pitch);
                value *= (-(x * ::std::f64::consts::PI).cos().signum() + 1.0) * 0.5;

                let x2 = r_outer * 1.5 * x1;
                let y2 = r_outer * 1.5 * y1;
                let rr = ((x0 - x2) * (x0 - x2) + (y0 - y2) * (y0 - y2)).sqrt();
                value *= 1.0 - soft_greater_than(rr, r_outer, pitch);
            } else {
                let theta = -grating_angle;
                let x1 = theta.cos();
                let y1 = theta.sin();

                // distance from line
                let d = (-x1 * (y1 - y0) - (x1 - x0) * -y1).abs() / (x1 * x1 + y1 * y1).sqrt();

                let a = grating_ratio.ln() / end;
                let x = (d * a).exp_m1() / (a * grating_width);
                value *= 1.0 - soft_greater_than(d, end, pitch);
                value *= (-(x * ::std::f64::consts::PI).cos().signum() + 1.0) * 0.5;

                let x2 = r_outer * 1.5 * x1;
                let y2 = r_outer * 1.5 * y1;
                let rr = ((x0 - x2) * (x0 - x2) + (y0 - y2) * (y0 - y2)).sqrt();
                value *= 1.0 - soft_greater_than(rr, r_outer, pitch);
            }
        } else {
            //vertical
            let a = grating_ratio.ln() / end;
            let x = (y0.abs() * a).exp_m1() / (a * grating_width);
            value *= 1.0 - soft_greater_than(y0.abs(), end, pitch);
            value *= (-(x * ::std::f64::consts::PI).cos().signum() + 1.0) * 0.5;

            let x2 = -r_outer * 1.5;
            let rr = ((x0 - x2) * (x0 - x2) + y0 * y0).sqrt();
            value *= 1.0 - soft_greater_than(rr, r_outer, pitch);
        }

        *e = Complex::new(value, 0.0);
    });

    Field {
        values: mask,
        pitch: (pitch, pitch),
    }
}

#[test]
fn bahtinov8() {
    let od = 0.0254 * 8.0;

    let mask_shape = 4096;
    let r_outer = od / 2.0;
    let r_co = od / 2.0 * 0.33;

    let input_field = generate_custom_bahtinov_mask(
        mask_shape,
        0.005,
        0.005,
        3.0,
        0.3333,
        20.0 / 180.0 * ::std::f64::consts::PI,
        r_outer,
        r_co,
    );

    println!("field_sum: {}", field_sum(&input_field));
    let input_intensity = input_field.values.map(|e| e.norm_sqr());
    save_grayscale_real_image("bahtinov8.png", input_intensity.view(), 1.0, true).unwrap();
}

/// generate a bahtinov with an exponential chirp grating with radial window function
pub fn generate_custom_bahtinov_mask(
    shape: usize,
    support_width: f64,
    grating_width: f64,
    grating_reduction_factor: f64,
    grating_area_fration: f64,
    grating_angle: f64,
    r_outer: f64,
    r_co: f64,
) -> Field {
    // size of the grating
    let end = r_outer * 0.4; // how far the outermost window is from the middle window of each grating
    let mid_distance = r_outer * 0.66; // how far the center of the middle window is from the center of the circle
    let mid_width = r_outer * 0.5; // how wide the middle window is in the radial direction

    let pitch = r_outer / (div_up(shape, 2) - 2) as f64;
    let c = (shape / 2) as f64;

    let mut mask = Array2::zeros([shape, shape]);

    let grating_pitch = 2.0 * grating_width;

    // exponential chirp
    let exp = |distance: f64| {
        let a = grating_reduction_factor.ln() / end;
        let adjusted_distance = (distance * a).exp_m1() / (a * grating_pitch);
        adjusted_distance
    };

    let exp_inv = |y: f64| {
        (end * ((grating_pitch
            * grating_reduction_factor.ln()
            * (end / (grating_pitch * grating_reduction_factor.ln()) + y))
            / end)
            .ln())
            / grating_reduction_factor.ln()
    };

    // quadratic chirp
    let quadratic = |distance: f64| {
        let adjusted_distance = (distance
            + (grating_reduction_factor - 1.0) * 0.5 * distance * distance / end)
            / grating_pitch;
        adjusted_distance
    };

    let quadratic_inv = |y: f64| {
        ((end * (2.0 * grating_reduction_factor * grating_pitch * y + end)).sqrt() - end)
            / grating_reduction_factor
    };

    let window_function = |d: f64| {
        if d > 1.0 || d < -1.0 {
            return 0.0;
        }
        // d is the normalised distance from center [-1, 1]
        let pow = 4;
        1.0 - 2.0 * d.powi(pow) / (1.0 + d.powi(pow))
    };

    let window_locations: Vec<_> = (0..)
        .map(|i| {
            let mid = i as f64 + 0.5;
            let lower = mid - grating_area_fration * 0.5;
            let upper = mid + grating_area_fration * 0.5;
            (exp_inv(lower), exp_inv(mid), exp_inv(upper))
        })
        .take_while(|(_, mid, _)| mid < &end)
        .collect();

    Zip::indexed(&mut mask).par_for_each(|(y, x), e| {
        let mut value = 1.0;

        let y0 = (y as f64 - c) * pitch;
        let x0 = (x as f64 - c) * pitch;

        // input supports
        value *= soft_greater_than(x0.abs(), support_width * 0.5, pitch);
        if x0 > 0.0 {
            value *= soft_greater_than(y0.abs(), support_width * 0.5, pitch);
        }

        let r = (x0 * x0 + y0 * y0).sqrt();

        value *= soft_greater_than(r, r_co, pitch);
        value *= 1.0 - soft_greater_than(r, r_outer, pitch);

        if x0 > 0.0 {
            // angled
            if y0 > 0.0 {
                let theta = grating_angle;
                let x1 = theta.cos();
                let y1 = theta.sin();

                // distance along and from grating center line
                let tangential_dist =
                    (-x1 * (y1 - y0) - (x1 - x0) * -y1).abs() / (x1 * x1 + y1 * y1).sqrt();
                let radial_dist = x0 * x1 + y0 * y1;

                let normalised_radial = 2.0 * (radial_dist - mid_distance) / mid_width;
                let tangential_index = min(
                    exp(tangential_dist).floor() as usize,
                    window_locations.len() - 1,
                );
                let (lower, mid, upper) = window_locations[tangential_index];

                if normalised_radial > 1.0 || normalised_radial < -1.0 {
                    value *= 0.0
                }
                value *= 1.0
                    - soft_greater_than(
                        tangential_dist - mid,
                        (upper - mid) * window_function(normalised_radial),
                        pitch,
                    );
                value *= 1.0
                    - soft_greater_than(
                        mid - tangential_dist,
                        (mid - lower) * window_function(normalised_radial),
                        pitch,
                    );
            } else {
                let theta = -grating_angle;
                let x1 = theta.cos();
                let y1 = theta.sin();

                // distance along and from grating center line
                let tangential_dist =
                    (-x1 * (y1 - y0) - (x1 - x0) * -y1).abs() / (x1 * x1 + y1 * y1).sqrt();
                let radial_dist = x0 * x1 + y0 * y1;

                let normalised_radial = 2.0 * (radial_dist - mid_distance) / mid_width;
                let tangential_index = min(
                    exp(tangential_dist).floor() as usize,
                    window_locations.len() - 1,
                );
                let (lower, mid, upper) = window_locations[tangential_index];

                if normalised_radial > 1.0 || normalised_radial < -1.0 {
                    value *= 0.0
                }
                value *= 1.0
                    - soft_greater_than(
                        tangential_dist - mid,
                        (upper - mid) * window_function(normalised_radial),
                        pitch,
                    );
                value *= 1.0
                    - soft_greater_than(
                        mid - tangential_dist,
                        (mid - lower) * window_function(normalised_radial),
                        pitch,
                    );
            }
        } else {
            //vertical

            // distance along and from grating center line
            let tangential_dist = y0.abs();
            let radial_dist = -x0;

            let normalised_radial = 2.0 * (radial_dist - mid_distance) / mid_width;
            let tangential_index = min(
                exp(tangential_dist).floor() as usize,
                window_locations.len() - 1,
            );
            let (lower, mid, upper) = window_locations[tangential_index];

            if normalised_radial > 1.0 || normalised_radial < -1.0 {
                value *= 0.0
            }
            value *= 1.0
                - soft_greater_than(
                    tangential_dist - mid,
                    (upper - mid) * window_function(normalised_radial),
                    pitch,
                );
            value *= 1.0
                - soft_greater_than(
                    mid - tangential_dist,
                    (mid - lower) * window_function(normalised_radial),
                    pitch,
                );
        }

        *e = Complex::new(value, 0.0);
    });

    Field {
        values: mask,
        pitch: (pitch, pitch),
    }
}

#[test]
fn bahtinov9() {
    let od = 0.0254 * 8.0;

    let mask_shape = 4096;
    let r_outer = od / 2.0;
    let r_co = od / 2.0 * 0.33;

    let input_field = generate_custom_bahtinov_mask2(
        mask_shape,
        0.005,
        0.005,
        4.0,
        0.3333,
        20.0 / 180.0 * ::std::f64::consts::PI,
        r_outer,
        r_co,
    );

    println!("field_sum: {}", field_sum(&input_field));
    let input_intensity = input_field.values.map(|e| e.norm_sqr());
    save_grayscale_real_image("bahtinov15.png", input_intensity.view(), 1.0, true).unwrap();
}

/// generate a bahtinov with an exponential chirp grating with radial window function
pub fn generate_custom_bahtinov_mask2(
    shape: usize,
    support_width: f64,
    grating_width: f64,
    grating_reduction_factor: f64,
    grating_area_fration: f64,
    grating_angle: f64,
    r_outer: f64,
    r_co: f64,
) -> Field {
    // width of the grating in the radial direction
    let max_width = 0.66 * r_outer;
    let min_width = 0.4 * r_outer;

    let circle2_distance = 2.0 * r_outer - max_width;

    let z = (circle2_distance - min_width) * 0.5;

    // distance from the radial midline to the ends of the grating (half width in the tangential direction)
    let end = (r_outer * r_outer - (min_width + z) * (min_width + z)).sqrt();
    // how far the center of the middle window is from the center of the circle
    let mid_distance = r_outer - max_width * 0.5;

    let pitch = r_outer / (div_up(shape, 2) - 2) as f64;
    let c = (shape / 2) as f64;

    let mut mask = Array2::zeros([shape, shape]);

    let grating_pitch = 2.0 * grating_width;

    // exponential chirp
    let exp = |distance: f64| {
        let a = grating_reduction_factor.ln() / end;
        let adjusted_distance = (distance * a).exp_m1() / (a * grating_pitch);
        adjusted_distance
    };

    let exp_inv = |y: f64| {
        (end * ((grating_pitch
            * grating_reduction_factor.ln()
            * (end / (grating_pitch * grating_reduction_factor.ln()) + y))
            / end)
            .ln())
            / grating_reduction_factor.ln()
    };

    // quadratic chirp
    let quadratic = |distance: f64| {
        let adjusted_distance = (distance
            + (grating_reduction_factor - 1.0) * 0.5 * distance * distance / end)
            / grating_pitch;
        adjusted_distance
    };

    let quadratic_inv = |y: f64| {
        ((end * (2.0 * grating_reduction_factor * grating_pitch * y + end)).sqrt() - end)
            / grating_reduction_factor
    };

    let window_function = |d: f64| {
        if d > 1.0 || d < -1.0 {
            return 0.0;
        }
        // d is the normalised distance from center [-1, 1]
        let pow = 4;
        1.0 - 2.0 * d.powi(pow) / (1.0 + d.powi(pow))
    };

    let window_locations: Vec<_> = (0..)
        .map(|i| {
            let mid = i as f64 + 0.5;
            let lower = mid - grating_area_fration * 0.5;
            let upper = mid + grating_area_fration * 0.5;
            (exp_inv(lower), exp_inv(mid), exp_inv(upper))
        })
        .take_while(|(_, mid, _)| mid < &end)
        .collect();

    Zip::indexed(&mut mask).par_for_each(|(y, x), e| {
        let mut value = 1.0;

        let y0 = (y as f64 - c) * pitch;
        let x0 = (x as f64 - c) * pitch;

        // input supports
        value *= soft_greater_than(x0.abs(), support_width * 0.5, pitch);
        if x0 > 0.0 {
            value *= soft_greater_than(y0.abs(), support_width * 0.5, pitch);
        }

        let r = (x0 * x0 + y0 * y0).sqrt();

        value *= soft_greater_than(r, r_co, pitch);
        value *= 1.0 - soft_greater_than(r, r_outer, pitch);

        if x0 > 0.0 {
            // angled
            if y0 > 0.0 {
                let theta = grating_angle;
                let x1 = theta.cos();
                let y1 = theta.sin();

                // distance along and from grating center line
                let tangential_dist =
                    (-x1 * (y1 - y0) - (x1 - x0) * -y1).abs() / (x1 * x1 + y1 * y1).sqrt();
                let radial_dist = x0 * x1 + y0 * y1;

                let mid_width = 2.0
                    * (r_outer * r_outer - tangential_dist * tangential_dist).sqrt()
                    - circle2_distance;

                let normalised_radial = 2.0 * (radial_dist - mid_distance) / mid_width;
                let tangential_index = min(
                    exp(tangential_dist).floor() as usize,
                    window_locations.len() - 1,
                );
                let (lower, mid, upper) = window_locations[tangential_index];

                if normalised_radial < 1.0 && normalised_radial > -1.0 {
                    value *= 1.0
                        - soft_greater_than(
                            tangential_dist - mid,
                            (upper - mid) * window_function(normalised_radial),
                            pitch,
                        );
                    value *= 1.0
                        - soft_greater_than(
                            mid - tangential_dist,
                            (mid - lower) * window_function(normalised_radial),
                            pitch,
                        );
                } else {
                    value *= 0.0
                }
                value *= 1.0 - soft_greater_than(tangential_dist, end, pitch);
            } else {
                let theta = -grating_angle;
                let x1 = theta.cos();
                let y1 = theta.sin();

                // distance along and from grating center line
                let tangential_dist =
                    (-x1 * (y1 - y0) - (x1 - x0) * -y1).abs() / (x1 * x1 + y1 * y1).sqrt();
                let radial_dist = x0 * x1 + y0 * y1;

                let mid_width = 2.0
                    * (r_outer * r_outer - tangential_dist * tangential_dist).sqrt()
                    - circle2_distance;

                let normalised_radial = 2.0 * (radial_dist - mid_distance) / mid_width;
                let tangential_index = min(
                    exp(tangential_dist).floor() as usize,
                    window_locations.len() - 1,
                );
                let (lower, mid, upper) = window_locations[tangential_index];

                if normalised_radial < 1.0 && normalised_radial > -1.0 {
                    value *= 1.0
                        - soft_greater_than(
                            tangential_dist - mid,
                            (upper - mid) * window_function(normalised_radial),
                            pitch,
                        );
                    value *= 1.0
                        - soft_greater_than(
                            mid - tangential_dist,
                            (mid - lower) * window_function(normalised_radial),
                            pitch,
                        );
                } else {
                    value *= 0.0
                }
                value *= 1.0 - soft_greater_than(tangential_dist, end, pitch);
            }
        } else {
            //vertical

            // distance along and from grating center line
            let tangential_dist = y0.abs();
            let radial_dist = -x0;

            let mid_width = 2.0 * (r_outer * r_outer - tangential_dist * tangential_dist).sqrt()
                - circle2_distance;

            let normalised_radial = 2.0 * (radial_dist - mid_distance) / mid_width;
            let tangential_index = min(
                exp(tangential_dist).floor() as usize,
                window_locations.len() - 1,
            );
            let (lower, mid, upper) = window_locations[tangential_index];

            if normalised_radial < 1.0 && normalised_radial > -1.0 {
                value *= 1.0
                    - soft_greater_than(
                        tangential_dist - mid,
                        (upper - mid) * window_function(normalised_radial),
                        pitch,
                    );
                value *= 1.0
                    - soft_greater_than(
                        mid - tangential_dist,
                        (mid - lower) * window_function(normalised_radial),
                        pitch,
                    );
            } else {
                value *= 0.0
            }
            value *= 1.0 - soft_greater_than(tangential_dist, end, pitch);
        }

        *e = Complex::new(value, 0.0);
    });

    Field {
        values: mask,
        pitch: (pitch, pitch),
    }
}

#[test]
fn bahtinov17() {
    let od = 0.0254 * 8.0;

    let mask_shape = 4096;
    let r_outer = od / 2.0;
    let r_co = od / 2.0 * 0.33;

    let input_field = generate_custom_bahtinov_mask3(
        mask_shape,
        0.005,
        0.005,
        5.0,
        0.3333,
        20.0 / 180.0 * ::std::f64::consts::PI,
        r_outer,
        r_co,
    );

    println!("field_sum: {}", field_sum(&input_field));
    let input_intensity = input_field.values.map(|e| e.norm_sqr());
    save_grayscale_real_image("bahtinov21.png", input_intensity.view(), 1.0, true).unwrap();
}

/// generate a bahtinov with an exponential chirp grating with radial window function
pub fn generate_custom_bahtinov_mask3(
    shape: usize,
    support_width: f64,
    grating_width: f64,
    grating_reduction_factor: f64,
    grating_area_fration: f64,
    grating_angle: f64,
    r_outer: f64,
    r_co: f64,
) -> Field {
    // width of the grating in the radial direction
    let max_width = 0.66 * r_outer;
    let min_width = 0.33 * r_outer;

    let circle2_distance = 2.0 * r_outer - max_width;

    let z = (circle2_distance - min_width) * 0.5;

    // distance from the radial midline to the ends of the grating (half width in the tangential direction)
    let end = (r_outer * r_outer - (min_width + z) * (min_width + z)).sqrt();
    // how far the center of the middle window is from the center of the circle
    let mid_distance = r_outer - max_width * 0.5;

    let pitch = r_outer / (div_up(shape, 2) - 2) as f64;
    let c = (shape / 2) as f64;

    let mut mask = Array2::zeros([shape, shape]);

    let grating_pitch = 2.0 * grating_width;

    // exponential chirp
    let exp = |distance: f64| {
        let a = grating_reduction_factor.ln() / end;
        let adjusted_distance = (distance * a).exp_m1() / (a * grating_pitch);
        adjusted_distance
    };

    let exp_inv = |y: f64| {
        (end * ((grating_pitch
            * grating_reduction_factor.ln()
            * (end / (grating_pitch * grating_reduction_factor.ln()) + y))
            / end)
            .ln())
            / grating_reduction_factor.ln()
    };

    // quadratic chirp
    let quadratic = |distance: f64| {
        let adjusted_distance = (distance
            + (grating_reduction_factor - 1.0) * 0.5 * distance * distance / end)
            / grating_pitch;
        adjusted_distance
    };

    let quadratic_inv = |y: f64| {
        ((end * (2.0 * grating_reduction_factor * grating_pitch * y + end)).sqrt() - end)
            / grating_reduction_factor
    };

    let window_function = |d: f64| {
        if d > 1.0 || d < -1.0 {
            return 0.0;
        }
        // d is the normalised distance from center [-1, 1]
        let pow = 6;
        1.0 - 2.0 * d.powi(pow) / (1.0 + d.powi(pow))
    };

    let window_locations: Vec<_> = (0..)
        .map(|i| {
            let mid = i as f64 + 0.5;
            let lower = mid - grating_area_fration * 0.5;
            let upper = mid + grating_area_fration * 0.5;
            (exp_inv(lower), exp_inv(mid), exp_inv(upper))
        })
        .take_while(|(_, mid, _)| mid < &end)
        .collect();

    Zip::indexed(&mut mask).par_for_each(|(y, x), e| {
        let mut value = 1.0;

        let y0 = (y as f64 - c) * pitch;
        let x0 = (x as f64 - c) * pitch;

        // input supports
        value *= soft_greater_than(x0.abs(), support_width * 0.5, pitch);
        if x0 > 0.0 {
            value *= soft_greater_than(y0.abs(), support_width * 0.5, pitch);
        }

        let r = (x0 * x0 + y0 * y0).sqrt();

        value *= soft_greater_than(r, r_co, pitch);
        value *= 1.0 - soft_greater_than(r, r_outer, pitch);

        if x0 > 0.0 {
            // angled
            if y0 > 0.0 {
                let theta = grating_angle;
                let x1 = theta.cos();
                let y1 = theta.sin();

                // distance along and from grating center line
                let tangential_dist =
                    (-x1 * (y1 - y0) - (x1 - x0) * -y1).abs() / (x1 * x1 + y1 * y1).sqrt();
                let radial_dist = x0 * x1 + y0 * y1;

                let mid_width = 2.0
                    * (r_outer * r_outer - tangential_dist * tangential_dist).sqrt()
                    - circle2_distance;

                let extra = (max_width - mid_width) * 0.5;

                let normalised_radial =
                    2.0 * (radial_dist - mid_distance + extra * 0.5) / (mid_width + extra);
                let tangential_index = min(
                    exp(tangential_dist).floor() as usize,
                    window_locations.len() - 1,
                );
                let (lower, mid, upper) = window_locations[tangential_index];

                if normalised_radial < 1.0 && normalised_radial > -1.0 {
                    value *= 1.0
                        - soft_greater_than(
                            tangential_dist - mid,
                            (upper - mid) * window_function(normalised_radial),
                            pitch,
                        );
                    value *= 1.0
                        - soft_greater_than(
                            mid - tangential_dist,
                            (mid - lower) * window_function(normalised_radial),
                            pitch,
                        );
                } else {
                    value *= 0.0
                }
                value *= 1.0 - soft_greater_than(tangential_dist, end, pitch);
            } else {
                let theta = -grating_angle;
                let x1 = theta.cos();
                let y1 = theta.sin();

                // distance along and from grating center line
                let tangential_dist =
                    (-x1 * (y1 - y0) - (x1 - x0) * -y1).abs() / (x1 * x1 + y1 * y1).sqrt();
                let radial_dist = x0 * x1 + y0 * y1;

                let mid_width = 2.0
                    * (r_outer * r_outer - tangential_dist * tangential_dist).sqrt()
                    - circle2_distance;

                let extra = (max_width - mid_width) * 0.5;

                let normalised_radial =
                    2.0 * (radial_dist - mid_distance + extra * 0.5) / (mid_width + extra);
                let tangential_index = min(
                    exp(tangential_dist).floor() as usize,
                    window_locations.len() - 1,
                );
                let (lower, mid, upper) = window_locations[tangential_index];

                if normalised_radial < 1.0 && normalised_radial > -1.0 {
                    value *= 1.0
                        - soft_greater_than(
                            tangential_dist - mid,
                            (upper - mid) * window_function(normalised_radial),
                            pitch,
                        );
                    value *= 1.0
                        - soft_greater_than(
                            mid - tangential_dist,
                            (mid - lower) * window_function(normalised_radial),
                            pitch,
                        );
                } else {
                    value *= 0.0
                }
                value *= 1.0 - soft_greater_than(tangential_dist, end, pitch);
            }
        } else {
            //vertical

            // distance along and from grating center line
            let tangential_dist = y0.abs();
            let radial_dist = -x0;

            let mid_width = 2.0 * (r_outer * r_outer - tangential_dist * tangential_dist).sqrt()
                - circle2_distance;

            let extra = (max_width - mid_width) * 0.5;

            let normalised_radial =
                2.0 * (radial_dist - mid_distance + extra * 0.5) / (mid_width + extra);
            let tangential_index = min(
                exp(tangential_dist).floor() as usize,
                window_locations.len() - 1,
            );
            let (lower, mid, upper) = window_locations[tangential_index];

            if normalised_radial < 1.0 && normalised_radial > -1.0 {
                value *= 1.0
                    - soft_greater_than(
                        tangential_dist - mid,
                        (upper - mid) * window_function(normalised_radial),
                        pitch,
                    );
                value *= 1.0
                    - soft_greater_than(
                        mid - tangential_dist,
                        (mid - lower) * window_function(normalised_radial),
                        pitch,
                    );
            } else {
                value *= 0.0
            }
            value *= 1.0 - soft_greater_than(tangential_dist, end, pitch);
        }

        *e = Complex::new(value, 0.0);
    });

    Field {
        values: mask,
        pitch: (pitch, pitch),
    }
}

fn huber_pow(x: f64) -> f64 {
    if x <= 0.0 || x >= 1.0 {
        return 0.0;
    };

    let x = (x - 0.5) * 2.0;
    let pow = 4;
    2.0 * (0.5 - x.powi(pow) / (1.0 + x.powi(pow)))
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

            let colour = Srgb::from_color(Lch::new(
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

fn field_sum(field: &Field) -> f64 {
    field.values.iter().fold(0.0, |acc, e| acc + e.norm_sqr()) / field.values.len() as f64
}

// #[cfg(test)]
// mod test {

//     #[test]
//     fn test_bahtinov_mask() {
//         let OD = 0.0254 * 8.0;

//         let mask_shape = 4096;
//         let r_outer = OD / 2.0;
//         let r_co = OD / 2.0 * 0.33;

//         let input_field = generate_bahtinov_mask(
//             mask_shape,
//             0.005,
//             0.005,
//             0.11111 * ::std::f64::consts::PI,
//             r_outer,
//             r_co,
//         );
//         println!("field_sum: {}", field_sum(input_field));
//         let input_intensity = input_field.values.map(|e| e.norm_sqr());
//         save_grayscale_real_image("bahtinov0.png", input_intensity.view(), 1.0, true).unwrap();
//     }

//     #[test]
//     fn test_bahtinov1_mask() {
//         let OD = 0.0254 * 8.0;

//         let mask_shape = 4096;
//         let r_outer = OD / 2.0;
//         let r_co = OD / 2.0 * 0.33;

//         let input_field = generate_bahtinov1_mask(
//             mask_shape,
//             0.005,
//             0.005,
//             0.11111 * ::std::f64::consts::PI,
//             r_outer,
//             r_co,
//         );
//         println!("field_sum: {}", field_sum(input_field));
//         let input_intensity = input_field.values.map(|e| e.norm_sqr());
//         save_grayscale_real_image("bahtinov1.png", input_intensity.view(), 1.0, true).unwrap();
//     }

//     #[test]
//     fn test_bahtinov2_mask() {
//         let OD = 0.0254 * 8.0;

//         let mask_shape = 4096;
//         let r_outer = OD / 2.0;
//         let r_co = OD / 2.0 * 0.33;

//         let input_field = generate_bahtinov2_mask(
//             mask_shape,
//             0.005,
//             0.005,
//             0.11111 * ::std::f64::consts::PI,
//             r_outer,
//             r_co,
//         );
//         println!("field_sum: {}", field_sum(input_field));
//         let input_intensity = input_field.values.map(|e| e.norm_sqr());
//         save_grayscale_real_image("bahtinov2.png", input_intensity.view(), 1.0, true).unwrap();
//     }

//     #[test]
//     fn test_bahtinov3_mask() {
//         let OD = 0.0254 * 8.0;

//         let mask_shape = 4096;
//         let r_outer = OD / 2.0;
//         let r_co = OD / 2.0 * 0.33;

//         let input_field = generate_bahtinov3_mask(
//             mask_shape,
//             0.005,
//             0.005,
//             0.11111 * ::std::f64::consts::PI,
//             r_outer,
//             r_co,
//         );
//         println!("field_sum: {}", field_sum(input_field));
//         let input_intensity = input_field.values.map(|e| e.norm_sqr());
//         save_grayscale_real_image("bahtinov3.png", input_intensity.view(), 1.0, true).unwrap();
//     }

//     #[test]
//     fn test_bahtinov4_mask() {
//         let OD = 0.0254 * 8.0;

//         let mask_shape = 4096;
//         let r_outer = OD / 2.0;
//         let r_co = OD / 2.0 * 0.33;

//         let input_field = generate_bahtinov4_mask(
//             mask_shape,
//             0.005,
//             0.005,
//             0.11111 * ::std::f64::consts::PI,
//             r_outer,
//             r_co,
//         );
//         println!("field_sum: {}", field_sum(input_field));
//         let input_intensity = input_field.values.map(|e| e.norm_sqr());
//         save_grayscale_real_image("bahtinov4.png", input_intensity.view(), 1.0, true).unwrap();
//     }

//     #[test]
//     fn test_bahtinov5_mask() {
//         let OD = 0.0254 * 8.0;

//         let mask_shape = 4096;
//         let r_outer = OD / 2.0;
//         let r_co = OD / 2.0 * 0.33;

//         let input_field = generate_bahtinov5_mask(
//             mask_shape,
//             0.005,
//             0.005,
//             0.11111 * ::std::f64::consts::PI,
//             r_outer,
//             r_co,
//         );
//         println!("field_sum: {}", field_sum(input_field));
//         let input_intensity = input_field.values.map(|e| e.norm_sqr());
//         save_grayscale_real_image("bahtinov5.png", input_intensity.view(), 1.0, true).unwrap();
//     }

//     #[test]
//     fn test_bahtinov6_mask() {
//         let OD = 0.0254 * 8.0;

//         let mask_shape = 4096;
//         let r_outer = OD / 2.0;
//         let r_co = OD / 2.0 * 0.33;

//         let input_field = generate_bahtinov6_mask(
//             mask_shape,
//             0.005,
//             0.005,
//             0.11111 * ::std::f64::consts::PI,
//             r_outer,
//             r_co,
//         );
//         println!("field_sum: {}", field_sum(input_field));
//         let input_intensity = input_field.values.map(|e| e.norm_sqr());
//         save_grayscale_real_image("bahtinov6.png", input_intensity.view(), 1.0, true).unwrap();
//     }
// }


#[test]
fn test_fresnel(){

    let od = 0.0254 * 8.0;

    let mask_shape = 4096;
    let r_outer = od / 2.0;
    let r_co = od / 2.0 * 0.33;
    let fl = 1.0;

    let input_field = generate_circle_mask(mask_shape, 1.0, r_outer, r_co);


    println!("field_sum: {}", field_sum(&input_field));
    let input_intensity = input_field.values.map(|e| e.norm_sqr());
    save_grayscale_real_image("circle.png", input_intensity.view(), 1.0, true).unwrap();


    let focal_field = fresnel(input_field, 1.0, 1.0, 700e-9, (2.0, 2.0));


    save_complex_image("fresnel.png", focal_field.values.view()).unwrap();
}