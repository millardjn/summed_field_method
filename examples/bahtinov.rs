use image::{Rgb, RgbImage};
use ndarray::{ArrayView2};
use ndarray::{Array2, Zip};
use num_complex::Complex;
use palette::{Lch, LinSrgb, Srgb};
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
        let input_intensity = input_field.values.map(|e| e.norm_sqr());
        save_grayscale_real_image("bahtinov1.png", input_intensity.view(), 1.0, true).unwrap();
 

        // let input_field = generate_bahtinov2_mask(
        //     mask_shape,
        //     0.005,
        //     0.005,
        //     0.11111 * ::std::f64::consts::PI,
        //     r_outer,
        //     r_co,
        // );
        // let input_intensity = input_field.values.map(|e| e.norm_sqr());
        // save_grayscale_real_image("bahtinov2.png", input_intensity.view(), 1.0, true).unwrap();


        // let input_field = generate_bahtinov3_mask(
        //     mask_shape,
        //     0.005,
        //     0.005,
        //     0.11111 * ::std::f64::consts::PI,
        //     r_outer,
        //     r_co,
        // );
        // let input_intensity = input_field.values.map(|e| e.norm_sqr());
        // save_grayscale_real_image("bahtinov3.png", input_intensity.view(), 1.0, true).unwrap();


        // let input_field = generate_bahtinov4_mask(
        //     mask_shape,
        //     0.005,
        //     0.005,
        //     0.11111 * ::std::f64::consts::PI,
        //     r_outer,
        //     r_co,
        // );
        // let input_intensity = input_field.values.map(|e| e.norm_sqr());
        // save_grayscale_real_image("bahtinov4.png", input_intensity.view(), 1.0, true).unwrap();


        let input_field = generate_lin_chirp_bahtinov_mask(
            mask_shape,
            0.005,
            0.005,
            0.11111 * ::std::f64::consts::PI,
            r_outer,
            r_co,
        );
        let input_intensity = input_field.values.map(|e| e.norm_sqr());
        save_grayscale_real_image("bahtinov5.png", input_intensity.view(), 1.0, true).unwrap();


        let input_field = generate_exp_chirp_bahtinov_mask(
            mask_shape,
            0.005,
            0.005,
            0.11111 * ::std::f64::consts::PI,
            r_outer,
            r_co,
        );
        let input_intensity = input_field.values.map(|e| e.norm_sqr());
        save_grayscale_real_image("bahtinov6.png", input_intensity.view(), 1.0, true).unwrap();
    
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

    Zip::indexed(&mut mask).par_apply(|(y, x), e| {
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

    Zip::indexed(&mut mask).par_apply(|(y, x), e| {
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

    Zip::indexed(&mut mask).par_apply(|(y, x), e| {
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
    grating_angle: f64,
    r_outer: f64,
    r_co: f64,
) -> Field {
    let end = r_outer / 2.0;

    let pitch = r_outer / (div_up(shape, 2) - 2) as f64;
    let c = (shape / 2) as f64;

    let mut mask = Array2::zeros([shape, shape]);

    Zip::indexed(&mut mask).par_apply(|(y, x), e| {
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

                let a = 5.0f64.ln() / end;
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

                let a = 5.0f64.ln() / end;
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
            let a = 5.0f64.ln() / end;
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

#[cfg(test)]
mod test {
    #[test]
    fn test_bahtinov_mask() {
        let OD = 0.0254 * 8.0;

        let mask_shape = 4096;
        let r_outer = OD / 2.0;
        let r_co = OD / 2.0 * 0.33;

        let input_field = generate_bahtinov_mask(
            mask_shape,
            0.005,
            0.005,
            0.11111 * ::std::f64::consts::PI,
            r_outer,
            r_co,
        );
        let input_intensity = input_field.values.map(|e| e.norm_sqr());
        save_grayscale_real_image("bahtinov0.png", input_intensity.view(), 1.0, true).unwrap();
    }

    #[test]
    fn test_bahtinov1_mask() {
        let OD = 0.0254 * 8.0;

        let mask_shape = 4096;
        let r_outer = OD / 2.0;
        let r_co = OD / 2.0 * 0.33;

        let input_field = generate_bahtinov1_mask(
            mask_shape,
            0.005,
            0.005,
            0.11111 * ::std::f64::consts::PI,
            r_outer,
            r_co,
        );
        let input_intensity = input_field.values.map(|e| e.norm_sqr());
        save_grayscale_real_image("bahtinov1.png", input_intensity.view(), 1.0, true).unwrap();
    }

    #[test]
    fn test_bahtinov2_mask() {
        let OD = 0.0254 * 8.0;

        let mask_shape = 4096;
        let r_outer = OD / 2.0;
        let r_co = OD / 2.0 * 0.33;

        let input_field = generate_bahtinov2_mask(
            mask_shape,
            0.005,
            0.005,
            0.11111 * ::std::f64::consts::PI,
            r_outer,
            r_co,
        );
        let input_intensity = input_field.values.map(|e| e.norm_sqr());
        save_grayscale_real_image("bahtinov2.png", input_intensity.view(), 1.0, true).unwrap();
    }

    #[test]
    fn test_bahtinov3_mask() {
        let OD = 0.0254 * 8.0;

        let mask_shape = 4096;
        let r_outer = OD / 2.0;
        let r_co = OD / 2.0 * 0.33;

        let input_field = generate_bahtinov3_mask(
            mask_shape,
            0.005,
            0.005,
            0.11111 * ::std::f64::consts::PI,
            r_outer,
            r_co,
        );
        let input_intensity = input_field.values.map(|e| e.norm_sqr());
        save_grayscale_real_image("bahtinov3.png", input_intensity.view(), 1.0, true).unwrap();
    }

    #[test]
    fn test_bahtinov4_mask() {
        let OD = 0.0254 * 8.0;

        let mask_shape = 4096;
        let r_outer = OD / 2.0;
        let r_co = OD / 2.0 * 0.33;

        let input_field = generate_bahtinov4_mask(
            mask_shape,
            0.005,
            0.005,
            0.11111 * ::std::f64::consts::PI,
            r_outer,
            r_co,
        );
        let input_intensity = input_field.values.map(|e| e.norm_sqr());
        save_grayscale_real_image("bahtinov4.png", input_intensity.view(), 1.0, true).unwrap();
    }

    #[test]
    fn test_bahtinov5_mask() {
        let OD = 0.0254 * 8.0;

        let mask_shape = 4096;
        let r_outer = OD / 2.0;
        let r_co = OD / 2.0 * 0.33;

        let input_field = generate_bahtinov5_mask(
            mask_shape,
            0.005,
            0.005,
            0.11111 * ::std::f64::consts::PI,
            r_outer,
            r_co,
        );
        let input_intensity = input_field.values.map(|e| e.norm_sqr());
        save_grayscale_real_image("bahtinov5.png", input_intensity.view(), 1.0, true).unwrap();
    }

    #[test]
    fn test_bahtinov6_mask() {
        let OD = 0.0254 * 8.0;

        let mask_shape = 4096;
        let r_outer = OD / 2.0;
        let r_co = OD / 2.0 * 0.33;

        let input_field = generate_bahtinov6_mask(
            mask_shape,
            0.005,
            0.005,
            0.11111 * ::std::f64::consts::PI,
            r_outer,
            r_co,
        );
        let input_intensity = input_field.values.map(|e| e.norm_sqr());
        save_grayscale_real_image("bahtinov6.png", input_intensity.view(), 1.0, true).unwrap();
    }
}
