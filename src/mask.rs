use crate::{div_up, Field};
use ndarray::{Array2, Zip};
use num_complex::Complex;

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

    Zip::indexed(&mut mask).par_apply(|(y, x), e| {
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

pub fn generate_bahtinov2_mask(
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

                let projected_r = x1 * x0 + y1 * y0;
                value *= soft_greater_than(projected_r, r_outer * 0.5, pitch);
            } else {
                let theta = -grating_angle;
                let x1 = theta.cos();
                let y1 = theta.sin();

                // distance from line
                let d = (-x1 * (y1 - y0) - (x1 - x0) * -y1).abs() / (x1 * x1 + y1 * y1).sqrt();

                value *= (-(d / grating_width * ::std::f64::consts::PI).cos().signum() + 1.0) * 0.5;

                let projected_r = x1 * x0 + y1 * y0;
                value *= soft_greater_than(projected_r, r_outer * 0.5, pitch);
            }
        } else {
            //vertical
            value *= (-(y0 / grating_width * ::std::f64::consts::PI).cos().signum() + 1.0) * 0.5;

            value *= soft_greater_than(x0.abs(), r_outer * 0.5, pitch);
        }

        *e = Complex::new(value, 0.0);
    });

    Field {
        values: mask,
        pitch: (pitch, pitch),
    }
}

pub fn generate_bahtinov3_mask(
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

                let x = (d + 0.5 * d * d / end) / grating_width;
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

                //let x = d/grating_width + (d/grating_width)*(d/grating_width);

                let x = (d + 0.5 * d * d / end) / grating_width;
                value *= 1.0 - soft_greater_than(d, end, pitch);
                value *= (-(x * ::std::f64::consts::PI).cos().signum() + 1.0) * 0.5;

                let x2 = r_outer * 1.5 * x1;
                let y2 = r_outer * 1.5 * y1;
                let rr = ((x0 - x2) * (x0 - x2) + (y0 - y2) * (y0 - y2)).sqrt();
                value *= 1.0 - soft_greater_than(rr, r_outer, pitch);
            }
        } else {
            //vertical
            let x = (y0.abs() + 0.5 * y0 * y0 / end) / grating_width;
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

pub fn generate_bahtinov4_mask(
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

                let x = (d + d * d / end) / grating_width;
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

                //let x = d/grating_width + (d/grating_width)*(d/grating_width);

                let x = (d + d * d / end) / grating_width;
                value *= 1.0 - soft_greater_than(d, end, pitch);
                value *= (-(x * ::std::f64::consts::PI).cos().signum() + 1.0) * 0.5;

                let x2 = r_outer * 1.5 * x1;
                let y2 = r_outer * 1.5 * y1;
                let rr = ((x0 - x2) * (x0 - x2) + (y0 - y2) * (y0 - y2)).sqrt();
                value *= 1.0 - soft_greater_than(rr, r_outer, pitch);
            }
        } else {
            //vertical
            let x = (y0.abs() + y0 * y0 / end) / grating_width;
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

pub fn generate_bahtinov5_mask(
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

                //let x = d/grating_width + (d/grating_width)*(d/grating_width);

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

pub fn generate_bahtinov6_mask(
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
