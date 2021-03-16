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
                value *= soft_greater_than(d, vane_width*0.5, pitch);
            }
        }
        *e = Complex::new(value, 0.0);
    });

    Field{
        values: mask,
        pitch: (pitch, pitch)
    }
}