use ndarray::Zip;
use num_complex::Complex;

use crate::{fft2::fft2c, scaling_czt3, Field};

pub fn fresnel_valid(A_xi: &Field, fl: f64, wavelength: f64) -> bool {
    let mid_x = A_xi.values.shape()[1] / 2;
    let mid_y = A_xi.values.shape()[0] / 2;
    // Which it is valid if ((x−x′)^2+(y−y′)^2)^2 << 8λL^3
    let limit = 8.0 * wavelength * fl * fl * fl;
    let distance_sqr = mid_x as f64 * mid_x as f64 * A_xi.pitch.1 * A_xi.pitch.1
        + mid_y as f64 * mid_y as f64 * A_xi.pitch.0 * A_xi.pitch.0;
    distance_sqr * distance_sqr < limit * 0.01 // needs to be in physical units not in pixel count
}

pub fn fresnel(mut A_xi: Field, fl: f64, z: f64, wavelength: f64, gamma: (f64, f64)) -> Field {
    // https://rafael-fuente.github.io/solving-the-diffraction-integral-with-the-fast-fourier-transform-fft-and-python.html

    let k = 2.0 * std::f64::consts::PI / wavelength;

    let mid_x = A_xi.values.shape()[1] / 2;
    let mid_y = A_xi.values.shape()[0] / 2;

    

    if z != fl {
        Zip::indexed(&mut A_xi.values)
        .par_for_each(|(yc, xc), value| {
                let x = (xc as f64 - mid_x as f64) * A_xi.pitch.1;
                let y = (yc as f64 - mid_y as f64) * A_xi.pitch.0;

                //let phase = k/(2.0*z) *(x*x + y*y) - k/(2.0*fl) *(x*x + y*y);
                let phase = k / (x * x + y * y) * (2.0 * z - 2.0 * fl);

                *value = Complex::new(0.0, phase).exp() * *value;
            });
    }

    // input dimensions
    let Lx = A_xi.values.shape()[1] as f64 * A_xi.pitch.1;
    let Ly = A_xi.values.shape()[0] as f64 * A_xi.pitch.0;

    // #screen size mm
    let dx_screen = z * wavelength / (2.0 * Lx);
    let dy_screen = z * wavelength / (2.0 * Ly);

    Field {
        values: scaling_czt3(A_xi.values, gamma), 
        // values: fft2c(A_xi.values),
        pitch: (dy_screen/ gamma.0, dx_screen/ gamma.1),
    }
}
