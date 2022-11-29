// [[file:../enm.note::d5052804][d5052804]]
use gut::prelude::*;
use nalgebra::{DMatrix, DVector};
use vecfx::*;

/// Anisotropic Network Model (ANM) analysis
///
/// # References
///
/// - Atilgan, A. R.; Durell, S. R.; Jernigan, R. L.; Demirel, M. C.;
/// Keskin, O.; Bahar, I. Anisotropy of Fluctuation Dynamics of
/// Proteins with an Elastic Network Model. Biophysical Journal 2001,
/// 80 (1), 505–515. <https://doi.org/10.1016/S0006-3495(01)76033-X>
pub struct AnisotropicNetworkModel {
    cutoff: f64,
    gamma: f64,
}

impl Default for AnisotropicNetworkModel {
    fn default() -> Self {
        Self {
            cutoff: 15.0,
            gamma: 1.0,
        }
    }
}

/// Calculates the normal modes by diagonalizing the Hessian matrix
/// `hessian`. Returns 3N-6 eigen values sorted in ascending order and
/// their associated eigen vectors with 6 translational and rotational
/// modes removed.
pub fn calculate_normal_modes(hessian: DMatrix<f64>) -> (Vec<f64>, Vec<DVector<f64>>) {
    let eigen = hessian.symmetric_eigen();
    let vectors = eigen.eigenvectors;
    let evalues = eigen.eigenvalues;

    // sort the eigenvalues in ascending order
    let indices: Vec<_> = evalues
        .iter()
        .enumerate()
        .sorted_by_key(|x| OrderedFloat(*x.1))
        .map(|x| x.0)
        .collect();

    // sort the corresponding eigenvectors in ascending order
    let mut evalues_ = vec![];
    let mut vectors_ = vec![];
    for &i in indices.iter() {
        evalues_.push(evalues[i]);
        vectors_.push(vectors.column(i).into_owned());
    }

    // skip the first 6 modes with zero eigenvalues for translation or rotation
    let evalues = evalues_.into_iter().skip(6).collect_vec();
    let vectors = vectors_.into_iter().skip(6).collect_vec();
    (evalues, vectors)
}

impl AnisotropicNetworkModel {
    /// Build Hessian matrix (3N*3N) for Cartesian `coords` of N atoms.
    pub fn build_hessian_matrix(&self, coords: &[[f64; 3]]) -> DMatrix<f64> {
        let n = coords.len();
        let data = vec![0.0; 3 * n * 3 * n];

        let gamma = self.gamma;
        let cutoff2 = self.cutoff.powi(2);

        let mut hessian = DMatrix::from_vec(3 * n, 3 * n, data);
        for i in 0..n {
            for j in 0..i {
                assert_ne!(i, j);
                let ri: Vector3f = coords[i].into();
                let rj: Vector3f = coords[j].into();
                let rij = rj - ri;
                let dist2 = (rj - ri).norm_squared();
                if dist2 < cutoff2 {
                    let super_element = -gamma / dist2 * rij * rij.transpose();
                    let mut sub = hessian.fixed_slice_mut::<3, 3>(i * 3, j * 3);
                    sub.copy_from(&super_element);
                    let mut sub = hessian.fixed_slice_mut::<3, 3>(j * 3, i * 3);
                    sub.copy_from(&super_element);
                    let mut sub = hessian.fixed_slice_mut::<3, 3>(i * 3, i * 3);
                    sub -= super_element;
                    let mut sub = hessian.fixed_slice_mut::<3, 3>(j * 3, j * 3);
                    sub -= super_element;
                }
            }
        }
        hessian
    }
}

#[test]
fn test_enm() {
    use approx::*;

    #[rustfmt::skip]
    let coords = [[ -1.72300000,   1.18800000,   1.85600000],
                  [ -3.40400000,   0.60000000,   1.76800000],
                  [ -4.67400000,  -1.11300000,   0.60100000],
                  [ -2.96700000,  -0.68200000,   0.54500000],
                  [ -3.09400000,   2.29500000,   1.39200000],
                  [ -2.51000000,   1.07900000,   0.26100000],
                  [ -4.25300000,   0.54000000,   0.15700000],
                  [ -3.85700000,  -0.76600000,  -0.99200000]];

    let hessian = AnisotropicNetworkModel::default().build_hessian_matrix(&coords);
    let (evalues, vectors) = calculate_normal_modes(hessian);

    assert_relative_eq!(evalues[0], 0.47256486306316137, epsilon = 1E-4);
    assert_relative_eq!(evalues[1], 0.824857, epsilon = 1E-4);
    assert_relative_eq!(evalues[2], 0.828897, epsilon = 1E-4);
    assert_relative_eq!(evalues[3], 1.051973, epsilon = 1E-4);

    assert_relative_eq!(vectors[0][0], 0.22011, epsilon = 1E-4);
    assert_relative_eq!(vectors[0][2], -0.36812, epsilon = 1E-4);
}
// d5052804 ends here
