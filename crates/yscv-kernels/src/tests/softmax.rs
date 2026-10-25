use yscv_tensor::{Tensor, TensorError};

use crate::{
    KernelError, ParallelElementwiseConfig, log_softmax_last_dim, log_softmax_last_dim_out,
    log_softmax_last_dim_with_config, logsumexp_last_dim, logsumexp_last_dim_with_config,
    softmax_last_dim, softmax_last_dim_out, softmax_last_dim_with_config,
};

use super::{assert_slice_close, build_tensor};

#[test]
fn softmax_last_dim_computes_expected_result() {
    let input = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let out = softmax_last_dim(&input).unwrap();
    assert_eq!(out.shape(), &[3]);
    assert_slice_close(out.data(), &[0.09003057, 0.24472848, 0.66524094], 1e-6);
}

#[test]
fn softmax_last_dim_normalizes_each_row() {
    let input = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 0.1, 0.2, 0.3]).unwrap();
    let out = softmax_last_dim(&input).unwrap();
    assert_eq!(out.shape(), &[2, 3]);
    let row0_sum = out.data()[0] + out.data()[1] + out.data()[2];
    let row1_sum = out.data()[3] + out.data()[4] + out.data()[5];
    assert!((row0_sum - 1.0).abs() <= 1e-6);
    assert!((row1_sum - 1.0).abs() <= 1e-6);
}

#[test]
fn softmax_last_dim_rejects_scalar_rank() {
    let input = Tensor::scalar(1.0);
    let err = softmax_last_dim(&input).unwrap_err();
    assert!(
        matches!(err, KernelError::InvalidSoftmaxRank { got_rank: 0 }),
        "{err:?}"
    );
}

#[test]
// Miri's software FP emulation is non-deterministic: two calls to f32::exp()
// with the same input can return different bit patterns (known Miri limitation).
#[cfg_attr(miri, ignore)]
fn softmax_last_dim_with_config_disabled_matches_default() {
    let input = build_tensor(&[64, 32], 0.52);
    let baseline = softmax_last_dim(&input).unwrap();
    let disabled =
        softmax_last_dim_with_config(&input, ParallelElementwiseConfig::disabled()).unwrap();
    assert_eq!(baseline, disabled);
}

#[test]
fn softmax_last_dim_out_matches_allocating_path() {
    let input = build_tensor(&[64, 32], 0.52);
    let baseline = softmax_last_dim(&input).unwrap();
    let mut output = Tensor::zeros(input.shape().to_vec()).unwrap();
    softmax_last_dim_out(&input, &mut output).unwrap();
    assert_eq!(baseline, output);
}

#[test]
fn softmax_last_dim_out_rejects_shape_mismatch() {
    let input = build_tensor(&[2, 3], 0.52);
    let mut output = Tensor::zeros(vec![3, 2]).unwrap();
    let err = softmax_last_dim_out(&input, &mut output).unwrap_err();
    assert!(matches!(
        err,
        KernelError::Tensor(TensorError::ShapeMismatch { .. })
    ));
}

#[test]
fn log_softmax_last_dim_computes_expected_result() {
    let input = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let out = log_softmax_last_dim(&input).unwrap();
    assert_eq!(out.shape(), &[3]);
    let log_denom = (1.0f32.exp() + 2.0f32.exp() + 3.0f32.exp()).ln();
    assert_slice_close(
        out.data(),
        &[1.0 - log_denom, 2.0 - log_denom, 3.0 - log_denom],
        1e-6,
    );
}

#[test]
fn log_softmax_last_dim_matches_ln_softmax() {
    let input = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 0.1, 0.2, 0.3]).unwrap();
    let softmax = softmax_last_dim(&input).unwrap();
    let log_softmax = log_softmax_last_dim(&input).unwrap();
    assert_eq!(softmax.shape(), log_softmax.shape());
    for index in 0..softmax.len() {
        assert!(
            (softmax.data()[index].ln() - log_softmax.data()[index]).abs() <= 1e-5,
            "mismatch at {index}: ln(softmax)={} log_softmax={}",
            softmax.data()[index].ln(),
            log_softmax.data()[index]
        );
    }
}

/// Softmax Collapse (Barron et al., 2026): a confidently-correct logit must keep
/// its (tiny, negative) log-probability instead of rounding to exactly `0.0`.
/// `1 + e^-18` is `1.0` in f32, which is why the standard formulation collapses.
#[test]
fn log_softmax_last_dim_does_not_collapse_at_high_margin() {
    let input = Tensor::from_vec(vec![3], vec![9.0, -9.0, -9.0]).unwrap();
    let out = log_softmax_last_dim(&input).unwrap();

    assert!(
        out.data()[0] < 0.0,
        "dominant logit collapsed: {}",
        out.data()[0]
    );
    assert!(
        (out.data()[0] + 3.045_995_8e-8).abs() <= 1e-14,
        "expected -3.046e-8, got {:e}",
        out.data()[0]
    );
    assert_eq!(out.data()[1], -18.0);
    assert_eq!(out.data()[2], -18.0);
}

#[test]
fn log_softmax_last_dim_preserves_tied_maxima() {
    // All logits equal: every log-probability is -ln(K).
    let input = Tensor::from_vec(vec![3], vec![0.0, 0.0, 0.0]).unwrap();
    let out = log_softmax_last_dim(&input).unwrap();
    let expected = -(3.0f32.ln());
    assert_slice_close(out.data(), &[expected, expected, expected], 1e-6);

    // Tied maxima plus a lower logit.
    let input = Tensor::from_vec(vec![3], vec![0.0, 0.0, -1.0]).unwrap();
    let out = log_softmax_last_dim(&input).unwrap();
    let top = -((2.0f32 + (-1.0f32).exp()).ln());
    let bottom = -1.0 - (2.0f32 + (-1.0f32).exp()).ln();
    assert_slice_close(out.data(), &[top, top, bottom], 1e-6);
}

#[test]
fn log_softmax_last_dim_keeps_mass_beyond_the_precision_cliff() {
    // One dominant logit, many non-max ones, across scalar/SIMD and tail widths.
    for len in [2usize, 3, 4, 7, 8, 16, 17, 32, 33, 1000] {
        let mut data = vec![-20.0f32; len];
        data[0] = 20.0;
        let input = Tensor::from_vec(vec![len], data).unwrap();
        let out = log_softmax_last_dim(&input).unwrap();

        // Margin 40: log_softmax(20) = -ln(1 + (len-1) * e^-40) ~= -(len-1) * e^-40.
        let expected = -((len - 1) as f64 * (-40.0f64).exp());
        let got = out.data()[0] as f64;
        assert!(got < 0.0, "len={len}: dominant logit collapsed to {got:e}");
        assert!(
            (got - expected).abs() <= expected.abs() * 1e-4,
            "len={len}: expected {expected:e}, got {got:e}"
        );
        for index in 1..len {
            assert_eq!(out.data()[index], -40.0, "len={len} index={index}");
        }
        // exp(log_softmax) still sums to one.
        let total: f32 = out.data().iter().map(|v| v.exp()).sum();
        assert!((total - 1.0).abs() <= 1e-6, "len={len}: total={total}");
    }
}

#[test]
fn log_softmax_last_dim_rejects_scalar_rank() {
    let input = Tensor::scalar(1.0);
    let err = log_softmax_last_dim(&input).unwrap_err();
    assert!(
        matches!(err, KernelError::InvalidLogSoftmaxRank { got_rank: 0 }),
        "{err:?}"
    );
}

#[test]
// Miri's software FP emulation is non-deterministic: two calls to f32::exp()
// with the same input can return different bit patterns (known Miri limitation).
#[cfg_attr(miri, ignore)]
fn log_softmax_last_dim_with_config_disabled_matches_default() {
    let input = build_tensor(&[64, 32], 0.66);
    let baseline = log_softmax_last_dim(&input).unwrap();
    let disabled =
        log_softmax_last_dim_with_config(&input, ParallelElementwiseConfig::disabled()).unwrap();
    assert_eq!(baseline, disabled);
}

#[test]
fn log_softmax_last_dim_out_matches_allocating_path() {
    let input = build_tensor(&[64, 32], 0.66);
    let baseline = log_softmax_last_dim(&input).unwrap();
    let mut output = Tensor::zeros(input.shape().to_vec()).unwrap();
    log_softmax_last_dim_out(&input, &mut output).unwrap();
    assert_eq!(baseline, output);
}

#[test]
fn logsumexp_last_dim_computes_expected_result() {
    let input = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let out = logsumexp_last_dim(&input).unwrap();
    assert_eq!(out.shape(), &[1]);
    let expected = (1.0f32.exp() + 2.0f32.exp() + 3.0f32.exp()).ln();
    assert_slice_close(out.data(), &[expected], 1e-5);
}

#[test]
fn logsumexp_last_dim_reduces_each_row() {
    let input = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 0.1, 0.2, 0.3]).unwrap();
    let out = logsumexp_last_dim(&input).unwrap();
    assert_eq!(out.shape(), &[2, 1]);
    let row0 = (1.0f32.exp() + 2.0f32.exp() + 3.0f32.exp()).ln();
    let row1 = (0.1f32.exp() + 0.2f32.exp() + 0.3f32.exp()).ln();
    assert_slice_close(out.data(), &[row0, row1], 1e-5);
}

#[test]
fn logsumexp_last_dim_rejects_scalar_rank() {
    let input = Tensor::scalar(1.0);
    let err = logsumexp_last_dim(&input).unwrap_err();
    assert!(
        matches!(err, KernelError::InvalidLogSumExpRank { got_rank: 0 }),
        "{err:?}"
    );
}

#[test]
// Miri's software FP emulation is non-deterministic: two calls to f32::exp()
// with the same input can return different bit patterns (known Miri limitation).
#[cfg_attr(miri, ignore)]
fn logsumexp_last_dim_with_config_disabled_matches_default() {
    let input = build_tensor(&[64, 32], 0.77);
    let baseline = logsumexp_last_dim(&input).unwrap();
    let disabled =
        logsumexp_last_dim_with_config(&input, ParallelElementwiseConfig::disabled()).unwrap();
    assert_eq!(baseline, disabled);
}
