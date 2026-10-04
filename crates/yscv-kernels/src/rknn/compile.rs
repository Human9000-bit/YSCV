use super::backend::RknnBackend;
#[cfg(target_os = "linux")]
use super::consts::RKNN_SUCC;
#[cfg(target_os = "linux")]
use super::ffi::RknnContext;
use crate::KernelError;

// ══════════════════════════════════════════════════════════════════════
// On-device ONNX → RKNN compiler (uses librknn_api.so from toolkit2-lite)
// ══════════════════════════════════════════════════════════════════════

/// Configuration for on-device ONNX → RKNN conversion.
///
/// **Important — what this config does NOT control:**
///
/// On-device `librknn_api.so` exposes only `rknn_init` / `rknn_build` /
/// `rknn_export_rknn`. There is no SDK function to set `target_platform`,
/// `quantized_dtype`, channel `mean_values` / `std_values`, output node
/// list, etc. — those parameters are exclusively part of the **offline
/// host-side Python toolkit2** (`from rknn.api import RKNN; rknn.config(...)`).
///
/// For full configuration (target SoC, quant scheme, normalization,
/// op-by-op precision tuning) compile on the host with rknn-toolkit2 and
/// deploy the resulting `.rknn` file. Use this on-device path only when
/// you can accept defaults: fp16 conversion (no calibration) or int8
/// post-training quantization with a calibration dataset.
#[derive(Default)]
pub struct RknnCompileConfig {
    /// Path to a calibration dataset description file (one preprocessed
    /// image path per line, as expected by `rknn_build`'s `dataset_path`
    /// argument).
    ///
    /// `Some(path)` → the SDK runs int8 post-training quantization using
    /// this dataset.
    /// `None` → fp16 export, no quantization.
    pub dataset_path: Option<std::path::PathBuf>,
}

/// Convert an ONNX model to RKNN format for NPU deployment.
///
/// Performs the conversion using a process-wide unique temp file
/// (`tempfile::NamedTempFile`, automatically cleaned up). Safe under
/// concurrent invocation from multiple threads.
///
/// Requires `librknn_api.so` on the device (ships with rknn-toolkit2-lite).
/// Returns `Err(KernelError::RknnCompilerUnsupportedPlatform)` on non-Linux hosts since the
/// toolkit2 runtime library is Linux-only.
pub fn compile_onnx_to_rknn(
    onnx_data: &[u8],
    config: &RknnCompileConfig,
) -> Result<Vec<u8>, KernelError> {
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (onnx_data, config);
        Err(KernelError::RknnCompilerUnsupportedPlatform)
    }

    #[cfg(target_os = "linux")]
    {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt as _;
        use std::sync::Arc;

        type RknnCompilerInit = unsafe extern "C" fn(*mut RknnContext, *const u8, u32, u32) -> i32;
        type RknnCompilerBuild = unsafe extern "C" fn(RknnContext, u32, *const u8) -> i32;
        type RknnCompilerExport = unsafe extern "C" fn(RknnContext, *const u8) -> i32;
        type RknnCompilerDestroy = unsafe extern "C" fn(RknnContext) -> i32;

        // Validate calibration dataset path (if any) before we touch the SDK.
        let dataset_cstr: Option<CString> = if let Some(path) = &config.dataset_path {
            if !path.exists() {
                return Err(KernelError::RknnCalibrationDataset {
                    path: path.clone(),
                    reason: "not found",
                });
            }
            let len = path
                .metadata()
                .map(|m| m.len())
                .map_err(|e| KernelError::RknnIo {
                    what: "cannot stat calibration dataset",
                    path: path.clone(),
                    source: Arc::new(e),
                })?;
            if len == 0 {
                return Err(KernelError::RknnCalibrationDataset {
                    path: path.clone(),
                    reason: "empty",
                });
            }
            Some(
                CString::new(path.as_os_str().as_bytes())
                    .map_err(|_| KernelError::RknnPathContainsNul { path: path.clone() })?,
            )
        } else {
            None
        };

        // SAFETY: dlopen probes without constructors.
        let handle = unsafe { libc::dlopen(c"librknn_api.so".as_ptr(), libc::RTLD_LAZY) };
        if handle.is_null() {
            return Err(KernelError::RknnLibraryUnavailable {
                library: "librknn_api.so (install rknn-toolkit2-lite for model compilation)",
            });
        }

        macro_rules! load_sym {
            ($name:expr, $ty:ty) => {{
                // SAFETY: handle valid; name null-terminated.
                let ptr = unsafe { libc::dlsym(handle, $name.as_ptr().cast()) };
                if ptr.is_null() {
                    // SAFETY: handle valid.
                    unsafe { libc::dlclose(handle) };
                    return Err(KernelError::RknnSymbolMissing {
                        symbol: std::str::from_utf8(&$name[..$name.len() - 1])
                            .unwrap_or("<invalid>"),
                    });
                }
                // SAFETY: ptr non-null, transmuted to matching function type.
                unsafe { std::mem::transmute_copy::<*mut libc::c_void, $ty>(&ptr) }
            }};
        }

        let fn_init: RknnCompilerInit = load_sym!(b"rknn_init\0", RknnCompilerInit);
        let fn_destroy: RknnCompilerDestroy = load_sym!(b"rknn_destroy\0", RknnCompilerDestroy);
        let fn_build: RknnCompilerBuild = load_sym!(b"rknn_build\0", RknnCompilerBuild);
        let fn_export: RknnCompilerExport = load_sym!(b"rknn_export_rknn\0", RknnCompilerExport);

        let mut ctx: RknnContext = 0;
        // SAFETY: onnx_data valid; ctx writable.
        let ret = unsafe {
            fn_init(
                &mut ctx as *mut RknnContext,
                onnx_data.as_ptr(),
                onnx_data.len() as u32,
                0,
            )
        };
        if ret != RKNN_SUCC {
            // SAFETY: handle valid.
            unsafe { libc::dlclose(handle) };
            return Err(KernelError::RknnCall {
                func: "rknn_init (compiler)",
                status: ret,
            });
        }

        // do_quantization = 1 if calibration dataset supplied, else 0 (fp16 export).
        let do_quant = if dataset_cstr.is_some() { 1u32 } else { 0u32 };
        let dataset_ptr = dataset_cstr
            .as_ref()
            .map(|c| c.as_ptr().cast::<u8>())
            .unwrap_or(std::ptr::null());

        // SAFETY: ctx valid; dataset_ptr is null or points into dataset_cstr
        // (which lives until end of this function).
        let ret = unsafe { fn_build(ctx, do_quant, dataset_ptr) };
        if ret != RKNN_SUCC {
            // SAFETY: ctx/handle valid.
            unsafe {
                fn_destroy(ctx);
                libc::dlclose(handle);
            }
            return Err(KernelError::RknnCall {
                func: "rknn_build",
                status: ret,
            });
        }

        // Export to a unique temp file. NamedTempFile RAII-removes on drop;
        // safe under concurrent invocation from multiple threads.
        let tmp = tempfile::Builder::new()
            .prefix("yscv-rknn-")
            .suffix(".rknn")
            .tempfile()
            .map_err(|e| KernelError::RknnIo {
                what: "failed to create temp file for compiled model in",
                path: std::env::temp_dir(),
                source: Arc::new(e),
            })?;
        let tmp_path_cstr = CString::new(tmp.path().as_os_str().as_bytes()).map_err(|_| {
            KernelError::RknnPathContainsNul {
                path: tmp.path().to_path_buf(),
            }
        })?;

        // SAFETY: tmp_path_cstr is null-terminated; ctx valid.
        let ret = unsafe { fn_export(ctx, tmp_path_cstr.as_ptr().cast()) };

        // SAFETY: ctx/handle valid.
        unsafe {
            fn_destroy(ctx);
            libc::dlclose(handle);
        }

        if ret != RKNN_SUCC {
            return Err(KernelError::RknnCall {
                func: "rknn_export_rknn",
                status: ret,
            });
        }

        let bytes = std::fs::read(tmp.path()).map_err(|e| KernelError::RknnIo {
            what: "failed to read exported .rknn at",
            path: tmp.path().to_path_buf(),
            source: Arc::new(e),
        })?;
        // tmp dropped here → file removed.
        Ok(bytes)
    }
}

/// Check whether the RKNN compiler (`librknn_api.so`) is available.
pub fn rknn_compiler_available() -> bool {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: dlopen probes only.
        let handle = unsafe { libc::dlopen(c"librknn_api.so".as_ptr(), libc::RTLD_LAZY) };
        if handle.is_null() {
            return false;
        }
        // SAFETY: handle non-null.
        unsafe { libc::dlclose(handle) };
        true
    }
    #[cfg(not(target_os = "linux"))]
    {
        false
    }
}

/// Convenience: load ONNX, compile to RKNN, return ready backend.
///
/// If `cache_path` is supplied and the file exists, loads the cached
/// `.rknn` directly without re-compiling.
pub fn load_onnx_as_rknn(
    onnx_data: &[u8],
    cache_path: Option<&str>,
    config: &RknnCompileConfig,
) -> Result<RknnBackend, KernelError> {
    if let Some(path) = cache_path
        && let Ok(cached) = std::fs::read(path)
    {
        return RknnBackend::load(&cached);
    }
    let rknn_data = compile_onnx_to_rknn(onnx_data, config)?;
    if let Some(path) = cache_path {
        let _ = std::fs::write(path, &rknn_data); // best-effort cache
    }
    RknnBackend::load(&rknn_data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_default_means_fp16() {
        let cfg = RknnCompileConfig::default();
        assert!(cfg.dataset_path.is_none());
    }

    #[test]
    fn compiler_unavailable_off_linux() {
        // On non-Linux hosts (CI macOS/Windows runners) the compiler library
        // never resolves; rknn_compiler_available() must report false.
        #[cfg(not(target_os = "linux"))]
        assert!(!rknn_compiler_available());
    }

    #[cfg(not(target_os = "linux"))]
    #[test]
    fn compile_returns_error_off_linux() {
        let cfg = RknnCompileConfig::default();
        let err = compile_onnx_to_rknn(b"not a model", &cfg).unwrap_err();
        assert!(
            matches!(err, KernelError::RknnCompilerUnsupportedPlatform),
            "{err:?}"
        );
        assert!(err.to_string().contains("only available on Linux"), "{err}");
    }
}
