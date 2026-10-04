//! SafeTensors file parser and pretrained weight loader.
//!
//! The [SafeTensors](https://huggingface.co/docs/safetensors/) format stores
//! tensors in a simple binary layout:
//!
//! 1. 8 bytes — little-endian `u64` header length
//! 2. N bytes — UTF-8 JSON header mapping tensor names to metadata
//! 3. Remaining bytes — contiguous raw tensor data

use crate::ModelError;
use rustc_hash::FxHashMap;
use std::path::Path;
use std::sync::Arc;
use yscv_tensor::Tensor;

// ── Public types ────────────────────────────────────────────────────

/// Supported element types in a SafeTensors file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafeTensorDType {
    F32,
    F16,
    BF16,
    I32,
    I64,
    U8,
    Bool,
}

impl SafeTensorDType {
    /// Number of bytes per element.
    const fn element_size(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::I64 => 8,
            Self::U8 | Self::Bool => 1,
        }
    }

    fn from_str(s: &str) -> Result<Self, ModelError> {
        match s {
            "F32" => Ok(Self::F32),
            "F16" => Ok(Self::F16),
            "BF16" => Ok(Self::BF16),
            "I32" => Ok(Self::I32),
            "I64" => Ok(Self::I64),
            "U8" => Ok(Self::U8),
            "BOOL" => Ok(Self::Bool),
            other => Err(ModelError::UnsupportedDType {
                dtype: other.to_string(),
            }),
        }
    }
}

/// Per-tensor metadata extracted from the SafeTensors JSON header.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: SafeTensorDType,
    pub shape: Vec<usize>,
    pub data_offsets: (usize, usize),
}

/// A parsed SafeTensors file backed by an in-memory byte buffer.
pub struct SafeTensorFile {
    /// Parsed tensor metadata, keyed by name.
    tensors: FxHashMap<String, TensorInfo>,
    /// The raw data section (everything after the JSON header).
    data: Vec<u8>,
}

impl SafeTensorFile {
    /// Parse a SafeTensors file from disk.
    ///
    /// Reads the entire file into memory. For very large models,
    /// the OS will return an error if insufficient memory is available.
    pub fn from_file(path: &Path) -> Result<Self, ModelError> {
        let bytes = std::fs::read(path).map_err(|source| ModelError::Io {
            path: path.to_path_buf(),
            source: Arc::new(source),
        })?;
        Self::from_bytes(&bytes)
    }

    /// Parse a SafeTensors file from an in-memory byte slice.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ModelError> {
        if bytes.len() < 8 {
            return Err(ModelError::Truncated {
                what: "safetensors file",
                needed: 8,
                len: bytes.len(),
            });
        }

        let header_len = u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]) as usize;

        // Saturates past any real buffer length, so an overflowing header
        // length is reported as truncation.
        let header_end = 8usize.saturating_add(header_len);

        if bytes.len() < header_end {
            return Err(ModelError::Truncated {
                what: "safetensors header",
                needed: header_end,
                len: bytes.len(),
            });
        }

        let header_str = std::str::from_utf8(&bytes[8..header_end])?;

        let header_map: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(header_str).map_err(|source| ModelError::Json {
                what: "safetensors header",
                source: Arc::new(source),
            })?;

        let mut tensors = FxHashMap::default();
        for (name, value) in &header_map {
            if name == "__metadata__" {
                continue;
            }
            let obj = value
                .as_object()
                .ok_or_else(|| ModelError::InvalidTensorHeader {
                    tensor: name.clone(),
                    field: "object",
                })?;

            let dtype_str = obj.get("dtype").and_then(|v| v.as_str()).ok_or_else(|| {
                ModelError::InvalidTensorHeader {
                    tensor: name.clone(),
                    field: "dtype",
                }
            })?;

            let dtype = SafeTensorDType::from_str(dtype_str)?;

            let shape_arr = obj.get("shape").and_then(|v| v.as_array()).ok_or_else(|| {
                ModelError::InvalidTensorHeader {
                    tensor: name.clone(),
                    field: "shape",
                }
            })?;

            let shape: Vec<usize> = shape_arr
                .iter()
                .map(|v| {
                    v.as_u64()
                        .map(|n| n as usize)
                        .ok_or_else(|| ModelError::InvalidTensorHeader {
                            tensor: name.clone(),
                            field: "shape",
                        })
                })
                .collect::<Result<_, _>>()?;

            let offsets_arr = obj
                .get("data_offsets")
                .and_then(|v| v.as_array())
                .ok_or_else(|| ModelError::InvalidTensorHeader {
                    tensor: name.clone(),
                    field: "data_offsets",
                })?;

            if offsets_arr.len() != 2 {
                return Err(ModelError::InvalidTensorHeader {
                    tensor: name.clone(),
                    field: "data_offsets",
                });
            }

            let start = offsets_arr[0]
                .as_u64()
                .ok_or_else(|| ModelError::InvalidTensorHeader {
                    tensor: name.clone(),
                    field: "data_offsets",
                })? as usize;
            let end = offsets_arr[1]
                .as_u64()
                .ok_or_else(|| ModelError::InvalidTensorHeader {
                    tensor: name.clone(),
                    field: "data_offsets",
                })? as usize;

            tensors.insert(
                name.clone(),
                TensorInfo {
                    dtype,
                    shape,
                    data_offsets: (start, end),
                },
            );
        }

        let data = bytes[header_end..].to_vec();

        Ok(Self { tensors, data })
    }

    /// Returns a list of all tensor names in the file.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }

    /// Returns metadata for a tensor by name.
    pub fn tensor_info(&self, name: &str) -> Option<TensorInfo> {
        self.tensors.get(name).cloned()
    }

    /// Load a single tensor by name, converting to F32 if necessary.
    ///
    /// F16 and BF16 data are converted to F32. I32, I64, U8, and Bool are
    /// converted to F32 by casting each element.
    pub fn load_tensor(&self, name: &str) -> Result<Tensor, ModelError> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| ModelError::WeightNotFound {
                name: name.to_string(),
            })?;

        let (start, end) = info.data_offsets;
        if end > self.data.len() || start > end {
            return Err(ModelError::TensorDataOutOfBounds {
                name: name.to_string(),
                start,
                end,
                len: self.data.len(),
            });
        }

        let raw = &self.data[start..end];
        let elem_size = info.dtype.element_size();
        let expected_elements: usize = info.shape.iter().copied().product();
        let expected_bytes = expected_elements * elem_size;

        if raw.len() != expected_bytes {
            return Err(ModelError::TensorByteLenMismatch {
                name: name.to_string(),
                expected: expected_bytes,
                got: raw.len(),
            });
        }

        match info.dtype {
            SafeTensorDType::F32 => {
                let f32_data: Vec<f32> = raw
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Ok(Tensor::from_vec(info.shape.clone(), f32_data)?)
            }
            SafeTensorDType::F16 => {
                // Load as u16 bit patterns, then use Tensor's built-in to_dtype for F16->F32
                let u16_data: Vec<u16> = raw
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();
                let f16_tensor = Tensor::from_f16(info.shape.clone(), u16_data)?;
                Ok(f16_tensor.to_dtype(yscv_tensor::DType::F32))
            }
            SafeTensorDType::BF16 => {
                let u16_data: Vec<u16> = raw
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();
                let bf16_tensor = Tensor::from_bf16(info.shape.clone(), u16_data)?;
                Ok(bf16_tensor.to_dtype(yscv_tensor::DType::F32))
            }
            SafeTensorDType::I32 => {
                let f32_data: Vec<f32> = raw
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|chunk| {
                        i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32
                    })
                    .collect();
                Ok(Tensor::from_vec(info.shape.clone(), f32_data)?)
            }
            SafeTensorDType::I64 => {
                let f32_data: Vec<f32> = raw
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|chunk| {
                        i64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ]) as f32
                    })
                    .collect();
                Ok(Tensor::from_vec(info.shape.clone(), f32_data)?)
            }
            SafeTensorDType::U8 | SafeTensorDType::Bool => {
                let f32_data: Vec<f32> = raw.iter().map(|&b| b as f32).collect();
                Ok(Tensor::from_vec(info.shape.clone(), f32_data)?)
            }
        }
    }
}

/// Load all tensors from a SafeTensors file into a name-to-tensor map.
///
/// All tensors are converted to F32.
pub fn load_state_dict(path: &Path) -> Result<FxHashMap<String, Tensor>, ModelError> {
    let file = SafeTensorFile::from_file(path)?;
    let mut map = FxHashMap::default();
    for name in file.tensor_names() {
        let name_owned = name.to_string();
        let tensor = file.load_tensor(name)?;
        map.insert(name_owned, tensor);
    }
    Ok(map)
}
