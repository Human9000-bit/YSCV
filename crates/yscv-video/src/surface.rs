//! Checked composition into caller-owned packed byte surfaces.
//!
//! Nearest-neighbor plans cache byte offsets, fuse scaling with conversion, and
//! copy repeated output rows. Rendering allocates nothing and uses no worker pool.

use crate::VideoError;

#[path = "surface_simd.rs"]
mod simd;

/// Byte order in memory, independent of host endianness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceFormat {
    /// R, G, B bytes.
    Rgb8,
    /// B, G, R, A bytes; writes use opaque alpha (255).
    Bgra8,
    /// Five red, six green, five blue bits, stored low byte first.
    Rgb565Le,
}

impl SurfaceFormat {
    /// Number of bytes occupied by one pixel.
    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Rgb8 => 3,
            Self::Bgra8 => 4,
            Self::Rgb565Le => 2,
        }
    }
}

/// A validated mutable view. Row padding is never modified by drawing operations.
pub struct SurfaceMut<'a> {
    bytes: &'a mut [u8],
    width: usize,
    height: usize,
    stride: usize,
    format: SurfaceFormat,
}

impl<'a> SurfaceMut<'a> {
    /// Validate nonzero dimensions, byte stride, buffer length and size overflow.
    /// Padding after the final visible row is optional.
    pub fn new(
        bytes: &'a mut [u8],
        width: usize,
        height: usize,
        stride: usize,
        format: SurfaceFormat,
    ) -> Result<Self, VideoError> {
        let row = packed_len(width, 1, format.bytes_per_pixel())?;
        let required = height
            .checked_sub(1)
            .and_then(|h| h.checked_mul(stride))
            .and_then(|n| n.checked_add(row))
            .ok_or(VideoError::InvalidDimensions {
                what: "surface",
                width,
                height,
            })?;
        check_len(stride, row, "surface stride")?;
        check_len(bytes.len(), required, "surface buffer")?;
        Ok(Self {
            bytes,
            width,
            height,
            stride,
            format,
        })
    }

    /// Visible width in pixels.
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Visible height in pixels.
    pub const fn height(&self) -> usize {
        self.height
    }

    /// Fill a rectangle clipped to the visible surface. Oversized coordinates and
    /// extents are safe; empty or wholly offscreen rectangles do nothing.
    pub fn fill_rect(
        &mut self,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
        rgb: (u8, u8, u8),
    ) {
        let width = width.min(self.width.saturating_sub(x));
        let height = height.min(self.height.saturating_sub(y));
        if width == 0 || height == 0 {
            return;
        }
        let bpp = self.format.bytes_per_pixel();
        let start = y * self.stride + x * bpp;
        let row = &mut self.bytes[start..start + width * bpp];
        let (r, g, b) = rgb;
        match self.format {
            SurfaceFormat::Rgb8 => row.as_chunks_mut::<3>().0.fill([r, g, b]),
            SurfaceFormat::Bgra8 => row.as_chunks_mut::<4>().0.fill([b, g, r, 255]),
            SurfaceFormat::Rgb565Le => row.as_chunks_mut::<2>().0.fill(rgb565(r, g, b)),
        }
        for dy in 1..height {
            self.bytes
                .copy_within(start..start + width * bpp, start + dy * self.stride);
        }
    }

    /// Draw a clipped rectangular outline, with thickness confined to its bounds.
    pub fn draw_rect(
        &mut self,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
        thickness: usize,
        rgb: (u8, u8, u8),
    ) {
        let tx = thickness.min(width);
        let ty = thickness.min(height);
        self.fill_rect(x, y, width, ty, rgb);
        self.fill_rect(
            x,
            y.saturating_add(height.saturating_sub(ty)),
            width,
            ty,
            rgb,
        );
        self.fill_rect(x, y, tx, height, rgb);
        self.fill_rect(
            x.saturating_add(width.saturating_sub(tx)),
            y,
            tx,
            height,
            rgb,
        );
    }
}

/// Reusable nearest-neighbor RGB8 scaling plan.
///
/// Pixel `(x, y)` reads `(floor(x * source_width / width),
/// floor(y * source_height / height))`. Construction allocates two offset maps;
/// [`Self::blit`] validates input before writing and allocates nothing.
pub struct NearestRgb8 {
    columns: Vec<usize>,
    rows: Vec<usize>,
    source_len: usize,
    source_stride: usize,
    same_width: bool,
    bgra_row: fn(&[u8], &mut [u8]),
}

impl NearestRgb8 {
    /// Construct a plan for nonzero source and destination dimensions.
    pub fn new(
        source_width: usize,
        source_height: usize,
        width: usize,
        height: usize,
    ) -> Result<Self, VideoError> {
        let source_len = packed_len(source_width, source_height, 3)?;
        packed_len(width, height, 4)?;
        packed_len(width, 1, std::mem::size_of::<usize>())?;
        packed_len(height, 1, std::mem::size_of::<usize>())?;
        Ok(Self {
            columns: offsets(source_width, width, 3),
            rows: offsets(source_height, height, source_width * 3),
            source_len,
            source_stride: source_width * 3,
            same_width: source_width == width,
            bgra_row: simd::select_bgra(),
        })
    }

    /// Scale and convert tightly packed RGB8 directly into `destination`, starting
    /// at `(x, y)` and clipping at its right and bottom edges. Row padding and all
    /// other pixels remain untouched. Extra bytes after the source image are ignored.
    pub fn blit(
        &self,
        source: &[u8],
        destination: &mut SurfaceMut<'_>,
        x: usize,
        y: usize,
    ) -> Result<(), VideoError> {
        check_len(source.len(), self.source_len, "RGB8 source")?;
        let width = self.columns.len().min(destination.width.saturating_sub(x));
        let height = self.rows.len().min(destination.height.saturating_sub(y));
        if width == 0 || height == 0 {
            return Ok(());
        }
        let bpp = destination.format.bytes_per_pixel();
        let row_len = width * bpp;
        let mut previous_source = None;
        let mut previous_start = 0;
        for (dy, &sy) in self.rows[..height].iter().enumerate() {
            let start = (y + dy) * destination.stride + x * bpp;
            if previous_source == Some(sy) {
                destination
                    .bytes
                    .copy_within(previous_start..previous_start + row_len, start);
            } else {
                let src = &source[sy..sy + self.source_stride];
                let dst = &mut destination.bytes[start..start + row_len];
                match destination.format {
                    SurfaceFormat::Rgb8 if self.same_width => dst.copy_from_slice(&src[..row_len]),
                    SurfaceFormat::Bgra8 if self.same_width => (self.bgra_row)(src, dst),
                    SurfaceFormat::Rgb8 => {
                        for (pixel, &sx) in dst.as_chunks_mut::<3>().0.iter_mut().zip(&self.columns)
                        {
                            pixel.copy_from_slice(&src[sx..sx + 3]);
                        }
                    }
                    SurfaceFormat::Bgra8 => {
                        for (pixel, &sx) in dst.as_chunks_mut::<4>().0.iter_mut().zip(&self.columns)
                        {
                            *pixel = [src[sx + 2], src[sx + 1], src[sx], 255];
                        }
                    }
                    SurfaceFormat::Rgb565Le => {
                        for (pixel, &sx) in dst.as_chunks_mut::<2>().0.iter_mut().zip(&self.columns)
                        {
                            *pixel = rgb565(src[sx], src[sx + 1], src[sx + 2]);
                        }
                    }
                }
            }
            previous_source = Some(sy);
            previous_start = start;
        }
        Ok(())
    }
}

// Quotient/remainder stepping avoids multiplication overflow and per-pixel
// division, including on 32-bit targets. packed_len bounds the error sum.
fn offsets(source: usize, destination: usize, stride: usize) -> Vec<usize> {
    let (step, remainder) = (source / destination, source % destination);
    let (mut position, mut error) = (0, 0);
    (0..destination)
        .map(|_| {
            let offset = position * stride;
            position += step;
            error += remainder;
            if error >= destination {
                error -= destination;
                position += 1;
            }
            offset
        })
        .collect()
}

const fn rgb565(r: u8, g: u8, b: u8) -> [u8; 2] {
    (((r as u16 >> 3) << 11) | ((g as u16 >> 2) << 5) | (b as u16 >> 3)).to_le_bytes()
}

fn packed_len(width: usize, height: usize, bpp: usize) -> Result<usize, VideoError> {
    width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(bpp))
        .filter(|&n| n > 0 && n <= isize::MAX as usize)
        .ok_or(VideoError::InvalidDimensions {
            what: "packed image",
            width,
            height,
        })
}

fn check_len(got: usize, expected: usize, what: &'static str) -> Result<(), VideoError> {
    if got < expected {
        Err(VideoError::BufferTooSmall {
            what,
            expected,
            got,
        })
    } else {
        Ok(())
    }
}

#[cfg(test)]
#[path = "tests/surface.rs"]
mod tests;
