//! Intermediate-buffer reuse pool and host<->device buffer helpers.

use crate::Buffer;
use crate::context::{MetalContext, read_t};
use half::{bf16, f16};
use kurumi_core::{DType, Storage};
use objc2_metal::{MTLDevice, MTLResourceOptions};
use std::ffi::c_void;
use std::ptr::NonNull;

impl MetalContext {
    // device-resident elementwise: buffers stay on the GPU across a chain of ops (no host
    // round-trip per op). The backend keeps results as device buffers, reading back only
    // at a host boundary. One commit+wait per op; batch the subgraph into one command
    // buffer when latency matters.

    /// A device buffer sized for `n` elements of `dt` (uninitialized). Drawn from the
    /// reuse pool when a buffer of that byte size is free, else freshly allocated.
    pub(crate) fn empty(&self, n: usize, dt: DType) -> Buffer {
        let bytes = n * dt.width();
        let buf = self.pool.borrow_mut().get_mut(&bytes).and_then(Vec::pop).unwrap_or_else(|| self.empty_bytes(bytes));
        self.inuse.borrow_mut().push((bytes, buf.clone()));
        buf
    }

    /// Return this step's intermediate buffers to the pool. Call at eval start: the
    /// prior step's `flush` already waited for the GPU, so its buffers are now free.
    pub(crate) fn recycle(&self) {
        let mut pool = self.pool.borrow_mut();
        for (bytes, buf) in self.inuse.borrow_mut().drain(..) {
            pool.entry(bytes).or_default().push(buf);
        }
    }
    /// Upload a host f32/f16/bf16 storage to a shared GPU buffer (zero-copy).
    pub(crate) fn upload(&self, s: &Storage) -> Buffer {
        match s {
            Storage::F32(v) => self.buffer_bytes(v),
            Storage::F16(v) => self.buffer_bytes(v),
            Storage::BF16(v) => self.buffer_bytes(v),
            Storage::BOOL(v) => self.buffer_bytes(v), // 1 byte (0/1): for device where/cmp
            Storage::U8(v) => self.buffer_bytes(v),
            Storage::U16(v) => self.buffer_bytes(v),
            Storage::U32(v) => self.buffer_bytes(v),
            Storage::U64(v) => self.buffer_bytes(v),
            Storage::I8(v) => self.buffer_bytes(v),
            Storage::I16(v) => self.buffer_bytes(v),
            Storage::I32(v) => self.buffer_bytes(v),
            Storage::I64(v) => self.buffer_bytes(v),
            Storage::C64(v) => self.buffer_bytes(v), // Complex<f32> = repr(C) (re, im) = float2
            _ => panic!("metal device path: unsupported dtype {:?}", s.dtype()),
        }
    }
    /// Read a device buffer (`n` elements of `dt`) back to a host storage.
    pub(crate) fn download(&self, buf: &Buffer, n: usize, dt: DType) -> Storage {
        match dt {
            DType::F32 => Storage::F32(read_t::<f32>(buf, n)),
            DType::F16 => Storage::F16(read_t::<f16>(buf, n)),
            DType::BF16 => Storage::BF16(read_t::<bf16>(buf, n)),
            DType::BOOL => Storage::BOOL(read_t::<u8>(buf, n).into_iter().map(|x| x != 0).collect()),
            DType::I64 => Storage::I64(read_t::<i64>(buf, n)), // argmax/argmin indices
            DType::I32 => Storage::I32(read_t::<i32>(buf, n)),
            DType::I16 => Storage::I16(read_t::<i16>(buf, n)),
            DType::I8 => Storage::I8(read_t::<i8>(buf, n)),
            DType::U64 => Storage::U64(read_t::<u64>(buf, n)),
            DType::U32 => Storage::U32(read_t::<u32>(buf, n)),
            DType::U16 => Storage::U16(read_t::<u16>(buf, n)),
            DType::U8 => Storage::U8(read_t::<u8>(buf, n)),
            DType::C64 => Storage::C64(read_t::<num_complex::Complex<f32>>(buf, n)),
            _ => panic!("metal device path: unsupported dtype {dt:?}"),
        }
    }

    // shared-storage buffer over `data` (unified memory: no copy on Apple Silicon)
    pub(crate) fn buffer_of(&self, data: &[f32]) -> Buffer {
        let bytes = std::mem::size_of_val(data);
        let ptr = NonNull::new(data.as_ptr() as *mut c_void).unwrap();
        unsafe { self.device.newBufferWithBytes_length_options(ptr, bytes, MTLResourceOptions::StorageModeShared) }
            .expect("buffer alloc")
    }

    pub(crate) fn empty_buffer(&self, len: usize) -> Buffer {
        let bytes = len * std::mem::size_of::<f32>();
        self.device.newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared).expect("buffer alloc")
    }

    // byte-buffer helpers over any element type (shared storage, unified memory)
    pub(crate) fn buffer_bytes<T>(&self, data: &[T]) -> Buffer {
        let bytes = std::mem::size_of_val(data);
        let ptr = NonNull::new(data.as_ptr() as *mut c_void).unwrap();
        unsafe { self.device.newBufferWithBytes_length_options(ptr, bytes, MTLResourceOptions::StorageModeShared) }
            .expect("buffer alloc")
    }
    pub(crate) fn empty_bytes(&self, bytes: usize) -> Buffer {
        self.device.newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared).expect("buffer alloc")
    }
}
