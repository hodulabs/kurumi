//! Metal device layer: device + command queue, buffer/pipeline helpers, and
//! the Rust-side launch of the GPU kernels (elementwise, GEMM f32/f16, cast).

mod dispatch;
mod profile;

use crate::{Buffer, Device, Pipeline, Queue};
use half::{bf16, f16};
use kurumi_core::{DType, Storage};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};
use objc2_metal_performance_shaders::MPSMatrixMultiplication;
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;

type CmdBuf = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
type Enc = Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>;

pub struct MetalContext {
    device: Device,
    queue: Queue,
    // device-resident batching: GPU ops encode into one command buffer; consecutive
    // custom kernels share ONE open serial compute encoder (`enc`), which runs its
    // dispatches in order with memory coherence -- dependents stay ordered without a
    // fresh encoder per op (create/end was a chunk of host encode). Closed only before
    // an MPS GEMM (encodes its own) or at `flush`. Commit+wait once per host boundary.
    pending: RefCell<Option<CmdBuf>>,
    enc: RefCell<Option<Enc>>,
    // fused-elementwise kernels compiled on demand, keyed by MSL source (a repeated
    // graph reuses them: the same fused chain emits the same source).
    fused: RefCell<HashMap<String, Pipeline>>,
    // intermediate-buffer pool keyed by byte size. A fixed graph re-evaluated (training
    // loop) allocates the same ~N intermediates each step; fresh MTLBuffers every time
    // dominate host encode. `inuse` holds this step's buffers; `recycle()` (at eval
    // start, after the prior flush -> GPU done) returns them to `pool`. Safe because
    // every kernel fully overwrites its output. Unbounded by design (buffer set bounded).
    pool: RefCell<HashMap<usize, Vec<Buffer>>>,
    inuse: RefCell<Vec<(usize, Buffer)>>,
    // MPS GEMM kernels keyed by (trans_l, trans_r, m, n, k, batch): made-once/encode-many.
    // Recreating them per matmul was a chunk of host encode (~33 GEMMs/step). The per-call
    // MPSMatrix wrappers still bind cheaply; this caches the heavy kernel object.
    mm_cache: RefCell<HashMap<MmKey, Retained<MPSMatrixMultiplication>>>,
}

// (trans_l, trans_r, m, n, k, batch): the shape signature of an MPS GEMM kernel.
pub(crate) type MmKey = (bool, bool, usize, usize, usize, usize);

impl MetalContext {
    /// `None` if no Metal device is available (e.g. headless CI).
    pub fn new() -> Option<Self> {
        let device = MTLCreateSystemDefaultDevice()?;
        let queue = device.newCommandQueue()?;
        Some(Self {
            device,
            queue,
            pending: RefCell::new(None),
            enc: RefCell::new(None),
            fused: RefCell::new(HashMap::new()),
            pool: RefCell::new(HashMap::new()),
            inuse: RefCell::new(Vec::new()),
            mm_cache: RefCell::new(HashMap::new()),
        })
    }

    /// Compile a generated kernel once per source: all device kernels (fused / reduce /
    /// strided / pad, each parameterized by element dtype) share this cache, so dtype
    /// variants compile separately and reuse across a training loop.
    pub(crate) fn cached(&self, src: &str, func: &str) -> Pipeline {
        let hit = self.fused.borrow().get(src).cloned(); // drop the borrow before borrow_mut
        hit.unwrap_or_else(|| {
            let p = self.pipeline(src, func);
            self.fused.borrow_mut().insert(src.to_string(), p.clone());
            p
        })
    }

    // the pending command buffer GPU ops encode into (created lazily).
    fn cmd(&self) -> CmdBuf {
        let mut p = self.pending.borrow_mut();
        if p.is_none() {
            *p = Some(self.queue.commandBuffer().expect("command buffer"));
        }
        p.clone().unwrap()
    }

    // the shared open serial compute encoder for custom kernels (created lazily on
    // the pending buffer). Consecutive dispatches into it stay ordered.
    fn encoder(&self) -> Enc {
        let mut e = self.enc.borrow_mut();
        if e.is_none() {
            *e = Some(self.cmd().computeCommandEncoder().expect("compute encoder"));
        }
        e.clone().unwrap()
    }

    // close the open encoder (before an MPS GEMM encodes its own, or before commit).
    fn end_encoder(&self) {
        if let Some(e) = self.enc.borrow_mut().take() {
            e.endEncoding();
        }
    }

    /// Commit + wait the pending GPU work, if any. Call before reading any device
    /// buffer back to the host (a host boundary or the final output).
    pub(crate) fn flush(&self) {
        self.end_encoder();
        if let Some(cmd) = self.pending.borrow_mut().take() {
            cmd.commit();
            cmd.waitUntilCompleted();
            if std::env::var_os("KURUMI_GPUTIME").is_some() {
                let gpu = cmd.GPUEndTime() - cmd.GPUStartTime();
                Self::bump_flush((gpu * 1e6) as u128);
            }
        }
    }

    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    // compile one MSL function into a compute pipeline state (cache this across runs)
    pub(crate) fn pipeline(&self, src: &str, func: &str) -> Pipeline {
        let lib = self
            .device
            .newLibraryWithSource_options_error(&NSString::from_str(src), None)
            .unwrap_or_else(|e| panic!("MSL compile failed: {e}"));
        let f = lib.newFunctionWithName(&NSString::from_str(func)).expect("function");
        self.device.newComputePipelineStateWithFunction_error(&f).expect("pipeline")
    }

    // shared-storage buffer over `data` (unified memory: no copy on Apple Silicon)
    fn buffer_of(&self, data: &[f32]) -> Buffer {
        let bytes = std::mem::size_of_val(data);
        let ptr = NonNull::new(data.as_ptr() as *mut c_void).unwrap();
        unsafe { self.device.newBufferWithBytes_length_options(ptr, bytes, MTLResourceOptions::StorageModeShared) }
            .expect("buffer alloc")
    }

    fn empty_buffer(&self, len: usize) -> Buffer {
        let bytes = len * std::mem::size_of::<f32>();
        self.device.newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared).expect("buffer alloc")
    }

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

    // encode a 1-D elementwise pso over `n` threads into the shared open encoder
    // (dispatches stay ordered; no per-op encoder, no commit: `flush` does that).
    fn run_1d(&self, pso: &Pipeline, bind: impl Fn(&ProtocolObject<dyn MTLComputeCommandEncoder>), n: usize) {
        let enc = self.encoder();
        enc.setComputePipelineState(pso);
        bind(&enc);
        let tg = pso.maxTotalThreadsPerThreadgroup().min(n.max(1));
        enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize { width: n, height: 1, depth: 1 },
            MTLSize { width: tg, height: 1, depth: 1 },
        );
    }

    // encode gx x gy threadgroups of tx x ty threads (for tiled kernels with threadgroup
    // memory) into the shared open encoder.
    fn run_groups(
        &self,
        pso: &Pipeline,
        bind: impl Fn(&ProtocolObject<dyn MTLComputeCommandEncoder>),
        gx: usize,
        gy: usize,
        tx: usize,
        ty: usize,
    ) {
        let enc = self.encoder();
        enc.setComputePipelineState(pso);
        bind(&enc);
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: gx, height: gy, depth: 1 },
            MTLSize { width: tx, height: ty, depth: 1 },
        );
    }

    // byte-buffer helpers over any element type (shared storage, unified memory)
    fn buffer_bytes<T>(&self, data: &[T]) -> Buffer {
        let bytes = std::mem::size_of_val(data);
        let ptr = NonNull::new(data.as_ptr() as *mut c_void).unwrap();
        unsafe { self.device.newBufferWithBytes_length_options(ptr, bytes, MTLResourceOptions::StorageModeShared) }
            .expect("buffer alloc")
    }
    fn empty_bytes(&self, bytes: usize) -> Buffer {
        self.device.newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared).expect("buffer alloc")
    }
}

fn read_t<T: Copy>(buf: &Buffer, len: usize) -> Vec<T> {
    let ptr = buf.contents().as_ptr() as *const T;
    unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
}

unsafe fn set_u32(enc: &ProtocolObject<dyn MTLComputeCommandEncoder>, v: u32, index: usize) {
    let ptr = NonNull::new(&v as *const u32 as *mut c_void).unwrap();
    unsafe { enc.setBytes_length_atIndex(ptr, 4, index) };
}

// bind a small u32 array (shape/strides) as a constant buffer argument
unsafe fn set_bytes(enc: &ProtocolObject<dyn MTLComputeCommandEncoder>, data: &[u32], index: usize) {
    let ptr = NonNull::new(data.as_ptr() as *mut c_void).unwrap();
    unsafe { enc.setBytes_length_atIndex(ptr, std::mem::size_of_val(data), index) };
}
