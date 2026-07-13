//! Metal device layer: device + command queue, buffer/pipeline helpers, and
//! the Rust-side launch of the GPU kernels (elementwise, GEMM f32/f16, cast).

mod dispatch;
mod encoder;
mod pipeline;
mod pool;
mod profile;

use crate::{Buffer, Device, Pipeline, Queue};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLComputeCommandEncoder, MTLCreateSystemDefaultDevice, MTLDevice};
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

    pub fn device_name(&self) -> String {
        self.device.name().to_string()
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
