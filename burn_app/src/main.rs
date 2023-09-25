#[cfg(any(
    feature = "ndarray",
    feature = "ndarray-blas-netlib",
    feature = "ndarray-blas-openblas",
    feature = "ndarray-blas-accelerate",
))]
mod ndarray {
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::NdArrayAutodiffBackend;
    use burn_app::training;

    pub fn run() {
        let device = NdArrayDevice::Cpu;
        training::run::<NdArrayAutodiffBackend>(device);
    }
}

#[cfg(feature = "tch-gpu")]
mod tch_gpu {
    use burn::backend::tch::TchDevice;
    use burn::backend::TchAutodiffBackend;
    use burn_app::training;

    pub fn run() {
        #[cfg(not(target_os = "macos"))]
        let device = TchDevice::Cuda(0);
        #[cfg(target_os = "macos")]
        let device = TchDevice::Mps;

        training::run::<TchAutodiffBackend>(device);
    }
}



mod tch_cpu {
    use burn::backend::tch::TchDevice;
    use burn::backend::TchAutodiffBackend;
    use burn_app::training;

    pub fn run() {
        let device = TchDevice::Cpu;
        training::run::<TchAutodiffBackend>(device);
    }
}

fn main() {
    #[cfg(any(
        feature = "ndarray",
        feature = "ndarray-blas-netlib",
        feature = "ndarray-blas-openblas",
        feature = "ndarray-blas-accelerate",
    ))]
    ndarray::run();
    #[cfg(feature = "tch-gpu")]
    tch_gpu::run();
    
    tch_cpu::run();
}
