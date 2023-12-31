mod data;
use image;
use burn::{backend::{wgpu::{Wgpu, WgpuDevice}, Autodiff, libtorch::{LibTorch, LibTorchDevice}}, optim::AdamConfig};
use mnist_local::training;

fn main() {

    let dataset_mnist = data::MNISTDataset::new("t10k","train",10);

    let mut imagebuf = image::ImageBuffer::new(28, 28);

    for (x, y , pixel) in imagebuf.enumerate_pixels_mut() {
        let pval = (dataset_mnist.dataset[0].image[x as usize][y as usize] * 255.0) as u8;
        *pixel = image::Rgb([pval,pval,pval]);
    }

    imagebuf.save("testmnist.png").unwrap();
    println!("Label: {}", dataset_mnist.dataset[0].label);
    println!("{:?}", dataset_mnist.dataset[0].image);
    println!("Hello, world!");

    let device = LibTorchDevice::Cpu;
    training::train::<Autodiff<LibTorch>>(".",training::TrainingConfig::new(AdamConfig::new()),device);
}
