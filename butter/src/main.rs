use burn::{backend::{libtorch::LibTorchDevice, Autodiff, LibTorch}, optim::AdamConfig};
use butter::training;
mod data;

fn main() {
    /*
    let butterdata = data::ButterDataset::new("", "train", 10);

    let mut imagebuf = image::ImageBuffer::new(224, 224);

    for (x, y , pixel) in imagebuf.enumerate_pixels_mut() {
        let pval = (butterdata.dataset[1000].image[0][x as usize][y as usize] * 255.0) as u8;
        let pval1 = (butterdata.dataset[1000].image[1][x as usize][y as usize] * 255.0) as u8;
        let pval2 = (butterdata.dataset[1000].image[2][x as usize][y as usize] * 255.0) as u8;
        *pixel = image::Rgb([pval,pval1,pval2]);
    }

    imagebuf.save("testmnist.png").unwrap();
    */
    let device = LibTorchDevice::Cpu;
    training::train::<Autodiff<LibTorch>>(".", training::TrainingConfig::new(AdamConfig::new()),device);
    println!("Hello, world!");
}
