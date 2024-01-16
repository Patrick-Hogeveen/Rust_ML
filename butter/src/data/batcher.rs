use burn::{
    data::{dataloader::batcher::Batcher},
    tensor::{backend::Backend, Data, ElementConversion, Int, Tensor},
};
use crate::data::dataset::*;

pub struct ButterBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> ButterBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct ButterBatch<B: Backend> {
    pub images: Tensor<B, 3>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<ButterItem, ButterBatch<B>> for ButterBatcher<B> {
    fn batch(&self, items: Vec<ButterItem>) -> ButterBatch<B> {
        let images = items
            .iter()
            .map(|item| Data::<f32, 3>::from(item.image))
            .map(|data| Tensor::<B, 3>::from_data(data.convert(), &self.device))
            .map(|tensor| tensor.reshape([3, 244, 244]))
            // Normalize: make between [0,1] and make the mean=0 and std=1
            // values mean=0.1307,std=0.3081 are from the PyTorch Butter example
            // https://github.com/pytorch/examples/blob/54f4572509891883a947411fd7239237dd2a39c3/Butter/main.py#L122
            .map(|tensor| ((tensor / 255) - 0.1307) / 0.3081)
            .collect();

        let targets = items
            .iter()
            .map(|item| Tensor::<B, 1, Int>::from_data(Data::from([(item.label as i64).elem()]), &self.device))
            .collect();

        let images = Tensor::cat(images, 0).to_device(&self.device);
        let targets = Tensor::cat(targets, 0).to_device(&self.device);

        ButterBatch { images, targets }
    }
}