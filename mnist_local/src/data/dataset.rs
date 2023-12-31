use burn::data::dataset::Dataset; 
use derive_new::new;
use std::fs::{self, File};
use crate::data::mnist_extract::MnistData;


const WIDTH: usize = 28;
const HEIGHT: usize = 28;

#[derive(new, Copy, Clone, Debug)]
pub struct MNISTItem {
    pub image: [[f32; WIDTH]; HEIGHT],

    pub label: usize,
}

pub struct MNISTDataset {
    pub dataset: Vec<MNISTItem>,
}

impl Dataset<MNISTItem> for MNISTDataset {
    fn get(&self, index:usize) -> Option<MNISTItem> {
        if index >= self.len() {
            return None;
        }

        let data = self.dataset[index];


 
        Some(data)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl MNISTDataset {
    pub fn new_with_contents(contents: Vec<MNISTItem>, split: &str, block_size: usize) -> Self {
        let size = contents.len();
        let mut train = contents;
        let test = train.split_off(size * 9 / 10); 

        let dataset = match split {
            "train" => train, 
            "test" => test, 
            _ => panic!("{} is not train or test", split),
        }; 

        Self {
            dataset: dataset, 
        }
    }

    pub fn new(data_file: &str, split: &str, batch_size: usize) -> Self {
        let contents = load_data(data_file).unwrap(); 
        Self::new_with_contents(contents, split, batch_size)
    }

    
}

pub fn load_data(dataset_name: &str) -> Result<Vec<MNISTItem>, std::io::Error> {
    
    let filename = format!("/tmp/emnist/MNIST/raw/{}-labels-idx1-ubyte.gz", dataset_name);
    println!("name: {}", filename);
    let label_data = &MnistData::new(&(File::open(filename))?)?;
    let filename = format!("/tmp/emnist/MNIST/raw/{}-images-idx3-ubyte.gz", dataset_name);
    let images_data = &MnistData::new(&(File::open(filename))?)?;
    let mut images: Vec<[[f32; WIDTH]; HEIGHT]> = Vec::new();
    let image_shape = (images_data.sizes[1] * images_data.sizes[2]) as usize;

    for i in 0..images_data.sizes[0] as usize {
        let start = i * image_shape;
        let image_data = images_data.data[start..start + image_shape].to_vec();
        let image_data: Vec<f64> = image_data.into_iter().map(|x| x as f64 / 255.).collect();
        let mut image_array = [[0f32; WIDTH]; HEIGHT];
        for (i, pixel) in image_data.iter().enumerate() {
            let x = i % WIDTH;
            let y = i / HEIGHT;
            image_array[x][y] = *pixel as f32;
        }
        
        images.push(image_array);
    }

    let classifications: Vec<usize> = label_data.data.clone().into_iter().map(|x| x as usize).collect();

    let mut ret: Vec<MNISTItem> = Vec::new();

    for (image, classification) in images.into_iter().zip(classifications.into_iter()) {
        ret.push(MNISTItem {
            image,
            label: classification,
        })
    }

    Ok(ret)
}