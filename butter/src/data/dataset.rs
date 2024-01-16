use burn::data::dataset::Dataset; 
use derive_new::new;
use image;
use image::GenericImageView;
use std::fs::{self, File};
use std::path::Path;



const WIDTH: usize = 224;
const HEIGHT: usize = 224;

#[derive(new, Copy, Clone, Debug)]
pub struct ButterItem {
    pub image: [[[f32; WIDTH]; HEIGHT]; 3],

    pub label: usize,
}

pub struct ButterDataset {
    pub dataset: Vec<ButterItem>,
}

impl Dataset<ButterItem> for ButterDataset {
    fn get(&self, index:usize) -> Option<ButterItem> {
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

impl ButterDataset {
    pub fn new_with_contents(contents: Vec<ButterItem>, split: &str, block_size: usize) -> Self {
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
        let contents = load_data(data_file, split).unwrap(); 
        Self::new_with_contents(contents, split, batch_size)
    }

    
}

pub fn load_data(dataset_name: &str, split: &str) -> Result<Vec<ButterItem>, std::io::Error> {
    
    let path = format!("./archive(1)/{}", split);
    print!("load_data: {:?}\n", path);
    let mut ret: Vec<ButterItem> = Vec::new();
    let paths = fs::read_dir(path.clone()).unwrap();

    
    let mut labelindex = 0;
    for entry in paths {
        let entry = entry?;
        let picpaths = fs::read_dir(entry.path()).unwrap();
        println!("{:?}\n", entry.path());

        for pic in picpaths{
            let pic = pic?;

            //let mut butter: ButterItem = ButterItem::new();
            //butter.label = labelindex;
            let mut img = image::open(pic.path()).unwrap();

            let mut butterarray = [[[0f32; WIDTH]; HEIGHT]; 3];
            //println!("{:?}", pic.path());
            for (i, pix) in img.pixels().enumerate() {
                /*println!("{}",pix.2[0]);
                println!("{}",pix.2[1]);
                println!("{}",pix.2[2]);*/
                let x = i % WIDTH;
                let y = i / HEIGHT;
                butterarray[0][x][y] = pix.2[0] as f32/255.0;
                butterarray[1][x][y] = pix.2[1] as f32/255.0;
                butterarray[2][x][y] = pix.2[2] as f32/255.0;
            }


            ret.push( ButterItem {
                image: butterarray,
                label: labelindex,
            })
        }

        labelindex+=1;

        
    }
    /* 
    let mut images: Vec<[[[f32; WIDTH]; HEIGHT]; 3]> = Vec::new();
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

    let mut ret: Vec<ButterItem> = Vec::new();

    for (image, classification) in images.into_iter().zip(classifications.into_iter()) {
        ret.push(ButterItem {
            image,
            label: classification,
        })
    }
    */
    Ok(ret)
}