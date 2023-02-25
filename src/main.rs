use crate::{model::TwoLayerNet, random::{Lcg, LcgTrait}, tensor::{Tensor2d, Tensor1d}};

//files
mod layer_trait;
mod layers;
mod load_mnist;
//main.rs
mod model;
mod random;
mod tensor;

fn main() {
    //load mnist and serialize
    println!("prepare data");
    let image_path = "./mnist/train-images-idx3-ubyte";
    let label_path = "./mnist/train-labels-idx1-ubyte";
    let (image_datas, label_datas) = load_mnist::load_minst(image_path, label_path);

    let test_size: usize = 10000;
    let train_size = image_datas.len() - test_size;
    println!("train size: {}, test size: {}", train_size, test_size);


    let x_test: Vec<[u8; 784]> = load_mnist::selialize_minst(&image_datas[0..test_size]);
    let t_test: Vec<usize> = label_datas[0..test_size].iter().map(|i| *i as usize).collect();
    let x_train: Vec<[u8; 784]> = load_mnist::selialize_minst(&image_datas[test_size..]);
    //println!("{:?}", x_train[0][0..50].to_vec());\
    let t_train: Vec<usize> = label_datas[test_size..].iter().map(|i| *i as usize).collect();
    
    
    //model setting
    const INPUT_SIZE: usize = 784;//28*28
    const HIDDEN_SIZE: usize = 50;
    const OUTPUT_SIZE: usize = 10;
    const BATCH_SIZE: usize = 10;
    let weight_init_std: f32 = 0.01;
    let mut lcg = Lcg::new(0);
    println!("create network: input_size: {}, hidden_size: {}, output_size: {}", INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    let mut model:TwoLayerNet<INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, BATCH_SIZE> 
        = TwoLayerNet::new(weight_init_std, &mut lcg);
    
    //train setting
    const EPOCH_NUM: usize = 100;
    const PRINT_INTERVAL_OF_EPOCH: usize = 1;
    const LEARNING_RATE: f32 = 0.1;
    println!("iter num: {}, batch size: {}, learning rate: {}", EPOCH_NUM, BATCH_SIZE, LEARNING_RATE);
    
    let mut train_loss_list = Vec::new();
    let mut train_acc_list = Vec::new();
    let mut test_acc_list = Vec::new();

    println!("---train start---");
    for epoch in 0..EPOCH_NUM {
        let (train_batch_data, train_batch_label): (Tensor2d<f32, 784, BATCH_SIZE>, Tensor1d<usize, BATCH_SIZE>) 
            = load_mnist::select_random_n(&x_train, &t_train, &mut lcg);


        //train
        model.update_gradient(&train_batch_data, &train_batch_label, LEARNING_RATE);


        if epoch % PRINT_INTERVAL_OF_EPOCH == 0 {
            let (test_batch, test_batch_label): (Tensor2d<f32, 784, BATCH_SIZE>, Tensor1d<usize, BATCH_SIZE>) 
                = load_mnist::select_random_n(&x_test, &t_test, &mut lcg);
            
            let train_loss = model.loss(&mut train_batch_data.clone(), &train_batch_label);
            let train_acc = model.accuracy(&train_batch_data, &train_batch_label);
            let test_acc = model.accuracy(&test_batch, &test_batch_label);
            
            train_loss_list.push(train_loss);
            train_acc_list.push(train_acc);
            test_acc_list.push(test_acc);

            println!("epoch{:6}: loss = {}, train_acc = {:.1}%, test_acc = {:.1}%", epoch, train_loss, train_acc * 100.0, test_acc * 100.0);
        }
    }

    println!("train_acc_list: {:?}", train_acc_list);
    println!("test_acc_list : {:?}", test_acc_list);
}
