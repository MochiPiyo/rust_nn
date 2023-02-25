use crate::random::{Lcg, LcgTrait};

use crate::tensor::TensorTrait;
use crate::{layer_trait::LayerTrait, layers::{SoftmaxWithLoss, AffineLayer, ReluLayer}, tensor::{Tensor2d, Tensor, Tensor1d}};



pub struct TwoLayerNet<
    const INPUT_SIZE: usize,
    const HIDDEN_SIZE: usize,
    const OUTPUT_SIZE: usize,
    const BATCH_SIZE: usize,
> {
    //layers
    affine1: AffineLayer<f32, INPUT_SIZE, HIDDEN_SIZE, BATCH_SIZE>,
    relu: ReluLayer<HIDDEN_SIZE, BATCH_SIZE>,
    affine2: AffineLayer<f32, HIDDEN_SIZE, OUTPUT_SIZE, BATCH_SIZE>,
    last_layer: SoftmaxWithLoss,

    //temp value store
    hidden: Tensor2d<f32, HIDDEN_SIZE, BATCH_SIZE>,
    hidden_relu: Tensor2d<f32, HIDDEN_SIZE, BATCH_SIZE>,
    out: Tensor2d<f32, OUTPUT_SIZE, BATCH_SIZE>,
}
impl<
    const INPUT_SIZE: usize,
    const HIDDEN_SIZE: usize,
    const OUTPUT_SIZE: usize,
    const BATCH_SIZE: usize,
> TwoLayerNet<INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, BATCH_SIZE> {
    pub fn new(weight_init_std: f32, rng: &mut Lcg) -> Self {
        /*
        layer1
            input_size -> hidden_size
            matrix.vertical: input_size
            matrix.horizontal: hidden_size
        layer2
            hidden_size -> output_size
            matrix.vertical: hidden_size
            matrix.horizontal: output_size
        */
        //random generator
        let mut lcg: Lcg = Lcg::new(0);

        //init weights with lcg
        let mut weight1: Tensor2d<f32, HIDDEN_SIZE, INPUT_SIZE> 
            = Tensor2d::new_fill_with(weight_init_std);
        for row in weight1.body.iter_mut() {
            for i in row.iter_mut() {
                *i *= lcg.gen_0to1();
            }
        }
        let mut bias1: Tensor1d<f32, HIDDEN_SIZE>
            = Tensor1d::new_fill_with(weight_init_std);
        for i in bias1.body.iter_mut() {
            *i *= lcg.gen_0to1();
        }


        let mut weight2: Tensor2d<f32, OUTPUT_SIZE, HIDDEN_SIZE>
            = Tensor2d::new_fill_with(weight_init_std);
        for row in weight2.body.iter_mut() {
            for i in row.iter_mut() {
                *i *= lcg.gen_0to1();
            }
        }
        let mut bias2: Tensor1d<f32, OUTPUT_SIZE>
            = Tensor1d::new_fill_with(weight_init_std);
        for i in bias2.body.iter_mut() {
            *i *= lcg.gen_0to1();
        }

        //make layers
        let affine1: AffineLayer<f32, INPUT_SIZE, HIDDEN_SIZE, BATCH_SIZE> = AffineLayer::new(weight1, bias1);
        let relu = ReluLayer::new();
        let affine2: AffineLayer<f32, HIDDEN_SIZE, OUTPUT_SIZE, BATCH_SIZE> = AffineLayer::new(weight2, bias2);
        let last_layer = SoftmaxWithLoss::new();

        let hidden = Tensor2d::<f32, HIDDEN_SIZE, BATCH_SIZE>::new();
        let hidden_relu = hidden.clone();
        let out = Tensor2d::<f32, OUTPUT_SIZE, BATCH_SIZE>::new();

        Self {
            affine1,
            relu,
            affine2,
            last_layer,

            hidden,
            hidden_relu,
            out,
        }
    }

    pub fn predict(&mut self, input: &Tensor2d<f32, INPUT_SIZE, BATCH_SIZE>) -> &Tensor2d<f32, OUTPUT_SIZE, BATCH_SIZE> {
        self.affine1.predict(&input, &mut self.hidden);
        self.relu.predict(&self.hidden, &mut self.hidden_relu);
        self.affine2.predict(&self.hidden_relu, &mut self.out);
        return &self.out;
    }

    //no grad
    pub fn loss(&mut self, input: &mut Tensor2d<f32, INPUT_SIZE, BATCH_SIZE>, label: &Tensor1d<usize, BATCH_SIZE>) -> f32 {
        let mut predict = self.predict(input);

        SoftmaxWithLoss::loss(input, label);

        let mut loss_sum: f64 = 0.0;
        for data in input.body.iter() {
            for i in data.iter() {
                loss_sum += (*i) as f64;
            }
        }
        return (loss_sum / BATCH_SIZE as f64) as f32;
    }

    //no grad
    pub fn accuracy(&mut self, input: &Tensor2d<f32, INPUT_SIZE, BATCH_SIZE>, label: &Tensor1d<usize, BATCH_SIZE>) -> f32 {
        let batch_predict = self.predict(input);

        let mut ok = 0;
        for (predict, label) in batch_predict.body.iter().zip(label.body.iter()) {
            let mut largest: f32 = 0.0;
            let mut largest_index = 0;
            for (index, value) in predict.iter().enumerate() {
                if *value > largest {
                    largest = *value;
                    largest_index = index;
                }
            }

            if largest_index == *label {
                ok += 1;
            }
        }
        return ok as f32 / BATCH_SIZE as f32;
    }

    pub fn update_gradient(&mut self, input: &Tensor2d<f32, INPUT_SIZE, BATCH_SIZE>, label: &Tensor1d<usize, BATCH_SIZE>, learning_rate: f32) {
        //forward
        self.affine1.forward(&input, &mut self.hidden);
        self.relu.forward(&self.hidden, &mut self.hidden_relu);
        self.affine2.forward(&self.hidden_relu, &mut self.out);

        //return self.out = doutput
        SoftmaxWithLoss::get_gradient(&mut self.out, &label);

        //backward
        self.affine2.backward(&mut self.hidden_relu, &self.out);
        self.relu.backward(&mut self.hidden, &self.hidden_relu);
        let mut dummy: Tensor2d<f32, INPUT_SIZE, BATCH_SIZE> = Tensor2d::new();
        self.affine1.backward(&mut dummy, &self.hidden);

        //update
        self.affine1.update_gradient(learning_rate);
        self.affine2.update_gradient(learning_rate);
    }

}