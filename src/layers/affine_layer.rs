use crate::{tensor::{Num, Tensor2d, Tensor1d, add_matrix_mul, add_all_row, TensorTrait}, layer_trait::LayerTrait};


pub struct AffineLayer<T: Num, const I: usize, const O: usize, const B: usize> {
    //TOI way, I is input_size, O is output_size
    weight: Tensor2d<T, O, I>,
    bias: Tensor1d<T, O>,
    //
    weight_transpozed: Tensor2d<T, I, O>,
    
    //B is batch size
    input_cach: Tensor2d<T, I, B>,
    
    //gradient buffer
    gradient_buffer_weight: Tensor2d<T, O, I>,
    gradient_buffer_bias: Tensor1d<T, O>,
}
impl<T: Num, const I: usize, const O: usize, const B: usize> AffineLayer<T, I, O, B> {
    pub fn new(weight: Tensor2d<T, O, I>, bias: Tensor1d<T, O>) -> Self {
        Self {
            weight: weight.clone(),
            bias,
            weight_transpozed: weight.transpose(),
            input_cach: Tensor2d::new(),
            gradient_buffer_weight: Tensor2d::new(),
            gradient_buffer_bias: Tensor1d::new(),
        }
    }
}
impl<T: Num, const I: usize, const O: usize, const B: usize> LayerTrait<T, I, O, B> for AffineLayer<T, I, O, B> {
    fn forward(&mut self, input: &Tensor2d<T, I, B>, output: &mut Tensor2d<T, O, B>) {
        self.input_cach = input.clone();
        self.gradient_buffer_weight = Tensor2d::new_fill_with(T::zero());
        self.gradient_buffer_bias = Tensor1d::new_fill_with(T::zero());

        //set ouput the gradient for next layer.
        self.predict(input, output);
    }

    fn backward(&mut self, doutput: &mut Tensor2d<T, I, B>, dinput: &Tensor2d<T, O, B>) {
        //get gradient for self parameter
        //weights
        add_matrix_mul(&self.input_cach.transpose(), dinput, &mut self.gradient_buffer_weight);
        //biases
        add_all_row(dinput, &mut self.gradient_buffer_bias);


        //gradient for next layer
        add_matrix_mul(dinput, &self.weight_transpozed, doutput);
        //(return doutput)
    }

    fn predict(&self, input: &Tensor2d<T, I, B>, output: &mut Tensor2d<T, O, B>) {
        //get doutput for next layer
        //init doutput value by bias (= add bias fist to prevent unnecessary initialization with 0)
        for data in output.body.iter_mut() {
            *data = self.bias.body.clone();
        }
        //output += mul(input, weight)
        add_matrix_mul(input, &self.weight, output);
    }

    fn update_gradient(&mut self, learning_rate: T) {
        self.gradient_buffer_weight.mul_scalar(learning_rate);
        self.gradient_buffer_bias.mul_scalar(learning_rate);
        
        //SubAssignは&で引数とらないので-=にするとcloneが必要
        //this .sub() is original tensor fnction
        self.weight.sub(&self.gradient_buffer_weight);
        self.bias.sub(&self.gradient_buffer_bias);
    }

    fn get_output_buffer() -> Tensor2d<T, O, B> {
        Tensor2d::new()
    }
}