use crate::{tensor::{Num, Tensor2d, Tensor1d, add_matrix_mul, add_all_row, TensorTrait}, layer_trait::LayerTrait};


pub struct AffineLayer<T: Num, const I: usize, const O: usize, const B: usize> {
    //TOI way, I is input_size, O is output_size
    weight: Tensor2d<T, O, I>,
    bias: Tensor1d<T, O>,
    
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
            input_cach: Tensor2d::new(),
            gradient_buffer_weight: Tensor2d::new(),
            gradient_buffer_bias: Tensor1d::new(),
        }
    }
}
impl<T: Num, const I: usize, const O: usize, const B: usize> LayerTrait<T, I, O, B> for AffineLayer<T, I, O, B> {
    fn forward(&mut self, input: &Tensor2d<T, I, B>, output: &mut Tensor2d<T, O, B>) {
        let input_clone = input.clone();
        self.input_cach = input_clone;
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
        add_matrix_mul(dinput, &self.weight.clone().transpose(), doutput);
        //(return doutput)
    }

    fn predict(&self, input: &Tensor2d<T, I, B>, output: &mut Tensor2d<T, O, B>) {
        //get doutput for next layer
        //init doutput value by bias (= add bias fist to prevent unnecessary initialization with 0)
        for data in output.body.iter_mut() {
            *data = *self.bias.body.clone();
        }
        //output += mul(input, weight)
        add_matrix_mul(input, &self.weight, output);
    }

    fn update_gradient(&mut self, batch_size: T, learning_rate: T) {
        //get average of gradient
        self.gradient_buffer_weight.div_scalar(batch_size);
        self.gradient_buffer_bias.div_scalar(batch_size);

        self.gradient_buffer_weight.mul_scalar(learning_rate);
        self.gradient_buffer_bias.mul_scalar(learning_rate);
       
        //SubAssignは&で引数とらないので-=にするとcloneが必要
        //this .sub() is original tensor fnction
        self.weight.sub(&self.gradient_buffer_weight);
        self.bias.sub(&self.gradient_buffer_bias);

        /*
        初回：
        affine1: gweight: //zero//->ok, gbias: ok
        affine2: gweight: ok, gbias: ok
        
        n:
        aff1: gw: zero, gbias: zero
        aff2: gw: zero, gbias: ok
        
        予想：問題は二つあって、初回以外reluがブロック＋weightの計算がおかしい
        →いや、それだと初回のaff2.gwがokなの説明つかない
        予想２：input_cacheが初回以外動いてない
        →いや、ではなぜ初回のaff1.gwは動かない？→動いている！
        ↓
        すると、初回は全部動くが、次回からはaff2.gbしか動かないと。
        
        わかった、二回目以降でreluが全部0にしてる
         */
    }

    fn get_output_buffer() -> Tensor2d<T, O, B> {
        Tensor2d::new()
    }
}