use crate::tensor::{Tensor1d ,Tensor2d};

use crate::tensor::Num;

//Type, Batchsize
pub trait LayerTrait<T: Num, const I: usize, const O: usize, const B: usize> {

    //warn: input and output are reversed
    //batch_io is standard for layer input and output.
    fn forward(&mut self, input: &Tensor2d<T, I, B>, output: &mut Tensor2d<T, O, B>);
    fn backward(&mut self, doutput: &mut Tensor2d<T, I, B>, dinput: &Tensor2d<T, O, B>);

    //do not save values for backward()
    fn predict(&self, input: &Tensor2d<T, I, B>, output: &mut Tensor2d<T, O, B>);

    //apply gradient update, if it have gradient buffer, 
    fn update_gradient(&mut self, learning_rate: T);

    //(weight, bias)
    fn seek_weights(&self) -> Option<(Tensor2d<T, I, O>, Tensor1d<T, O>)> {
        None
    }

    fn get_output_buffer() -> Tensor2d<T, O, B>;
}
