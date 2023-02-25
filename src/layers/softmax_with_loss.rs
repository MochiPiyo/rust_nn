use crate::{tensor::{Num, Tensor2d, Tensor1d, TensorTrait, self}};

pub struct SoftmaxWithLoss {
}
impl SoftmaxWithLoss {
    pub fn new() -> Self {
        Self {}
    }
    pub fn loss<T: Num, const IO: usize, const B: usize>(input_and_loss_out: &mut Tensor2d<T, IO, B>, label: &Tensor1d<usize, B>) {
        let mut softmax_result: Tensor2d<T, IO, B> = Tensor2d::new();
        tensor::softmax_per_batch(&input_and_loss_out, &mut softmax_result);
        tensor::cross_entropy_error_per_batch(softmax_result, label, input_and_loss_out);
    }
    
    pub fn get_gradient<T: Num, const IO: usize, const B: usize>(input_and_dout: &mut Tensor2d<T, IO, B>, label: &Tensor1d<usize, B>) {
        let mut softmax_result: Tensor2d<T, IO, B> = Tensor2d::new();
        tensor::softmax_per_batch(&input_and_dout, &mut softmax_result);

        //doutput = (softmaxe_result - onehot_label: Tensor2d<usize, IO, B>);
        for (data, label) in softmax_result.body.iter_mut().zip(label.body.iter()) {
            for (index, data_i) in data.iter_mut().enumerate() {
                if index == *label {
                    *data_i -= T::one();
                    break;
                }
            }
        }
        //return doutput
        *input_and_dout = softmax_result;
    }

    //softmaxWithLoss layer is not used for predict() without gradient to backprop 
}

