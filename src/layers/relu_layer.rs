use crate::{tensor::{Num, Tensor2d, TensorTrait}, layer_trait::LayerTrait};


pub struct ReluLayer<const IO: usize, const B: usize> {
    mask_cache: Tensor2d<bool, IO, B>,
}
impl<const IO: usize, const B: usize> ReluLayer<IO, B> {
    pub fn new() -> Self {
        Self {
            mask_cache: Tensor2d::new_bool(),
        }
    }
}impl<T: Num, const IO: usize, const B: usize> LayerTrait<T, IO, IO, B> for ReluLayer<IO, B> {
    fn forward(&mut self, input: &Tensor2d<T, IO, B>, output: &mut Tensor2d<T, IO, B>) {
        self.mask_cache = Tensor2d::new_bool();
        for i in 0..B {
            for j in 0..IO {
                output.body[i][j] = if input.body[i][j] > T::zero() {
                    self.mask_cache.body[i][j] = false;
                    input.body[i][j].clone()
                } else {
                    self.mask_cache.body[i][j] = true;
                    T::zero()
                };
            }
        }
       
        /*
        for b in 0..B {
            for io in 0..IO {
                //dbg!(input.body[b][io]);
                if input.body[b][io] < T::zero() {
                    self.mask_cache.body[b][io] = true;
                    output.body[b][io] = T::zero();
                }else {
                    self.mask_cache.body[b][io] = false;
                    output.body[b][io] = input.body[b][io];
                }
            }
        }*/
        /*for (mask_cache_row, (input_row, output_row)) in self.mask_cache.body.iter_mut().zip(input.body.iter().zip(output.body.iter_mut())) {
            for (m, (i , o)) in mask_cache_row.iter_mut().zip(input_row.iter().zip(output_row.iter_mut())) {
                if *i < T::zero() {
                    //mask
                    *m = true;
                    *o = T::zero();
                }else {
                    //not mask
                    *m = false;
                    *o = *i;
                    /*
                    試しに*o = T::one()にしてみたら一回目は全部１だけど次からは０。
                    
                     */
                }
            }
        ?*/
    }

    fn backward(&mut self, doutput: &mut Tensor2d<T, IO, B>, dinput: &Tensor2d<T, IO, B>) {
        for (mask_cache_row, (input_row, output_row)) in self.mask_cache.body.iter_mut().zip(dinput.body.iter().zip(doutput.body.iter_mut())) {
            for (m, (i , o)) in mask_cache_row.iter_mut().zip(input_row.iter().zip(output_row.iter_mut())) {
                if *m == true {
                    //if masked
                    *o = T::zero();
                }else {
                    //not masked
                    *o = *i;
                }
            }
        }
    }

    fn predict(&self, input: &Tensor2d<T, IO, B>, output: &mut Tensor2d<T, IO, B>) {
        for (input_row, output_row) in input.body.iter().zip(output.body.iter_mut()) {
            for (i , o) in input_row.iter().zip(output_row.iter_mut()) {
                if *i < T::zero() {
                    //mask
                    *o = T::zero();
                }else {
                    //not mask
                    *o = *i;
                }
            }
        }
    }

    fn update_gradient(&mut self, _batch_size: T,_learning_rate: T) {
        //nothing to update
    }

    fn get_output_buffer() -> Tensor2d<T, IO, B> {
        Tensor2d::new()
    }
}