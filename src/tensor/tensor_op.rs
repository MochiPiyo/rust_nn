use super::{Num, tensor1d::Tensor1d, tensor2d::Tensor2d};


pub fn inner_product<T: Num, const N: usize>
    (left: Tensor1d<T, N>, right: Tensor1d<T, N>, zero: T) -> T
{
    let mut sum = zero;
    for &l in left.body.iter() {
        for &r in right.body.iter() {
            sum += l * r;
        }
    }
    return sum;
}

//ex: Tnsor2d<T, I, B> for add all input: I in batches: B
pub fn add_all_row<T: Num, const N: usize, const B: usize>
    (source: &Tensor2d<T, N, B>, target: &mut Tensor1d<T, N>)
{
    for row in source.body.iter() {
        for (target_i, row_i) in target.body.iter_mut().zip(row.iter()) {
            *target_i += *row_i;
        }
    }
}

//output += matmul(left, right)
pub fn add_matrix_mul<T: Num, const N: usize, const M: usize, const S: usize>
    (left: &Tensor2d<T, N, M>, right: &Tensor2d<T, S, N>, output: &mut Tensor2d<T, S, M>)
{
    //i, k, j 高速化
    for i_n in 0..N {
        for k_m in 0..M {
            for j_s in 0..S {
                output.body[i_n][j_s] = left.body[i_n][k_m] * right.body[k_m][j_s];
            }
        }
    }
}


pub fn softmax_per_batch<T: Num, const IO: usize, const B: usize>
    (input: &Tensor2d<T, IO, B>, softmax_result: &mut Tensor2d<T, IO, B>)
{
    for (input_per_data, softmax_result_per_data) in input.body.iter().zip(softmax_result.body.iter_mut()) {
        //for prevent overflow at exp()
        let mut largest: T = T::min_const();
        for i in input_per_data.iter() {
            if largest < *i {
                largest = *i;
            }
        }
        
        //exp
        let mut exp_sum = T::zero();
        for (data_i, exp) in input_per_data.iter().zip(softmax_result_per_data.iter_mut()) {
            *exp = ((*data_i) - largest).exp();
            exp_sum += *exp;
        }

        //(exp/exp_sum)
        for i in softmax_result_per_data.iter_mut() {
            *i = *i / exp_sum;
        }
    }
    
}

pub fn cross_entropy_error_per_batch<T: Num, const IO: usize, const B: usize>
    (softmax_result: Tensor2d<T, IO, B>, label: &Tensor1d<usize, B>, loss_output: &mut Tensor2d<T, IO, B>)
{
    for (error_per_data, (softmax_result_per_data, &label_this)) in loss_output.body.iter_mut().zip(softmax_result.body.iter().zip(label.body.iter())) {
        
        for (i, error_i) in error_per_data.iter_mut().enumerate() {
            if i == label_this {
                *error_i = -(softmax_result_per_data[label_this] + T::min_const()).ln();
            }else {
                *error_i = T::zero();
            }
        }
    }
}