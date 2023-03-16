

mod tensor_op;
mod tensor1d;
mod tensor2d;


use std::fmt::Debug;

pub use tensor1d::Tensor1d;
pub use tensor2d::Tensor2d;
pub use tensor_op::{_inner_product, add_matrix_mul, add_all_row, softmax_per_batch, cross_entropy_error_per_batch};

pub trait Num<Rhs = Self, Output = Self>:
    //+, -, *, /, %
    std::ops::Add<Rhs, Output = Output>
    + std::ops::Sub<Rhs, Output = Output>
    + std::ops::Mul<Rhs, Output = Output>
    + std::ops::Div<Rhs, Output = Output>
    + std::ops::Rem<Rhs, Output = Output>// % operator

    //+=, -=, *=, /=, %=
    + std::ops::AddAssign
    + std::ops::SubAssign
    + std::ops::MulAssign
    + std::ops::DivAssign
    + std::ops::RemAssign

    //(-x)
    + std::ops::Neg<Output = Self>
    
    + Default
    + Copy
    + PartialOrd
    + Debug
    
{
    fn zero() -> Self;
    fn one() -> Self;
    fn min_const() -> Self;
    fn ln(self) -> Self;
    fn exp(self) -> Self;
}

impl Num for f32 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn min_const() -> Self {
        std::f32::MIN
    }
    fn ln(self) -> Self {
        f32::ln(self)
    }
    fn exp(self) -> Self {
        f32::exp(self)
    }
}

pub trait TensorTrait<T: Num> {
    //init by Default::default() value
    fn new() -> Self;
    
    //+, -, *, /
    fn add(&mut self, other: &Self);
    fn sub(&mut self, other: &Self);
    fn mul(&mut self, other: &Self);
    fn div(&mut self, other: &Self);
    fn rem(&mut self, otehr: &Self);

    fn add_scalar(&mut self, scalar: T);
    fn sub_scalar(&mut self, scalar: T);
    fn mul_scalar(&mut self, scalar: T);
    fn div_scalar(&mut self, scalar: T);
    fn rem_scalar(&mut self, scalar: T);
}



//for easy usage
//別々にimplしたtraitが全部ついてるかのチェックにもなる
pub trait Tensor<T: Num, Rhs = Self, Output = Self>:
    TensorTrait<T>

    //+, -, *, /, %
    + std::ops::Add<Rhs, Output = Output>
    + std::ops::Sub<Rhs, Output = Output>
    + std::ops::Mul<Rhs, Output = Output>
    + std::ops::Div<Rhs, Output = Output>
    + std::ops::Rem<Rhs, Output = Output>

    //+=, -=, *=, /=, %=
    + std::ops::AddAssign
    + std::ops::SubAssign
    + std::ops::MulAssign
    + std::ops::DivAssign
    + std::ops::RemAssign


    + Clone
{}





