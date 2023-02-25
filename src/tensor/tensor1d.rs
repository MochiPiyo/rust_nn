use super::{Num, TensorTrait, Tensor2d};

#[derive(Clone)]
pub struct Tensor1d<T, const N: usize> {
    pub body: Box<[T; N]>,

    pub length: usize,
}
impl<T: Num, const N: usize> Tensor1d<T, N> {
    pub fn new_fill_with(init: T) -> Self {
        Self {
            body: Box::new([init; N]),
            length: N,
        }
    }

    pub fn new_from_array(array: [T; N]) -> Self {
        Self {
            body: Box::new(array),
            length: N,
        }
    }

    pub fn new_from_vec(vec: &Vec<T>) -> Result<Self, String> {
        if vec.len() < N {
            return Err(format!("vec is to short. vec.len() = {}, while expect length is {}", vec.len(), N));
        }else if vec.len() > N {
            return Err(format!("vec is to long. vec.len() = {}, while expect length is {}", vec.len(), N));
        }
        let mut body = Box::new([T::default(); N]);
        for i in 0..N {
            body[i] = vec[i];
        }
        
        Ok(Self {
            body,
            length: N,
        })
    }
    
    pub fn to_tensor2d_as_row(&self) -> Tensor2d<T, N, 1> {
        Tensor2d::<T, N, 1> {
            body: Box::new([*self.body.clone(); 1]),
            length: [1, N],
        }
    }

    pub fn to_tensor2d_as_col(&self) -> Tensor2d<T, 1, N> {
        let mut body = Box::new([[T::default();1]; N]);
        for (body_i, self_i) in body.iter_mut().zip(self.body.iter()) {
            *body_i = [*self_i];
        }
        Tensor2d::<T, 1, N> {
            body,
            length: [1, N],
        }
    }
}

impl<T: Num, const N: usize> TensorTrait<T> for Tensor1d<T, N> {
    fn new() -> Self {
        Self {
            body: Box::new([T::default(); N]),
            length: N,
        }
    }

    fn add(&mut self, other: &Self) {
        for (self_i, other_i) in self.body.iter_mut().zip(other.body.iter()) {
            *self_i += *other_i;
        }
    }

    fn sub(&mut self, other: &Self) {
        for (self_i, other_i) in self.body.iter_mut().zip(other.body.iter()) {
            *self_i -= *other_i;
        }
    }

    fn mul(&mut self, other: &Self) {
        for (self_i, other_i) in self.body.iter_mut().zip(other.body.iter()) {
            *self_i *= *other_i;
        }
    }

    fn div(&mut self, other: &Self) {
        for (self_i, other_i) in self.body.iter_mut().zip(other.body.iter()) {
            *self_i /= *other_i;
        }
    }

    fn rem(&mut self, other: &Self) {
        for (self_i, other_i) in self.body.iter_mut().zip(other.body.iter()) {
            *self_i %= *other_i;
        }
    }

    fn add_scalar(&mut self, scalar: T) {
        for self_i in self.body.iter_mut() {
            *self_i += scalar;
        }
    }

    fn sub_scalar(&mut self, scalar: T) {
        for self_i in self.body.iter_mut() {
            *self_i -= scalar;
        }
    }

    fn mul_scalar(&mut self, scalar: T) {
        for self_i in self.body.iter_mut() {
            *self_i *= scalar;
        }
    }

    fn div_scalar(&mut self, scalar: T) {
        for self_i in self.body.iter_mut() {
            *self_i /= scalar;
        }
    }

    fn rem_scalar(&mut self, scalar: T) {
        for self_i in self.body.iter_mut() {
            *self_i %= scalar;
        }
    }
}


//---impl Ops --------------------------------------------------------------
impl<T: Num, const N: usize> std::ops::Add for Tensor1d<T, N> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut returns = self.clone();
        for (self_i, other_i) in returns.body.iter_mut().zip(rhs.body.iter()) {
            *self_i += *other_i;
        }
        returns
    }
}
impl<T: Num, const N: usize> std::ops::Sub for Tensor1d<T, N> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut returns = self.clone();
        for (self_i, other_i) in returns.body.iter_mut().zip(rhs.body.iter()) {
            *self_i -= *other_i;
        }
        returns
    }
}
impl<T: Num, const N: usize> std::ops::Mul for Tensor1d<T, N> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut returns = self.clone();
        for (self_i, other_i) in returns.body.iter_mut().zip(rhs.body.iter()) {
            *self_i *= *other_i;
        }
        returns
    }
}
impl<T: Num, const N: usize> std::ops::Div for Tensor1d<T, N> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let mut returns = self.clone();
        for (self_i, other_i) in returns.body.iter_mut().zip(rhs.body.iter()) {
            *self_i /= *other_i;
        }
        returns
    }
}
impl<T: Num, const N: usize> std::ops::Rem for Tensor1d<T, N> {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        let mut returns = self.clone();
        for (self_i, other_i) in returns.body.iter_mut().zip(rhs.body.iter()) {
            *self_i %= *other_i;
        }
        returns
    }
}

//---impl OpsAssign --------------------------------------------------------------
impl<T: Num, const N: usize> std::ops::AddAssign for Tensor1d<T, N> {
    fn add_assign(&mut self, rhs: Self) {
        self.add(&rhs)
    }
}
impl<T: Num, const N: usize> std::ops::SubAssign for Tensor1d<T, N> {
    fn sub_assign(&mut self, rhs: Self) {
        self.sub(&rhs)
    }
}
impl<T: Num, const N: usize> std::ops::MulAssign for Tensor1d<T, N> {
    fn mul_assign(&mut self, rhs: Self) {
        self.mul(&rhs)
    }
}
impl<T: Num, const N: usize> std::ops::DivAssign for Tensor1d<T, N> {
    fn div_assign(&mut self, rhs: Self) {
        self.div(&rhs)
    }
}
impl<T: Num, const N: usize> std::ops::RemAssign for Tensor1d<T, N> {
    fn rem_assign(&mut self, rhs: Self) {
        self.rem(&rhs)
    }
}

impl<const N: usize> Tensor1d<usize, N> {
    pub fn new_usize() -> Self {
        Self {
            body: Box::new([0; N]),
            length: N,
        }
    }
}