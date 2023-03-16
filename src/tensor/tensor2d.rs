use super::{Num, TensorTrait};


/*
Tensor2d<T, H, V>
-> body[[T; H]; V]
because TVH way can arrayaccess like this
: tensor2d[V][H]


for batch io, TIB or TOB way, so that self.body.iter() can be iteration of eatch data
Tensor2d<T, I, B> or Tensor2d<T, O, B>
-> body[[T; data.len()]; B]

for layer weights, TOI way
Tensor2d<T, O, I>
-> body: [[T; O]; I]
 */



#[derive(Clone, PartialEq, Debug)]
pub struct Tensor2d<T, const H: usize, const V: usize> {
    //vertical len: V, horizontal len: H
    pub body: Box<[[T; H]; V]>,
    pub length: [usize; 2],
}
impl<T: Num, const H: usize, const V: usize> Tensor2d<T, H, V> {
    pub fn new_fill_with(init: T) -> Self {
        Self {
            body: Box::new([[init; H]; V]),
            length: [V, H],
        }
    }
    
    pub fn new_from_array(array: [[T; H]; V]) -> Self {
        Self {
            body: Box::new(array),
            length: [V, H],
        }
    }

    pub fn new_from_vec(vec: Vec<T>) -> Self {
        let mut body = Box::new([[T::default(); H]; V]);
        let mut iter = vec.iter();
        for self_row in body.iter_mut() {
            for self_i in self_row.iter_mut() {
                *self_i = *iter.next().unwrap();
            }
        }
        Self {
            body,
            length: [V, H],
        }
    }

    pub fn new_from_slices(slices: &[&[T]]) -> Self {
        let mut body = Box::new([[T::default(); H]; V]);
        for (self_row, slice_row) in body.iter_mut().zip(slices.iter()) {
            for (self_i, slice_i) in self_row.iter_mut().zip(slice_row.iter()) {
                *self_i = *slice_i;
            }
        }
        Self {
            body,
            length: [V, H],
        }
    }

    pub fn transpose(&self) -> Tensor2d<T, V, H> {
        let mut transposed: Tensor2d<T, V, H> = Tensor2d::new();
        for (v, self_row) in self.body.iter().enumerate() {
            for (h, self_i) in self_row.iter().enumerate() {
                transposed.body[h][v] = *self_i;
            }
        }
        return transposed;
    }
}

impl<T: Num, const H: usize, const V: usize> TensorTrait<T> for Tensor2d<T, H, V> {
    fn new() -> Self {
        Self {
            body: Box::new([[T::default(); H]; V]),
            length: [V, H],
        }
    }

    fn add(&mut self, other: &Self) {
        for (self_row, other_row) in self.body.iter_mut().zip(other.body.iter()) {
            for (self_i, other_i) in self_row.iter_mut().zip(other_row.iter()) {
                *self_i += *other_i;
            }
        }
    }

    fn sub(&mut self, other: &Self) {
        for (self_row, other_row) in self.body.iter_mut().zip(other.body.iter()) {
            for (self_i, other_i) in self_row.iter_mut().zip(other_row.iter()) {
                *self_i -= *other_i;
            }
        }
    }

    fn mul(&mut self, other: &Self) {
        for (self_row, other_row) in self.body.iter_mut().zip(other.body.iter()) {
            for (self_i, other_i) in self_row.iter_mut().zip(other_row.iter()) {
                *self_i *= *other_i;
            }
        }
    }

    fn div(&mut self, other: &Self) {
        for (self_row, other_row) in self.body.iter_mut().zip(other.body.iter()) {
            for (self_i, other_i) in self_row.iter_mut().zip(other_row.iter()) {
                *self_i /= *other_i;
            }
        }
    }

    fn rem(&mut self, other: &Self) {
        for (self_row, other_row) in self.body.iter_mut().zip(other.body.iter()) {
            for (self_i, other_i) in self_row.iter_mut().zip(other_row.iter()) {
                *self_i %= *other_i;
            }
        }
    }


    

    fn add_scalar(&mut self, scalar: T) {
        for self_row in self.body.iter_mut() {
            for self_i in self_row.iter_mut() {
                *self_i += scalar;
            }
        }
    }

    fn sub_scalar(&mut self, scalar: T) {
        for self_row in self.body.iter_mut() {
            for self_i in self_row.iter_mut() {
                *self_i -= scalar;
            }
        }
    }

    fn mul_scalar(&mut self, scalar: T) {
        for self_row in self.body.iter_mut() {
            for self_i in self_row.iter_mut() {
                *self_i *= scalar;
            }
        }
    }

    fn div_scalar(&mut self, scalar: T) {
        for self_row in self.body.iter_mut() {
            for self_i in self_row.iter_mut() {
                *self_i /= scalar;
            }
        }
    }

    fn rem_scalar(&mut self, scalar: T) {
        for self_row in self.body.iter_mut() {
            for self_i in self_row.iter_mut() {
                *self_i %= scalar;
            }
        }
    }
}



//---impl Ops --------------------------------------------------------------
impl<T: Num, const H: usize, const V: usize> std::ops::Add for Tensor2d<T, H, V> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut returns = self.clone();
        for (self_row, other_row) in returns.body.iter_mut().zip(rhs.body.iter()) {
            for (self_i, other_i) in self_row.iter_mut().zip(other_row.iter()) {
                *self_i += *other_i;
            }
        }
        returns
    }
}
impl<T: Num, const H: usize, const V: usize> std::ops::Sub for Tensor2d<T, H, V> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut returns = self.clone();
        for (self_row, other_row) in returns.body.iter_mut().zip(rhs.body.iter()) {
            for (self_i, other_i) in self_row.iter_mut().zip(other_row.iter()) {
                *self_i -= *other_i;
            }
        }
        returns
    }
}
impl<T: Num, const H: usize, const V: usize> std::ops::Mul for Tensor2d<T, H, V> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut returns = self.clone();
        for (self_row, other_row) in returns.body.iter_mut().zip(rhs.body.iter()) {
            for (self_i, other_i) in self_row.iter_mut().zip(other_row.iter()) {
                *self_i *= *other_i;
            }
        }
        returns
    }
}
impl<T: Num, const H: usize, const V: usize> std::ops::Div for Tensor2d<T, H, V> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let mut returns = self.clone();
        for (self_row, other_row) in returns.body.iter_mut().zip(rhs.body.iter()) {
            for (self_i, other_i) in self_row.iter_mut().zip(other_row.iter()) {
                *self_i /= *other_i;
            }
        }
        returns
    }
}
impl<T: Num, const H: usize, const V: usize> std::ops::Rem for Tensor2d<T, H, V> {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        let mut returns = self.clone();
        for (self_row, other_row) in returns.body.iter_mut().zip(rhs.body.iter()) {
            for (self_i, other_i) in self_row.iter_mut().zip(other_row.iter()) {
                *self_i %= *other_i;
            }
        }
        returns
    }
}



//---impl OpsAssign --------------------------------------------------------------
impl<T: Num, const H: usize, const V: usize> std::ops::AddAssign for Tensor2d<T, H, V> {
    fn add_assign(&mut self, rhs: Self) {
        self.add(&rhs)
    }
}
impl<T: Num, const H: usize, const V: usize> std::ops::SubAssign for Tensor2d<T, H, V> {
    fn sub_assign(&mut self, rhs: Self) {
        self.sub(&rhs)
    }
}
impl<T: Num, const H: usize, const V: usize> std::ops::MulAssign for Tensor2d<T, H, V> {
    fn mul_assign(&mut self, rhs: Self) {
        self.mul(&rhs)
    }
}
impl<T: Num, const H: usize, const V: usize> std::ops::DivAssign for Tensor2d<T, H, V> {
    fn div_assign(&mut self, rhs: Self) {
        self.div(&rhs)
    }
}
impl<T: Num, const H: usize, const V: usize> std::ops::RemAssign for Tensor2d<T, H, V> {
    fn rem_assign(&mut self, rhs: Self) {
        self.rem(&rhs)
    }
}


impl<const H: usize, const V: usize> Tensor2d<bool, H, V> {
    pub fn new_bool() -> Self {
        Self {
            body: Box::new([[false; H]; V]),
            length: [H, V],
        }
    }
}