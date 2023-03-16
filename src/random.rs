

/*
No time seed. because our(this code) policy is that not using any crate, so time crate is unabailable
It's not problem in this case, get initial value for "training" of NeuralNetworl.

 */

use crate::tensor::Tensor1d;

pub trait LcgTrait<T = Self> {
    fn new(seed: u32) -> Lcg;
    fn gen(&mut self) -> u32;
    fn gen_0to1(&mut self) -> T;
}

//線形合同法
//Linear Congruential Generator: LCG
pub struct Lcg {
    temp: u32,
}
impl LcgTrait<f32> for Lcg {
    fn new(seed: u32) -> Lcg {
        Lcg {
            temp: seed,
        }
    }
    fn gen_0to1(&mut self) -> f32 {
        let a: u32 = 1664525;
        let c: u32 = 1013904223;
        let m: u32 = 214783647; //2^31 - 1

        let gen: u32 = (self.temp as u64 * a as u64 + c as u64) as u32 & m;
        self.temp = gen;

        return gen as f32 / u32::MAX as f32;
    }

    fn gen(&mut self) -> u32 {
        let a: u32 = 1664525;
        let c: u32 = 1013904223;
        let m: u32 = 214783647; //2^31 - 1

        let gen: u32 = (self.temp as u64 * a as u64 + c as u64) as u32 & m;
        self.temp = gen;
        
        return gen;
    }
}


fn generate_normal_distribution(average: f32, std_diviation: f32, rng: &mut Lcg) -> f32 {
    let u1 = rng.gen() as f32 / (u32::MAX as f32 + 1.0);
    let u2 = rng.gen() as f32 / (u32::MAX as f32 + 1.0);

    //Box-Muller法で正規分布を作成する
    let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();

    average + std_diviation * z0
}

pub fn he_init(size: usize, rng: &mut Lcg) -> Vec<f32> {
    let std_diviation = (2.0 / size as f32).sqrt();
    let mut weights = Vec::with_capacity(size);
    for _ in 0..size {
        let weight = generate_normal_distribution(0.0, std_diviation, rng);
        weights.push(weight);
    }
    weights
}