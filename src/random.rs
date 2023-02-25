

/*
No time seed. because our(this code) policy is that not using any crate, so time crate is unabailable
It's not problem in this case, get initial value for "training" of NeuralNetworl.

 */

pub trait LcgTrait<T = Self> {
    fn new(seed: u32) -> Lcg;
    fn gen(&mut self) -> u32;
    fn gen_0to1(&mut self) -> T;
}

//線形合同法
//Linear Congruential Generator: LCG
pub struct Lcg {
    seed: u32,
}
impl LcgTrait<f32> for Lcg {
    fn new(seed: u32) -> Lcg {
        Lcg {
            seed,
        }
    }
    fn gen_0to1(&mut self) -> f32 {
        let a: u32 = 1664525;
        let c: u32 = 1013904223;
        let m: u32 = 214783647; //2^31 - 1

        let gen: u32 = (self.seed as u64 * a as u64 + c as u64) as u32 & m;

        if self.seed + 1 == u32::MAX {
            dbg!("seed is u32 max");
            self.seed = 0;
        }else {
            self.seed += 1;
        }
        return gen as f32 / u32::MAX as f32;
    }

    fn gen(&mut self) -> u32 {
        let a: u32 = 1664525;
        let c: u32 = 1013904223;
        let m: u32 = 214783647; //2^31 - 1

        let gen: u32 = (self.seed as u64 * a as u64 + c as u64) as u32 & m;

        if self.seed + 1 == u32::MAX {
            dbg!("seed is u32 max");
            self.seed = 0;
        }else {
            self.seed += 1;
        }
        return gen;
    }
}


//pub struct MersenneTwister {

//}