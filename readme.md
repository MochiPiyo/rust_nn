# Non dependence Neural Net with Rust
## Second challenge
2023_02_26

## problems of previous cuallenge 
the previous challenge was failed to make it "learn" the network because code was too complicated to find where the bug code is. The biggest factor is that "greedy translating" python code, especially numpy, made the code worse archtectured for Rust.

## improvement points !
So, this time, 
- define tensor and tensor operation
- const generics for layer size and their io tensor operation
- native batch io support functions, layers

## result
Good point

- Well understanding of Deep learning, especially gradient backprops.
- Super Type Safe nn model. Python never can't be like.
- no dependences, so run everywhere compiler support, no changing for library update.

Bad point

- a lot of code, time, thinking resource
- Every execution of model learning is suspicious for bug while python libraries offer well validated components.
- too much restriction from const generics.
    - layers can't in a `Vec<>` even if `Vec<Box<dyn Trait>>` because LayerTrait has const generics, this means parameter is necessary like `dyn Traint<T, I, O>` but size of const generics I and O are different for eatch layer. 
    - can't change model size after compiled. for instance, command line argument and dynamically size changing algorithm (ex: genetic learning) are outof case.


# future 
- GPU: wgpu seems interesting because wide support OS, Hardware, even if Browsers(future). but auto generation of wgsl is challenging.
- combination with cpu algorithms. written in rust, quite fast lang, we can write down low level algorithms in completery combined way.