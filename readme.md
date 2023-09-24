Non-Dependent Neural Network with Rust
Second Challenge
log Created on February 26, 2023 Learning completed on March 16, 2023, with He initialization.

Problems from the Previous Challenge
The previous challenge failed to make the network "learn" due # Non-Dependent Neural Network with Rust
## Second Challenge

`log
Created on February 26, 2023

Learning completed on March 16, 2023, with He initialization.
`

## Problems from the Previous Challenge 
The previous challenge failed to make the network "learn" due to the complexity of the code, making it challenging to locate and fix bugs. The primary issue stemmed from attempting to directly translate Python code, especially numpy code, which resulted in suboptimal architecture for Rust.

## Improved Points 
This time, we have made the following improvements:

- Separate tensors and tensor operations by define them with struct and trait (optimized for Rust).
- Utilized const generics for layer sizes and their input-output tensor operations.
- Rewrite functions and layers for batched data.

## Results 
Positive Aspects:

- Profound understanding of deep learning, particularly gradient backpropagation.
- A strongly type safe neural network model that surpasses Python's capabilities.
- Zero external dependencies, making it compatible with any compiler that Rust supports.

Challenges Faced:

- Significant investment in code development, time, and intellectual resources.
- Suspicion of bugs with each model learning execution, in contrast to Python libraries offering well-validated components.
- Excessive constraints from const generics:
    - Layers cannot be stored in a `Vec<>` even if using `Vec<Box<dyn Trait>>` because `LayerTrait` employs const generics. This necessitates a parameter such as `dyn Trait<T, I, O>`. However, the sizes of const generics I and O vary for each layer.
    - Inability to change the model size after compilation, preventing features like command-line arguments and dynamically adjusting algorithms (e.g., genetic learning).

# Future Endeavors 
- GPU Integration: The wgpu framework appears promising due to its wide-ranging support for operating systems, hardware, and even browsers in the future. However, generating wgsl (WebGPU Shading Language) automatically presents a challenge.
- Synergy with CPU Algorithms: Written in Rust, a highly efficient language, we can seamlessly combine low-level algorithms.