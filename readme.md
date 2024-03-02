# Learn MNIST with No Crate in Rust
Second Challenge

```
log
2023/2/26 Created

2023/3/16 Learning completed, with He initialization.

2024/3/12 Edit readme
```

## Problems from the Previous Challenge
The previous challenge failed to enable the network to "learn" effectively due to the complexity of the code, making it difficult to locate and fix bugs. The primary issue was the attempt to directly translate Python code, especially numpy, which resulted in a bad architecture for Rust.

## Improved Points
This time(second challenge), we have made the following improvements:

- Separated tensors and tensor operations by defining them with structs and traits (suitable for Rust).
- Utilized const generics for layer sizes and their input-output tensor operations.
- Rewrote functions and layers to handle batched data.

## Results
### Positive Aspects:
- A profound understanding of deep learning, particularly gradient backpropagation.
- A strongly type-safe neural network model that surpasses Python's capabilities.
- Zero external dependencies, making it compatible with any OS and any hardware that Rust supports.

### Challenges Faced:
- Significant investment in code development, time, and intellectual resources.
- Suspected bugs with each model learning execution, in contrast to Python libraries that offer well-validated components.
- Excessive constraints from const generics:
    - Layers cannot be stored in a `Vec<>`, even if using `Vec<Box<dyn Trait>>`, because `LayerTrait` employs const generics. This necessitates a parameter such as `dyn Trait<T, I, O>`. However, the sizes of const generics I and O vary for each layer.
    - Inability to change the model size after compilation, preventing features like command-line arguments and dynamically adjusting algorithms (e.g., genetic learning).

## Future Endeavors
- **GPU Integration:** The wgpu framework appears promising due to its wide-ranging support for operating systems, hardware, and even browsers in the future. However, generating wgsl (WebGPU Shading Language) is a challenge.
- **Synergy with CPU Algorithms:** Written in Rust, a highly efficient language, we can seamlessly combine low-level algorithms.
