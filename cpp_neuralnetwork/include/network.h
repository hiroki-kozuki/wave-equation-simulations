#pragma once 

#include <iostream>
#include <torch/torch.h>

struct NetImpl : torch::nn::Module {
    // Constructor to initialize the layers
    NetImpl(int fc1_dims, int fc2_dims)
       : fc1(fc1_dims, fc1_dims), fc2(fc2_dims, fc2_dims), out(fc2_dims, 1) {
          // Register the layers as submodules
          register_module("fc1", fc1);
          register_module("fc2", fc2);
          register_module("out", out);
    }

    // Forward pass
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1(x)); // Apply ReLU activation after fc1
        x = torch::relu(fc2(x)); // Apply ReLU activation after fc2
        x = out(x);              // Output layer (no activation)
        return x;
    }

    // Layer definitions
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, out{nullptr};
};

// Define a module holder for the network
TORCH_MODULE(Net);
