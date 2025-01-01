#include "include/network.h"
#include <torch/torch.h>
#include <iostream>

int main() {
    // Create an instance of the Net class
    Net network = Net(50, 10);

    // Print network details
    std::cout << network << "\n\n";

    // Create input tensor
    torch::Tensor x = torch::randn({2, 50});

    // Perform a forward pass
    torch::Tensor output = network->forward(x);

    // Print the output tensor
    std::cout << output << std::endl;

    return 0;
}
