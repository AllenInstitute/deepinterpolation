## This is a FAQ for DeepInterpolation | Frequently asked questions about DeepInterpolation

This document includes questions raised when getting DeepInterpolation to run independently. 

## FAQ
**Question:** I am getting error related to GPU memory allocation, for instance : "ResourceExhaustedError: OOM when allocating tensor with shape..."
how should I deal with these?

**Answer:** For Deep learning, there are 2 types of memory that are critical, the RAM memory associated with your CPU and the GPU RAM associated with your GPU unit. Filling either will yield different kind of issues. The GPU RAM is used to store the model being learned as well as input and output data in a given batch of data. The model size is driven by the computational complexity of the task while the batch size guides how well you will measure the loss gradient at each iteration. Two-photon imaging models are typically larger than Electrophysiological models as we simultaneously record from more samples, expect therefore to need both more RAM and GPU RAM for Two-photon models. 

In addition, the larger the batch size, the more GPU RAM you will need. You will see in the method section of the paper that we adjusted the batch size accordingly. With DeepInterpolation, we are using a noisy loss to calculate the gradient so ideally you want the batch to be as large as possible, within the limits of your available GPU RAM. 

##

Template:
**Q:**
**A:**
##