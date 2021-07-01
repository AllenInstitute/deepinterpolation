## This is a FAQ for DeepInterpolation

**Question: What kind of hardware/platform do you recommend to get started?**

**Answer:** This will depend on many different factors. The example notebook provided (https://github.com/AllenInstitute/deepinterpolation/tree/master/examples) were tested on a 2015 MacBook Pro with 16GB of RAM but such a machine will be insufficient for any serious work. They were provided to facilitate the first few experimentations and familiarity with the code. We recommend a machine with at least 30 to 50GB of RAM and a dedicated GPU such as one listed in the method section of the publication. The fMRI and Electrophysiological DeepInterpolation models require less memory than the two-photon imaging DeepInterpolation models. We trained our Two-photon models on a machine with 100+ GB of RAM and a single dedicated GPU. For training, it helps to have a CPU that can support 16 threads or more as data fetching from files is multi-threaded allowing to increase the bandwidth onto the GPU. Training and inference have very different requirements. We implemented inference on CPU only as we had access to many machines (100+) to parallelize the process and process sub-parts of files in parallel. You will for sure need a dedicated GPU for Training. Training speed depends on a lot of factors but with our hardware, it would take one day or more to train our DeepInterpolation models from a large collection of files.   

##

**Question: Can I train in the cloud?**

**Answer:** We are currently experimenting with this and will report our experience. We recently had good success using Google Colab to train electrophysiological models, even with the free instances with GPU. For two photon models, you will likely need to upgrade to have access to more RAM. All fMRI work was done on AWS with P2 or P3 instances (like a p2.xlarge, see https://aws.amazon.com/ec2/instance-types/p2/). Whether this is practical depends on avaibility in your AWS area, P2 and P3 instances have been in very high demand lately. 

##

**Question: Do you have more example training and inference code?**

**Answer:** A large number of scripts are provided here : https://github.com/AllenInstitute/deepinterpolation/tree/master/examples/paper_generation_code
Those are the scripts we used to train and do inference for the publication, using the DeepInterpolation classes. Some of the code in this folder is related to how our cluster operates. 
The files in this folder : https://github.com/AllenInstitute/deepinterpolation/tree/master/examples/paper_generation_code/cluster_lib
show how we parallelized the submission of jobs on our cluster infrastructure. You could reuse this code within your own institution but it will require adjustements as it is using pbstools, an internal library designed to work with our cluster. For example, the generic_ophys_process_sync.py file is the master job that is calling out smaller section inference with single_ophys_section_inferrence.py. Therefore you will likely need to update *_sync.py files. 

##
**Question: I am getting error related to GPU memory allocation, for instance : "ResourceExhaustedError: OOM when allocating tensor with shape...".
how should I deal with these?**

**Answer:** For Deep learning, there are 2 types of memory that are critical, the RAM memory associated with your CPU and the GPU RAM associated with your GPU unit. Filling either will yield different kind of issues. The GPU RAM is used to store the model being learned as well as input and output data in a given batch of data. The model size is driven by the computational complexity of the task while the batch size guides how well you will measure the loss gradient at each iteration. Two-photon imaging models are typically larger than Electrophysiological models as we simultaneously record from more samples, expect therefore to need both more RAM and GPU RAM for Two-photon models. 

In addition, the larger the batch size, the more GPU RAM you will need. You will see in the method section of the paper that we adjusted the batch size accordingly. With DeepInterpolation, we are using a noisy loss to calculate the gradient so ideally you want the batch to be as large as possible, within the limits of your available GPU RAM. 

##

**Question: Can I do transfer learning from your published models to accelerate training and require less training data ?**

**Answer:** Yes! We are working to release more quantification and help on this front. There is an example script on how to do this using DeepInterpolation classes: 
https://github.com/AllenInstitute/deepinterpolation/blob/master/examples/example_tiny_ephys_transfer_training.py
Feel free to reach out if you are interested in working on this and have data you can share: We are looking for more diverse data to evaluate performance during the transfer!

##

**Question: Will the models from the Allen Institute work directly on my data?**

**Answer:** Yes and No. We have heard about some successes directly applying our models with external data. Some other laboratories reported issues in the quality of the reconstruction. It will depend how close your recording conditions are to ours. Obviously taking our models directly and applying them is the easiest approach at the moment. We are currently doing more in depth characterization of transfer performance. It is likely the final solution we will adopt as a field is to have pre-trained reconstruction models broadly available that we fine-tune on individual consistent datasets with a minimal amount of training. 

**Question:How can I contribute to the codebase?**

**Answer:** Feel free to do so! Just make a PR request and we will review it shortly. We are building up a continuous integration system with a battery of tests to facilitate this effort.  

**Question:Why is my output movie size smaller than the input ?**

**Answer:** This denoising framework uses surrounding frames (pre_frame and post_frame parameters) to predict a center frame. As a result, the first set of frames at the beginning and end of the movies are missing key values for the network to operate. We chose to only train our DeepInterpolation networks to expect full size inputs and excluded boundary conditions. As a result the current inference code truncate the first few and final frames of data. This is very similar to boundary conditions with rolling averages. However in this case, artifactual frames at tne onset and end of the movie could be more severe due to the non-linear nature of DeepInterpolation. 
It is possible to train specific networks to deal with those boundary conditions, for example by training only forward (or backward) looking inputs but we have not done so yet.

**Question:Why is my output data type float32 instead of the original input type (uint16 or others) ?**

**Answer:** First, Deep Learning methods typically convert input to float so as to have differentiable functions. Second, while the output values are mapped back on the same range as the input, the bit depth could potentially improve through denoising as DeepInterpolation is pulling more information than is present in a single pixel. We chose not to enforce the original bit depth because of that. Arguably some datasets are collected with excessive bit depth to begin with so it is possible that your data will not benefit from being represented with the higher precision of float32. We recommend to exercise good judgement when choosing your final bit depth based on these criterias. 

Template:
**Q:**
**A:**
##
