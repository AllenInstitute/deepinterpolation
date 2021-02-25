.. image:: https://circleci.com/gh/AllenInstitute/deepinterpolation.svg?style=svg
    :target: https://circleci.com/gh/AllenInstitute/deepinterpolation

Deep Interpolation
========================

*deepinterpolation* is a Python library to denoise data by removing independent noise. Importantly training does **NOT** require ground truth. This repository is currently meant to support the bioRxiv publication results : https://www.biorxiv.org/content/10.1101/2020.10.15.341602v1

Principle of Deep Interpolation
========================
.. image:: /docs/principle.png
    :alt: principle of deep interpolation
    :width: 100 px
**Figure 1** - Schematic introducing the principles of deep interpolation.  **A**. An interpolation model is trained to predict a noisy block from other blocks with independent noise. The loss is the difference between the predicted data and a new noisy block. **B**. The interpolation model is used to create a noiseless version of the input data. 

For more information, consult the associated bioRxiv publication : https://www.biorxiv.org/content/10.1101/2020.10.15.341602v1

Support
========================

For bug and issues, please submit issue tickets on this repository. 
For installation and running support, join the slack channel (if invitation has expired: email to Jerome): https://join.slack.com/t/deepinterpolation/shared_invite/zt-n36he49d-6fw1LNqqe5jgbhrk2ktGMg

Installation
========================

The following outlines how to install on your local machine. This should take no more than a few minutes. This was tested on a macOS Catalina but should be adapted depending on your final environment (institution cluster, AWS EC2 instance, ...). Tensorflow made a lot of progress lately to install GPU dependencies. However, you might have to consult tensorflow documentation to enable your GPU. The small training example below works on both CPU and GPU architecture (ie. even a small macbook). If you are not familiar with using deep learning, we recommend to play with smaller datasets first, such as the example Neuropixels data provided. 

1. Clone the repository locally on a directory 'local_dir'

.. code-block:: bash

	git clone https://github.com/AllenInstitute/deepinterpolation.git

2. Go to that directory

.. code-block:: bash

	cd 'local_dir'

3. Create new conda environment called 'local_env'

.. code-block:: bash

	conda create -n local_env python=3.8

4. activate environment

.. code-block:: bash

	conda activate local_env

5. install necessary packages

.. code-block:: bash

	make init

6. install deepinterpolation package

.. code-block:: bash

	python setup.py install

General code description
========================
The files in the deepinterpolation folder contains the core classes for training, inferrence, loss calculation and network generations. Those are called 'Collection'. Each collection is essentially a local list of functions that are used to create different type of objects and can be extended on one another. 
For instance, the network_collection.py contains a list of networks that can be generated for training. This allows for quick iteration and modification of an architecture while keeping the code organized. 

FAQ
========================
See here : https://github.com/AllenInstitute/deepinterpolation/tree/master/faq

Training
========================
To adapt DeepInterpolation to a new dataset, you will need to use or recreate a generator in 'generator_collection.py'. Those are all constructed from a core class called 'DeepGenerator'. The 'CollectorGenerator' class allows to group generators if your dataset is distributed across many files/folder/sources. 
This system was designed to allow to train very large DeepInterpolation models from TB of data distributed on a network infrastructure. 

To try out training your own DeepInterpolation network, we recommend to start with this file: https://github.com/AllenInstitute/deepinterpolation/blob/master/examples/example_tiny_ephys_training.py

In this file, you will need to edit the jobdir variable, in particular change "/Users/jeromel/test" to a local folder appropriate to save your models. 

Then, activate your conda env called 'local_env'

.. code-block:: bash

	conda activate local_env
	
then run 

.. code-block:: bash

	python example_tiny_ephys_training.py

If everything runs correctly, you should see the following in just a few minutes : 

.. code-block:: bash

	2020-10-19 18:01:03.735098: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)to use the following CPU instructions in performance-critical operations:  AVX2 FMA
	To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
	sh: sysctl: command not found
	2020-10-19 18:01:03.749184: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7f9b1f115860 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
	2020-10-19 18:01:03.749202: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
	WARNING:tensorflow:`period` argument is deprecated. Please use `save_freq` to specify the frequency in number of batches seen.
	Epoch 1/5
	10/10 [==============================] - 19s 2s/step - loss: 0.4597 - val_loss: 0.3987
	Epoch 2/5
	10/10 [==============================] - 20s 2s/step - loss: 0.3796 - val_loss: 0.3785
	Epoch 3/5
	10/10 [==============================] - 22s 2s/step - loss: 0.3646 - val_loss: 0.3709
	Epoch 4/5
	10/10 [==============================] - 21s 2s/step - loss: 0.3797 - val_loss: 0.3698
	Epoch 5/5
	10/10 [==============================] - 21s 2s/step - loss: 0.3835 - val_loss: 0.3675
	Saved model to disk

This is a toy example but you can increase the number of training frames to increase the quality of the model. 
All parameters are commented in the file. To adjust to a larger dataset, change the train_path parameters, the start_frame and end_frame parameters. 

Inference
========================

Raw pre-trained models are available either as part of Tensorflow ModelServer in an AWS docker environment or as a separate h5 file on Dropbox. 

The following models are currently available : 

**Two-photon Ai93 excitatory line DeepInterpolation network:**

Key recording parameters: 

- 30Hz sampling rate, 400x400 μm2 field of view, 512x512 pixels.
- 0.8 NA objective.
- 910 nm excitation wavelength.
- Gcamp6f calcium indicator.
- Ai93 reporter line expressed in excitatory neurons.
- Docker hub id : 245412653747/deep_interpolation:allen_400um_512pix_30hz_ai93
- Dropbox link : https://www.dropbox.com/sh/vwxf1uq2j60uj9o/AAC9BQI1bdfmAL3OFO0lmVb1a?dl=0
- Training data : https://github.com/AllenInstitute/deepinterpolation/blob/master/examples/paper_generation_code/json_data/2019-09-05-train-very-large-single-plane-Ai93-norm.json

**Two-photon Ai148 excitatory line DeepInterpolation network:**

Key recording parameters: 

- 30 Hz sampling rate, 400x400 μm2 field of view, 512x512 pixels.
- 0.8 NA objective.
- 910 nm excitation wavelength.
- Gcamp6f calcium indicator.
- Ai148 reporter line expressed in excitatory neurons.
- Pre-processing: Individual movies were motion corrected. Each movie recording was mean-centered and normalized with a single pair of value for all pixels 
- Docker hub id : 245412653747/deep_interpolation:allen_400um_512pix_30hz_ai148
- Dropxbox link : https://www.dropbox.com/sh/u9h9mhppkmku5bs/AAD9UoomhB3D4JfLV7zT9Y_Ca?dl=0
- Training data : https://github.com/AllenInstitute/deepinterpolation/blob/master/examples/paper_generation_code/json_data/2019-09-05-train-very-large-single-plane-Ai148-norm.json

**Neuropixel DeepInterpolation network:**

Key recording parameters: 

- Neuropixels Phase 3a probes
- 374 simultaneous recording sites across 3.84 mm, 10 reference channels
- Four-column checkerboard site layout with 20 µm spacing between rows
- 30 kHz sampling rate
- 500x hardware gain setting
- 500 Hz high pass filter in hardware, 150 Hz high-pass filter applied offline. 
- Pre-processing: Median subtraction was applied to individual probes to remove signals that were common across all recording sites. Each probe recording was mean-centered and normalized with a single pair of value for all nodes on the probe. 
- Docker hub id : 245412653747/deep_interpolation:allen_neuropixel
- Dropxbox link : https://www.dropbox.com/sh/tm3epzil44ybalq/AACyKxfvvA2T_Lq_rnpHnhFma?dl=0

**fMRI DeepInterpolation network:**

Key recording parameters: 

- TR, 3000 ms; TE, 30 ms; flip angle, 80°; voxel size, 3 × 3 × 3 mm; FOV, 192 × 192 mm; number of slices, 50, slice gap, 0 mm
- Pre-processing: N/A
- Docker hub id : 245412653747/deep_interpolation:allen_3_3_3_tr_3000_fmri
- Dropxbox link : https://www.dropbox.com/sh/ngx5plndmd4jsca/AAAkR-_4_E7VyL8WzEC7twuza?dl=0



To start inference, we recommend to start with this file: https://github.com/AllenInstitute/deepinterpolation/blob/master/examples/example_tiny_ephys_inference.py

In this file, you will need to edit the train_path, model_path and output_file variable to fit your local paths. 

Then, activate your conda env called 'local_env'

.. code-block:: bash

	conda activate local_env
	
then run 

.. code-block:: bash

	python example_tiny_ephys_inference.py

If everything runs correctly, you should see the following in just a few minutes : 

.. code-block:: bash

	2020-10-20 14:10:37.549061: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)to use the following CPU instructions in performance-critical operations:  AVX2 FMA
	To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
	sh: sysctl: command not found
	2020-10-20 14:10:37.564133: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7f82ada8a520 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
	2020-10-20 14:10:37.564156: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version

This is a toy example but you can increase the start_frame and end_frame variable for larger data. 
It is important to keep in mind that this process is easily parallelizable. In practice, we wrapped this code with additional routines to leverage 20 to 100 cluster CPU nodes to accelerate this process. You could also use GPU nodes as well, we just had access to a much larger number of CPU machines quickly.  

More on using the Tensorflow ModelServer soon. Those are usefull to deploy to AWS and/or avoid installing GPUs related packages. 

License
========================

Allen Institute Software License – This software license is the 2-clause BSD 
license plus clause a third clause that prohibits redistribution and use for 
commercial purposes without further permission. 

Copyright © 2019. Allen Institute.  All rights reserved.

Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this 
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, 
this list of conditions and the following disclaimer in the documentation 
and/or other materials provided with the distribution.

3. Redistributions and use for commercial purposes are not permitted without 
the Allen Institute’s written permission. For purposes of this license, 
commercial purposes are the incorporation of the Allen Institute's software 
into anything for which you will charge fees or other compensation or use of 
the software to perform a commercial service for a third party. Contact 
terms@alleninstitute.org for commercial licensing opportunities.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE 
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
