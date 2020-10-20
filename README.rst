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

Installation
========================

The following outlines how to install on your local machine. This was tested on a macOS Catalina but should be adapted depending on your final environment (institution cluster, AWS EC2 instance, ...). Tensorflow made a lot of progress lately to install GPU dependencies. However, you might have to consult tensorflow documentation to enable your GPU. The small training example below works on both CPU and GPU architecture (ie. even a small macbook). If you are not familiar with using deep learning, we recommend to play with smaller datasets first, such as the example Neuropixels data provided. 

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

Training
========================
To adapt DeepInterpolation to a new dataset, you will need to use or recreate a generator in 'generator_collection.py'. Those are all constructed from a core class called 'DeepGenerator'. The 'CollectorGenerator' class allows to group generators if your dataset is distributed across many files/folder/sources. 
This system was designed to allow to train very large DeepInterpolation models from TB of data distributed on a network infrastructure. 

More details coming...

Inference
========================

More details coming...

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
