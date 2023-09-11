# Dehazing using GAN

## Overview

This project implements a Generative Adversarial Network (GAN) for dehazing images. It includes several components for training and testing the model, as well as a pre-processing step. The goal is to take hazy images as input and produce dehazed images as output.

## Dataset

The project uses the [O-Haze Dataset](https://data.vision.ee.ethz.ch/cvl/ntire18//o-haze/) for training and testing the dehazing model. You can download the dataset from the provided link and place it in the appropriate directory as described in the instructions below. Note that the dataset will consist of 2 set of folders: Hazy Images and Ground Truth (Dehazed Images). Keep them seprate for training purposes.

## Files

1. **pix2pix_model.py**:
   - This file contains the implementation of the GAN model architecture used for dehazing.
   - It defines the generator and discriminator networks, loss functions, and training procedures.

2. **preprocessing.py**:
   - This script is responsible for pre-processing the dataset.
   - It applies any necessary data transformations or enhancements before training the model.

3. **train_model.py**:
   - This script uses the pix2pix model and preprocessed images to train the dehazing model.
   - The trained model is saved for later use during testing.

4. **test_model.py**:
   - This script loads the previously saved dehazing model.
   - It takes hazy images as input and returns dehazed images using the trained model.

## Getting Started

Follow these steps to get started with the project:

1. **Dataset**: Download the [O-Haze Dataset](https://data.vision.ee.ethz.ch/cvl/ntire18//o-haze/).

2. **Environment**: Create a virtual environment and install the required dependencies. You can do this using `requirements.txt`.

   ```bash
   pip install -r requirements.txt
   ```

3. **Training**: Train the dehazing model by running the `train_model.py` script.

   ```bash
   python train_model.py
   ```

4. **Testing**: Test the trained model using the `test_model.py` script. You can use your own images here, but for testing purpose some images are given in the `Testing Images` folder.

   ```bash
   python test_model.py
   ```

## Results

We are still in the initial phase of testing our model. The results with the present architecture are as shown below:
![image](https://github.com/devika-pedada/Dehazing_Using_GAN/assets/132656055/ac62e807-1d0f-489b-920c-93c68150cb11)
![image](https://github.com/devika-pedada/Dehazing_Using_GAN/assets/132656055/68835e91-cc8a-4eb0-9ff2-3146d5b16b93)

We have noted few problems with our present model:
<ol>
  <li>The images are decolorised</li>
  <li>Some of the image charcteristics are not as intended (such as contrast, hue, saturation, etc)</li>
</ol>
We are aiming to tackle these problems in the near future.

## Contributing

If you would like to contribute to this project, please open an issue or create a pull request with your changes.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- The GAN model which we have used for this project is Pix2Pix GAN. The Pix2Pix model architecture is based on the implementation by Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros [(Paper)](https://doi.org/10.48550/arXiv.1611.07004). The preprocessing code is adapted from [bnsreenu](https://github.com/bnsreenu/python_for_microscopists/tree/master/251_satellite_image_to_maps_translation). We would like to express our gratitude to these authors for their contributions.
- The [O-Haze Dataset](https://data.vision.ee.ethz.ch/cvl/ntire18//o-haze/) for providing the dataset for this project.
