# Dog Breed Classification with Deep Learning

## About This Project

In this project I built an image classifier that identifies a dog's breed from a photo. The idea came from a simple use case: taking a picture of a dog at a cafe and wanting to know what breed it is.

I used the Kaggle Dog Breed Identification dataset, which has 10,222 labeled training images covering 120 breeds, plus 10,357 unlabeled test images. The work is done in Python in a Google Colab notebook, using TensorFlow 2 and TensorFlow Hub.

## How I Built It

1. **Data preparation:** I matched each image to its breed label, converted the labels to one-hot arrays, and wrote a function that decodes each JPEG, scales pixel values to 0-1, and resizes it to 224 x 224. I fed the data to the model in batches of 32.
2. **Model:** I used transfer learning. The base is MobileNetV2 pretrained on ImageNet (from TensorFlow Hub), with a softmax output layer of 120 units added on top. I trained it with the Adam optimizer and categorical cross-entropy loss.
3. **Experiment on a subset:** To check that everything worked, I first trained on 1,000 images (800 for training, 200 for validation), using TensorBoard to track progress and early stopping to prevent overfitting. This model reached about 67% validation accuracy across the 120 breeds.
4. **Evaluating predictions:** I wrote helper functions that show each validation image with its predicted breed, confidence, and true label, and a chart of the model's top 10 predictions per image.
5. **Full training:** I saved and reloaded models to confirm the workflow, then trained a second model on all 10,222 labeled images. It trained for 22 epochs before early stopping, reaching about 99.5% training accuracy. Because this model had no validation set, that figure is training accuracy only; the 67% above is the held-out result.
6. **Test data:** I prepared the 10,357 test images in the same batch format, ready for prediction.

## Tools and Skills Used

- Python, TensorFlow 2.8, TensorFlow Hub, Keras
- Transfer learning, multi-class image classification
- pandas, NumPy, scikit-learn, Matplotlib
- TensorBoard, early stopping, model saving and loading
- Google Colab with GPU

**File:** [Dog_vision.ipynb](Dog_vision.ipynb)
