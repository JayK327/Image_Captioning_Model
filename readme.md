# Image Captioning Project

This project uses the **Flickr8k dataset** for image captioning experiments. The dataset contains images and multiple captions per image, along with expert and CrowdFlower human annotations.

## Dataset

The project expects the dataset to be organized in the **parent directory** like this:

```md
parent_directory/
├── Flickr8k_text/
└── Flickr8k_Dataset/

You can download the dataset from Kaggle:  
[Flickr8k Dataset on Kaggle](https://www.kaggle.com/datasets/dibyansudiptiman/flickr-8k)

### After downloading:

- Place all text files (captions, splits, annotations) in `Flickr8k_text/`  
- Place all images in `Flickr8k_Dataset/`

# Image Captioning using CNN + LSTM

This project implements an end-to-end Image Captioning system that generates natural language descriptions for images using Deep Learning. The model combines Computer Vision and Natural Language Processing by using a Convolutional Neural Network (CNN) to extract image features and a Long Short-Term Memory (LSTM) network to generate captions word by word. The Flickr8k dataset is used, which contains 8,000 images and five human-written captions per image.

Images are first processed using a pre-trained Xception model (transfer learning) to extract a 2048-dimensional feature vector. These features represent high-level visual information and remove the need to train a CNN from scratch. Text captions are cleaned by converting to lowercase, removing punctuation, numbers, and short words, and then adding special tokens `<start>` and `<end>` to mark sentence boundaries. A tokenizer converts words into integer sequences, and captions are padded to the maximum caption length.

Training data is generated dynamically using a generator. For each image and caption, multiple input-output pairs are created using teacher forcing. For example, given `<start> a dog running <end>`, the model learns to predict `a`, then `dog`, then `running`, and finally `<end>`, always conditioned on the image features and previous words. The output word is one-hot encoded across the vocabulary.

The model has two inputs: image features of shape `(2048,)` and text sequences of shape `(max_length,)`. The image branch passes through a Dense layer, while the text branch uses an Embedding layer followed by an LSTM. Both branches are merged and passed through Dense layers with a Softmax output to predict the next word. The model is trained using categorical cross-entropy loss and the Adam optimizer. Training is done in batches using a TensorFlow data generator to avoid memory issues.

To run the project, create a virtual environment, install dependencies from `requirements.txt` (use `Pillow` instead of `PIL`), and execute `python3 main.py`. The model is trained for multiple epochs and saved after each epoch. Common errors include a typo in `to_categorical` (`num_classes` must be used) and tqdm notebook issues, which can be fixed by using standard tqdm in scripts.

This project demonstrates key concepts such as transfer learning, CNN feature extraction, sequence modeling, teacher forcing, and multimodal learning. It serves as a strong foundation for more advanced image captioning systems using attention mechanisms or transformers.
