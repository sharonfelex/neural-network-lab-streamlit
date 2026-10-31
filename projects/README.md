# Three tiny neural network projects

Each folder is a separate full-stack demo with a FastAPI backend, a browser frontend, and a PyTorch model.

The combined Streamlit frontend is in [STREAMLIT.md](STREAMLIT.md) and `streamlit_app.py`; its RNN tab classifies short reviews as positive or negative.

- [GAN Studio](gan-studio/) is a compact FastAPI sample; the improved image GAN is in the Streamlit app's GAN tab.
- [CNN Image Classifier](cnn-image-classifier/README.md) trains a small convolutional network on synthetic shapes and predicts square, circle, or cross from an uploaded image.
- [RNN Text Generator](rnn-text-generator/README.md) trains a small character-level GRU on an example corpus and continues a text seed.

The CNN sample images are in `cnn-samples/` and can also be downloaded from the CNN tab in the Streamlit app.

Install and run each project separately; each README has the commands. All use http://127.0.0.1:8000, so stop one server before starting another. These are compact learning demos, not production models.
