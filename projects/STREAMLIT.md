# Streamlit launcher for the three demos

The combined app is `streamlit_app.py`. From this folder, run it with Python 3.14:

```powershell
py -3.14 -m streamlit run streamlit_app.py
```

The app uses Streamlit, PyTorch, NumPy, and Pillow. Install from `requirements-streamlit.txt` if needed. Streamlit prints the local URL (usually http://localhost:8501); the app running in this session is at http://localhost:8502.

The GAN tab trains a conditional convolutional GAN to generate square, circle, and triangle images, with a seed control and downloadable sample grid. The CNN tab explains what the classifier recognizes and includes sample square, circle, and cross images. The RNN tab is a simple positive/negative review classifier using a vanilla recurrent neural network.

For public hosting on Streamlit Community Cloud, push this folder to a GitHub repository, then create a Community Cloud app pointing at `streamlit_app.py` and `requirements-streamlit.txt`.
