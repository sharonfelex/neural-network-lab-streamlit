import re
import io
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from torch import nn
from PIL import Image, ImageOps

st.set_page_config(page_title="Neural Network Lab", page_icon="🧠", layout="centered")
st.title("🧠 Neural Network Lab")
st.caption("Three small, interactive PyTorch projects: GAN · CNN · RNN")
torch.set_num_threads(1)


@st.cache_resource
def train_gan():
    """Train a small conditional GAN on varied synthetic 24x24 shape images."""
    torch.manual_seed(7)
    side, classes, per_class = 24, 3, 240
    yy, xx = torch.meshgrid(torch.arange(side), torch.arange(side), indexing="ij")
    images = torch.zeros(classes * per_class, 1, side, side)
    labels = torch.arange(classes).repeat_interleave(per_class)
    for i, label in enumerate(labels):
        radius = int(torch.randint(5, 9, ()))
        cx, cy = int(torch.randint(9, 15, ())), int(torch.randint(9, 15, ()))
        if label == 0:
            shape = ((xx-cx).abs() <= radius) & ((yy-cy).abs() <= radius)
        elif label == 1:
            shape = (xx-cx)**2 + (yy-cy)**2 <= radius**2
        else:
            shape = (yy >= cy-radius) & (yy <= cy+radius) & ((xx-cx).abs() <= (yy-(cy-radius))//2)
        images[i, 0] = shape.float() * 2 - 1

    def one_hot(y):
        return torch.nn.functional.one_hot(y, classes).float()

    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(35, 128*6*6), nn.BatchNorm1d(128*6*6), nn.ReLU(),
                nn.Unflatten(1, (128, 6, 6)),
                nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Tanh(),
            )

        def forward(self, noise, y):
            return self.layers(torch.cat([noise, one_hot(y)], dim=1))

    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(1+classes, 32, 4, 2, 1), nn.LeakyReLU(.2),
                nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(.2),
                nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(.2),
                nn.Flatten(), nn.Linear(128*3*3, 1),
            )

        def forward(self, image, y):
            condition = one_hot(y)[:, :, None, None].expand(-1, -1, side, side)
            return self.layers(torch.cat([image, condition], dim=1)).squeeze(1)

    generator, discriminator = Generator(), Discriminator()
    g_opt = torch.optim.Adam(generator.parameters(), lr=.0002, betas=(.5, .999))
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=.0002, betas=(.5, .999))
    loss = nn.MSELoss()
    batch_size = 64
    for _ in range(120):
        order = torch.randperm(len(images))
        for indices in order.split(batch_size):
            real, y = images[indices], labels[indices]
            noise = torch.randn(len(indices), 32)
            fake = generator(noise, y).detach()
            d_opt.zero_grad()
            real_loss = loss(discriminator(real, y), torch.ones(len(indices)))
            fake_loss = loss(discriminator(fake, y), torch.zeros(len(indices)))
            (real_loss + fake_loss).backward()
            d_opt.step()
            g_opt.zero_grad()
            generated = generator(torch.randn(len(indices), 32), y)
            g_loss = loss(discriminator(generated, y), torch.ones(len(indices)))
            g_loss.backward()
            g_opt.step()
    generator.eval()
    return generator


@st.cache_resource
def train_cnn():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten(), nn.Linear(1024, 3),
            )

        def forward(self, x):
            return self.layers(x)

    torch.manual_seed(12)
    model = Net()
    images = torch.zeros(900, 1, 32, 32)
    yy, xx = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    labels = torch.arange(900) % 3
    for i, label in enumerate(labels):
        radius = int(torch.randint(5, 11, ()))
        cx, cy = int(torch.randint(10, 23, ())), int(torch.randint(10, 23, ()))
        outline = bool(torch.randint(0, 2, ()))
        if label == 0:
            outer = ((xx-cx).abs() <= radius) & ((yy-cy).abs() <= radius)
            inner = ((xx-cx).abs() < radius-2) & ((yy-cy).abs() < radius-2)
            shape = outer & ~inner if outline else outer
        elif label == 1:
            distance = torch.sqrt((xx-cx)**2 + (yy-cy)**2)
            shape = (distance <= radius) & (distance >= radius-2) if outline else distance <= radius
        else:
            shape = (((yy-cy).abs() <= radius) & ((xx-cx).abs() <= 1)) | (((xx-cx).abs() <= radius) & ((yy-cy).abs() <= 1))
        images[i, 0] = shape.float() + torch.rand(32, 32) * .015
    optimizer = torch.optim.Adam(model.parameters(), lr=.004)
    loss = nn.CrossEntropyLoss()
    for _ in range(100):
        optimizer.zero_grad()
        value = loss(model(images), labels)
        value.backward()
        optimizer.step()
    model.eval()
    return model


@st.cache_resource
def train_rnn():
    positive_words = ["love", "like", "enjoy", "recommend", "adore"]
    positive_adjectives = ["wonderful", "great", "good", "excellent", "amazing", "delightful", "pleasant", "perfect", "fun", "impressive", "beautiful", "happy"]
    negative_words = ["hate", "dislike", "regret", "avoid", "detest"]
    negative_adjectives = ["awful", "bad", "terrible", "poor", "boring", "horrible", "disappointing", "unpleasant", "broken", "dull", "sad", "frustrating"]
    templates = [
        "i {verb} this movie", "i {verb} this product", "i {verb} this experience",
        "this movie is {adj}", "this product is {adj}", "this experience is {adj}",
        "what a {adj} day", "the service was {adj}", "the food tastes {adj}",
        "i am {adj} with this result", "this place feels {adj}", "the story was {adj}",
        "it was a {adj} experience", "the quality is {adj}", "i feel {adj} about it",
    ]
    examples = []
    for label, verbs, adjectives in [(1, positive_words, positive_adjectives), (0, negative_words, negative_adjectives)]:
        for template in templates:
            terms = verbs if "{verb}" in template else adjectives
            for term in terms:
                examples.append((template.format(verb=term, adj=term), label))
    tokenized = [re.findall(r"[a-z']+", text.lower()) for text, _ in examples]
    vocabulary = ["<pad>", "<unk>"] + sorted({word for sentence in tokenized for word in sentence})
    word_ids = {word: i for i, word in enumerate(vocabulary)}
    lengths = torch.tensor([len(sentence) for sentence in tokenized])
    inputs = torch.zeros(len(tokenized), int(lengths.max()), dtype=torch.long)
    for i, sentence in enumerate(tokenized):
        inputs[i, :len(sentence)] = torch.tensor([word_ids[word] for word in sentence])
    labels = torch.tensor([label for _, label in examples])

    class SentimentRNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(len(vocabulary), 20, padding_idx=0)
            self.rnn = nn.RNN(20, 32, batch_first=True)
            self.output = nn.Linear(32, 2)

        def forward(self, x, seq_lengths):
            embedded = self.embedding(x)
            packed = nn.utils.rnn.pack_padded_sequence(embedded, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, hidden = self.rnn(packed)
            return self.output(hidden[-1])

    torch.manual_seed(24)
    model = SentimentRNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    loss = nn.CrossEntropyLoss()
    for _ in range(80):
        optimizer.zero_grad()
        value = loss(model(inputs, lengths), labels)
        value.backward()
        optimizer.step()
    model.eval()
    return model, word_ids


gan, cnn, rnn = st.tabs(["🎨 GAN · Generate", "🔍 CNN · Classify", "📝 RNN · Sentiment"])
with gan:
    st.subheader("Generate geometric images")
    st.write("This conditional GAN learns to create tiny shape images. Choose a class: random noise is combined with your choice, and the generator learns to fool a discriminator trained on example images.")
    st.info("The training set is created locally from varied synthetic examples. The first run trains the GAN; later generations reuse the trained model.")
    styles = {"Circle": 1, "Square": 0, "Triangle": 2}
    left, right = st.columns(2)
    style = left.selectbox("Shape", list(styles))
    count = right.slider("Images", 4, 12, 8)
    seed = st.number_input("Random seed (change it for new variations)", min_value=0, max_value=99999, value=7, step=1)
    if st.button("Train GAN & generate", key="gan"):
        with st.spinner("Training the image GAN on synthetic shape examples…"):
            generator = train_gan()
            torch.manual_seed(int(seed))
            shape_ids = torch.full((count,), styles[style], dtype=torch.long)
            with torch.no_grad():
                raw = generator(torch.randn(count, 32), shape_ids).squeeze(1).numpy()
                generated = (raw > .25).astype(np.uint8) * 255
        cols, padding, scale = 4, 8, 12
        cell = 24 * scale
        rows = (count + cols - 1) // cols
        sheet = Image.new("L", (cols*cell+(cols+1)*padding, rows*cell+(rows+1)*padding), 36)
        for i, pixels in enumerate(generated):
            tile = Image.fromarray(pixels).resize((cell, cell), Image.Resampling.NEAREST)
            x = padding + (i % cols) * (cell + padding)
            y = padding + (i // cols) * (cell + padding)
            sheet.paste(tile, (x, y))
        st.image(sheet, caption=f"Generated {count} {style.lower()} variations", width=600)
        download = io.BytesIO()
        sheet.save(download, format="PNG")
        st.download_button("Download generated image grid", download.getvalue(), file_name=f"gan-{style.lower()}-samples.png", mime="image/png")
        st.caption("Each image comes from random latent noise plus the selected shape label. Change the seed to sample different outputs.")

with cnn:
    st.subheader("What does the CNN do?")
    st.write("A convolutional neural network (CNN) looks at an image in small patches. Its layers learn visual patterns—from edges and curves to whole shapes—and use them to choose a label.")
    st.info("This demo recognizes three simple classes: square, circle, and cross. It handles transparent backgrounds and outlined shapes; clear images with one main shape work best.")
    sample_dir = Path(__file__).parent / "cnn-samples"
    columns = st.columns(3)
    for column, name in zip(columns, ["square", "circle", "cross"]):
        sample = (sample_dir / f"{name}.png").read_bytes()
        column.image(sample, caption=name.title(), width=110)
        column.download_button(f"Download {name}", sample, file_name=f"cnn-{name}.png", mime="image/png", key=f"sample-{name}")
    uploaded = st.file_uploader("Or upload your own image", type=["png", "jpg", "jpeg"], key="image")
    if uploaded:
        image = ImageOps.exif_transpose(Image.open(uploaded))
        preview = Image.new("RGBA", image.size, "white")
        preview.alpha_composite(image.convert("RGBA"))
        preview = preview.convert("RGB")
        st.image(preview, caption="Uploaded image", width=240)
        if st.button("Classify image", key="cnn"):
            with st.spinner("Training CNN and classifying…"):
                model = train_cnn()
                gray = np.asarray(preview.convert("L"), dtype=np.float32) / 255
                border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
                background = float(np.median(border))
                foreground = gray < background - .16 if background >= .5 else gray > background + .16
                if foreground.sum() < 12:
                    st.warning("I could not find a clear shape. Try a high-contrast image with one centered shape.")
                else:
                    ys, xs = np.where(foreground)
                    mask = Image.fromarray((foreground * 255).astype(np.uint8))
                    crop = mask.crop((xs.min(), ys.min(), xs.max()+1, ys.max()+1))
                    crop.thumbnail((18, 18), Image.Resampling.NEAREST)
                    normalized = Image.new("L", (32, 32), 0)
                    normalized.paste(crop, ((32-crop.width)//2, (32-crop.height)//2))
                    tensor = torch.from_numpy(np.asarray(normalized, dtype=np.float32).copy() / 255).reshape(1, 1, 32, 32)
                    with torch.no_grad():
                        probabilities = model(tensor).softmax(1)[0].numpy()
                    labels = ["Square", "Circle", "Cross"]
                    index = int(np.argmax(probabilities))
                    st.metric("Prediction", labels[index], f"{probabilities[index]:.0%} model score")
                    st.caption("The score reflects this small demo model and is not a calibrated probability.")
                    st.bar_chart({"Model score": dict(zip(labels, probabilities))})

with rnn:
    st.subheader("Simple review sentiment")
    st.write("Type a short review. A basic recurrent neural network (RNN) reads the words in order and classifies the overall sentiment as positive or negative.")
    st.info("This is a teaching demo trained on short example phrases. It recognizes simple wording and is not a general-purpose sentiment model.")
    if "review_text" not in st.session_state:
        st.session_state.review_text = "I love this movie, it was wonderful and fun"
    if st.button("Use positive example", key="positive-example"):
        st.session_state.review_text = "I love this movie, it was wonderful and fun"
    if st.button("Use negative example", key="negative-example"):
        st.session_state.review_text = "I hate this movie, it was boring and awful"
    review = st.text_input("Review", key="review_text")
    if st.button("Analyze sentiment", key="rnn"):
        with st.spinner("Training the simple RNN…"):
            model, word_ids = train_rnn()
            words = re.findall(r"[a-z']+", review.lower())
            known = [word_ids.get(word, word_ids["<unk>"]) for word in words]
            if not known:
                st.warning("Please enter a few words to analyze.")
            else:
                sequence = torch.tensor([known], dtype=torch.long)
                with torch.no_grad():
                    probabilities = torch.softmax(model(sequence, torch.tensor([len(known)])), dim=1)[0]
                result = int(probabilities.argmax())
                if result == 1:
                    st.success(f"Positive sentiment · {probabilities[result]:.0%} model confidence")
                else:
                    st.error(f"Negative sentiment · {probabilities[result]:.0%} model confidence")
                st.bar_chart({"Model score": {"Negative": float(probabilities[0]), "Positive": float(probabilities[1])}})

st.divider()
st.caption("Streamlit · Fast local demos · Models train on first use")
