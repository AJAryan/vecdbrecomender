import gradio as gr
from transformers import CLIPModel, CLIPProcessor

MODEL_ID = "openai/clip-vit-base-patch32"

device = "cpu"

model = CLIPModel.from_pretrained(MODEL_ID)
processor = CLIPProcessor.from_pretrained(MODEL_ID)

import io

from lancedb.pydantic import LanceModel, vector
import PIL


class Image(LanceModel):
    image: bytes
    label: int
    vector: vector(512)

    def to_pil(self):
        return PIL.Image.open(io.BytesIO(self.image))

    @classmethod
    def pil_to_bytes(cls, img) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


def process_image(batch: dict) -> dict:
    image = processor(text=None, images=batch["image"], return_tensors="pt")[
        "pixel_values"
    ].to(device)
    img_emb = model.get_image_features(image)
    batch["vector"] = img_emb.cpu()
    batch["image_bytes"] = [Image.pil_to_bytes(img) for img in batch["image"]]
    return batch


import lancedb
from datasets import load_dataset


def datagen() -> list[Image]:
    dataset = load_dataset("zh-plus/tiny-imagenet")["valid"]
    return [
        Image(image=b["image_bytes"], label=b["label"], vector=b["vector"])
        for b in dataset.map(process_image, batched=True, batch_size=256)
    ]


db = lancedb.connect("~/.lancedb")
TABLE_NAME = "image_search"
try:
    table = db.open_table(TABLE_NAME)
except:
    # db.drop_table(TABLE_NAME, ignore_missing=True)
    table = db.create_table(TABLE_NAME, schema=Image.to_arrow_schema())
    data = datagen()
    table.add(data)

from transformers import CLIPTokenizerFast

# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPTokenizerFast.from_pretrained(MODEL_ID)


def embed_func(query):
    inputs = tokenizer([query], padding=True, return_tensors="pt")
    text_features = model.get_text_features(**inputs)
    return text_features.detach().numpy()[0]


def find_images(query):
    emb = embed_func(query)
    rs = table.search(emb).limit(9).to_pydantic(Image)
    return [m.to_pil() for m in rs]


import gradio as gr


with gr.Blocks() as demo:
    with gr.Row():
        vector_query = gr.Textbox(value="fish", show_label=False)
        b1 = gr.Button("Submit")
    with gr.Row():
        gallery = gr.Gallery(
            label="Found images",
            show_label=False,
            elem_id="gallery",
            columns=[3],
            rows=[3],
            object_fit="contain",
            height="auto",
        )

    b1.click(find_images, inputs=vector_query, outputs=gallery)

demo.launch()
