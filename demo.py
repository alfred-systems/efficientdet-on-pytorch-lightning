# import gradio as gr

# def greet(name, is_morning, temperature):
#     salutation = "Good morning" if is_morning else "Good evening"
#     greeting = f"{salutation} {name}. It is {temperature} degrees today"
#     celsius = (temperature - 32) * 5 / 9
#     return greeting, round(celsius, 2)

# demo = gr.Interface(
#     fn=greet,
#     inputs=["text", "checkbox", gr.Slider(0, 100)],
#     outputs=["text", "number"],
# )
# demo.launch()
import time
from functools import partial

import cv2
import open_clip
import torch
import numpy as np
import gradio as gr
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms


from src.lightning_model import VisGenome_FuseDet
from src.dataset.train_dataset import VisualGenomeFuseDet
from src.dataset.bbox_augmentor import eval_augmentor


preprocess = transforms.Compose([
    transforms.Resize([512, 512]),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
# preprocess = eval_augmentor(512)
tokenizer = open_clip.get_tokenizer("convnext_large_d")
model = VisGenome_FuseDet.load_from_checkpoint("last.ckpt").cuda().to(torch.float16)
model.eval()


def plot_det(input_image, scores, boxes, label, score_threshold = 0.1):
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(input_image)
    # ax.imshow(input_image, extent=(0, 1, 1, 0))
    ax.set_axis_off()

    for score, box in zip(scores, boxes):
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = box.cpu().detach().numpy()
        h = y2 - y1
        w = x2 - x1
        cx = (x2 + x1) / 2
        cy = (y2 + y1) / 2
        ax.plot([cx - w / 2, cx + w / 2, cx + w / 2, cx - w / 2, cx - w / 2],
                [cy - h / 2, cy - h / 2, cy + h / 2, cy + h / 2, cy - h / 2], 'r')
        ax.text(
            cx - w / 2,
            cy + h / 2 + 0.015,
            f'{label}: {score:1.2f}',
            ha='left',
            va='top',
            color='red',
            bbox={
                'facecolor': 'white',
                'edgecolor': 'red',
                'boxstyle': 'square,pad=.3'
            })
    

    fig.canvas.draw()
    # Now we can save it to a numpy array.
    fig_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    fig_np = fig_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return fig_np


def sepia(query_text, threshold, input_img, inference=True, plot=True):
    global processor, model

    if not inference:
        return input_img

    input_ids = tokenizer(query_text).to('cuda')
    img_tensor = preprocess(Image.fromarray(input_img))
    # img_tensor = preprocess(cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR))['image']
    # img_tensor = preprocess(input_img)['image']
    img_tensor = img_tensor.float().cuda()
    img_tensor = torch.unsqueeze(img_tensor, dim=0)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            t1 = time.time()
            outputs = model.inference_step(img_tensor, input_ids)

    print(f"inference time: {(time.time() - t1) * 1000} ms")
    i = 0  # Retrieve predictions for the first image for the corresponding text queries

    boxes, scores = outputs
    print('max scores: ', scores.max())
    boxes = boxes[0]
    scores = scores[0]

    if plot:        
        return plot_det(
            Image.fromarray(input_img).resize((512, 512)), 
            scores, boxes, query_text, score_threshold=threshold/100)
    else:
        json_obj = []
        for box, score in zip(boxes, scores):
            if score >= threshold/100:
                box = [round(i, 2) for i in box.tolist()]
                json_obj.append({
                    "box": box,
                    "score": round(score.item(), 3),
                })
        print(json_obj)
        return json_obj


with gr.Blocks() as demo:
    gr.Markdown("Flip text or image files using this demo.")
    
    with gr.Tab("Image Files"):
        
        with gr.Row():
            with gr.Column():
                input_ui = [
                    gr.Text(label="query", value="find all objects"), 
                    gr.Slider(10, 100, label='threshold', value=20), 
                    gr.Image()
                ]
                text_button = gr.Button("Submit")
            image_output = gr.Image()
    
    with gr.Tab("Image Files 2 JSON"):
        
        with gr.Row():
            with gr.Column():
                json_input_ui = [
                    gr.Text(label="query", value="find all objects"), 
                    gr.Slider(10, 100, label='threshold', value=20), 
                    gr.Image()
                ]
                json_button = gr.Button("Submit")
            json_output = gr.JSON()
    
    with gr.Tab("WebCam"):
        with gr.Row():
            with gr.Column():
                live_input_ui = [
                    gr.Text(label="query", value="find all objects"), 
                    gr.Slider(10, 100, label='threshold', value=20), 
                    gr.Image(source='webcam', streaming=True),
                    gr.Checkbox(value=False, label='Live Streaming')
                ]
                text_button2 = gr.Button("Submit")
            live_image_output = gr.Image()

    text_button.click(sepia, inputs=input_ui, outputs=image_output)
    text_button.click(sepia, inputs=input_ui, outputs=image_output)
    
    json_button.click(partial(sepia, plot=False), inputs=json_input_ui, outputs=json_output, api_name='detect_json')
    
    live_input_ui[2].stream(sepia, inputs=live_input_ui, outputs=live_image_output)

demo.launch(share=True, server_name="0.0.0.0", server_port=6006)
# sepia('a test', 20, np.zeros([512, 512, 3], dtype=np.uint8))


def sanity_check():

    # A = np.ones([512, 512, 3], dtype=np.uint8)
    A = cv2.imread('/home/ron_zhu/visual_genome/VG_100K/2315427.jpg')
    aug = eval_augmentor(512)
    B1 = aug(A)['image']
    
    A = Image.open('/home/ron_zhu/visual_genome/VG_100K/2315427.jpg')
    B2 = preprocess(A)
    
    diff = torch.abs(B1 - B2)
    input_ids = tokenizer(['find all objects'])
    print((diff < 1e-5).all())
    print(B1.shape)
    print(input_ids.shape)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            embed = torch.frombuffer(VisualGenomeFuseDet.find_all_objects, dtype=torch.float32)
            embed = embed.unsqueeze(0)
            print(embed.shape)

            B1 = B1.to('cuda').unsqueeze(0)
            B2 = B2.to('cuda').unsqueeze(0)
            
            outputs = model.inference_step(
                B1, 
                input_ids.to('cuda'),
                text_embed=embed.to('cuda'),
            )
            
            outputs = model.inference_step(
                B2, 
                input_ids.to('cuda')
            )

    breakpoint()
    print(B1)
    print(B2)

# sanity_check()