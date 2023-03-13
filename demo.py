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

import open_clip
import torch
import numpy as np
import gradio as gr
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms


from src.lightning_model import COCO_EfficientDet, Laion400m_EfficientDet, VisGenome_EfficientDet, VisGenome_FuseDet


preprocess = transforms.Compose([
    transforms.Resize([512, 512]),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
tokenizer = open_clip.get_tokenizer("convnext_base_w")
model = VisGenome_FuseDet.load_from_checkpoint("last.ckpt").cuda().to(torch.float16)


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
        score_threshold = 0.1
        
        return plot_det(
            np.asarray(input_img) / 255, 
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
                    gr.Text(label="query", value="person, cup, phone, video game  controller, headphone, box"), 
                    gr.Slider(10, 100, label='threshold', value=20), 
                    gr.Image()
                ]
                text_button = gr.Button("Submit")
            image_output = gr.Image()
    
    with gr.Tab("Image Files 2 JSON"):
        
        with gr.Row():
            with gr.Column():
                json_input_ui = [
                    gr.Text(label="query", value="person, cup, phone, video game  controller, headphone, box"), 
                    gr.Slider(10, 100, label='threshold', value=20), 
                    gr.Image()
                ]
                json_button = gr.Button("Submit")
            json_output = gr.JSON()
    
    with gr.Tab("WebCam"):
        with gr.Row():
            with gr.Column():
                live_input_ui = [
                    gr.Text(label="query", value="person, cup, phone, video game  controller, headphone, box"), 
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
    
# examples = [
#     ['bottle and brush, bottle', 10, 'img/gettyimages-510693044-1550590816.jpg'],
#     ['construction worker, truck, building, window', 20, 'img/272648782_238578478447028_8952244765357561494_n.jpg'],
#     ['fedex, men, woman', 20, 'img/POD-fy22_sustainability_-37.jpg'],
#     ['floor, wall, kid, toy on the floor', 20, 'img/istockphoto-1185942848-612x612.jpg'],
#     ['traffic cone, constrution worker, truck, building, window', 20, 'img/web1_vka-viewstreet-13264.jpg'],
#     ['plastic bag, trash can, mug, TV, book, scissors, disk, box', 20, 'img/000000396765.jpg'],
#     ['chocolate donut, glasses, cup, hand', 20, 'img/000000014791.jpg'],
# ]
# input_ui = [
#     gr.Text(label="query", value="person, cup, phone, video game  controller, headphone, box"), 
#     gr.Slider(10, 100, label='threshold', value=20), 
#     gr.Image()
# ]
# demo = gr.Interface(sepia, input_ui, "image", examples=examples, live=True)
# live_demo = gr.Interface(
#     partial(sepia, "person, cup, phone, video game  controller, headphone, box", 20), 
#     gr.Image(source='webcam', streaming=True), 
#     "image")

demo.launch(share=True, server_name="0.0.0.0")
# sepia('a test', 20, np.zeros([512, 512, 3], dtype=np.uint8))

