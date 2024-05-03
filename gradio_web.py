import gradio as gr
import os
from utils.llm import load_llm
from unet import predict

llm = load_llm()


def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)


def add_text(history, text):  # 录入历史文本
    history = history + [(text, None)]
    return history, gr.Textbox(value="", interactive=False)


def add_file(history, file):  # 录入文件
    history = history + [((file.name,), None)]
    print(history)
    return history


def is_TF(pt):  # 判断是否返回图片、表格
    pt = """如果以下的问题不是要求返回文字回答而是要求返回其他类型，如图片、表格等信息的话，请你回复“True”，否则回复“False”，
    请你一定要准许这个回复规则，我只想在你的回答中看到一次True或者一次False，不可以都同时出现。问题：""" + pt + "答案(True or False):"
    response = llm.invoke(pt) # 是不是图片由llm判断，可能判断错
    # response = "是不是图片"
    return response


# def prompt(file):  # 问题补充
#     contents, merge_img, counts = predict(file)
#     pt = "帮我重新简明的分析一下一张菌落图片的分析数据：" + contents
#     print(pt)
#     return pt


def bot(history):
    question = history[-1][0]
    if isinstance(question, tuple):  # 如果是元组说明有图片吗
        if os.path.exists(question[0]):
            # print(question[0])
            from PIL import Image
            image = Image.open(question[0])
            contents, merge_img = predict(image)  # 预测出信息
            question = "帮我重新简明的分析一下一张菌落图片的分析数据：" + contents  # 信息存放在question
    print(question)
    if "True" in is_TF(question): # 如果问题是要放回预测图片则进入这个循环
        import cv2
        save_img = "predicted_img.jpg"
        cv2.imwrite(save_img, merge_img)
        history[-1][1] = (save_img,)
    else:
        response = llm.invoke(question)
        # response = "123"
        history[-1][1] = response
    return history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        avatar_images=(None, (os.path.join(os.path.dirname(__file__), "R-C.jpg"))),
    )
    with gr.Row():
        txt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Enter text and press enter, or upload an image",
            container=False,
        )
        btn = gr.UploadButton("📁", file_types=["image", "video", "audio"])

    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, chatbot, chatbot, api_name="bot_response"
    )
    txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)
    file_msg = btn.upload(add_file, [chatbot, btn], [chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    chatbot.like(print_like_dislike, None, None)

demo.queue()
demo.launch(share=True)
