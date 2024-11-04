from vertexai.preview.vision_models import ImageTextModel
from vertexai.preview.vision_models import Image
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
import gradio as gr
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

# Go to https://github.com/jggomez/generativeimagevertexai
# vertexai.init(project="project_id", location="us-central1")


MODEL_IMAGE = "imagen-3.0-generate-001"
MODEL_IMAGE_TEXT = "imagetext@001"
MODEL_IMAGE_GENERATION_2 = "imagegeneration@002"
OUTPUT_IMAGE = "images/image.png"
OUTPUT_EDIT_IMAGE = "images/edit_image.png"
PROMPT_QA = "Can you specify details about the background of the image?"
PROMPT_QA_DEFAULT = "Can you specify details about the image?"
PROMPT_CHAT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a helpful assistant and you can chat with this image that contains {visual_caption}"),
        ("user", "{message}")
    ]
)

visual_captioning_chat = ""

instance_model_image_generation = ImageGenerationModel.from_pretrained(
    MODEL_IMAGE)
instance_model_image_text = ImageTextModel.from_pretrained(
    MODEL_IMAGE_TEXT)
instance_image_generation_2 = ImageGenerationModel.from_pretrained(
    MODEL_IMAGE_GENERATION_2)


llm = ChatOllama(
    model="gemma2:2b",
    base_url="URL",
)


def get_question_answer(input_file, prompt):
    source_img = Image.load_from_file(location=input_file)
    answers = instance_model_image_text.ask_question(
        image=source_img,
        question=prompt,
        number_of_results=1,
    )
    if (len(answers) > 0):
        return answers[0]
    return "NOT ANSWER"


def get_captions(input_file):
    source_img = Image.load_from_file(location=input_file)
    captions = instance_model_image_text.get_captions(
        image=source_img,
        language="en",
        number_of_results=1,
    )
    if (len(captions) > 0):
        return captions[0]
    return "NOT FOUND CAPTION"


def get_image(prompt, aspect_ratio="1:1", prompt_negative=""):
    images = instance_model_image_generation.generate_images(
        prompt=prompt,
        number_of_images=1,
        language="en",
        aspect_ratio=aspect_ratio,
        safety_filter_level="block_few",
        person_generation="allow_adult",
        negative_prompt=prompt_negative,
    )

    images[0].save(location=OUTPUT_IMAGE, include_generation_parameters=False)
    caption = get_captions(OUTPUT_IMAGE)

    return caption, gr.Image(OUTPUT_IMAGE),


def get_image_information(path_file, question):
    caption = get_captions(path_file)
    if question:
        answer = get_question_answer(path_file, question)
    else:
        answer = get_question_answer(path_file, PROMPT_QA_DEFAULT)
    return caption, answer


def edit_image(path_file, prompt):
    base_img = Image.load_from_file(location=path_file)
    images = instance_image_generation_2.edit_image(
        base_image=base_img,
        prompt=prompt,
        guidance_scale=21,
        number_of_images=1,
    )

    images[0].save(location=OUTPUT_EDIT_IMAGE,
                   include_generation_parameters=False)

    return gr.Image(OUTPUT_EDIT_IMAGE)


def get_chat_answer(message, chat_history, state_ui):
    print(message)
    print(state_ui)
    chain = PROMPT_CHAT | llm
    response = chain.invoke(
        {"message": message, "visual_caption": state_ui})
    chat_history.append((message, response.content))
    return "", chat_history


def get_captions_chat(input_file, state_ui):
    captions = get_captions(input_file)
    return captions, captions


ui_blocks = gr.Blocks()
chat_ui_block = gr.Blocks()

create_images_ui = gr.Interface(
    fn=get_image,
    inputs=[gr.Textbox(label="Write your prompt... (Generate Images)", lines=3),
            gr.Radio(["1:1", "9:16", "16:9", "3:4", "4:3"],
                     label="Aspect Ratio"),
            gr.Textbox(label="Negative prompt... (Define what you don't want to see)", lines=3),],
    outputs=[gr.Textbox(label="Visual captioning", lines=2),
             gr.Image(),],
    allow_flagging="never")

get_image_information_ui = gr.Interface(
    fn=get_image_information,
    inputs=[gr.Image(type="filepath", sources="upload"),
            gr.Textbox(label="Visual Question...", lines=3),],
    outputs=[gr.Textbox(label="Visual captioning", lines=2),
             gr.Textbox(label="Visual Question Answering (VQA)", lines=3),],
    allow_flagging="never")

get_edit_image = gr.Interface(
    fn=edit_image,
    inputs=[gr.Image(type="filepath", sources="upload"),
            gr.Textbox(label="Write your prompt... (What do you want to edit)", lines=3),],
    outputs=[gr.Image()],
    allow_flagging="never")

with chat_ui_block:
    state_ui = gr.State([])

    gr.Interface(
        fn=get_captions_chat,
        inputs=[gr.Image(type="filepath", sources="upload"),
                state_ui],
        outputs=[gr.Textbox(label="Visual captioning",
                            lines=1), state_ui],
        allow_flagging="never")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Upload an image first and then enter the instructions here",
                     value="Create a poem with this image",
                     lines=1,)
    msg.submit(get_chat_answer, [msg, chatbot, state_ui], [msg, chatbot])


with ui_blocks:
    gr.TabbedInterface(
        [create_images_ui,
         get_image_information_ui,
         get_edit_image,
         chat_ui_block],
        ["Create Images",
         "Get Image Information",
         "Edit Image",
         "Chatting with your image"],
    )

if __name__ == "__main__":
    ui_blocks.launch()
