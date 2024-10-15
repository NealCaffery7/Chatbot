import os
import gradio as gr
import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv
from PIL import Image

load_dotenv(find_dotenv(), override=True)

api_key = os.environ.get('GOOGLE_API_KEY')
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')


def response_stream(user_input, user_image, history):
    try:
        danger_keywords = [
            "suicide", "self-harm", "kill myself", "harm myself",
            "end my life", "take my life", "cut myself", "overdose",
            "jump off", "hang myself", "hurt myself", "no reason to live",
            "final goodbye", "want to die", "want to disappear",
        ]

        if any(keyword in user_input.lower() for keyword in danger_keywords):
            print("Warning: Dangerous keywords detected in user input. Please intervene immediately!")

        if not user_input and not user_image:
            history.append(("System", "Please provide either text or an image."))
            yield history

        expert_prompt = (
            "You are an experienced mental health counselor with expertise in helping individuals with emotional issues"
            "Your goal is to provide thoughtful, empathetic, and professional advice to users"
            "You are supposed to good at dealing with mental health challenges"
            "You should base your responses on psychological principles and practical solutions. "
            "Read and understand the users' questions, write responses at good levels of empathic understanding. "
            "Limit each response to a minimum of 100 words and a maximum of 200 words. "
        )
        prompt = f"{expert_prompt} User question: {user_input}" if user_input else expert_prompt

        if user_image:
            prompt += " The user also uploaded an image."

        history.append((user_input, ""))
        yield history

        gemini_response = model.generate_content(prompt, stream=True)

        reply = ""
        for chunk in gemini_response:
            reply += chunk.text
            history[-1] = (user_input, reply)
            yield history

    except Exception as e:
        history.append((user_input, f"An error occurred: {str(e)}"))
        yield history


def undo(history):
    if history:
        history.pop()
    return history


def retry(history):
    if history:
        last_input = history[-1][0]
        return response_stream(last_input, None, history[:-1])
    return history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(label="I'm your trustworthy friend, always here and ready to chat whenever you need!")
    with gr.Row():
        text_input = gr.Textbox(label="You are welcome to discuss anything with me :)",
                                placeholder="How is it going today?")
        image_input = gr.Image(label="Upload an image", type="pil")

    with gr.Row():
        submit_button = gr.Button("Send")
        undo_button = gr.Button("Undo")
        retry_button = gr.Button("Retry")

    submit_button.click(fn=response_stream,
                        inputs=[text_input, image_input, chatbot],
                        outputs=chatbot)

    undo_button.click(fn=undo, inputs=chatbot, outputs=chatbot)

    retry_button.click(fn=retry, inputs=chatbot, outputs=chatbot)

if __name__ == "__main__":
    demo.launch()
