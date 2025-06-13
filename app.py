from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai
from google.api_core import retry
from io import BytesIO
import gradio as gr
import base64
import time
import uuid
import os

# Define a retry policy
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

if not hasattr(genai.models.Models.generate_content, '__wrapped__'):
  genai.models.Models.generate_content = retry.Retry(
      predicate=is_retriable)(genai.models.Models.generate_content)

# Load Google API Key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Load LLM
vis_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-preview-image-generation")

# Image to Base64 Converter
def image_to_base64(image_path):
  with open(image_path, 'rb') as img:
    encoded_string = base64.b64encode(img.read())
  return encoded_string.decode('utf-8')

# Base64 to Image Converter
def base64_to_image(base64_img, filename):
  with open(filename, "wb") as f:
    f.write(base64.b64decode(base64_img))
  return filename

# Function that takes user inputs and displays it on chatUI
def query_message(message, history):
  if message["files"] is not None:
    for img in message["files"]:
      history.append({"role": "user", "content": {"path": img}})
  if message["text"] is not None:
    history.append({"role": "user", "content": message["text"]})
  return gr.MultimodalTextbox(value=None, interactive=False), history

# Function that takes user inputs, generate response and displays on chatUI
def chat_bot(history):
  messages = history.copy()

  # Format with system prompt
  system_prompt = """
                   system: You are CreativeMuse, a warm, imaginative, and encouraging creative assistant.
                   Your purpose is to help users overcome creative blocks and spark new ideas across various fields
                   — writing, design, music, visual arts, filmmaking, and more. When a user feels stuck, you offer unconventional prompts,
                   inspiring questions, or curated brainstorming techniques to get their creativity flowing again.
                   - Tailor your suggestions to their medium and mood, mixing practical strategies with whimsical, unexpected twists to encourage exploration.
                   Always be supportive, non-judgmental, and affirming—help them feel safe to experiment.
                   You can also suggest small creative exercises, constraints, "what if" scenarios, or draw connections between unrelated concepts to stimulate new thinking.
                   If a user shares part of a project, offer suggestions that expand on their work without taking it over.
                   - Keep your response short and straight to the point, and only generate an image on the user's request. 
                   """

  formatted_prompt = system_prompt + "\n\n"

  # Add conversation history
  for msg in messages:
    if "path" in msg["content"] and isinstance(msg["content"], dict):
      encoded_img = image_to_base64(msg["content"]["path"])
      formatted_prompt += f"{msg['role']}: data:image/jpeg.base64,{encoded_img}"
    else:
      formatted_prompt += f"{msg['role']}: {msg['content']}\n\n"

  # Get response from model
  response = vis_llm.invoke(formatted_prompt,
                            generation_config=dict(response_modalities=["TEXT", "IMAGE"]))

  # Add response to history
  history.append({"role": "assistant", "content": ""})

  for block in response.content:
    if isinstance(block, str):
      history[-1]["content"] += block
      time.sleep(0.02)
      yield history
    elif isinstance(block, dict) and block.get("image_url"):
      url = block["image_url"].get("url").split(",")[-1]
      image = base64_to_image(url, f"{uuid.uuid4()}.png")
      history.append({"role": "assistant", "content": {"path": image}})
      yield history

welcome_message = "Hey, I’m CreativeMuse—your creative companion when inspiration runs dry. Stuck on an idea? Let’s shake things up and spark something new together."

# Interface Code
with gr.Blocks() as app:
  chatbot = gr.Chatbot(
      value=[{"role": "assistant", "content": welcome_message}],
      label="CreativeMuse",
      show_label=True,
      type='messages',
      bubble_full_width=False,
      avatar_images=(None, "1eb05f325ec50a15c8b045f3428d6d5e.jpg")
    )
  text_box = gr.MultimodalTextbox(
          placeholder="Enter text or upload an image...",
          container=False,
          file_types=["image"],
          file_count="multiple",
          interactive=True,
      )

  chat_msg = text_box.submit(query_message, [text_box,chatbot], [text_box, chatbot])
  bot_msg = chat_msg.then(chat_bot, chatbot, chatbot)
  bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, text_box)

if __name__ == "__main__":
    app.queue() 
    app.launch(share=True)