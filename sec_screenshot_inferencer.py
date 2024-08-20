import pandas as pd
import os
import torch
import openai
import regex as re
import json
import requests
from tqdm import tqdm
from transformers import pipeline
import tiktoken
import pandas as pd
import re
from urllib import parse
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from itertools import islice
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from itertools import islice
import base64
import requests
from pathlib import Path
from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field, validator
from langchain.llms import OpenAI
import requests
import requests
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
import nltk
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
nltk.download('punkt')
import base64
import requests

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def encode_image(image_path):
   with open(image_path, "rb") as image_file:
      return base64.b64encode(image_file.read()).decode('utf-8')

def summarize_images_in_folder(folder_path, api_key):
  summaries = []
  for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
      image_path = os.path.join(folder_path, filename)
      base64_image = encode_image(image_path)

      headers = {
          "Content-Type": "application/json",
          "Authorization": f"Bearer {api_key}"
      }

      payload = {
          "model": "gpt-4-vision-preview",
          "messages": [
              {
                  "role": "user",
                  "content": [
                      {"type": "text", "text": "Summarize in bullet points"},
                      {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                  ]
              }
          ],
          "max_tokens": 300
      }

      response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
      if response.status_code == 200:
          summaries.append(response.json()['choices'][0]['message']['content'])
      else:
          summaries.append(f"Error processing {filename}: {response.text}")

  return summaries

def sec_filing_inferencer(input_to_llm):
  response_schemas = [
      ResponseSchema(name="SEC Filing Inference", description="List of key bullet points about the SEC Filing")
  ]
  output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
  format_instructions = output_parser.get_format_instructions()

  chat_model = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.5, openai_api_key = "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")

  prompt = ChatPromptTemplate(
      messages=[
          HumanMessagePromptTemplate.from_template("""
        Assume the role as a leading Technical Analysis (TA) expert in the stock market, a modern counterpart to Charles Dow, John Bollinger, and Alan Andrews. Analyze the SEC Filing report and extract key elements:

        SEC Filing: {hist}

        Instructions for Analysis:
        - Begin with a summary of the SEC Filing's primary focus.
        - Detail the examination of its components: financial statements, management discussion, market risk factors, etc.
        - Conclude with your professional insights into the probable market outcomes based on this filing.
        - Remember, your analysis should help in understanding the broader market implications of these filings and aid in decision-making processes for various stakeholders.

        Apply these format instructions to analyze and report on the SEC Filing:
        {format_instructions}""")
      ],
      input_variables=["hist"],
      partial_variables={"format_instructions": format_instructions}
  )

  if num_tokens_from_string(input_to_llm, "gpt-3.5-turbo-16k") < 16000:
    _input = prompt.format_prompt(hist=input_to_llm)
    output = chat_model(_input.to_messages())
  else:
    chunks = split_into_chunks(input_to_llm)
    outputs = []
    for chunk in chunks:
      _input = prompt.format_prompt(hist=chunk)
      output = chat_model(_input.to_messages())
      outputs.append(output.content)
    final_output = ', '.join(outputs)
    _input = prompt.format_prompt(hist=final_output)
    output = chat_model(_input.to_messages())

  return(output)


# Example usage
folder_path = "screenshots033-4248424724650"
api_key = "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"
summaries = summarize_images_in_folder(folder_path, api_key)
screenshots_summary = "; ".join(summaries)

print(sec_filing_inferencer(screenshots_summary))