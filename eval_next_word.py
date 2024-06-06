import os
from openai import AzureOpenAI

# Azure OpenAI Variables
client = AzureOpenAI(
  azure_endpoint = "https://llmtaskforce-collabathon-openai-rg.openai.azure.com/", 
  api_key="97c8d59b86eb4d6f8eb056e4a356668b",
  api_version="2024-02-01"
)

prefix = "Based on the risk assessment results, EFSA concluded that the proposed temporary MRL is"
prefix = "Swissmedic is the"

def evaluate(prefix,next_term):
    response = client.chat.completions.create(
        model="gpt-4", # model = "gpt-35-turbo" or "gpt-4".
        max_tokens=5,
        temperature=1,   
        n=10,
        messages=[
         {"role": "user", "content": prefix},

        ]
    )
    all_predicted_next_words = set([ choice.message.content.split()[0] for choice in response.choices])
    if next_term in all_predicted_next_words:
       return 1
    else: 
      return 0


for choice in response.choices:
 print(choice.message.content)


evaluate("Swissmedic is the","greatest")







