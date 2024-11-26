import requests
import json

response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": f"Bearer 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",
    "HTTP-Referer": "SUHANGSSALMUK", # Optional, for including your app on openrouter.ai rankings.
    "X-Title": "SUHANGSSALMUK", # Optional. Shows in rankings on openrouter.ai.
  },
  data=json.dumps({
    "model": "qwen/qwen-2-7b-instruct:free", # Optional
    "messages": [
      {
        "role": "user",
        "content": "What is the meaning of life?"
      }
    ]
  })
)
print(response.text)