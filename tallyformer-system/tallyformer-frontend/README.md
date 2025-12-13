# TallyFormer Frontend

This is a simple frontend application built to interact with the **TallyFormer LLM API**.  
It provides a user interface to send prompts and receive generated text from the backend models.

---

## 🚀 Backend API Overview

The frontend communicates with a FastAPI backend that exposes the following endpoints:

### 🔹 Welcome Endpoint
**GET** `/welcome`

Returns basic information about the loaded model.

Example response:
```
{
  "message": "Welcome to TallyFormer LLM inference API!",
  "model_name": "TallyFormer",
  "model_size": "51M parameters",
  "context_length": 256,
  "layers": 8,
  "heads": 8,
  "hidden_dim": 512
} 
``` 

### 🔹 Text Generation Endpoint
POST /generate,
Generates text based on a user prompt and inference parameters.

Request Body
```
{
  "prompt": "Explain what inflation is",
  "model_name": "sft",
  "max_new_tokens": 50,
  "temperature": 0.7,
  "topk": 500,
  "topp": 0.9,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "mode": "combined",
  "return_metrics": false
}
```
Response
```
{
  "generated_text": "Inflation is an economic phenomenon where..."
}
```
If return_metrics is enabled, performance metrics are also returned.



## 🛠 Frontend Responsibilities
The frontend should:

- Collect user prompts

- Allow model and parameter selection

- Send requests to /generate

- Display generated text

- Optionally display inference metrics

## 📌 Notes
Ensure the backend server is running before starting the frontend.