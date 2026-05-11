# bedrock-local-server

A local server that exposes a chat completions endpoint backed by AWS Bedrock.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
HOST=0.0.0.0
PORT=8010
```


## Run

Start the server:

```bash
python app.py
```

After startup, copy a model ID from the CLI table and paste it into the prompt.

Open the API docs:

```text
http://localhost:8010/docs
```

List available Bedrock models:

```bash
curl http://localhost:8010/v1/models
```

Send a chat completion request:

```bash
curl -X POST http://localhost:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Say hello in one sentence."
      }
    ],
    "max_tokens": 128,
    "temperature": 0.7
  }'
```
