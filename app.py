import json
import os
import time
import uuid
from typing import Optional

import boto3
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()


def env_value(name):
    value = os.getenv(name)
    return value if value else None


AWS_REGION = env_value("AWS_REGION") or "us-east-1"
HOST = env_value("HOST") or "0.0.0.0"
PORT = int(env_value("PORT") or "8010")
SELECTED_MODEL = None


def fetch_available_models():
    try:
        bedrock = boto3.client(
            "bedrock",
            region_name=AWS_REGION,
            aws_access_key_id=env_value("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=env_value("AWS_SECRET_ACCESS_KEY"),
        )
        models = []
        foundation_response = bedrock.list_foundation_models(byOutputModality="TEXT")
        for model in foundation_response.get("modelSummaries", []):
            inference_types = model.get("inferenceTypesSupported", [])
            if inference_types and "ON_DEMAND" not in inference_types:
                continue
            models.append(
                {
                    "modelId": model.get("modelId", ""),
                    "modelName": model.get("modelName", model.get("modelId", "")),
                    "providerName": model.get("providerName", "unknown"),
                }
            )

        next_token = None
        while True:
            kwargs = {"maxResults": 1000}
            if next_token:
                kwargs["nextToken"] = next_token
            profile_response = bedrock.list_inference_profiles(**kwargs)
            for profile in profile_response.get("inferenceProfileSummaries", []):
                if profile.get("status") != "ACTIVE":
                    continue
                profile_id = profile.get("inferenceProfileId", "")
                models.append(
                    {
                        "modelId": profile_id,
                        "modelName": profile.get("inferenceProfileName", profile_id),
                        "providerName": "inference-profile",
                    }
                )
            next_token = profile_response.get("nextToken")
            if not next_token:
                break

        by_id = {model["modelId"]: model for model in models if model.get("modelId")}
        return sorted(by_id.values(), key=lambda model: model.get("modelId", ""))
    except Exception as e:
        print(f"\nCould not fetch models: {e}")
        print("Check your AWS credentials and region.\n")
        return []


def prompt_model_selection(models):
    if not models:
        manual = input("Enter model ID manually, or press Enter to exit: ").strip()
        return manual if manual else None

    rows = [
        (model.get("modelId", ""), model.get("modelName", model.get("modelId", "")))
        for model in models
    ]
    model_id_width = max(len("Model ID"), *(len(model_id) for model_id, _ in rows))
    model_name_width = max(len("Model name"), *(len(model_name) for _, model_name in rows))

    print()
    print(f"  {'Model ID':<{model_id_width}}  {'Model name':<{model_name_width}}")
    print(f"  {'-' * model_id_width}  {'-' * model_name_width}")
    for model_id, model_name in rows:
        print(f"  {model_id:<{model_id_width}}  {model_name:<{model_name_width}}")

    print("\n" + "-" * 70)
    print("  Copy a model ID from the table and paste it below.")
    print("-" * 70)

    available_model_ids = {model_id.lower(): model_id for model_id, _ in rows}

    while True:
        choice = input("\n  Model ID: ").strip()
        if not choice:
            continue
        selected = available_model_ids.get(choice.lower())
        if selected:
            print(f"\n  Selected: {selected}\n")
            return selected
        print("  Invalid model ID. Paste a value from the Model ID column.")


def get_runtime_client():
    return boto3.client(
        "bedrock-runtime",
        region_name=AWS_REGION,
        aws_access_key_id=env_value("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=env_value("AWS_SECRET_ACCESS_KEY"),
    )


def get_provider(model_id: str) -> str:
    parts = model_id.split(".")
    if len(parts) > 1 and parts[0] in ("us", "eu", "ap", "apac", "global"):
        return parts[1].lower()
    return parts[0].lower()


def build_payload(model_id, messages, system, max_tokens, temperature):
    provider = get_provider(model_id)
    msgs = [message for message in messages if message["role"] != "system"]
    sys_text = system or next((message["content"] for message in messages if message["role"] == "system"), None)

    if provider == "anthropic":
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": msgs,
        }
        if sys_text:
            body["system"] = sys_text
        return body

    if provider == "meta":
        prompt = ""
        if sys_text:
            prompt += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{sys_text}<|eot_id|>"
        for message in msgs:
            prompt += f"<|start_header_id|>{message['role']}<|end_header_id|>\n{message['content']}<|eot_id|>"
        prompt += "<|start_header_id|>assistant<|end_header_id|>"
        return {"prompt": prompt, "max_gen_len": max_tokens, "temperature": temperature}

    if provider == "mistral":
        formatted = ""
        for message in msgs:
            if message["role"] == "user":
                formatted += f"[INST] {message['content']} [/INST]"
            else:
                formatted += f" {message['content']} "
        return {"prompt": formatted, "max_tokens": max_tokens, "temperature": temperature}

    if provider == "amazon":
        input_text = "\n".join(f"{message['role']}: {message['content']}" for message in msgs)
        return {
            "inputText": input_text,
            "textGenerationConfig": {"maxTokenCount": max_tokens, "temperature": temperature},
        }

    if provider == "cohere":
        last_user = next((message["content"] for message in reversed(msgs) if message["role"] == "user"), "")
        history = [{"role": message["role"], "message": message["content"]} for message in msgs[:-1]]
        return {"message": last_user, "chat_history": history, "max_tokens": max_tokens, "temperature": temperature}

    if provider == "ai21":
        prompt = "\n".join(f"{message['role']}: {message['content']}" for message in msgs)
        return {"prompt": prompt, "maxTokens": max_tokens, "temperature": temperature}

    return None


def extract_text(model_id, body):
    provider = get_provider(model_id)
    if provider == "anthropic":
        return body.get("content", [{}])[0].get("text", "")
    if provider == "meta":
        return body.get("generation", "")
    if provider == "mistral":
        return body.get("outputs", [{}])[0].get("text", "")
    if provider == "amazon":
        results = body.get("results") or []
        return results[0].get("outputText", "") if results else ""
    if provider == "cohere":
        return body.get("text", "")
    if provider == "ai21":
        return body.get("completions", [{}])[0].get("data", {}).get("text", "")

    for key in ("content", "generation", "text", "outputText"):
        if key in body:
            value = body[key]
            if isinstance(value, list):
                return value[0].get("text", str(value[0]))
            return str(value)
    return json.dumps(body)


def call_converse(model_id, messages, system, max_tokens, temperature):
    rt = get_runtime_client()
    converse_msgs = [
        {"role": message["role"], "content": [{"text": message["content"]}]}
        for message in messages
        if message["role"] != "system"
    ]
    kwargs = {
        "modelId": model_id,
        "messages": converse_msgs,
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
    }
    if system:
        kwargs["system"] = [{"text": system}]
    resp = rt.converse(**kwargs)
    return resp["output"]["message"]["content"][0]["text"]


app = FastAPI(title="Bedrock Universal Local Server", version="2.0.0")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: list[Message]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False
    system: Optional[str] = None


@app.get("/")
def root():
    return {"status": "ok", "selected_model": SELECTED_MODEL, "region": AWS_REGION}


@app.get("/v1/models")
def list_models():
    models = fetch_available_models()
    return {
        "object": "list",
        "data": [
            {"id": model["modelId"], "object": "model", "owned_by": model.get("providerName", "unknown")}
            for model in models
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    model_id = req.model or SELECTED_MODEL
    if not model_id:
        raise HTTPException(status_code=400, detail="No model selected.")

    messages = [{"role": message.role, "content": message.content} for message in req.messages]
    sys_text = req.system or next((message["content"] for message in messages if message["role"] == "system"), None)

    try:
        rt = get_runtime_client()
        payload = build_payload(model_id, messages, sys_text, req.max_tokens, req.temperature)

        if req.stream:

            def stream_gen():
                if payload:
                    resp = rt.invoke_model_with_response_stream(
                        modelId=model_id,
                        contentType="application/json",
                        body=json.dumps(payload),
                    )
                    for event in resp["body"]:
                        chunk_data = json.loads(event["chunk"]["bytes"])
                        text = ""
                        if chunk_data.get("type") == "content_block_delta":
                            text = chunk_data.get("delta", {}).get("text", "")
                        elif "generation" in chunk_data:
                            text = chunk_data["generation"]
                        elif "outputText" in chunk_data:
                            text = chunk_data["outputText"]
                        if text:
                            yield f"data: {json.dumps({'choices': [{'delta': {'content': text}}]})}\n\n"
                else:
                    converse_msgs = [
                        {"role": message["role"], "content": [{"text": message["content"]}]}
                        for message in messages
                        if message["role"] != "system"
                    ]
                    kwargs = {
                        "modelId": model_id,
                        "messages": converse_msgs,
                        "inferenceConfig": {"maxTokens": req.max_tokens, "temperature": req.temperature},
                    }
                    if sys_text:
                        kwargs["system"] = [{"text": sys_text}]
                    resp = rt.converse_stream(**kwargs)
                    for event in resp["stream"]:
                        if "contentBlockDelta" in event:
                            text = event["contentBlockDelta"]["delta"].get("text", "")
                            if text:
                                yield f"data: {json.dumps({'choices': [{'delta': {'content': text}}]})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(stream_gen(), media_type="text/event-stream")

        if payload:
            response = rt.invoke_model(
                modelId=model_id,
                contentType="application/json",
                body=json.dumps(payload),
            )
            content = extract_text(model_id, json.loads(response["body"].read()))
        else:
            content = call_converse(model_id, messages, sys_text, req.max_tokens, req.temperature)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  Bedrock Universal Local Server")
    print("=" * 70)
    print(f"\n  Region : {AWS_REGION}")

    print("\n  Fetching available models from Bedrock...\n")
    models = fetch_available_models()
    SELECTED_MODEL = prompt_model_selection(models)

    if not SELECTED_MODEL:
        print("  No model selected. Exiting.")
        raise SystemExit(1)

    print(f"  Server starting: http://localhost:{PORT}")
    print(f"  API docs       : http://localhost:{PORT}/docs\n")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
