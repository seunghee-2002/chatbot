import requests
import json

url = "http://192.168.0.2:11434/api/chat"

payload = {
    "model": "llama31",
    "messages": [
        {"role": "user", "content": "스트리밍으로 한 글자씩 출력해줘"}
    ],
    "stream": True
}

with requests.post(url, json=payload, stream=True) as r:
    for line in r.iter_lines():
        if line:
            data = json.loads(line.decode("utf-8"))
            if "message" in data:
                print(data["message"]["content"], end="", flush=True)