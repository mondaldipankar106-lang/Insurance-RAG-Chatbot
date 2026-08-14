import json
import urllib.request

payload = {"data": ["What is my policy coverage?"], "fn_index": 0}
body = json.dumps(payload).encode("utf-8")
req = urllib.request.Request("http://127.0.0.1:7860/api/predict", data=body, headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        print(resp.read().decode())
except Exception as e:
    print("REQUEST_FAILED")
    print(e)
