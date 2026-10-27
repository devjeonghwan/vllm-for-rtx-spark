"""Small live-server check; run after starting vllm serve."""
import argparse
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request


def request(base, body):
    query = urllib.request.Request(base + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(query, timeout=180) as response:
        value = json.load(response)
    choice = value["choices"][0]
    if choice["finish_reason"] != "stop" or value["usage"]["completion_tokens"] <= 0:
        raise RuntimeError("Response did not finish normally: " + json.dumps(value, ensure_ascii=False))
    return {"seconds": time.perf_counter() - started, "response": value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    parsed = urllib.parse.urlparse(base)
    if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1"} or parsed.path:
        raise ValueError("Use the local test server")
    if args.output and args.output.exists():
        raise FileExistsError(args.output)
    result = {"status": "started", "requests": [], "performance_benchmark": False}
    try:
        with urllib.request.urlopen(base + "/health", timeout=5) as response:
            if response.status != 200:
                raise RuntimeError("Server is not healthy")
        context = "서울은 대한민국의 수도입니다. 이 정보에 따라 한국어로 답하세요. " * 12
        body = {"model": "Qwen/Qwen3.8-Flash-Next", "temperature": 0, "max_tokens": 160,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "system", "content": context},
                             {"role": "user", "content": "서울을 소개하는 짧은 문장 세 개를 써 주세요."}]}
        for _ in range(2):
            row = request(base, body)
            text = row["response"]["choices"][0]["message"]["content"]
            if "서울" not in text or len(text) < 10:
                raise RuntimeError("Unexpected text response: " + text)
            result["requests"].append(row)
        body = {**body, "max_tokens": 64, "messages": [
            {"role": "user", "content": "대한민국의 수도를 JSON으로 답하세요."}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "capital", "strict": True,
                "schema": {"type": "object", "properties": {"capital": {"type": "string", "enum": ["서울"]}},
                           "required": ["capital"], "additionalProperties": False}}}}
        row = request(base, body)
        if json.loads(row["response"]["choices"][0]["message"]["content"]) != {"capital": "서울"}:
            raise RuntimeError("Structured response failed")
        result["requests"].append(row)
        result["status"] = "passed_text_repeat_and_json_requests"
    except Exception as error:
        result.update(status="failed", error=repr(error))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if result["status"].startswith("passed_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
