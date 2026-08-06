import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise SystemExit("ANTHROPIC_API_KEY is missing — set it in .env")

print(f"Key: {api_key[:7]}... (length={len(api_key)})")

client = Anthropic(api_key=api_key)

resp = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=50,
    messages=[{"role": "user", "content": "Reply with exactly: API connection verified."}],
)

print(resp.content[0].text)
print(f"Input tokens: {resp.usage.input_tokens}")
print(f"Output tokens: {resp.usage.output_tokens}")
