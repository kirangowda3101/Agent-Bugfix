import asyncio
import subprocess
from mcp import Client
import mcp_server

INSTANCE = "django__django-10999"
IMAGE = "swebench/sweb.eval.x86_64.django_1776_django-10999:latest"
NAME = f"swe_agent_{INSTANCE}"

subprocess.run(["docker", "rm", "-f", NAME], capture_output=True)
subprocess.run(["docker", "run", "-d", "--name", NAME, IMAGE, "sleep", "600"], check=True)

mcp_server.set_container(NAME)


async def main():
    async with Client(mcp_server.mcp) as client:
        tools = await client.list_tools()
        print("Tools:", [t.name for t in tools.tools])

        r = await client.call_tool("search_code", {"pattern": "standard_duration_re"})
        print("\nsearch_code:")
        print(r.content[0].text)

        r = await client.call_tool("read_file", {"filename": "django/utils/dateparse.py"})
        print("\nread_file (first 200 chars):")
        print(r.content[0].text[:200])


asyncio.run(main())
subprocess.run(["docker", "rm", "-f", NAME], capture_output=True)