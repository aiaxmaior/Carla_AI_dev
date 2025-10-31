# jetson_ws_server.py
import asyncio, json, logging
import websockets

logging.basicConfig(level=logging.INFO)
QUEUE_MAXSIZE = 1000
incoming_queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)

# Optional: simple bearer token
AUTH_TOKEN = "changeme-very-secret"  # set same token on client
PORT = 8765

async def consumer_loop():
    """Example: consume parsed JSON objects and hand them to your LLM pipeline."""
    while True:
        obj = await incoming_queue.get()
        # TODO: swap this with your real hook into the LLM / analytics
        # e.g., enqueue to another process, write to Kafka/MQTT, call a local API, etc.
        # For demo:
        print("LLM-IN:", obj)
        incoming_queue.task_done()

async def handle(ws, path):
    # --- very simple auth ---
    proto = ws.request_headers.get("Sec-WebSocket-Protocol")
    if AUTH_TOKEN and proto != AUTH_TOKEN:
        await ws.close(code=4401, reason="Unauthorized")
        return

    peer = ws.remote_address
    logging.info(f"Client connected: {peer}")
    try:
        buffer = ""
        async for message in ws:
            # We expect text frames containing one or multiple JSON lines
            buffer += message
            # Split by newline; keep trailing partial in buffer
            *lines, buffer = buffer.split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    logging.warning(f"Bad JSONL from {peer}: {e}; line={line[:200]}")
                    continue
                # Backpressure: wait if queue is full
                await incoming_queue.put(obj)
        logging.info(f"Client disconnected: {peer}")
    except websockets.ConnectionClosed:
        logging.info(f"Client closed: {peer}")
    except Exception as e:
        logging.exception(f"Handler error for {peer}: {e}")

async def main():
    # Start consumer(s)
    asyncio.create_task(consumer_loop())
    # Start server
    async with websockets.serve(
        handle, "0.0.0.0", PORT,
        ping_interval=20, ping_timeout=20, max_size=10*1024*1024
    ):
        logging.info(f"WS server listening on ws://0.0.0.0:{PORT}")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())

