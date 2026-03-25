import asyncio
import msgpack
import msgpack_numpy as m
import struct
from typing import Any, Dict, Optional

m.patch()  # allow numpy arrays

def encode_msg(obj: dict) -> bytes:
    return msgpack.packb(obj, use_bin_type=True)

def decode_msg(raw: bytes) -> dict:
    return msgpack.unpackb(raw, raw=True)

async def send_framed(writer: asyncio.StreamWriter, obj: dict):
    payload = encode_msg(obj)
    header = struct.pack("!I", len(payload))
    writer.write(header + payload)
    await writer.drain()

async def recv_framed(reader: asyncio.StreamReader) -> dict:
    header = await reader.readexactly(4)
    (msg_len,) = struct.unpack("!I", header)
    payload = await reader.readexactly(msg_len)
    return decode_msg(payload)


class MsgpackNumpyServer:
    """TCP server that stores the latest observation + action for a robot loop."""

    def __init__(self, host: str = "0.0.0.0", port: int = 9000):
        self.host = host
        self.port = port

        # Shared state with the robot loop
        self.latest_observation: Optional[Dict[str, Any]] = None
        self.latest_action: Dict[str, Any] = {}

    async def start(self):
        server = await asyncio.start_server(self.handle, self.host, self.port)
        print(f"[SERVER] listening on {self.host}:{self.port}")
        async with server:
            await server.serve_forever()

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        print("[SERVER] client connected")
        try:
            while True:
                request = await recv_framed(reader)

                # Store latest observation from client
                self.latest_observation = request

                # Respond with latest action (to be filled by robot loop)
                await send_framed(writer, self.latest_action)
        except asyncio.IncompleteReadError:
            print("[SERVER] client disconnected")
        except Exception as e:
            print("[SERVER] error:", e)


class MsgpackNumpyClient:
    def __init__(self, host="0.0.0.0", port=9000):
        self.host = host
        self.port = port
        self.reader = None
        self.writer = None

    async def connect(self):
        self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
        print("[CLIENT] connected")

        return self

    async def send_request(self, data: dict) -> dict:
        await send_framed(self.writer, data)
        response = await recv_framed(self.reader)
        return response

    async def close(self):
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()

async def main():
    server = MsgpackNumpyServer(host="0.0.0.0", port=9001)
    await server.start()

if __name__ == "__main__":
    asyncio.run(main())