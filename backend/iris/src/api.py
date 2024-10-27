import asyncio
import websockets
import caption
from inference import GeminiBackend


async def receive_frames(websocket):
    while True:
        try:
            frame = await websocket.recv()

            if isinstance(frame, bytes):
                is_hazard, description = await asyncio.gather(
                    caption.is_hazardous_frame(frame, backend=GeminiBackend),
                    caption.describe_frame(frame, backend=GeminiBackend),
                )

                if is_hazard:
                    await websocket.send(description)

        except websockets.ConnectionClosed as e:
            print(f"Connection closed: {e}")
            break


start_server = websockets.serve(receive_frames, "10.91.29.98", 8000)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
