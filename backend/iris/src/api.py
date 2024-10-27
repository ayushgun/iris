import asyncio
import websockets
import caption
import inference


async def receive_frames(websocket):
    print("Client connected")

    while True:
        try:
            frame = await websocket.recv()

            if isinstance(frame, bytes):
                is_hazard, description = await asyncio.gather(
                    caption.is_hazardous_frame(frame, backend=inference.ClaudeBackend),
                    caption.describe_frame(frame, backend=inference.GeminiBackend),
                )

                print(f"Result: {is_hazard} -- {description}")

                if is_hazard:
                    await websocket.send(description)

        except websockets.ConnectionClosed as e:
            print(f"Connection closed: {e}")
            break


start_server = websockets.serve(receive_frames, "172.20.10.5", 8000)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
