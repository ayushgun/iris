import asyncio
import websockets
import caption


async def receive_frames(websocket):
    while True:
        try:
            frame = await websocket.recv()

            if isinstance(frame, bytes) and await caption.is_hazardous_frame(frame):
                description = await caption.describe_frame(frame)
                await websocket.send(description)

        except websockets.ConnectionClosed as e:
            print(f"Connection closed: {e}")
            break


start_server = websockets.serve(receive_frames, "localhost", 8000)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
