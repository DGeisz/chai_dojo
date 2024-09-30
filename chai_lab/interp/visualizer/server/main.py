import uvicorn

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from typing import List


class ResidueVis(BaseModel):
    index: int
    chain: int
    residue: str


class ChainVis(BaseModel):
    index: int
    sequence: str


class ProteinToVisualize(BaseModel):
    pdb_id: str
    activation: float
    chains: List[ChainVis]
    residues: List[ResidueVis]


class VisualizationCommand(BaseModel):
    feature_index: int
    label: str
    proteins: List[ProteinToVisualize]


class Response(BaseModel):
    res_type: str
    data: str | VisualizationCommand


# This will store all the connected WebSocket clients
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        print("Got websocket connection!")
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        print("Disconnecting websocket!")
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


app = FastAPI()

manager = ConnectionManager()


@app.post("/visualize/")
async def visualize(command: VisualizationCommand):
    print("Got command: ", command.model_dump_json())
    res = Response(res_type="visualize", data=command)

    await manager.broadcast(res.model_dump_json())

    return {"message": "Visualizing!"}


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)

    res = Response(res_type="connected", data="Connected!")

    await websocket.send_text(res.model_dump_json())

    try:
        while True:
            # Keep the WebSocket connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=4200)
