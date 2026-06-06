from fastapi import APIRouter
from pydantic import BaseModel


class EchoRequest(BaseModel):
    name: str


router = APIRouter(prefix="", tags=["Echo"])


@router.post("/echo")
async def echo(body: EchoRequest) -> EchoRequest:
    return body
