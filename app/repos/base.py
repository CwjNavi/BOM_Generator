from typing import Protocol, Any


class DBRepository(Protocol):
    async def query(self, q: str) -> Any:
        ...