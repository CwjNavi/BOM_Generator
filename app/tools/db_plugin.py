from semantic_kernel.functions import kernel_function
from app.repos.base import DBRepository


class DBPlugin:
    def __init__(self, repo: DBRepository) -> None:
        self._repo = repo

    @kernel_function(
        name="db_query",
        description="Query the application database. Input should be a simple query key for now, e.g. 'users' or 'orders'.",
    )
    async def db_query(self, query: str) -> str:
        result = await self._repo.query(query)
        return str(result)