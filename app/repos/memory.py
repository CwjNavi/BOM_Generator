from typing import Any


class InMemoryRepository:
    def __init__(self) -> None:
        # fake “db”
        self._data = {
            "users": [{"id": 1, "name": "Ada"}, {"id": 2, "name": "Grace"}],
            "orders": [{"id": 101, "user_id": 1, "total": 42.50}],
        }

    async def query(self, q: str) -> Any:
        """
        Very dumb placeholder:
        - if q == "users" -> return users table
        - if q == "orders" -> return orders table
        - else -> return empty
        """
        return self._data.get(q.strip().lower(), [])