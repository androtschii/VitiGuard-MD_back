import uuid
from datetime import datetime

from sqlalchemy import select, update

from app.models.auth import RefreshToken
from app.repositories.base import Repository


class RefreshTokenRepository(Repository[RefreshToken]):
    model = RefreshToken
    not_found_message = "Refresh-токен не найден"

    async def get_by_hash(
        self, token_hash: str, *, for_update: bool = False
    ) -> RefreshToken | None:
        query = select(RefreshToken).where(RefreshToken.token_hash == token_hash)
        if for_update:
            # Блокировка строки до конца транзакции: два одновременных обновления
            # с одним токеном выполнятся по очереди, и второе увидит его отозванным
            query = query.with_for_update()
        return await self.session.scalar(query)

    async def revoke(self, token: RefreshToken, now: datetime) -> None:
        token.revoked_at = now
        await self.session.flush()

    async def revoke_all_for_user(self, user_id: uuid.UUID, now: datetime) -> None:
        await self.session.execute(
            update(RefreshToken)
            .where(RefreshToken.user_id == user_id, RefreshToken.revoked_at.is_(None))
            .values(revoked_at=now)
        )
