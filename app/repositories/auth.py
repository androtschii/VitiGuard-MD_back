from app.models.auth import RefreshToken
from app.repositories.base import Repository


class RefreshTokenRepository(Repository[RefreshToken]):
    model = RefreshToken
    not_found_message = "Refresh-токен не найден"
