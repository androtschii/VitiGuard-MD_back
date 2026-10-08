from email.message import EmailMessage
from typing import Any, ClassVar

import pytest

from app.core import mail
from app.core.config import Settings
from app.tasks import mail as mail_tasks


class FakeSMTP:
    """Подменяет smtplib.SMTP и запоминает, что с ним делали."""

    instances: ClassVar[list["FakeSMTP"]] = []

    def __init__(self, host: str, port: int, timeout: float) -> None:
        self.address = (host, port)
        self.started_tls = False
        self.login_args: tuple[str, str] | None = None
        self.sent: list[EmailMessage] = []
        FakeSMTP.instances.append(self)

    def __enter__(self) -> "FakeSMTP":
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def starttls(self) -> None:
        self.started_tls = True

    def login(self, username: str, password: str) -> None:
        self.login_args = (username, password)

    def send_message(self, message: EmailMessage) -> None:
        self.sent.append(message)


@pytest.fixture
def smtp(monkeypatch: pytest.MonkeyPatch) -> type[FakeSMTP]:
    FakeSMTP.instances = []
    monkeypatch.setattr("smtplib.SMTP", FakeSMTP)
    return FakeSMTP


def test_email_goes_to_configured_server(smtp: type[FakeSMTP]) -> None:
    settings = Settings(_env_file=None, smtp_host="mailpit", smtp_port=1025)

    mail.send_email(settings, "grower@example.md", "Тема", "Текст письма")

    (server,) = smtp.instances
    assert server.address == ("mailpit", 1025)
    assert server.started_tls is False
    assert server.login_args is None
    (message,) = server.sent
    assert message["To"] == "grower@example.md"
    assert message["From"] == "VitiGuard MD <no-reply@vitiguard.md>"
    assert message["Subject"] == "Тема"
    assert message.get_content().strip() == "Текст письма"


def test_starttls_and_login_when_configured(smtp: type[FakeSMTP]) -> None:
    settings = Settings(
        _env_file=None,
        smtp_starttls=True,
        smtp_username="mailer",
        smtp_password="secret",  # type: ignore[arg-type]
    )

    mail.send_email(settings, "grower@example.md", "Тема", "Текст")

    (server,) = smtp.instances
    assert server.started_tls is True
    assert server.login_args == ("mailer", "secret")


def test_reset_task_sends_link_with_lifetime(smtp: type[FakeSMTP]) -> None:
    mail_tasks.send_password_reset_email(
        "grower@example.md", "https://app.vitiguard.md/reset-password?token=abc"
    )

    (message,) = smtp.instances[0].sent
    body = message.get_content()
    assert message["Subject"] == "Восстановление пароля VitiGuard MD"
    assert "https://app.vitiguard.md/reset-password?token=abc" in body
    assert "60 мин" in body


def test_reset_task_is_retried_on_smtp_failure() -> None:
    assert mail_tasks.send_password_reset_email.max_retries == 5
