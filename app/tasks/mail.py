import smtplib

from celery import shared_task

from app.core.config import get_settings
from app.core.mail import send_email

RESET_SUBJECT = "Восстановление пароля VitiGuard MD"
RESET_BODY = """Здравствуйте!

Чтобы задать новый пароль, перейдите по ссылке:
{url}

Ссылка действует {minutes} мин и сработает один раз. Если вы не запрашивали
восстановление пароля, просто проигнорируйте это письмо.
"""


# Сбой почтового сервера не должен терять письмо: задача повторяется с растущей паузой
@shared_task(
    name="mail.password_reset",
    autoretry_for=(OSError, smtplib.SMTPException),
    retry_backoff=True,
    max_retries=5,
)
def send_password_reset_email(to: str, reset_url: str) -> None:
    settings = get_settings()
    body = RESET_BODY.format(url=reset_url, minutes=settings.password_reset_ttl_minutes)
    send_email(settings, to, RESET_SUBJECT, body)
