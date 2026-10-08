import smtplib
from email.message import EmailMessage

from app.core.config import Settings


def send_email(settings: Settings, to: str, subject: str, body: str) -> None:
    message = EmailMessage()
    message["From"] = settings.mail_from
    message["To"] = to
    message["Subject"] = subject
    message.set_content(body)

    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=10) as smtp:
        if settings.smtp_starttls:
            smtp.starttls()
        if settings.smtp_username and settings.smtp_password:
            smtp.login(
                settings.smtp_username, settings.smtp_password.get_secret_value()
            )
        smtp.send_message(message)
