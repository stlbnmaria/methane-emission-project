import requests
from typing import Tuple, Any


def send_email_via_sendgrid(subject: str, body: str, to_email: str) -> Tuple[int, str]:
    """
    Sends an email via the SendGrid API.

    Args:
    - subject (str): The subject line of the email.
    - body (str): The main content/body of the email.
    - to_email (str): The recipient's email address.

    Returns:
    - Tuple[int, str]: A tuple containing the status code of the API request and the response text.
    """

    SENDGRID_API_URL = "https://api.sendgrid.com/v3/mail/send"
    SENDGRID_API_KEY = "YOUR_SENDGRID_API_KEY"

    headers = {
        "Authorization": f"Bearer {SENDGRID_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "personalizations": [{"to": [{"email": to_email}], "subject": subject}],
        "from": {"email": "stevemoses13@gmail.com"},
        "content": [{"type": "text/plain", "value": body}],
    }

    response = requests.post(SENDGRID_API_URL, headers=headers, json=data)
    return response.status_code, response.text


send_email_via_sendgrid(
    "Methane Lekage Alert",
    "There has been a leakage of methane",
    "mosessteve04@gmail.com",
)
