import requests

# Function to altert methane leakage via email 
def send_email_via_sendgrid(subject, body, to_email):
    SENDGRID_API_URL = "https://api.sendgrid.com/v3/mail/send"
    SENDGRID_API_KEY = "YOUR_SENDGRID_API_KEY"

    headers = {
        "Authorization": f"Bearer {SENDGRID_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "personalizations": [{
            "to": [{"email": to_email}],
            "subject": subject
        }],
        "from": {"email": "stevemoses13@gmail.com"},
        "content": [{"type": "text/plain", "value": body}]
    }

    response = requests.post(SENDGRID_API_URL, headers=headers, json=data)
    return response.status_code, response.text

send_email_via_sendgrid('Methane Lekage Alert', 'There has been a leakage of methane', 'mosessteve04@gmail.com')
