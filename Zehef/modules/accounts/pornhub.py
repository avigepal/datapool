from bs4 import BeautifulSoup
import requests

def pornhub(target: str):
    """Checks if an email is associated with a Pornhub account"""
    try:
        from modules.accounts import get_logo  # Import get_logo function
        with requests.Session() as s:
            # Step 1: Get signup page and extract CSRF token
            r = s.get("https://fr.pornhub.com/signup", timeout=10)
            soup = BeautifulSoup(r.text, 'html.parser')
            token_element = soup.find(attrs={'name': 'token'})

            if not token_element:
                return {"source": "Pornhub", "status": "error", "message": "CSRF token not found", "logo": get_logo("Pornhub")}

            token = token_element.get('value')

            # Step 2: Check email existence
            data = {
                'token': token,
                'check_what': 'email',
                'email': target
            }

            api = s.post("https://fr.pornhub.com/user/create_account_check", data=data, timeout=10)
            response_json = api.json()

            if response_json.get('email') == "create_account_failed":
                return {"source": "Pornhub", "status": "found", "logo": get_logo("Pornhub")}
            else:
                return {"source": "Pornhub", "status": "not found", "logo": get_logo("Pornhub")}

    except requests.exceptions.Timeout:
        return {"source": "Pornhub", "status": "error", "message": "Request timed out", "logo": get_logo("Pornhub")}
    except Exception as e:
        return {"source": "Pornhub", "status": "error", "message": str(e), "logo": get_logo("Pornhub")}
