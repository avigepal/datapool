from lib.Requests import Request
from datetime import datetime
import json
import re
import requests


async def generate_auth_cookie():
    """Generates authentication cookie for ProtonMail API"""
    url_session = "https://account.proton.me/api/auth/v4/sessions"
    url_cookies = "https://account.proton.me/api/core/v4/auth/cookies"

    data_session = {
        "x-pm-appversion": "web-account@5.0.153.3",
        "x-pm-locale": "en_US",
        "x-enforce-unauthsession": "true"
    }

    try:
        with requests.Session() as session:
            response = session.post(url_session, headers=data_session, timeout=10)
            json_dump = response.json()

            access_token = json_dump.get('AccessToken')
            refresh_token = json_dump.get('RefreshToken')
            uid = json_dump.get('UID')

            if not access_token or not refresh_token or not uid:
                return None, None

            data_cookie = {
                "x-pm-appversion": "web-account@5.0.153.3",
                "x-pm-locale": "en_US",
                "x-pm-uid": uid,
                "Authorization": f"Bearer {access_token}"
            }

            request_data = {
                "GrantType": "refresh_token",
                "Persistent": 0,
                "RedirectURI": "https://protonmail.com",
                "RefreshToken": refresh_token,
                "ResponseType": "token",
                "State": "C72g4sTNltu4TAL5bUQlnvUT",
                "UID": uid
            }

            response = session.post(url_cookies, headers=data_cookie, json=request_data, timeout=10)

            auth_cookie = None
            for cookie in response.cookies:
                if "AUTH" in str(cookie):
                    auth_cookie = str(cookie).split(" ")[1]
                    break

            return uid, auth_cookie

    except requests.exceptions.Timeout:
        return None, None
    except Exception:
        return None, None

async def protonmail(target: str):
    """Checks if an email is associated with a ProtonMail account"""
    try:
        from modules.accounts import get_logo  # Import the get_logo function
        if target.split('@')[1] not in ['pm.me', 'proton.me', 'protonmail.com', 'protonmail.ch']:
            return {"source": "ProtonMail", "status": "not checked", "message": "Not a ProtonMail domain", "logo": get_logo("ProtonMail")}

        uid, auth_cookie = await generate_auth_cookie()

        if not uid or not auth_cookie:
            return {"source": "ProtonMail", "status": "error", "message": "Failed to generate authentication cookie", "logo": get_logo("ProtonMail")}

        headers = {
            "x-pm-appversion": "web-account@5.0.153.3",
            "x-pm-locale": "en_US",
            "x-pm-uid": uid,
            "Cookie": auth_cookie
        }

        params = {
            "Name": target,
            "ParseDomain": "1"
        }

        with requests.Session() as session:
            r = session.get("https://account.proton.me/api/core/v4/users/available", headers=headers, params=params, timeout=10)
        
        if '"Suggestions":[]' in r.text or '"Code":1000' in r.text:
            return {"source": "ProtonMail", "status": "not found", "logo": get_logo("ProtonMail")}

        # Check account creation date via ProtonMail's PGP lookup
        api = f"https://api.protonmail.ch/pks/lookup?op=index&search={target}"
        r = await Request(api).get()
        match = re.search(r'\b\d{10}\b', r.text)

        if match:
            timestamp = int(match.group())
            date_of_creation = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

            return {
                "source": "ProtonMail",
                "status": "found",
                "created_on": date_of_creation,
                "logo": get_logo("ProtonMail")
            }
        
        return {"source": "ProtonMail", "status": "found", "logo": get_logo("ProtonMail")}

    except requests.exceptions.Timeout:
        return {"source": "ProtonMail", "status": "error", "message": "Request timed out", "logo": get_logo("ProtonMail")}
    except Exception as e:
        return {"source": "ProtonMail", "status": "error", "message": str(e), "logo": get_logo("ProtonMail")}
