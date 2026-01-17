import requests


def imgur(target: str):
    """Checks if an email is associated with an Imgur account"""
    try:
        from modules.accounts import get_logo 
        with requests.Session() as s:
            r = s.post("https://imgur.com/signin/ajax_email_available", data={'email': target})
            response_json = r.json()

            logo = get_logo("Imgur")

            if response_json.get('data', {}).get('available') is False:
                return {"source": "Imgur", "status": "found", "logo": logo}
            else:
                return {"source": "Imgur", "status": "not found", "logo": logo}

    except requests.exceptions.RequestException as e:
        return {"source": "Imgur", "status": "error", "message": str(e), "logo": get_logo("Imgur")}
