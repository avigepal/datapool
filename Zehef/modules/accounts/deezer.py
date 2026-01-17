from lib.Requests import Request


def deezer(target: str):
    """Checks if an email is associated with a Deezer account"""
    try:
        from modules.accounts import get_logo 
        s = Request(url=None).Session()

        # Get authentication token
        r = s.post("https://www.deezer.com/ajax/gw-light.php?method=deezer.getUserData&input=3&api_version=1.0&api_token=&cid=")
        token = r.json().get('results', {}).get('checkForm')

        if not token:
            s.close()
            return {"source": "Deezer", "status": "error", "message": "Failed to get API token", "logo": get_logo("Deezer")}

        # Check email availability
        params = {
            'method': 'deezer.emailCheck',
            'input': 3,
            'api_version': 1.0,
            'api_token': token,
        }

        api = s.post("https://www.deezer.com/ajax/gw-light.php", params=params, data=f'{{"EMAIL":"{target}"}}')
        availability = api.json().get('results', {}).get('availability')

        s.close()  # Close session

        logo = get_logo("Deezer")  # Get logo path

        if availability is False:
            return {"source": "Deezer", "status": "found", "logo": logo}
        else:
            return {"source": "Deezer", "status": "not found", "logo": logo}

    except Exception as e:
        s.close()
        return {"source": "Deezer", "status": "error", "message": str(e), "logo": get_logo("Deezer")}
