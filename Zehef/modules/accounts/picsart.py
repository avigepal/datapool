from lib.Requests import Request


async def picsart(target: str):
    """Checks if an email is associated with a PicsArt account"""
    try:
        from modules.accounts import get_logo 
        params = {
            'email_encoded': 1,
            'emails': target
        }

        r = await Request("https://api.picsart.com/users/email/existence", params=params).get()
        response_json = r.json()
        logo = get_logo("PicsArt")

        if response_json.get('status') == 'success':
            if response_json.get('response'):
                return {"source": "PicsArt", "status": "found", "logo": logo}
            else:
                return {"source": "PicsArt", "status": "not found", "logo": logo}

        return {"source": "PicsArt", "status": "error", "message": "Invalid API response", "logo": logo}

    except Exception as e:
        return {"source": "PicsArt", "status": "error", "message": str(e), "logo": get_logo("PicsArt")}
