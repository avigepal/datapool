from lib.Requests import Request


async def duolingo(target: str):
    """Checks if an email is associated with a Duolingo account"""
    try:
        from modules.accounts import get_logo 
        r = await Request(
            "https://www.duolingo.com/2017-06-30/users",
            params={'email': target}
        ).get()

        response_json = r.json()
        logo = get_logo("Duolingo")  # Get the Duolingo logo path

        if response_json.get('users'):
            username = response_json['users'][0].get('username', 'N/A')
            return {"source": "Duolingo", "status": "found", "username": username, "logo": logo}
        else:
            return {"source": "Duolingo", "status": "not found", "logo": logo}

    except Exception as e:
        return {"source": "Duolingo", "status": "error", "message": str(e), "logo": get_logo("Duolingo")}
