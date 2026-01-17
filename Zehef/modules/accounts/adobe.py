from lib.Requests import Request


async def adobe(target: str):
    """Checks if an email is associated with an Adobe account"""
    data = {
        "username": target,
        "usernameType": "EMAIL"
    }

    headers = {
        'x-ims-clientid': 'homepage_milo',
        'content-type': 'application/json'
    }

    try:
        from modules.accounts import get_logo 
        r = await Request("https://auth.services.adobe.com/signin/v2/users/accounts", headers=headers, json=data).post()
        response_json = r.json()

        logo = get_logo("Adobe")  # Get Adobe logo path

        if response_json and isinstance(response_json, list) and response_json[0].get('authenticationMethods'):
            return {"source": "Adobe", "status": "found", "logo": logo}
        else:
            return {"source": "Adobe", "status": "not found", "logo": logo}

    except Exception as e:
        return {"source": "Adobe", "status": "error", "message": str(e), "logo": get_logo("Adobe")}
