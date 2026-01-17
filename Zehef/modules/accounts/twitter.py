from lib.Requests import Request


async def x(target: str):
    """Checks if an email is associated with an X (Twitter) account"""
    try:
        from modules.accounts import get_logo 
        r = await Request(f"https://api.twitter.com/i/users/email_available.json?email={target}").get()
        response_json = r.json()

        if response_json.get('taken') is True:
            return {"source": "X (Twitter)", "status": "found", "logo": get_logo("X (Twitter)")}
        else:
            return {"source": "X (Twitter)", "status": "not found", "logo": get_logo("X (Twitter)")}

    except Exception as e:
        return {"source": "X (Twitter)", "status": "error", "message": str(e), "logo": get_logo("X (Twitter)")}
