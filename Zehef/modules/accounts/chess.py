from lib.Requests import Request


async def chess(target: str):
    """Checks if an email is associated with a Chess.com account"""
    try:
        from modules.accounts import get_logo 
        r = await Request(
            f"https://www.chess.com/callback/email/available?email={target}"
        ).post()

        response_json = r.json()
        logo = get_logo("Chess.com")  # Fetch Chess.com logo path

        if response_json.get('isEmailAvailable') is False:
            return {"source": "Chess.com", "status": "found", "logo": logo}
        else:
            return {"source": "Chess.com", "status": "not found", "logo": logo}

    except Exception as e:
        return {"source": "Chess.com", "status": "error", "message": str(e), "logo": get_logo("Chess.com")}
