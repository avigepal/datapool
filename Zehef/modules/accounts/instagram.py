from lib.Requests import Request

async def instagram(target: str):
    """Checks if an email is associated with an Instagram account"""
    try:
        from modules.accounts import get_logo 
        # Step 1: Get CSRF token from Instagram signup page
        req = await Request("https://www.instagram.com/accounts/emailsignup/").get()
        csrf_token = req.cookies.get('csrftoken')

        if not csrf_token:
            return {"source": "Instagram", "status": "error", "message": "CSRF token not found"}

        # Step 2: Attempt to sign up with the email to check availability
        data = {
            'email': target,
            'first_name': '',
            'username': '',
            'opt_into_one_tap': False
        }

        r = await Request(
            "https://www.instagram.com/api/v1/web/accounts/web_create_ajax/attempt/",
            headers={'x-csrftoken': csrf_token, 'referer': 'https://www.instagram.com/'},
            data=data
        ).post()

        response_json = r.json()
        logo = get_logo("Instagram")

        # Step 3: Check response for email status
        code = response_json.get('errors', {}).get('email', [{}])[0].get('code')

        if code == 'email_is_taken':
            return {"source": "Instagram", "status": "found", "logo": logo}
        else:
            return {"source": "Instagram", "status": "not found", "logo": logo}

    except Exception as e:
        return {"source": "Instagram", "status": "error", "message": str(e), "logo": get_logo("Instagram")}
