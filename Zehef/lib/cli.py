import asyncio , re
from .colors import *
from .update import Version_Checker
from .emails_gen import Email_Gen
from modules import *   

async def parser(email: str):
    """Search for email data across multiple services"""
    await Version_Checker.checker()

    EMAIL_REGEX = r'[a-zA-Z0-9._+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

    if not re.match(EMAIL_REGEX, email):
        return {"error": "Not a valid email address."}

    results = []

    # Async functions
    results.append(await adobe(email))
    results.append(await bandlab(email))
    results.append(await chess(email))
    results.append(await duolingo(email))
    results.append(await flickr(email))
    results.append(await github(email))
    results.append(await gravatar(email))
    results.append(await instagram(email))
    results.append(await picsart(email))
    results.append(await pinterest(email))
    results.append(await protonmail(email))
    results.append(await spotify(email))
    results.append(await strava(email))
    results.append(await x(email))

    # Sync functions (wrapped in async)
    loop = asyncio.get_running_loop()
    results.append(await loop.run_in_executor(None, deezer, email))
    results.append(await loop.run_in_executor(None, imgur, email))
    results.append(await loop.run_in_executor(None, pornhub, email))

    return {"data": [res for res in results if res]}
