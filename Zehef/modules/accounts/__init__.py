from modules.accounts.adobe import adobe 
from modules.accounts.bandlab import bandlab
from modules.accounts.chess import chess
from modules.accounts.deezer import deezer
from modules.accounts.duolingo import duolingo
from modules.accounts.flickr import flickr
from modules.accounts.github import github
from modules.accounts.gravatar import gravatar
from modules.accounts.imgur import imgur
from modules.accounts.instagram import instagram
from modules.accounts.picsart import picsart
from modules.accounts.pinterest import pinterest
from modules.accounts.pornhub import pornhub
from modules.accounts.protonmail import protonmail
from modules.accounts.spotify import spotify
from modules.accounts.strava import strava
from modules.accounts.twitter import x




# Define logo mapping
LOGO_PATH = "/assets/"  # Change this to the correct assets path

account_logos = {
    "Adobe": LOGO_PATH + "adobe.png",
    "BandLab": LOGO_PATH + "bandlab.png",
    "Chess.com": LOGO_PATH + "chess.png",
    "Deezer": LOGO_PATH + "deezer.png",
    "Duolingo": LOGO_PATH + "duolingo.png",
    "Flickr": LOGO_PATH + "flickr.png",
    "GitHub": LOGO_PATH + "github.png",
    "Gravatar": LOGO_PATH + "gravatar.png",
    "Imgur": LOGO_PATH + "imgur.png",
    "Instagram": LOGO_PATH + "instagram.png",
    "PicsArt": LOGO_PATH + "picsart.png",
    "Pinterest": LOGO_PATH + "pinterest.png",
    "Pornhub": LOGO_PATH + "pornhub.png",
    "ProtonMail": LOGO_PATH + "protonmail.png",
    "Spotify": LOGO_PATH + "spotify.png",
    "Strava": LOGO_PATH + "strava.png",
    "X (Twitter)": LOGO_PATH + "twitter.png"
}

# Function to get logo for a given source
def get_logo(source):
    return account_logos.get(source, None)