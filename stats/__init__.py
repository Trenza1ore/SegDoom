"""
A module for plotting stats and sending stats via discord webhook
"""

from stats.helper_func import *
from stats.discord_webhook import DiscordWebhook

# Class alias for backward compatibility
discord_bot = DiscordWebhook