"""
Author: Matěj Křenek (xkrenem00)
Contact: xkrenem00@vutbr.cz
File: __init__.py
Responsibility: Exposes Discord notification utilities for training and evaluation processes.
"""

from .discord_notifier import DiscordNotifier

__all__ = [
    "DiscordNotifier",
]
