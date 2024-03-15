"""
This plugin searches for Coinbase Access Token
"""
import re

from detect_secrets.plugins.base import RegexBasedDetector


class CoinbaseAccessTokenDetector(RegexBasedDetector):
    """Scans for Coinbase Access Token."""

    secret_type = "Coinbase Access Token"

    denylist = [
        # For Coinbase Access Token
        re.compile(
            r"""(?i)(?:coinbase)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-z0-9_-]{64})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
        ),
    ]
