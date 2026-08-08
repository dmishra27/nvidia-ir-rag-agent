"""Slack Bolt app entry point (Socket Mode): `/nvidia-search` slash command
+ 👍/👎 HITL feedback on the bot's own answers.

Socket Mode (not the HTTP Events API) so this needs no public URL or request
signing -- matching this project's local-dev-first posture (same reasoning
as docker-compose.yml's other services). Requires SLACK_BOT_TOKEN (xoxb-...)
and SLACK_APP_TOKEN (xapp-... with the `connections:write` scope) in .env.

Run with: `python -m slackbot.app`
"""

from __future__ import annotations

import os

import structlog
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from slackbot import feedback_handler, handlers

load_dotenv()
log = structlog.get_logger()


def build_app() -> App:
    app = App(token=os.environ["SLACK_BOT_TOKEN"])
    handlers.register(app)
    feedback_handler.register(app)
    return app


def main() -> None:
    app = build_app()
    log.info("slackbot_start", stage="slackbot_app")
    # slack_bolt.adapter.socket_mode.SocketModeHandler.start ships without a
    # type annotation upstream.
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()  # type: ignore[no-untyped-call]


if __name__ == "__main__":
    main()
