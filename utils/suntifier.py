from dataclasses import dataclass, field
from datetime import datetime

from typing import Any
import os
import requests
import time

@dataclass
class NotiInfo:
    experiment_name: str
    success: bool
    runtime: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None
    finished_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

class Suntifier:
    def __init__(self, webhook_url:str | None = None):
        self.webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")

        if not self.webhook_url:
            raise ValueError("DISCORD_WEBHOOK_URL is missing.")

    def send(self, result: NotiInfo):
        msg = self._format_message(result)

        response = requests.post(
            self.webhook_url,
            json=msg,
            timeout=10
        )

        if response.status_code not in (200, 204):
            print("Discord notification failed:", response.status_code, response.text)

    def _format_message(self, r:NotiInfo):

        if r.success:
            color = 0x2ECC71
            title = "🎆 EXPERIMENT DONE 🎆\n\u200b"
            #description = "✨ Training completed successfully. Check the summary below ✨\n\u200b"
        else:
            color = 0xE74C3C
            title = "😫 EXPERIMENT FAILED 😫\n\u200b"
            #description = "⚠️ Training stopped before completion. Check the error below ⚠️\n\u200b"

        fields = [
            {
                "name": "Experiment",
                "value": self._fmt(r.experiment_name),
                "inline": False,    # display side by side when possible
            },
        ]

        if r.success:
            fields.append({
                "name": "Runtime",
                "value": self._fmt(r.runtime),
                "inline": False,
            })

            # add metrics depends on the models
            for key, value in list(r.metrics.items())[:4]:
                if isinstance(value, float):
                    value = f"{value:.4f}"

                fields.append({
                    "name": str(key),
                    "value": str(value),
                    "inline": True,
                })

        if not r.success and r.error_message:
            fields.append({
                "name": "error",
                "value": self._fmt(r.error_message, max_len=250),
                "inline": False,
            })

        payload = {
            "embeds": [
                {
                    "title": title,
                    #"description": description,
                    "color": color,
                    "fields": fields,
                }
            ]
        }

        return payload

    def _fmt(self, value: Any, max_len: int = 120) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.4f}"

        text = str(value).strip()
        if not text:
            return "-"

        if len(text) > max_len:
            return text[:max_len - 3] + "..."

        return text

    def format_runtime(self, seconds):
        return time.strftime("%H:%M:%S", time.gmtime(seconds))

    """DUMPED CONVERTING TO IMAGE"""
    def _send_image(self, result: NotiInfo, image_path: str):
        msg = "🎆" if result.success else "😫"

        with open(image_path, "rb") as f:
            response = requests.post(
                self.webhook_url,
                data={"content": msg},
                files={"file": ("sunfire_result.png", f, "image/png")},
                timeout=15
            )

        if response.status_code not in (200, 204):
            print("Discord notification failed:", response.status_code, response.text)