from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen


@dataclass(frozen=True)
class StatsApiClient:
    base_url: str = "https://statsapi.mlb.com/api/v1"
    sleep_seconds: float = 0.15

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        query = f"?{urlencode(params or {})}" if params else ""
        url = f"{self.base_url}{path}{query}"
        with urlopen(url, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        return payload

    def schedule(self, start_date: date | str, end_date: date | str) -> dict[str, Any]:
        return self.get(
            "/schedule",
            {
                "sportId": 1,
                "startDate": str(start_date),
                "endDate": str(end_date),
                "hydrate": "probablePitcher,team",
            },
        )

    def boxscore(self, game_pk: int) -> dict[str, Any]:
        return self.get(f"/game/{game_pk}/boxscore")

