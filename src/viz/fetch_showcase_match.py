import pandas as pd
from statsbombpy import sb
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def fetch_showcase_match(match_id, competition_id, season_id, out_path):
    """
    Collecte un match isolé hors corpus d'entraînement (showcase out-of-distribution).
    Ne pas utiliser fetch_events_for_competitions() : celle-ci fetch toute une
    compétition/saison, pas un match unique.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    logging.info(f"Fetching match_id={match_id} (comp={competition_id}, season={season_id})...")
    events = sb.events(match_id=match_id)
    events['competition_id'] = competition_id
    events['season_id'] = season_id
    events.to_parquet(out_path, index=False)
    logging.info(f"Saved {len(events)} events -> {out_path}")
    return events


if __name__ == "__main__":
    fetch_showcase_match(
        match_id=18243,
        competition_id=16,
        season_id=27,
        out_path="data/raw/showcase/comp16_match18243.parquet",
    )
