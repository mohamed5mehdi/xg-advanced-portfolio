import pandas as pd
from statsbombpy import sb
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_events_for_competitions(competitions_dict, out_dir):
    """
    Fetch events for a list of competitions and seasons, saving each to a Parquet checkpoint.
    competitions_dict format: {competition_id: season_id}
    """
    os.makedirs(out_dir, exist_ok=True)
    summary = {}
    
    for comp_id, season_id in competitions_dict.items():
        checkpoint_path = f"{out_dir}/comp{comp_id}_season{season_id}.parquet"
        if os.path.exists(checkpoint_path):
            logging.info(f"Already collected: {checkpoint_path}, skipping.")
            # Still record in summary to show complete status, assuming 100% since it exists
            # We don't have the exact failed matches but we can just note it as skipped.
            summary[comp_id] = "Skipped (already exists)"
            continue
            
        logging.info(f"Fetching matches for comp_id={comp_id}, season_id={season_id}...")
        try:
            matches = sb.matches(competition_id=comp_id, season_id=season_id)
            match_ids = matches['match_id'].tolist()
            comp_events = []
            failed = []
            
            for mid in match_ids:
                try:
                    events = sb.events(match_id=mid)
                    events['competition_id'] = comp_id
                    events['season_id'] = season_id
                    comp_events.append(events)
                except Exception as e:
                    failed.append(mid)
                    logging.warning(f"Failed match_id={mid}: {e}")
            
            if comp_events:
                pd.concat(comp_events, ignore_index=True).to_parquet(checkpoint_path, index=False)
            
            summary_str = f"{len(match_ids) - len(failed)}/{len(match_ids)} matches OK, {len(failed)} failed"
            if failed:
                summary_str += f" -> {failed}"
            summary[comp_id] = summary_str
            logging.info(f"Comp {comp_id}: {summary_str}")
            
        except Exception as e:
            logging.error(f"Failed to fetch matches for comp_id={comp_id}, season_id={season_id}: {e}")
            summary[comp_id] = f"Failed to fetch matches: {e}"
            
    return summary

def main():
    # Train leagues (2015/16)
    train_leagues = {
        11: 27, # La Liga
        2: 27,  # Premier League
        9: 27,  # Bundesliga
        12: 27  # Serie A
    }
    
    # Test league (2015/16)
    test_league = {
        7: 27 # Ligue 1
    }
    
    logging.info("Starting data collection for Train leagues...")
    train_summary = fetch_events_for_competitions(train_leagues, out_dir='data/raw/train')
    
    logging.info("Starting data collection for Test league...")
    test_summary = fetch_events_for_competitions(test_league, out_dir='data/raw/test')
    
    logging.info("=== FINAL COLLECTION SUMMARY ===")
    logging.info("Train Leagues:")
    for comp_id, status in train_summary.items():
        logging.info(f"  - Comp {comp_id}: {status}")
        
    logging.info("Test League:")
    for comp_id, status in test_summary.items():
        logging.info(f"  - Comp {comp_id}: {status}")

if __name__ == "__main__":
    main()
