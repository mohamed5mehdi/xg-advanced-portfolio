import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_distance(x, y, goal_x=120, goal_y=40):
    return np.sqrt((x - goal_x)**2 + (y - goal_y)**2)

def calculate_angle(x, y, goal_x=120):
    # Goal posts are at (120, 36) and (120, 44)
    post1_y = 36
    post2_y = 44
    
    v1_x, v1_y = goal_x - x, post1_y - y
    v2_x, v2_y = goal_x - x, post2_y - y
    
    dot = v1_x * v2_x + v1_y * v2_y
    mag1 = np.sqrt(v1_x**2 + v1_y**2)
    mag2 = np.sqrt(v2_x**2 + v2_y**2)
    
    mag_prod = mag1 * mag2
    if mag_prod == 0:
        return 0
        
    cos_angle = dot / mag_prod
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad)

def extract_freeze_frame_features(freeze_frame, shot_x, shot_y):
    # Check if freeze_frame is valid using duck-typing to avoid Parquet list->ndarray issues
    if not hasattr(freeze_frame, '__iter__') or isinstance(freeze_frame, str):
        return np.nan, np.nan, np.nan, None  # Missing data, not fabricated values; None = no freeze_frame

    defenders_in_cone = 0
    gk_x, gk_y = 120, 40  # Placeholder — overwritten below si un vrai gardien est trouve, jamais
    # consomme dans le calcul final si gk_found reste False (voir branche NaN plus bas).
    gk_found = False

    def point_in_triangle(pt, v1, v2, v3):
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        d1 = sign(pt, v1, v2)
        d2 = sign(pt, v2, v3)
        d3 = sign(pt, v3, v1)
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (has_neg and has_pos)

    for player in freeze_frame:
        try:
            # Depending on how pyarrow parses dicts inside arrays
            if hasattr(player, 'keys'):
                loc = player.get('location')
                teammate = player.get('teammate')
                
                # 'position' might be a dict or a string depending on serialization
                pos = player.get('position')
                if isinstance(pos, dict):
                    position_name = pos.get('name', '')
                else:
                    position_name = str(pos)
                    
            elif isinstance(player, tuple):  # NamedTuple or similar from pyarrow
                # Extremely defensive fallback if pyarrow converts to tuples
                # We assume we can't parse it easily, continue
                continue
            else:
                continue
                
            if loc is None or not hasattr(loc, '__iter__'):
                continue
                
            if not teammate:
                p_x, p_y = loc[0], loc[1]
                
                if 'Goalkeeper' in position_name:
                    gk_x, gk_y = p_x, p_y
                    gk_found = True
                else:
                    if point_in_triangle((p_x, p_y), (shot_x, shot_y), (120, 36), (120, 44)):
                        defenders_in_cone += 1
        except Exception:
            continue
            
    if gk_found:
        gk_dist = calculate_distance(shot_x, shot_y, gk_x, gk_y)

        v_goal_x, v_goal_y = 120 - shot_x, 40 - shot_y
        v_gk_x, v_gk_y = gk_x - shot_x, gk_y - shot_y

        mag_goal = np.sqrt(v_goal_x**2 + v_goal_y**2)
        mag_gk = np.sqrt(v_gk_x**2 + v_gk_y**2)

        if mag_goal == 0 or mag_gk == 0:
            gk_angle = 0
        else:
            dot = v_goal_x * v_gk_x + v_goal_y * v_gk_y
            cos_angle = np.clip(dot / (mag_goal * mag_gk), -1.0, 1.0)
            gk_angle = np.degrees(np.arccos(cos_angle))
    else:
        # Freeze frame present mais gardien non detecte -> NaN honnete plutot qu'un repli
        # fabrique sur (120,40). Concerne 32/37488 tirs (0.0854%). XGBoost gere nativement
        # les NaN (deja le cas pour les tirs sans freeze_frame du tout).
        gk_dist = np.nan
        gk_angle = np.nan

    return defenders_in_cone, gk_dist, gk_angle, gk_found

def process_events(df):
    logging.info(f"Processing {len(df)} events...")
    
    # Safely extract pass properties
    passes = df[df['type'] == 'Pass'].copy()
    pass_cols = ['id']
    for col in ['pass_technique', 'pass_height', 'pass_type']:
        if col in passes.columns:
            pass_cols.append(col)
            
    passes = passes[pass_cols].copy()
    passes.rename(columns={'id': 'shot_key_pass_id'}, inplace=True)
    
    shots = df[df['type'] == 'Shot'].copy()
    
    if 'shot_type' in shots.columns:
        shots = shots[shots['shot_type'] != 'Penalty']
        
    logging.info(f"Found {len(shots)} non-penalty shots.")
    
    # Élimination dynamique des collisions de colonnes avant le merge
    collision_cols = [c for c in passes.columns if c != 'shot_key_pass_id' and c in shots.columns]
    shots = shots.drop(columns=collision_cols)
    
    # Merge
    shots = shots.merge(passes, on='shot_key_pass_id', how='left')
    
    features = []
    no_freeze_frame_count = 0
    valid_ff_no_gk_count = 0
    
    for _, row in shots.iterrows():
        try:
            loc = row.get('location')
            if loc is None or not hasattr(loc, '__iter__') or isinstance(loc, str):
                continue
                
            shot_x, shot_y = loc[0], loc[1]
            dist = calculate_distance(shot_x, shot_y)
            angle = calculate_angle(shot_x, shot_y)
            
            ff = row.get('shot_freeze_frame')
            def_in_cone, gk_dist, gk_angle, gk_found = extract_freeze_frame_features(ff, shot_x, shot_y)
            
            if gk_found is None:
                no_freeze_frame_count += 1
            elif not gk_found:
                valid_ff_no_gk_count += 1
            
            is_goal = 1 if row.get('shot_outcome') == 'Goal' else 0
            
            # Extract features safely
            pass_tech = row.get('pass_technique')
            pass_tech = str(pass_tech) if pd.notna(pass_tech) else 'None'
            pass_height = row.get('pass_height')
            pass_height = str(pass_height) if pd.notna(pass_height) else 'None'
            pass_type = row.get('pass_type')
            pass_type = str(pass_type) if pd.notna(pass_type) else 'None'
            shot_body_part = str(row.get('shot_body_part', 'None')) if 'shot_body_part' in row else 'None'
            
            f_dict = {
                'match_id': row['match_id'],
                'competition_id': row['competition_id'],
                'team': row.get('team'),              # métadonnée
                'player': row.get('player'),          # métadonnée
                'minute': row.get('minute'),          # métadonnée
                'x': shot_x,                          # métadonnée
                'y': shot_y,                          # métadonnée
                'shot_outcome': row.get('shot_outcome'),  # métadonnée
                'distance': dist,
                'angle': angle,
                'defenders_in_cone': def_in_cone,
                'gk_distance': gk_dist,
                'gk_angle': gk_angle,
                'gk_found': gk_found,  # Audit-only flag — False = freeze_frame valide mais gardien non détecté (fallback 120,40). Doit être exclu de X_train/X_test (voir train.py cols_to_drop).
                'under_pressure': 1 if row.get('under_pressure') == True else 0,
                'shot_technique': str(row.get('shot_technique', 'None')),
                'play_pattern': str(row.get('play_pattern', 'None')),
                'pass_technique': pass_tech,
                'pass_height': pass_height,
                'pass_type': pass_type,
                'shot_body_part': shot_body_part,
                'is_goal': is_goal,
                'statsbomb_xg': float(row.get('shot_statsbomb_xg', 0.0)) if 'shot_statsbomb_xg' in row and pd.notna(row['shot_statsbomb_xg']) else 0.0
            }
            features.append(f_dict)
            
        except Exception as e:
            logging.warning(f"Error processing shot in match {row.get('match_id')}: {e}")
            
    df_features = pd.DataFrame(features)
    logging.info(f"Generated {len(df_features)} features.")
    logging.info(f"DIAGNOSTIC (pre-Bundesliga-exclusion population) — Shots with NO freeze_frame (NaN forced): {no_freeze_frame_count}")
    logging.info(f"DIAGNOSTIC (pre-Bundesliga-exclusion population) — Shots with valid freeze_frame but NO Goalkeeper detected: {valid_ff_no_gk_count}")
    return df_features

def main():
    import glob
    os.makedirs('data/processed', exist_ok=True)
    
    # Train
    train_files = glob.glob('data/raw/train/*.parquet')
    train_dfs = []
    for f in train_files:
        logging.info(f"Reading {f}...")
        train_dfs.append(pd.read_parquet(f))
        
    if train_dfs:
        df_train_raw = pd.concat(train_dfs, ignore_index=True)
        df_train_features = process_events(df_train_raw)
        df_train_features.to_parquet('data/processed/train_features.parquet', index=False)
        logging.info(f"Saved Train features: {len(df_train_features)} shots.")
        
    # Test
    test_files = glob.glob('data/raw/test/*.parquet')
    test_dfs = []
    for f in test_files:
        logging.info(f"Reading {f}...")
        test_dfs.append(pd.read_parquet(f))
        
    if test_dfs:
        df_test_raw = pd.concat(test_dfs, ignore_index=True)
        df_test_features = process_events(df_test_raw)
        df_test_features.to_parquet('data/processed/test_features.parquet', index=False)
        logging.info(f"Saved Test features: {len(df_test_features)} shots.")

if __name__ == "__main__":
    main()