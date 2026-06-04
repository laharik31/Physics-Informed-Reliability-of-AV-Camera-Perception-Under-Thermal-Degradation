import os
import pandas as pd
import numpy as np

THRESHOLD = 0.060
TOTAL_TIME = 600.0
HEATER_START = 180.0

def calculate_metrics(df_surface):
    # Sort by time just in case
    df = df_surface.sort_values(by='t_s').reset_index(drop=True)
    times = df['t_s'].values
    maps = df['mAP50'].values

    mttf = np.inf
    recovery_time_abs = 0.0
    mttr = 0.0
    failed = False

    # 1. Calculate MTTF (first time it drops below threshold)
    for i in range(len(times) - 1):
        t1, t2 = times[i], times[i+1]
        m1, m2 = maps[i], maps[i+1]
        
        if m1 >= THRESHOLD and m2 < THRESHOLD:
            # Linearly interpolate exact failure time
            mttf = t1 + (t2 - t1) * (THRESHOLD - m1) / (m2 - m1)
            failed = True
            break
        elif m1 < THRESHOLD:
            # It was already below threshold at t=0 (shouldn't happen, but just in case)
            mttf = t1
            failed = True
            break

    # 2. Calculate Absolute Recovery Time (first time it goes back above threshold AFTER heater starts)
    if failed:
        for i in range(len(times) - 1):
            t1, t2 = times[i], times[i+1]
            m1, m2 = maps[i], maps[i+1]
            
            # We only look for recovery during the heating phase (t >= 180)
            if t1 >= HEATER_START and m1 < THRESHOLD and m2 >= THRESHOLD:
                recovery_time_abs = t1 + (t2 - t1) * (THRESHOLD - m1) / (m2 - m1)
                # MTTR is the time it took to recover *after* the heater was turned on
                mttr = recovery_time_abs - HEATER_START
                break
            elif t1 >= HEATER_START and m1 >= THRESHOLD:
                # Recovered exactly at the start of the interval (or never dropped)
                recovery_time_abs = t1
                mttr = recovery_time_abs - HEATER_START
                break

    # 3. Calculate Availability
    if not failed:
        availability = 1.0
        mttf_str = "∞"
        mttr_str = "0.0"
    else:
        downtime = recovery_time_abs - mttf
        uptime = TOTAL_TIME - downtime
        availability = uptime / TOTAL_TIME
        mttf_str = f"{mttf:.1f}"
        mttr_str = f"{mttr:.1f}"

    return mttf_str, mttr_str, availability * 100

def main():
    experiments = [
        {"file": "results/bdd_results_uniform_rh80.csv", "mode": "Uniform", "rh": 80},
        {"file": "results/bdd_results_patchy_rh80.csv", "mode": "Patchy", "rh": 80},
        {"file": "results/bdd_results_patchy_rh90.csv", "mode": "Patchy", "rh": 90},
    ]

    out_data = []

    print(f"\\n{'='*80}")
    print(f" RAMS 2027 LAYER 3: RELIABILITY METRICS (Threshold mAP@50 = {THRESHOLD})")
    print(f"{'='*80}")
    
    # Markdown table header
    md_table = "| Mode | RH (%) | Surface Coating | MTTF (s) | MTTR (s) | Availability (%) |\\n"
    md_table += "|------|--------|-----------------|----------|----------|------------------|\\n"

    for exp in experiments:
        if not os.path.exists(exp["file"]):
            print(f"Warning: {exp['file']} not found. Skipping.")
            continue
            
        df = pd.read_csv(exp["file"])
        surfaces = df['surface'].unique()
        
        for surface in surfaces:
            df_surface = df[df['surface'] == surface]
            mttf, mttr, avail = calculate_metrics(df_surface)
            
            out_data.append({
                "Mode": exp["mode"],
                "RH(%)": exp["rh"],
                "Surface": surface,
                "MTTF(s)": mttf,
                "MTTR(s)": mttr,
                "Availability(%)": f"{avail:.1f}"
            })
            
            md_table += f"| {exp['mode']} | {exp['rh']} | {surface} | {mttf} | {mttr} | {avail:.1f}% |\\n"

    print(md_table)

    # Save to CSV
    out_df = pd.DataFrame(out_data)
    out_csv = "results/reliability_metrics.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\\n✅ Metrics saved to {out_csv}")

if __name__ == "__main__":
    main()
