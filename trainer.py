import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import config
import data_manager
from spin_glass import SpinGlass

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} (Spin Glass) ===")
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty or len(returns) < config.ROLLING_WINDOW + 2:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        sg = SpinGlass(returns, window=config.ROLLING_WINDOW)
        result = sg.run()
        # Get local fields and corresponding assets
        local_fields = result["local_fields"]
        assets = result["assets"]
        # Sort by local field descending (high = aligned with metastable state)
        sorted_idx = np.argsort(local_fields)[::-1]
        top_etfs = []
        full_scores = {}
        for i, idx in enumerate(sorted_idx):
            ticker = assets[idx]
            field = local_fields[idx]
            full_scores[ticker] = field
            if i < config.TOP_N:
                top_etfs.append({
                    "ticker": ticker,
                    "local_field": float(field)
                })
        print(f"  Top 3 ETFs by local field: {[e['ticker'] for e in top_etfs]}")
        print(f"  Magnetisation (consensus): {result['magnetisation']:.3f}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "full_scores": full_scores,
            "magnetisation": result["magnetisation"],
            "energy": result["energy"],
            "effective_temperature": result["effective_temperature"],
            "run_date": today
        }

    # Save results
    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/spin_glass_{today}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": today, "universes": all_results}, f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Spin Glass / Ising Market Model complete ===")

if __name__ == "__main__":
    main()
