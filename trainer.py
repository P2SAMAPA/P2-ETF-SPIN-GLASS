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
        print(f"\n=== Universe: {universe_name} (Spin Glass multi‑window) ===")
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty or len(returns) < max(config.WINDOWS) + 2:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        # Store best local field per ETF across windows
        best_field = {ticker: -np.inf for ticker in tickers}
        best_window = {ticker: None for ticker in tickers}

        for win in config.WINDOWS:
            print(f"  Rolling window: {win} days")
            if len(returns) < win + 1:
                continue
            sg = SpinGlass(returns, window=win)
            result = sg.run()
            local_fields = result["local_fields"]
            assets = result["assets"]
            for i, ticker in enumerate(assets):
                field = local_fields[i]
                if field > best_field[ticker]:
                    best_field[ticker] = field
                    best_window[ticker] = win

        # Build list of ETFs with their best field and window
        etf_scores = [{"ticker": t, "local_field": best_field[t], "window": best_window[t]} for t in tickers if best_window[t] is not None]
        if not etf_scores:
            print("  No valid windows")
            all_results[universe_name] = {"top_etfs": []}
            continue

        # Sort by local field descending
        etf_scores.sort(key=lambda x: x["local_field"], reverse=True)
        top_etfs = []
        full_scores = {}
        for item in etf_scores:
            full_scores[item["ticker"]] = {
                "local_field": item["local_field"],
                "window": item["window"]
            }
        for item in etf_scores[:config.TOP_N]:
            top_etfs.append({
                "ticker": item["ticker"],
                "local_field": float(item["local_field"]),
                "window": item["window"]
            })

        print(f"  Top 3 ETFs by best local field across windows:")
        for etf in top_etfs:
            print(f"    {etf['ticker']}: field={etf['local_field']:.4f} (window={etf['window']}d)")

        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "full_scores": full_scores,
            "run_date": today
        }

    # Save results
    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/spin_glass_{today}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": today, "universes": all_results}, f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Spin Glass / Ising Market Model (multi‑window) complete ===")

if __name__ == "__main__":
    main()
