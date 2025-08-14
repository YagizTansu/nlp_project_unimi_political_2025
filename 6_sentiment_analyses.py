import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Basic column presence check
    required = {"Author", "party", "predicted_emotions"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing column(s): {missing}")
    return df

def explode_emotions(df: pd.DataFrame) -> pd.DataFrame:
    # predicted_emotions may contain comma-separated values; explode to long form
    df = df.copy()
    df["predicted_emotions"] = (
        df["predicted_emotions"]
        .astype(str)
        .str.split(",")
    )
    df = df.explode("predicted_emotions")
    df["predicted_emotions"] = (
        df["predicted_emotions"]
        .str.strip()
        .str.capitalize()
    )
    df = df[df["predicted_emotions"] != ""]
    return df

def count_party_emotions(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df.groupby(["party", "predicted_emotions"])
          .size()
          .reset_index(name="count")
    )
    return counts

def summarize_top3(counts: pd.DataFrame):
    top3 = (
        counts.sort_values(["party", "count"], ascending=[True, False])
              .groupby("party")
              .head(3)
    )
    print("\nTop 3 emotions per party:")
    for party, grp in top3.groupby("party"):
        parts = [f"{r.predicted_emotions}={int(r['count'])}" for _, r in grp.iterrows()]
        print(f"  {party}: " + ", ".join(parts))

def plot_counts(counts: pd.DataFrame, output_dir: Path, index: int, label: str):
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=counts,
        x="party",
        y="count",
        hue="predicted_emotions",
        edgecolor="black"
    )
    plt.title("Emotion Distribution by Party")
    plt.ylabel("Tweet Count")
    plt.xlabel("Party")
    plt.legend(title="Emotion", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_label = label.lower().replace(" ", "_")
    out_file = output_dir / f"{index}_{safe_label}.png"
    plt.savefig(out_file, dpi=300)
    print(f"Figure saved to: {out_file}")
    plt.close()

# --- NEW: Party-Topic-Emotion helpers ---
def count_party_topic_emotions(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["topic", "party", "predicted_emotions"])
          .size()
          .reset_index(name="count")
    )

def summarize_topic_party_top_emotion(counts_pt: pd.DataFrame, topics):
    print("\nMost frequent emotion per party within each topic:")
    for topic in topics:
        sub = counts_pt[counts_pt["topic"] == topic]
        if sub.empty:
            continue
        print(f"\nTopic: {topic}")
        for party, g in sub.groupby("party"):
            top = g.sort_values("count", ascending=False).head(1)
            r = top.iloc[0]
            print(f"  {party}: {r.predicted_emotions}={int(r['count'])}")

def plot_party_topic_for_topic(counts_pt: pd.DataFrame, topic: str, output_dir: Path, index: int):
    subset = counts_pt[counts_pt["topic"] == topic]
    if subset.empty:
        return
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=subset,
        x="party",
        y="count",
        hue="predicted_emotions",
        edgecolor="black"
    )
    plt.title(f"Topic: {topic} | Emotion Distribution by Party")
    plt.ylabel("Tweet Count")
    plt.xlabel("Party")
    plt.legend(title="Emotion", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    safe_topic = str(topic).lower().replace(" ", "_")
    out_file = output_dir / f"{index}_party_topic_{safe_topic}.png"
    plt.savefig(out_file, dpi=300)
    print(f"Figure saved to: {out_file}")
    plt.close()

# --- NEW: Political Side emotion helpers ---
def count_side_emotions(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["political_side", "predicted_emotions"])
          .size()
          .reset_index(name="count")
    )

def summarize_side_top(counts_side: pd.DataFrame, top_n: int = 3):
    print("\nTop emotions by political side:")
    top = (
        counts_side.sort_values(["political_side", "count"], ascending=[True, False])
                   .groupby("political_side")
                   .head(top_n)
    )
    for side, grp in top.groupby("political_side"):
        parts = [f"{r.predicted_emotions}={int(r['count'])}" for _, r in grp.iterrows()]
        print(f"  {side}: " + ", ".join(parts))

def plot_side_counts(counts_side: pd.DataFrame, output_dir: Path, index: int, label: str):
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=counts_side,
        x="political_side",
        y="count",
        hue="predicted_emotions",
        edgecolor="black"
    )
    plt.title("Emotion Distribution by Political Side")
    plt.ylabel("Tweet Count")
    plt.xlabel("Political Side")
    plt.legend(title="Emotion", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_label = label.lower().replace(" ", "_")
    out_file = output_dir / f"{index}_{safe_label}.png"
    plt.savefig(out_file, dpi=300)
    print(f"Figure saved to: {out_file}")
    plt.close()

# --- NEW: Topic-level emotion pie charts ---
def plot_topic_emotion_pies(df_long: pd.DataFrame, output_dir: Path, start_index: int,
                            topics: list[str] | None = None, max_topics: int = 4) -> int:
    if "topic" not in df_long.columns:
        print("\nTopic column not found; skipping topic emotion pie charts.")
        return start_index
    if topics:
        selected = [t for t in topics if t in df_long["topic"].unique()]
    else:
        selected = (
            df_long.groupby("topic")["predicted_emotions"]
                   .count()
                   .sort_values(ascending=False)
                   .head(max_topics)
                   .index.tolist()
        )
    if not selected:
        print("\nNo topics selected for pie charts.")
        return start_index
    output_dir.mkdir(parents=True, exist_ok=True)
    for topic in selected:
        sub = (df_long[df_long["topic"] == topic]
               .groupby("predicted_emotions")
               .size()
               .reset_index(name="count")
               .sort_values("count", ascending=False))
        if sub.empty:
            continue
        # Küçük dilimleri birleştirme (opsiyonel)
        total = sub["count"].sum()
        sub["share"] = sub["count"] / total
        small = sub[sub["share"] < 0.02]
        if len(small) > 1:
            merged = pd.DataFrame({
                "predicted_emotions": ["Other"],
                "count": [small["count"].sum()],
                "share": [small["share"].sum()]
            })
            sub = pd.concat([sub[sub["share"] >= 0.02], merged], ignore_index=True)
        plt.figure(figsize=(6, 6))
        plt.pie(
            sub["count"],
            labels=sub["predicted_emotions"],
            autopct=lambda p: f"{p:.1f}%\n({int(round(p*total/100))})" if p >= 5 else "",
            startangle=90,
            counterclock=False
        )
        plt.title(f"Topic: {topic} | Emotion Distribution")
        plt.tight_layout()
        safe_topic = str(topic).lower().replace(" ", "_")
        out_file = output_dir / f"{start_index}_topic_emotions_{safe_topic}.png"
        plt.savefig(out_file, dpi=300)
        print(f"Figure saved to: {out_file}")
        plt.close()
        start_index += 1
    return start_index

# --- NEW: Selected emotion time series (Fear, Anger) ---
SELECTED_EMOTIONS = ["Fear", "Anger"]

def detect_datetime_column(df: pd.DataFrame):
    candidates = ["date", "created_at", "createdAt", "timestamp", "Datetime", "Date"]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def prepare_time_series(df_long: pd.DataFrame) -> pd.DataFrame | None:
    col = detect_datetime_column(df_long)
    if not col:
        print("\nNo datetime column detected; skipping time series.")
        return None
    ts = df_long.copy()
    ts[col] = pd.to_datetime(ts[col], errors="coerce")
    ts = ts.dropna(subset=[col])
    ts = ts[ts["predicted_emotions"].isin(SELECTED_EMOTIONS)]
    if ts.empty:
        print("\nNo data for selected emotions in time series.")
        return None
    span_days = (ts[col].max() - ts[col].min()).days
    ts["ts_period"] = ts[col].dt.to_period("M").dt.to_timestamp() if span_days > 90 else ts[col].dt.date
    grouped = (
        ts.groupby(["ts_period", "predicted_emotions"])
          .size()
          .reset_index(name="count")
    )
    return grouped

def plot_time_series(grouped: pd.DataFrame, output_dir: Path, index: int) -> int:
    if grouped is None or grouped.empty:
        return index
    pivoted = grouped.pivot(index="ts_period", columns="predicted_emotions", values="count").fillna(0)
    # Preserve SELECTED_EMOTIONS order where present
    cols = [c for c in SELECTED_EMOTIONS if c in pivoted.columns]
    pivoted = pivoted[cols]
    plt.figure(figsize=(12, 5))
    for emotion in pivoted.columns:
        plt.plot(pivoted.index, pivoted[emotion], marker="o", linewidth=2, label=emotion)
    plt.title("Time Series of Selected Emotions (Fear / Anger / Pride)")
    plt.ylabel("Tweet Count")
    plt.xlabel("Time")
    plt.legend(title="Emotion")
    plt.xticks(rotation=30)
    plt.tight_layout()
    out_file = output_dir / f"{index}_time_series_selected_emotions.png"
    plt.savefig(out_file, dpi=300)
    print(f"Figure saved to: {out_file}")
    plt.close()
    return index + 1

# --- NEW: Political Side time series for selected emotions ---
def prepare_time_series_by_political_side(df_long: pd.DataFrame) -> pd.DataFrame | None:
    col = detect_datetime_column(df_long)
    if not col:
        print("\nNo datetime column detected; skipping political side time series.")
        return None
    
    if "political_side" not in df_long.columns:
        print("\nColumn 'political_side' not found; skipping political side time series.")
        return None
    
    ts = df_long.copy()
    ts[col] = pd.to_datetime(ts[col], errors="coerce")
    ts = ts.dropna(subset=[col])
    ts = ts[ts["predicted_emotions"].isin(SELECTED_EMOTIONS)]
    
    if ts.empty:
        print("\nNo data for selected emotions in political side time series.")
        return None
    
    span_days = (ts[col].max() - ts[col].min()).days
    ts["ts_period"] = ts[col].dt.to_period("M").dt.to_timestamp() if span_days > 90 else ts[col].dt.date
    
    grouped = (
        ts.groupby(["ts_period", "political_side", "predicted_emotions"])
          .size()
          .reset_index(name="count")
    )
    return grouped

def plot_time_series_by_political_side(grouped: pd.DataFrame, output_dir: Path, index: int) -> int:
    if grouped is None or grouped.empty:
        return index
    
    # Her political side için ayrı subplot oluştur
    sides = grouped["political_side"].unique()
    emotions = [e for e in SELECTED_EMOTIONS if e in grouped["predicted_emotions"].unique()]
    
    if len(sides) == 0 or len(emotions) == 0:
        return index
    
    fig, axes = plt.subplots(len(sides), 1, figsize=(12, 4 * len(sides)), sharex=True)
    if len(sides) == 1:
        axes = [axes]
    
    for i, side in enumerate(sides):
        side_data = grouped[grouped["political_side"] == side]
        pivoted = side_data.pivot(index="ts_period", columns="predicted_emotions", values="count").fillna(0)
        
        # Preserve emotion order
        cols = [c for c in emotions if c in pivoted.columns]
        if cols:
            pivoted = pivoted[cols]
            
            for emotion in pivoted.columns:
                axes[i].plot(pivoted.index, pivoted[emotion], marker="o", linewidth=2, label=emotion)
            
            axes[i].set_title(f"Political Side: {side} | Selected Emotions Over Time")
            axes[i].set_ylabel("Tweet Count")
            axes[i].legend(title="Emotion")
            axes[i].tick_params(axis='x', rotation=30)
    
    plt.xlabel("Time")
    plt.suptitle("Time Series of Selected Emotions by Political Side", y=0.98)
    plt.tight_layout()
    
    out_file = output_dir / f"{index}_time_series_emotions_by_political_side.png"
    plt.savefig(out_file, dpi=300)
    print(f"Figure saved to: {out_file}")
    plt.close()
    return index + 1

# --- NEW: Party time series for selected emotions ---
def prepare_time_series_by_party(df_long: pd.DataFrame) -> pd.DataFrame | None:
    col = detect_datetime_column(df_long)
    if not col:
        print("\nNo datetime column detected; skipping party time series.")
        return None
    
    if "party" not in df_long.columns:
        print("\nColumn 'party' not found; skipping party time series.")
        return None
    
    ts = df_long.copy()
    ts[col] = pd.to_datetime(ts[col], errors="coerce")
    ts = ts.dropna(subset=[col])
    ts = ts[ts["predicted_emotions"].isin(SELECTED_EMOTIONS)]
    
    if ts.empty:
        print("\nNo data for selected emotions in party time series.")
        return None
    
    span_days = (ts[col].max() - ts[col].min()).days
    ts["ts_period"] = ts[col].dt.to_period("M").dt.to_timestamp() if span_days > 90 else ts[col].dt.date
    
    grouped = (
        ts.groupby(["ts_period", "party", "predicted_emotions"])
          .size()
          .reset_index(name="count")
    )
    return grouped

def plot_time_series_by_party(grouped: pd.DataFrame, output_dir: Path, index: int) -> int:
    if grouped is None or grouped.empty:
        return index
    
    # Her party için ayrı subplot oluştur
    parties = grouped["party"].unique()
    emotions = [e for e in SELECTED_EMOTIONS if e in grouped["predicted_emotions"].unique()]
    
    if len(parties) == 0 or len(emotions) == 0:
        return index
    
    fig, axes = plt.subplots(len(parties), 1, figsize=(12, 4 * len(parties)), sharex=True)
    if len(parties) == 1:
        axes = [axes]
    
    for i, party in enumerate(parties):
        party_data = grouped[grouped["party"] == party]
        pivoted = party_data.pivot(index="ts_period", columns="predicted_emotions", values="count").fillna(0)
        
        # Preserve emotion order
        cols = [c for c in emotions if c in pivoted.columns]
        if cols:
            pivoted = pivoted[cols]
            
            for emotion in pivoted.columns:
                axes[i].plot(pivoted.index, pivoted[emotion], marker="o", linewidth=2, label=emotion)
            
            axes[i].set_title(f"Party: {party} | Selected Emotions Over Time")
            axes[i].set_ylabel("Tweet Count")
            axes[i].legend(title="Emotion")
            axes[i].tick_params(axis='x', rotation=30)
    
    plt.xlabel("Time")
    plt.suptitle("Time Series of Selected Emotions by Party", y=0.98)
    plt.tight_layout()
    
    out_file = output_dir / f"{index}_time_series_emotions_by_party.png"
    plt.savefig(out_file, dpi=300)
    print(f"Figure saved to: {out_file}")
    plt.close()
    return index + 1

def main():
    # ---- Configuration ----
    CSV_PATH = "/home/yagiz/Desktop/nlp_project/3_tweets_with_emotions/all_cleaned_tweets_with_topics_and_emotions.csv"
    OUTPUT_DIR = Path("4_sentiment_plots")
    plot_idx = 1  # sıralı dosya isimleri

    df = load_data(CSV_PATH)
    df_long = explode_emotions(df)
    counts = count_party_emotions(df_long)
    if counts.empty:
        print("Warning: no emotion data found.")
        return
    summarize_top3(counts)
    plot_counts(counts, OUTPUT_DIR, plot_idx, "party_emotions_bar")
    plot_idx += 1

    # --- Second run: exclude specific topics ---
    EXCLUDE_TOPICS = ["condolence", "congratulation"]
    if "topic" in df_long.columns:
        df_filtered = df_long[~df_long["topic"].isin(EXCLUDE_TOPICS)]
        counts_filtered = count_party_emotions(df_filtered)
        if not counts_filtered.empty:
            print(f"\nExcluded topics: {', '.join(EXCLUDE_TOPICS)}")
            summarize_top3(counts_filtered)
            plot_counts(
                counts_filtered,
                OUTPUT_DIR,
                plot_idx,
                "party_emotions_bar_excluded_condolence_congratulation"
            )
            plot_idx += 1
        else:
            print("\nAll data removed after excluding topics; skipping second plot.")
    else:
        print("\nColumn 'topic' not found; skipping filtered second plot.")

    # --- NEW: Party & Topic emotion intersection ---
    if "topic" in df_long.columns:
        counts_pt = count_party_topic_emotions(df_long)
        if not counts_pt.empty:
            # Konu seçimi: buraya istediğin konuları yaz (ör: ["economy", "security"])
            FOCUS_TOPICS = ["economy","migration" , "education" , "justice"]  # "x" yerine ilgilendiğin konuyu ekle
            existing_focus = [t for t in FOCUS_TOPICS if t in counts_pt["topic"].unique()]
            if not existing_focus:
                existing_focus = (
                    counts_pt.groupby("topic")["count"].sum()
                             .sort_values(ascending=False)
                             .head(3)
                             .index.tolist()
                )
            summarize_topic_party_top_emotion(counts_pt, existing_focus)
            for topic in existing_focus:
                plot_party_topic_for_topic(counts_pt, topic, OUTPUT_DIR, plot_idx)
                plot_idx += 1
            # --- NEW: topic-level emotion pie charts (same focus topics; fallback handled inside) ---
            plot_idx = plot_topic_emotion_pies(df_long, OUTPUT_DIR, plot_idx, topics=existing_focus)
        else:
            print("\nNo data for party-topic-emotion intersection.")
    else:
        print("\nColumn 'topic' not found; skipping party-topic-emotion intersection and pies.")

    # --- NEW: Political Side emotion analysis ---
    if "political_side" in df_long.columns:
        counts_side = count_side_emotions(df_long)
        if not counts_side.empty:
            summarize_side_top(counts_side, top_n=3)
            plot_side_counts(counts_side, OUTPUT_DIR, plot_idx, "political_side_emotions_bar")
            plot_idx += 1
        else:
            print("\nPolitical Side data empty.")
    else:
        print("\nColumn 'political_side' not found; skipping political side emotion analysis.")

    # --- NEW: Time series analysis for selected emotions ---
    ts_grouped = prepare_time_series(df_long)
    plot_idx = plot_time_series(ts_grouped, OUTPUT_DIR, plot_idx)
    
    # --- NEW: Political Side time series for selected emotions ---
    ts_side_grouped = prepare_time_series_by_political_side(df_long)
    plot_idx = plot_time_series_by_political_side(ts_side_grouped, OUTPUT_DIR, plot_idx)
    
    # --- NEW: Party time series for selected emotions ---
    ts_party_grouped = prepare_time_series_by_party(df_long)
    plot_idx = plot_time_series_by_party(ts_party_grouped, OUTPUT_DIR, plot_idx)

if __name__ == "__main__":
    main()