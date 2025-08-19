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
    # Calculate percentages
    counts_pct = counts.copy()
    total_counts = counts_pct.groupby('party')['count'].transform('sum')
    counts_pct['percentage'] = (counts_pct['count'] / total_counts) * 100
    
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=counts_pct,
        x="party",
        y="percentage",
        hue="predicted_emotions",
        edgecolor="black"
    )
    plt.title("Emotion Distribution by Party (Percentage)")
    plt.ylabel("Percentage (%)")
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
    
    # Calculate percentages
    subset_pct = subset.copy()
    total_counts = subset_pct.groupby('party')['count'].transform('sum')
    subset_pct['percentage'] = (subset_pct['count'] / total_counts) * 100
    
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=subset_pct,
        x="party",
        y="percentage",
        hue="predicted_emotions",
        edgecolor="black"
    )
    plt.title(f"Topic: {topic} | Emotion Distribution by Party (Percentage)")
    plt.ylabel("Percentage (%)")
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
    # Calculate percentages
    counts_pct = counts_side.copy()
    total_counts = counts_pct.groupby('political_side')['count'].transform('sum')
    counts_pct['percentage'] = (counts_pct['count'] / total_counts) * 100
    
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=counts_pct,
        x="political_side",
        y="percentage",
        hue="predicted_emotions",
        edgecolor="black"
    )
    plt.title("Emotion Distribution by Political Side (Percentage)")
    plt.ylabel("Percentage (%)")
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
def detect_datetime_column(df: pd.DataFrame):
    candidates = ["date", "created_at", "createdAt", "timestamp", "Datetime", "Date"]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def prepare_time_series(df_long: pd.DataFrame, selected_emotions: list[str]) -> pd.DataFrame | None:
    col = detect_datetime_column(df_long)
    if not col:
        print("\nNo datetime column detected; skipping time series.")
        return None
    ts = df_long.copy()
    ts[col] = pd.to_datetime(ts[col], errors="coerce")
    ts = ts.dropna(subset=[col])
    ts = ts[ts["predicted_emotions"].isin(selected_emotions)]
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

def plot_time_series(grouped: pd.DataFrame, output_dir: Path, index: int, selected_emotions: list[str]) -> int:
    if grouped is None or grouped.empty:
        return index
    pivoted = grouped.pivot(index="ts_period", columns="predicted_emotions", values="count").fillna(0)
    cols = [c for c in selected_emotions if c in pivoted.columns]
    pivoted = pivoted[cols]
    plt.figure(figsize=(12, 5))
    for emotion in pivoted.columns:
        plt.plot(pivoted.index, pivoted[emotion], marker="o", linewidth=2, label=emotion)
    plt.title("Time Series of Selected Emotions (Tweet Counts)")
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


# --- NEW: Election window (± months) time series by party ---
def prepare_time_series_by_party_election_window(
    df_long: pd.DataFrame,
    election_date: str = "2023-05-14",
    months_window: int = 3,
    emotions: list[str] | None = None,
    parties: list[str] | None = None,
    selected_emotions: list[str] | None = None,
) -> pd.DataFrame | None:
    """
    Build weekly emotion time series for parties within [election - window, election + window].
    """
    col = detect_datetime_column(df_long)
    if not col:
        print("\nNo datetime column detected; skipping election window party time series.")
        return None

    ts = df_long.copy()
    ts[col] = pd.to_datetime(ts[col], errors="coerce")
    ts = ts.dropna(subset=[col])

    # --- FIX: Normalize timezone to avoid tz-aware vs naive comparison ---
    if ts[col].dt.tz is not None:
        ts[col] = ts[col].dt.tz_convert(None)  # drop timezone -> make naive

    election_ts = pd.to_datetime(election_date)  # naive
    start = election_ts - pd.DateOffset(months=months_window)
    end = election_ts + pd.DateOffset(months=months_window)

    ts = ts[(ts[col] >= start) & (ts[col] <= end)]
    if ts.empty:
        print("\nNo data in specified election window.")
        return None

    # Emotion selection
    if emotions is not None and len(emotions) > 0:
        norm_emotions = {e.strip().capitalize() for e in emotions if isinstance(e, str)}
        available_emotions = set(ts["predicted_emotions"].unique())
        matched = norm_emotions & available_emotions
        if not matched:
            print(f"\nWarning: None of the specified ELECTION_EMOTIONS found in data: {sorted(norm_emotions)}")
            return None
        missing = norm_emotions - matched
        if missing:
            print(f"Info: Missing emotions ignored: {sorted(missing)}")
        ts = ts[ts["predicted_emotions"].isin(matched)]
        print(f"Applied ELECTION_EMOTIONS filter: {sorted(matched)}")
    elif "predicted_emotions" in ts.columns and selected_emotions:
        available = set(selected_emotions) & set(ts["predicted_emotions"].unique())
        if available:
            ts = ts[ts["predicted_emotions"].isin(available)]
            print(f"Fallback to SELECTED_EMOTIONS: {sorted(available)}")

    # Party selection
    print(f"\nFiltering for parties: {parties}")
    print(f"Available parties in data: {ts['party'].unique()}")
    if parties is not None and len(parties) > 0 and "party" in ts.columns:
        ts = ts[ts["party"].isin(parties)]

    if ts.empty:
        print("\nNo data after filtering for selected emotions/parties in election window.")
        return None

    ts["ts_period"] = ts[col].dt.to_period("W").dt.start_time

    grouped = (
        ts.groupby(["ts_period", "party", "predicted_emotions"])
          .size()
          .reset_index(name="count")
    )
    return grouped

def plot_time_series_by_party_election_window(
    grouped: pd.DataFrame,
    election_date: str,
    output_dir: Path,
    index: int
) -> int:
    if grouped is None or grouped.empty:
        return index

    parties = grouped["party"].unique()
    if len(parties) == 0:
        return index

    emotions = grouped["predicted_emotions"].unique().tolist()
    election_ts = pd.to_datetime(election_date)

    fig, axes = plt.subplots(len(parties), 1, figsize=(12, 4 * len(parties)), sharex=True)
    if len(parties) == 1:
        axes = [axes]

    for i, party in enumerate(parties):
        party_data = grouped[grouped["party"] == party]
        pivoted = party_data.pivot(index="ts_period", columns="predicted_emotions", values="count").fillna(0)

        cols = [c for c in emotions if c in pivoted.columns]
        pivoted = pivoted[cols]

        for emotion in pivoted.columns:
            axes[i].plot(pivoted.index, pivoted[emotion], marker="o", linewidth=2, label=emotion)

        # Election day marker
        axes[i].axvline(election_ts, color="red", linestyle="--", linewidth=1,
                        label="Election Day" if i == 0 else None)

        axes[i].set_title(f"Party: {party} | Emotions Around Election (Weekly Tweet Counts)")
        axes[i].set_ylabel("Tweet Count")
        axes[i].legend(title="Emotion")
        axes[i].tick_params(axis="x", rotation=30)

    plt.xlabel("Week Start")
    plt.suptitle("Selected Emotions by Party (± Election Window) - Tweet Counts", y=0.98)
    plt.tight_layout()
    out_file = output_dir / f"{index}_time_series_emotions_by_party_election_window.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300)
    print(f"Figure saved to: {out_file}")
    plt.close()
    return index + 1

def main():
    # ---- Configuration ----
    CSV_PATH = "/home/yagiz/Desktop/nlp_project/outputs/all_cleaned_tweets_with_topics_and_emotions.csv"
    OUTPUT_DIR = Path("visualizations")
    plot_idx = 1  # sıralı dosya isimleri
    # --- NEW CONFIG: Election window analysis (edit as needed) ---
    ELECTION_DATE = "2023-05-14"
    ELECTION_WINDOW_MONTHS = 2
    # Leave empty list [] to include all available; otherwise list e.g. ["PartyA","PartyB"]
    ELECTION_PARTIES: list[str] = ["MHP", "CHP","DEM"]  # örnek: ["AKP","CHP"]
    # Emotions to include (leave [] for auto / SELECTED_EMOTIONS fallback). Example: ["Fear","Anger","Joy"]
    ELECTION_EMOTIONS: list[str] = ["Anger" , "Happy"]  # örnek: ["Fear","Anger","Joy"]
    SELECTED_EMOTIONS: list[str] = ["Anger", "Happy"]  # main içinde tanımlandı

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
            FOCUS_TOPICS = ["economy","migration" ,"health","justice"]  # "x" yerine ilgilendiğin konuyu ekle
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
    ts_grouped = prepare_time_series(df_long, SELECTED_EMOTIONS)
    plot_idx = plot_time_series(ts_grouped, OUTPUT_DIR, plot_idx, SELECTED_EMOTIONS)

    # --- NEW CALL: Election date ± window (weekly) party time series ---
    election_grouped = prepare_time_series_by_party_election_window(
        df_long,
        election_date=ELECTION_DATE,
        months_window=ELECTION_WINDOW_MONTHS,
        emotions=ELECTION_EMOTIONS,
        parties=ELECTION_PARTIES,
        selected_emotions=SELECTED_EMOTIONS
    )
    plot_idx = plot_time_series_by_party_election_window(
        election_grouped,
        election_date=ELECTION_DATE,
        output_dir=OUTPUT_DIR,
        index=plot_idx
    )

if __name__ == "__main__":
    main()