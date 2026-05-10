"""
SBR NFL Odds Scraper
====================
Scrapes raw NFL betting data from sportsbookreviewsonline.com.
Outputs raw data from the source plus a line_quality flag.

TABLE STRUCTURE (verified by direct page inspection):
  Each game = 2 consecutive rows: Visitor (V) then Home (H)
  Columns: Date | Rot | VH | Team | 1st | 2nd | 3rd | 4th | Final | Open | Close | ML | 2H

  V row: Open/Close = opening/closing TOTAL (over-under)
  H row: Open/Close = opening/closing SPREAD (unsigned magnitude)
  ML col: moneyline for that team (negative = favorite)

  Note: SBR shows spread as an unsigned magnitude. Sign is determined
  by the moneyline and applied in preprocessing, not here.

Output columns:
  season        - e.g. "2007-08"
  date          - MMDD from SBR (e.g. 906 = Sept 6)
  home_team     - SBR short team name
  visitor       - SBR short team name
  score_home    - final score, home team
  score_visitor - final score, visitor team
  open_total    - opening over/under (from visitor row)
  close_total   - closing over/under (from visitor row)
  open_spread   - opening spread magnitude (from home row, unsigned)
  close_spread  - closing spread magnitude (from home row, unsigned)
  ml_home       - home team moneyline
  ml_visitor    - visitor moneyline
  line_quality  - data quality flag:
                    0 = reliable
                    1 = closing values swapped (close_spread looks like total)
                    2 = opening values swapped only (open_spread looks like total)
                    3 = opening total recorded as 0 due to PK on visitor row
                    4 = extreme moneyline value (|ML| > 2000)

Usage:
  pip install requests beautifulsoup4 pandas
  python scrape_sbr_nfl.py
"""

import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

SEASONS = [
    "2007-08", "2008-09", "2009-10", "2010-11", "2011-12",
    "2012-13", "2013-14", "2014-15", "2015-16", "2016-17",
    "2017-18", "2018-19", "2019-20", "2020-21", "2021-22",
]

BASE_URL = "https://sportsbookreviewsonline.com/scoresoddsarchives/nfl-odds-{season}/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Thresholds for line_quality flag
SPREAD_MAX = 20    # spreads beyond this are suspicious
ML_EXTREME = 2000  # moneylines beyond this magnitude are extreme


def fetch_page(season: str):
    url = BASE_URL.format(season=season)
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        return BeautifulSoup(r.text, "html.parser")
    except requests.RequestException as e:
        print(f"  ERROR fetching {season}: {e}")
        return None


def parse_raw_value(val: str):
    """
    Parse a raw numeric value from SBR exactly as shown on the page.
    - "pk" / "PK" -> 0.0 (pick'em)
    - "½" suffix  -> .5
    - Empty string -> None
    Returns float or None. Does NOT apply sign or interpret meaning.
    """
    if not val:
        return None
    val = val.strip().upper().replace("\u00bd", ".5").replace("½", ".5")
    if val in ("PK", "PKS", "EVEN", "EV", ""):
        return 0.0
    try:
        return float(val)
    except ValueError:
        return None


def compute_line_quality(open_spread, close_spread, open_total, close_total,
                         ml_home, ml_visitor) -> int:
    """
    Assign a line_quality flag based on data anomalies.

    0 = reliable
    1 = closing values swapped (close_spread looks like a total >= 20,
        and/or close_total looks like a spread <= 20). Closing line unreliable.
        Note: may co-occur with open swap — flagged as 1 regardless.
    2 = only opening values swapped (open_spread > 20, open_total <= 20).
        Closing line is still reliable.
    3 = open_total recorded as 0 because visitor row showed PK.
        Opening total is unknown; closing total may still be valid.
    4 = extreme moneyline (|ML| > 2000). Spread may still be valid.
    """
    def is_spread_like(v):
        return v is not None and abs(v) <= SPREAD_MAX

    def is_total_like(v):
        return v is not None and abs(v) > SPREAD_MAX

    open_swap  = is_total_like(open_spread)  and is_spread_like(open_total)
    close_swap = is_total_like(close_spread) and is_spread_like(close_total)

    is_pk      = (open_total == 0.0 and not open_swap)
    is_extreme = (ml_home    is not None and abs(ml_home)    > ML_EXTREME or
                  ml_visitor is not None and abs(ml_visitor) > ML_EXTREME)

    if close_swap:  return 1
    if open_swap:   return 2
    if is_pk:       return 3
    if is_extreme:  return 4
    return 0


def parse_season(soup, season_label: str) -> list[dict]:
    table = soup.find("table")
    if table is None:
        print(f"  No table found for {season_label}")
        return []

    rows = table.find_all("tr")
    records = []
    i = 1 if (rows and rows[0].find("th")) else 0

    while i < len(rows) - 1:
        v_cells = rows[i].find_all("td")
        h_cells = rows[i + 1].find_all("td")

        def cell(cells, idx):
            try:
                return cells[idx].get_text(strip=True)
            except IndexError:
                return ""

        if cell(v_cells, 2) != "V" or cell(h_cells, 2) != "H":
            i += 1
            continue

        try:
            score_visitor = int(cell(v_cells, 8))
            score_home    = int(cell(h_cells, 8))
        except ValueError:
            i += 2
            continue

        open_total   = parse_raw_value(cell(v_cells, 9))
        close_total  = parse_raw_value(cell(v_cells, 10))
        open_spread  = parse_raw_value(cell(h_cells, 9))
        close_spread = parse_raw_value(cell(h_cells, 10))
        ml_home      = parse_raw_value(cell(h_cells, 11))
        ml_visitor   = parse_raw_value(cell(v_cells, 11))

        quality = compute_line_quality(
            open_spread, close_spread, open_total, close_total,
            ml_home, ml_visitor
        )

        records.append({
            "season":        season_label,
            "date":          cell(v_cells, 0),
            "home_team":     cell(h_cells, 3),
            "visitor":       cell(v_cells, 3),
            "score_home":    score_home,
            "score_visitor": score_visitor,
            "open_total":    open_total,
            "close_total":   close_total,
            "open_spread":   open_spread,
            "close_spread":  close_spread,
            "ml_home":       ml_home,
            "ml_visitor":    ml_visitor,
            "line_quality":  quality,
        })

        i += 2

    return records


def main():
    all_records = []

    for season in SEASONS:
        print(f"Fetching {season}...")
        soup = fetch_page(season)
        if soup is None:
            continue
        records = parse_season(soup, season)
        print(f"  {len(records)} games")
        all_records.extend(records)
        time.sleep(1.5)

    if not all_records:
        print("No records collected. Check network access.")
        return

    df = pd.DataFrame(all_records)

    print("\n=== SANITY CHECKS ===")
    print(f"Total games: {len(df)}")
    print(f"\nline_quality distribution:")
    print(df["line_quality"].value_counts().sort_index())
    print(f"\nSeasons:\n{df['season'].value_counts().sort_index()}")
    print(f"\nNulls:\n{df.isnull().sum()}")

    df.to_csv("sbr_nfl_odds_raw.csv", index=False)
    print(f"\nSaved {len(df)} rows to sbr_nfl_odds_raw.csv")
    print("Columns:", list(df.columns))


if __name__ == "__main__":
    main()
