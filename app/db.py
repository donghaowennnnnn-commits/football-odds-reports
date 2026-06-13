"""SQLite 存取层。所有时间字段存 UTC ISO 字符串。"""
import sqlite3
from datetime import datetime, timezone

from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    competition TEXT,
    kickoff_utc TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'tracking',   -- tracking | finished
    sport_key TEXT,
    odds_api_event_id TEXT,
    home_team_en TEXT,
    away_team_en TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS odds_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL REFERENCES matches(id),
    fetched_at TEXT NOT NULL,
    source TEXT NOT NULL,            -- odds_api | oddsportal
    bookmaker TEXT NOT NULL,
    market TEXT NOT NULL,            -- 1x2 | ah | ou
    line REAL,                       -- 让球线 / 大小球线；1x2 为 NULL
    home_odds REAL,
    draw_odds REAL,                  -- 仅 1x2 有；ou 时 home=大 away=小
    away_odds REAL
);
CREATE INDEX IF NOT EXISTS idx_snap_match ON odds_snapshots(match_id, fetched_at);
CREATE TABLE IF NOT EXISTS team_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    source TEXT NOT NULL,
    recent_matches_json TEXT,
    injuries_json TEXT
);
CREATE TABLE IF NOT EXISTS scrape_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT,                     -- ok | partial | error
    detail TEXT
);
CREATE TABLE IF NOT EXISTS paper_bets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL REFERENCES matches(id),
    placed_at TEXT NOT NULL,
    market TEXT NOT NULL,            -- ah | ou | cs
    pick TEXT NOT NULL,              -- 如 主让 -1.25 / 大 2.25 / 1-0
    bookmaker TEXT,
    line REAL,
    side TEXT,                       -- home/away/over/under；cs 为空
    odds REAL NOT NULL,              -- cs 记模型公平赔率（市价不可得）
    ev REAL,
    result TEXT,                     -- 赢/赢半/走/输半/输（未结算为 NULL）
    pnl REAL,                        -- 每注本金 1
    settled_at TEXT
);
"""


def utcnow_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def _migrate(conn):
    """老库平滑加列（赛果回填功能引入）。"""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(matches)")}
    for col, ddl in [
        ("home_score", "ALTER TABLE matches ADD COLUMN home_score INTEGER"),
        ("away_score", "ALTER TABLE matches ADD COLUMN away_score INTEGER"),
        ("result_source", "ALTER TABLE matches ADD COLUMN result_source TEXT"),
        ("result_attempts",
         "ALTER TABLE matches ADD COLUMN result_attempts INTEGER NOT NULL DEFAULT 0"),
    ]:
        if col not in cols:
            conn.execute(ddl)
    pb_cols = {r["name"] for r in conn.execute("PRAGMA table_info(paper_bets)")}
    if pb_cols and "stake" not in pb_cols:
        # 注额列：波胆 0.1，其余 1（高赔率玩法小仓位，符合实际投注习惯）
        conn.execute("ALTER TABLE paper_bets ADD COLUMN stake REAL NOT NULL DEFAULT 1")
    if pb_cols and "strategy" not in pb_cols:
        # 策略列：ev=EV最优入口；flow=顺职业资金方向（平行实验组）
        conn.execute("ALTER TABLE paper_bets ADD COLUMN strategy TEXT NOT NULL"
                     " DEFAULT 'ev'")
    for col in ("venue_name", "venue_city"):  # 球场缓存（静态，抓一次即可）
        if col not in cols:
            conn.execute(f"ALTER TABLE matches ADD COLUMN {col} TEXT")
    conn.commit()


# ---------- matches ----------

def add_match(conn, home, away, kickoff_utc, competition=None, sport_key=None):
    cur = conn.execute(
        "INSERT INTO matches (home_team, away_team, competition, kickoff_utc,"
        " sport_key, created_at) VALUES (?,?,?,?,?,?)",
        (home, away, competition, kickoff_utc, sport_key, utcnow_iso()),
    )
    conn.commit()
    return cur.lastrowid


def get_match(conn, match_id):
    return conn.execute("SELECT * FROM matches WHERE id=?", (match_id,)).fetchone()


def get_match_by_event(conn, event_id):
    return conn.execute(
        "SELECT * FROM matches WHERE odds_api_event_id=?", (event_id,)
    ).fetchone()


def add_passive_match(conn, home_en, away_en, kickoff_utc, sport_key, event_id):
    """批量响应里顺带存档的比赛：不调度、不显示，只攒数据。"""
    cur = conn.execute(
        "INSERT INTO matches (home_team, away_team, kickoff_utc, status,"
        " sport_key, odds_api_event_id, home_team_en, away_team_en, created_at)"
        " VALUES (?,?,?,'passive',?,?,?,?,?)",
        (home_en, away_en, kickoff_utc, sport_key, event_id,
         home_en, away_en, utcnow_iso()),
    )
    conn.commit()
    return cur.lastrowid


def promote_match(conn, match_id, home, away, competition, kickoff_utc):
    """把被动存档的比赛升级为正式跟踪（历史快照自动保留）。"""
    conn.execute(
        "UPDATE matches SET home_team=?, away_team=?, competition=?,"
        " kickoff_utc=?, status='tracking' WHERE id=?",
        (home, away, competition, kickoff_utc, match_id),
    )
    conn.commit()


def list_matches(conn, status=None):
    if status:
        return conn.execute(
            "SELECT * FROM matches WHERE status=? ORDER BY kickoff_utc", (status,)
        ).fetchall()
    return conn.execute("SELECT * FROM matches ORDER BY kickoff_utc").fetchall()


def remove_match(conn, match_id):
    conn.execute("DELETE FROM odds_snapshots WHERE match_id=?", (match_id,))
    conn.execute("DELETE FROM matches WHERE id=?", (match_id,))
    conn.commit()


def set_event(conn, match_id, sport_key, event_id, home_en=None, away_en=None):
    conn.execute(
        "UPDATE matches SET sport_key=?, odds_api_event_id=?,"
        " home_team_en=COALESCE(?, home_team_en),"
        " away_team_en=COALESCE(?, away_team_en) WHERE id=?",
        (sport_key, event_id, home_en, away_en, match_id),
    )
    conn.commit()


def update_kickoff(conn, match_id, kickoff_utc):
    conn.execute("UPDATE matches SET kickoff_utc=? WHERE id=?",
                 (kickoff_utc, match_id))
    conn.commit()


def set_venue(conn, match_id, venue_name, venue_city):
    conn.execute("UPDATE matches SET venue_name=?, venue_city=? WHERE id=?",
                 (venue_name, venue_city, match_id))
    conn.commit()


def set_status(conn, match_id, status):
    conn.execute("UPDATE matches SET status=? WHERE id=?", (status, match_id))
    conn.commit()


# ---------- 赛果回填 ----------

def matches_needing_result(conn, kickoff_before_iso, limit=10):
    """已开赛但还没有比分的比赛（含被动存档），跟踪中的优先。"""
    return conn.execute(
        "SELECT * FROM matches WHERE home_score IS NULL AND kickoff_utc < ?"
        " AND result_attempts < 20"   # 提前到110min起每15min一问，点球赛需更多次
        " ORDER BY (status != 'passive') DESC, kickoff_utc DESC LIMIT ?",
        (kickoff_before_iso, limit),
    ).fetchall()


def set_result(conn, match_id, home_score, away_score, source):
    conn.execute(
        "UPDATE matches SET home_score=?, away_score=?, result_source=? WHERE id=?",
        (home_score, away_score, source, match_id),
    )
    conn.commit()


def bump_result_attempts(conn, match_id):
    conn.execute(
        "UPDATE matches SET result_attempts = result_attempts + 1 WHERE id=?",
        (match_id,),
    )
    conn.commit()


# ---------- 模拟下注 ----------

def get_paper_bets(conn, match_id):
    return conn.execute(
        "SELECT * FROM paper_bets WHERE match_id=? ORDER BY market", (match_id,)
    ).fetchall()


def insert_paper_bet(conn, match_id, market, pick, bookmaker, line, side,
                     odds, ev, stake=1.0, strategy="ev"):
    conn.execute(
        "INSERT INTO paper_bets (match_id, placed_at, market, pick, bookmaker,"
        " line, side, odds, ev, stake, strategy) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (match_id, utcnow_iso(), market, pick, bookmaker, line, side, odds, ev,
         stake, strategy),
    )
    conn.commit()


def unsettled_paper_bets(conn):
    """有赛果但未结算的注单。"""
    return conn.execute(
        "SELECT b.*, m.home_score AS hs, m.away_score AS aws FROM paper_bets b"
        " JOIN matches m ON m.id = b.match_id"
        " WHERE b.result IS NULL AND m.home_score IS NOT NULL",
    ).fetchall()


def settle_paper_bet(conn, bet_id, result, pnl):
    conn.execute(
        "UPDATE paper_bets SET result=?, pnl=?, settled_at=? WHERE id=?",
        (result, pnl, utcnow_iso(), bet_id),
    )
    conn.commit()


def finished_with_results(conn):
    """有比分且有赔率快照的比赛，校准验证用。"""
    return conn.execute(
        "SELECT m.* FROM matches m WHERE m.home_score IS NOT NULL"
        " AND EXISTS (SELECT 1 FROM odds_snapshots s WHERE s.match_id = m.id"
        "             AND s.market = '1x2')"
        " ORDER BY m.kickoff_utc",
    ).fetchall()


# ---------- odds_snapshots ----------

def insert_snapshots(conn, match_id, source, rows, fetched_at=None):
    """rows: [{bookmaker, market, line, home, draw, away}]，只追加不覆盖。"""
    ts = fetched_at or utcnow_iso()
    conn.executemany(
        "INSERT INTO odds_snapshots (match_id, fetched_at, source, bookmaker,"
        " market, line, home_odds, draw_odds, away_odds) VALUES (?,?,?,?,?,?,?,?,?)",
        [
            (match_id, ts, source, r["bookmaker"], r["market"], r.get("line"),
             r.get("home"), r.get("draw"), r.get("away"))
            for r in rows
        ],
    )
    conn.commit()
    return ts


def last_snapshot_time(conn, match_id):
    row = conn.execute(
        "SELECT MAX(fetched_at) AS t FROM odds_snapshots WHERE match_id=?",
        (match_id,),
    ).fetchone()
    return row["t"]


def get_snapshots(conn, match_id):
    return conn.execute(
        "SELECT * FROM odds_snapshots WHERE match_id=?"
        " ORDER BY market, bookmaker, fetched_at",
        (match_id,),
    ).fetchall()


# ---------- team_stats ----------

def insert_team_stats(conn, team, source, recent_json, injuries_json):
    conn.execute(
        "INSERT INTO team_stats (team, fetched_at, source, recent_matches_json,"
        " injuries_json) VALUES (?,?,?,?,?)",
        (team, utcnow_iso(), source, recent_json, injuries_json),
    )
    conn.commit()


def last_team_stats_time(conn, team):
    row = conn.execute(
        "SELECT MAX(fetched_at) AS t FROM team_stats WHERE team=?", (team,)
    ).fetchone()
    return row["t"]


# ---------- scrape_runs ----------

def start_run(conn):
    cur = conn.execute(
        "INSERT INTO scrape_runs (started_at) VALUES (?)", (utcnow_iso(),)
    )
    conn.commit()
    return cur.lastrowid


def finish_run(conn, run_id, status, detail=""):
    conn.execute(
        "UPDATE scrape_runs SET finished_at=?, status=?, detail=? WHERE id=?",
        (utcnow_iso(), status, detail, run_id),
    )
    conn.commit()
