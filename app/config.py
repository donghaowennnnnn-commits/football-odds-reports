"""全局配置。API key 放 .env，不要写进代码。"""
import os
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_DIR = Path(__file__).resolve().parent
DB_PATH = PROJECT_DIR / "football.db"
LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

TZ = ZoneInfo("Asia/Shanghai")  # 所有输入/显示按北京时间，库内存 UTC


def _load_env():
    env_file = PROJECT_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


_load_env()

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
# 备用 key：主 key 额度耗尽(401)时自动启用，详见 sources/odds_api.py
ODDS_API_KEY_FALLBACK = os.environ.get("ODDS_API_KEY_FALLBACK", "")

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

# The Odds API 不提供 bet365；bet365 只能通过 OddsPortal 降级源获得。
# 这里最多 10 家（10 家以内额度消耗相同：每次调用 = 市场数 3）。
BOOKMAKERS = [
    "pinnacle", "marathonbet", "williamhill", "unibet", "betfair_ex_eu",
    "betvictor", "betsson", "onexbet", "skybet", "coral",
]
MARKETS = "h2h,spreads,totals"  # 欧赔 1X2 / 亚洲让球 / 大小球

# 调度规则：launchd 每 30 分钟唤醒一次 scrape.py，由下面的规则决定是否真的抓
# 抓取节奏（2026-06 精简）：预测=开赛前最后一帧收盘价，远端盘口对预测无意义、
# 只烧额度，故大幅收窄。每场约 1（远）+3（近）帧，较旧 13 帧省 ~60% 额度，
# 而喂给模型的收盘价一帧不少。详见会话结论。
WINDOW_FAR_HOURS = 8      # 开赛前 8 小时以外不抓（旧 48h 的早盘对预测无意义）
INTERVAL_FAR_MIN = 360    # 8h ~ 1.5h：每 6 小时（约 1 帧，让网页提前有内容）
WINDOW_NEAR_HOURS = 1.5
INTERVAL_NEAR_MIN = 30    # 1.5h 以内：每 30 分钟（约 3 帧，末帧=收盘价，多帧容错）
TEAM_STATS_INTERVAL_MIN = 720  # 球队数据每 12 小时刷新一次
REPORT_WINDOW_HOURS = 8        # 开赛前 8 小时起生成报告（对齐抓取窗口，有赔率才出页）

# 自动跟踪整个赛事：列表中赛事的所有未开赛场次都按正式比赛的频率密集抓取
# （球队数据和自动 PDF 仍只针对手动 add 的比赛）
AUTO_TRACK_SPORTS = ["soccer_fifa_world_cup"]

# FIFA 世界排名前 10（英文名，对齐 The Odds API）。涉及任一强队的比赛
# 自动升格为"重点"（抓球队数据 + 重点标记）。排名静态、可随官方更新调整。
TOP_TEAMS = {
    "Argentina", "Spain", "France", "England", "Brazil",
    "Portugal", "Netherlands", "Belgium", "Italy", "Germany",
}

# API 额度低于此值时弹 macOS 系统通知（每 24 小时最多一次）
# 全量密集模式日耗约 75~80，150 约等于提前 2 天预警，便于及时更换 key
QUOTA_WARN_THRESHOLD = 150

# 抓取礼貌性设置
FBREF_MIN_DELAY = 4.0     # FBref robots 要求 >=3s，留余量
SCRAPE_DELAY_RANGE = (2.0, 6.0)  # 网页抓取源的随机间隔
