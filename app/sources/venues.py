"""2026 世界杯球场静态库 + 免费天气（open-meteo，无需 API key）。

球场属性是固定的，预先列好；天气按球场坐标实时取赛日预报。
这些数据全部免费/可计算，不消耗 The Odds API 额度。
"""
import json
import logging
import urllib.request
from urllib.parse import urlencode

log = logging.getLogger("venues")

# 东道主（用于主场优势判断）
HOSTS = {"USA", "Canada", "Mexico"}

# stadium_key -> 属性。key 用小写去空格匹配 ESPN 的 venue fullName。
# alt=海拔(米), roof=顶棚, cap=容量(千)
VENUES = {
    "mercedesbenzstadium": {"city": "亚特兰大", "lat": 33.755, "lon": -84.401, "alt": 320, "roof": "可开合顶棚", "cap": 71, "country": "USA"},
    "gillettestadium": {"city": "福克斯堡", "lat": 42.091, "lon": -71.264, "alt": 90, "roof": "露天", "cap": 65, "country": "USA"},
    "attstadium": {"city": "阿灵顿", "lat": 32.747, "lon": -97.093, "alt": 180, "roof": "可开合顶棚", "cap": 80, "country": "USA"},
    "nrgstadium": {"city": "休斯顿", "lat": 29.685, "lon": -95.411, "alt": 15, "roof": "可开合顶棚", "cap": 72, "country": "USA"},
    "arrowheadstadium": {"city": "堪萨斯城", "lat": 39.049, "lon": -94.484, "alt": 270, "roof": "露天", "cap": 76, "country": "USA"},
    "sofistadium": {"city": "英格尔伍德", "lat": 33.953, "lon": -118.339, "alt": 40, "roof": "固定顶棚(侧开)", "cap": 70, "country": "USA"},
    "hardrockstadium": {"city": "迈阿密", "lat": 25.958, "lon": -80.239, "alt": 3, "roof": "遮阳棚", "cap": 65, "country": "USA"},
    "metlifestadium": {"city": "东卢瑟福", "lat": 40.814, "lon": -74.074, "alt": 5, "roof": "露天", "cap": 82, "country": "USA"},
    "lincolnfinancialfield": {"city": "费城", "lat": 39.901, "lon": -75.168, "alt": 12, "roof": "露天", "cap": 69, "country": "USA"},
    "levisstadium": {"city": "圣克拉拉", "lat": 37.403, "lon": -121.970, "alt": 4, "roof": "露天", "cap": 68, "country": "USA"},
    "lumenfield": {"city": "西雅图", "lat": 47.595, "lon": -122.332, "alt": 10, "roof": "露天", "cap": 69, "country": "USA"},
    "bmofield": {"city": "多伦多", "lat": 43.633, "lon": -79.418, "alt": 80, "roof": "露天", "cap": 45, "country": "Canada"},
    "bcplace": {"city": "温哥华", "lat": 49.277, "lon": -123.112, "alt": 5, "roof": "可开合顶棚", "cap": 54, "country": "Canada"},
    "estadioakron": {"city": "瓜达拉哈拉", "lat": 20.682, "lon": -103.463, "alt": 1560, "roof": "露天", "cap": 49, "country": "Mexico"},
    "estadioazteca": {"city": "墨西哥城", "lat": 19.303, "lon": -99.150, "alt": 2240, "roof": "露天", "cap": 87, "country": "Mexico"},
    "estadiobanorte": {"city": "墨西哥城", "lat": 19.303, "lon": -99.150, "alt": 2240, "roof": "露天", "cap": 87, "country": "Mexico"},
    "estadiobbva": {"city": "蒙特雷", "lat": 25.669, "lon": -100.244, "alt": 500, "roof": "露天", "cap": 53, "country": "Mexico"},
}


def _norm(s):
    return "".join(c for c in (s or "").lower() if c.isalnum())


def lookup(venue_name, city=None):
    """按球场名（必要时城市）匹配静态库，返回属性 dict 或 None。"""
    key = _norm(venue_name)
    if key in VENUES:
        return dict(VENUES[key], stadium=venue_name)
    for k, v in VENUES.items():  # 名称包含匹配
        if k in key or key in k:
            return dict(v, stadium=venue_name)
    if city:  # 城市兜底
        cn = _norm(city)
        for v in VENUES.values():
            if _norm(v["city"]) in cn:
                return dict(v, stadium=venue_name)
    return None


def get_weather(lat, lon, date_yyyymmdd):
    """open-meteo 免费预报（无 key），返回 {tmin,tmax,rain,wind} 或 None。"""
    try:
        q = urlencode({
            "latitude": lat, "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,"
                     "precipitation_probability_max,wind_speed_10m_max",
            "timezone": "auto",
            "start_date": date_yyyymmdd, "end_date": date_yyyymmdd,
        })
        url = f"https://api.open-meteo.com/v1/forecast?{q}"
        with urllib.request.urlopen(url, timeout=15) as r:
            d = json.load(r)
        dy = d.get("daily", {})
        def g(k):
            v = dy.get(k) or []
            return v[0] if v else None
        return {
            "tmin": g("temperature_2m_min"), "tmax": g("temperature_2m_max"),
            "rain": g("precipitation_probability_max"),
            "wind": g("wind_speed_10m_max"),
        }
    except Exception as e:
        log.warning("天气获取失败 (%.2f,%.2f): %s", lat, lon, e)
        return None
