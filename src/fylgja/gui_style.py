"""GUI 样式：Windows 11 Fluent 双主题（浅色/深色自动切换）。

设计 token：白卡圆角、强调色 #0067C0、微软雅黑 UI、细边框、轻分隔线。
"""

ACCENT = "#0067C0"
ACCENT_HOVER = "#1A7AD1"
ACCENT_DOWN = "#005A9E"

LIGHT = {
    "bg": "#F3F3F3", "card": "#FFFFFF", "border": "#E5E5E5",
    "text": "#1A1A1A", "text2": "#616161", "field": "#FFFFFF",
    "hover": "#F0F0F0", "pressed": "#E8E8E8", "line": "#E8E8E8",
    "preview": "#171717", "banner_bg": "#FFF4CE", "banner_fg": "#6B4E00",
    "banner_bd": "#F0D77B", "disabled_fg": "#A0A0A0", "disabled_bg": "#F6F6F6",
    "slider_groove": "#E0E0E0", "stop_fg": "#C42B1C", "stop_hover": "#FDF1F0",
}

DARK = {
    "bg": "#202020", "card": "#2B2B2B", "border": "#3A3A3A",
    "text": "#FFFFFF", "text2": "#A0A0A0", "field": "#333333",
    "hover": "#383838", "pressed": "#404040", "line": "#3A3A3A",
    "preview": "#101010", "banner_bg": "#3B3200", "banner_fg": "#F5D97B",
    "banner_bd": "#5C4D00", "disabled_fg": "#6E6E6E", "disabled_bg": "#262626",
    "slider_groove": "#4A4A4A", "stop_fg": "#FF897D", "stop_hover": "#3A2A28",
}


def is_system_dark() -> bool:
    """读取 Windows 个性化设置（AppsUseLightTheme：0=深色）。"""
    try:
        import winreg
        k = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize")
        val, _ = winreg.QueryValueEx(k, "AppsUseLightTheme")
        return val == 0
    except Exception:
        return False


def build_qss(dark: bool) -> str:
    c = DARK if dark else LIGHT
    return f"""
* {{
    font-family: "Microsoft YaHei UI";
    font-size: 9.5pt;
    color: {c['text']};
}}
QMainWindow, QWidget#root {{ background: {c['bg']}; }}
QWidget#card {{
    background: {c['card']};
    border: 1px solid {c['border']};
    border-radius: 10px;
}}
QLabel#title {{ font-size: 15pt; font-weight: 600; }}
QLabel#hint {{ font-size: 8pt; color: {c['text2']}; }}
QLabel#mini {{ font-size: 8pt; color: {c['text2']}; font-weight: 600; }}
QPushButton {{
    background: {c['card']};
    border: 1px solid {c['border']};
    border-radius: 8px;
    padding: 6px 12px;
    font-weight: 600;
}}
QPushButton:hover {{ background: {c['hover']}; }}
QPushButton:pressed {{ background: {c['pressed']}; }}
QPushButton:disabled {{ color: {c['disabled_fg']}; background: {c['disabled_bg']}; }}
QPushButton:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
    color: white;
}}
QPushButton#startBtn {{
    background: {ACCENT};
    border: 1px solid {ACCENT};
    color: white;
    font-size: 11pt;
    padding: 10px 18px;
}}
QPushButton#startBtn:hover {{ background: {ACCENT_HOVER}; border-color: {ACCENT_HOVER}; }}
QPushButton#startBtn:pressed {{ background: {ACCENT_DOWN}; }}
QPushButton#stopBtn {{
    background: {c['card']};
    border: 1px solid {c['stop_fg']};
    color: {c['stop_fg']};
    font-size: 11pt;
    padding: 10px 18px;
}}
QPushButton#stopBtn:hover {{ background: {c['stop_hover']}; }}
QPushButton#stopBtn:disabled {{
    border-color: {c['border']};
    color: {c['disabled_fg']};
    background: {c['disabled_bg']};
}}
QComboBox, QLineEdit, QDoubleSpinBox {{
    background: {c['field']};
    border: 1px solid {c['border']};
    border-radius: 6px;
    padding: 5px 8px;
    selection-background-color: {ACCENT};
}}
QComboBox:hover, QLineEdit:hover {{ border-color: {c['text2']}; }}
QComboBox:focus, QLineEdit:focus {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background: {c['card']};
    border: 1px solid {c['border']};
    selection-background-color: {ACCENT};
    selection-color: white;
}}
QCheckBox {{ spacing: 6px; background: transparent; }}
QCheckBox::indicator {{
    width: 16px; height: 16px;
    border-radius: 4px;
    border: 1px solid {c['text2']};
    background: {c['field']};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
}}
QSlider::groove:horizontal {{
    height: 4px;
    background: {c['slider_groove']};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    width: 14px; height: 14px;
    margin: -5px 0;
    border-radius: 7px;
    background: {ACCENT};
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT};
    border-radius: 2px;
}}
QFrame#previewFrame {{
    background: {c['preview']};
    border-radius: 10px;
}}
QLabel#banner {{
    background: {c['banner_bg']};
    color: {c['banner_fg']};
    border: 1px solid {c['banner_bd']};
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 9pt;
}}
QStatusBar {{ background: transparent; color: {c['text2']}; font-size: 8.5pt; }}
QScrollBar:vertical {{ width: 10px; background: transparent; }}
QScrollBar::handle:vertical {{
    background: {c['text2']}; border-radius: 5px; min-height: 30px;
}}
"""
