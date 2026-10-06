"""Operator model names and compatibility for already-started rounds."""

import re


_ALPHANUMERIC = r"A-Za-z0-9\u3400-\u4dbf\u4e00-\u9fff"
_ALLOWED = re.compile(rf"[{_ALPHANUMERIC}()\-]+")
_FIRST = re.compile(rf"[{_ALPHANUMERIC}]")
_LAST = re.compile(rf"[{_ALPHANUMERIC})]")
_INVALID_PATH_CHARACTER = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = re.compile(r"(?:CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])", re.IGNORECASE)


def validate_product_model(value: str, *, allow_legacy: bool = False) -> str:
    """Return the accepted name; never rewrite a restored round's identity."""
    model = value if allow_legacy else value.strip()
    if not model.strip():
        raise ValueError("请先填写型号。")
    if _INVALID_PATH_CHARACTER.search(model) or model.endswith((" ", ".")):
        raise ValueError("型号包含文件路径非法字符，或以空格、点结尾。")
    if _RESERVED.fullmatch(model.split(".", 1)[0]):
        raise ValueError("型号不能使用 CON、NUL、COM1 等 Windows 保留名称。")
    if allow_legacy:
        return model
    if len(model) > 80:
        raise ValueError("型号长度不能超过 80 个字符。")
    if not _ALLOWED.fullmatch(model):
        raise ValueError("型号仅允许中文汉字、英文字母、数字、半角短横线 - 和括号 ()；不允许下划线、点或内部空白。")
    if not _FIRST.fullmatch(model[0]) or not _LAST.fullmatch(model[-1]):
        raise ValueError("型号必须以汉字、字母或数字开头，以汉字、字母、数字或右括号结尾。")
    openings = []
    for index, char in enumerate(model):
        if char == "(":
            openings.append(index)
        elif char == ")":
            if not openings:
                raise ValueError("型号括号必须成对且顺序正确。")
            if index == openings.pop() + 1:
                raise ValueError("型号括号内不能为空。")
    if openings:
        raise ValueError("型号括号必须成对且顺序正确。")
    return model
