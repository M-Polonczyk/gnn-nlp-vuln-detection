import re

NAMING_PATTERNS = {
    "snake_case": re.compile(r"^[a-z]+(_[a-z0-9]+)*$"),
    "camelCase": re.compile(r"^[a-z][a-z0-9]*([A-Z][a-z0-9]*)+$"),
    "PascalCase": re.compile(r"^[A-Z][a-z0-9]*([A-Z][a-z0-9]*)*$"),
    "SCREAMING_SNAKE": re.compile(r"^[A-Z]+(_[A-Z0-9]+)*$"),
    "kebab-case": re.compile(r"^[a-z]+(-[a-z0-9]+)*$"),
    "lower.dot.case": re.compile(r"^[a-z]+(\.[a-z0-9]+)*$"),
}

BAD_NAMES = {
    "temp",
    "asdf",
    "foo",
    "bar",
    "baz",
    "data",
    "value",
    "val",
    "thing",
    "stuff",
    "var",
    "obj",
    "x",
    "y",
    "z",
}


def identify_naming_convention(name: str) -> str:
    for style, regex in NAMING_PATTERNS.items():
        if regex.match(name):
            return style
    return "unknown"


def identify_naming_conventions(name: str) -> list[str]:
    styles = []
    for style, regex in NAMING_PATTERNS.items():
        if regex.match(name):
            styles.append(style)
    return styles
