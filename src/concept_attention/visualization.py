from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize


_PUNCT_NO_SPACE_BEFORE = set(list(".,;:!?%)]}")) | {"'s"}
_PUNCT_NO_SPACE_AFTER = set(list("([{"))


def _is_punct(token: str) -> bool:
    return bool(re.fullmatch(r"[^\w\s]+", token)) or token in _PUNCT_NO_SPACE_BEFORE or token in _PUNCT_NO_SPACE_AFTER


def _needs_space(prev_word: str, cur_word: str) -> bool:
    if prev_word == "":
        return False
    if cur_word in _PUNCT_NO_SPACE_BEFORE:
        return False
    if prev_word in _PUNCT_NO_SPACE_AFTER:
        return False
    if _is_punct(cur_word):
        return False
    return True


@dataclass
class WordPiece:
    text: str
    score: float
    raw_token: str


@dataclass
class WordGroup:
    pieces: list[WordPiece]
    word_text: str
    leading_space: bool


def merge_wordpieces(tokens: list[str], scores: list[float]) -> list[WordGroup]:
    assert len(tokens) == len(scores)
    groups: list[WordGroup] = []
    cur_pieces: list[WordPiece] = []
    prev_word_text = ""

    for tok, score in zip(tokens, scores):
        is_cont = tok.startswith("##")
        piece_text = tok[2:] if is_cont else tok

        if not cur_pieces or not is_cont:
            if cur_pieces:
                word_text = "".join(piece.text for piece in cur_pieces)
                groups.append(WordGroup(cur_pieces, word_text, _needs_space(prev_word_text, word_text)))
                prev_word_text = word_text
            cur_pieces = [WordPiece(piece_text, float(score), tok)]
        else:
            cur_pieces.append(WordPiece(piece_text, float(score), tok))

    if cur_pieces:
        word_text = "".join(piece.text for piece in cur_pieces)
        groups.append(WordGroup(cur_pieces, word_text, _needs_space(prev_word_text, word_text)))

    return groups


def _get_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        font_path = font_manager.findfont("DejaVu Sans", fallback_to_default=True)
        return ImageFont.truetype(font_path, font_size)
    except Exception:
        return ImageFont.load_default()


def _text_width(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, text: str) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return int(bbox[2] - bbox[0])


def _blend(bg_rgb: tuple[int, int, int], fg_rgb: tuple[int, int, int], alpha: float) -> tuple[int, int, int]:
    return tuple(int(round((1 - alpha) * b + alpha * f)) for b, f in zip(bg_rgb, fg_rgb))


def make_white_to_blue_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "white_lightblue_darkblue",
        [(1.0, 1.0, 1.0), (0.70, 0.85, 1.0), (0.10, 0.35, 0.85)],
        N=256,
    )


def render_token_scores_png(
    tokens: list[str],
    scores: list[float],
    out_path: str,
    *,
    width_px: int = 1200,
    margin_px: int = 20,
    font_size: int = 20,
    line_gap_px: int = 6,
    piece_pad_x: int = 2,
    piece_pad_y: int = 2,
    space_px: Optional[int] = None,
    alpha: float = 0.95,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    text_rgb: tuple[int, int, int] = (0, 0, 0),
    bg_rgb: tuple[int, int, int] = (255, 255, 255),
    cmap: Optional[LinearSegmentedColormap] = None,
) -> str:
    """Render token-level attention scores as a blue highlight PNG."""
    groups = merge_wordpieces(tokens, scores)
    font = _get_font(font_size)

    tmp = Image.new("RGB", (width_px, 200), bg_rgb)
    draw = ImageDraw.Draw(tmp)

    if space_px is None:
        space_px = _text_width(draw, font, " ")

    arr = np.asarray(scores, dtype=float)
    if vmin is None:
        vmin = float(np.nanmin(arr)) if arr.size else 0.0
    if vmax is None:
        vmax = float(np.nanmax(arr)) if arr.size else 1.0
    if vmax == vmin:
        vmax = vmin + 1e-6

    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = cmap or make_white_to_blue_cmap()

    lines: list[list[WordGroup]] = []
    cur_line: list[WordGroup] = []
    cur_x = margin_px
    max_x = width_px - margin_px

    for group in groups:
        prefix_w = space_px if group.leading_space else 0
        word_w = sum(_text_width(draw, font, piece.text) + 2 * piece_pad_x for piece in group.pieces)
        needed = prefix_w + word_w

        if cur_line and cur_x + needed > max_x:
            lines.append(cur_line)
            cur_line = [group]
            cur_x = margin_px + word_w
        else:
            cur_line.append(group)
            cur_x += needed

    if cur_line:
        lines.append(cur_line)

    bbox = draw.textbbox((0, 0), "Hg", font=font)
    line_h = (bbox[3] - bbox[1]) + 2 * piece_pad_y + line_gap_px
    height_px = margin_px * 2 + max(1, len(lines)) * line_h

    img = Image.new("RGB", (width_px, height_px), bg_rgb)
    draw = ImageDraw.Draw(img)

    y = margin_px
    for line in lines:
        x = margin_px
        for group in line:
            if group.leading_space:
                x += space_px

            for piece in group.pieces:
                txt = piece.text
                w = _text_width(draw, font, txt)
                rect_w = w + 2 * piece_pad_x
                color = tuple(int(c * 255) for c in cmap(norm(piece.score))[:3])
                fill = _blend(bg_rgb, color, alpha)
                draw.rectangle(
                    [x, y, x + rect_w, y + line_h - line_gap_px],
                    fill=fill,
                    outline=None,
                )
                draw.text((x + piece_pad_x, y + piece_pad_y), txt, font=font, fill=text_rgb)
                x += rect_w

        y += line_h

    img.save(out_path)
    return out_path
