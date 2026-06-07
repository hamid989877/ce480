"""
HW6 Breakwater layout design worksheet.

Edit the USER_INPUTS block, especially the values that come from HW5:
    - design wave height in front of the breakwater, H_i
    - corresponding design wave period, T
    - wave angle alpha / incident angle to the breakwater
    - diffraction coefficient K_d read from the correct diffraction diagram

Then run:
    python hw6_breakwater_design.py

The script prints the main result and regenerates:
    HW6_Breakwater_Report.pdf
"""

from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


G = 9.81
OUTPUT_DIR = Path(__file__).resolve().parent
PAGE_SIZE = (1240, 1754)  # A4 at about 150 dpi
MARGIN = 70
TEXT_WIDTH = PAGE_SIZE[0] - 2 * MARGIN


@dataclass(frozen=True)
class BreakwaterInputs:
    assignment_code: str
    project_name: str
    student_name: str
    group_name: str
    group_members: tuple[str, ...]
    course_name: str
    harbor_type: str

    # HW5 / project values. Replace the example values with your own.
    design_wave_height_m: float
    design_wave_period_s: float
    water_depth_m: float
    incident_angle_to_breakwater_deg: float
    diffraction_coefficient_from_diagram: float

    # Simple layout coordinates for geometry reporting.
    # Use any consistent local coordinate system in meters.
    breakwater_axis_deg: float
    breakwater_tip_xy_m: tuple[float, float]
    quay_wall_point_xy_m: tuple[float, float]

    # Set this to False after replacing the example values.
    example_values: bool = True


USER_INPUTS = BreakwaterInputs(
    assignment_code="CE480- HW6",
    project_name="Breakwater Layout Design Report",
    student_name="Group 3",
    group_name="Group 3",
    group_members=(
        "Hamidreza Khalaj Zahraei (300204093)",
        "Ali Özuysal (310204059)",
    ),
    course_name="Coastal Engineering",
    harbor_type="marina",  # marina/fishery: 0.30 m limit, container: 0.50 m limit
    design_wave_height_m=1.20,  # H_i from HW5, in meters
    design_wave_period_s=6.50,  # T from HW5, in seconds
    water_depth_m=8.00,  # local depth near the breakwater/quay, in meters
    incident_angle_to_breakwater_deg=35.0,  # alpha or beta used to choose the diagram
    diffraction_coefficient_from_diagram=0.22,  # K_d read from diffraction diagram
    breakwater_axis_deg=0.0,  # direction of breakwater axis in this local plan
    breakwater_tip_xy_m=(0.0, 0.0),
    quay_wall_point_xy_m=(85.0, -65.0),
    example_values=True,
)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    regular_candidates = [
        Path("C:/Windows/Fonts/times.ttf"),
        Path("C:/Windows/Fonts/timesnewroman.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    bold_candidates = [
        Path("C:/Windows/Fonts/timesbd.ttf"),
        Path("C:/Windows/Fonts/timesnewromanbold.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
    ]
    for path in (bold_candidates if bold else regular_candidates):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


# At 150 dpi, 12 pt is approximately 25 px.
FONT_TITLE = font(46, bold=True)
FONT_H1 = font(36, bold=True)
FONT_H2 = font(29, bold=True)
FONT_BODY = font(25)
FONT_BODY_BOLD = font(25, bold=True)
FONT_SMALL = font(21)
FONT_SMALL_BOLD = font(21, bold=True)


def allowable_wave_height_m(harbor_type: str) -> float:
    key = harbor_type.strip().lower()
    if key in {"marina", "fishery", "fishery harbor", "fishing", "fishing harbor"}:
        return 0.30
    if key in {"container", "container terminal", "container harbor"}:
        return 0.50
    raise ValueError(
        "Unknown harbor_type. Use 'marina', 'fishery', or 'container'."
    )


def solve_wavelength_m(period_s: float, depth_m: float, tol: float = 1e-10) -> float:
    """Solve finite-depth linear wave dispersion: L = L0 tanh(2 pi h / L)."""
    if period_s <= 0:
        raise ValueError("Wave period must be positive.")
    if depth_m <= 0:
        raise ValueError("Water depth must be positive.")

    deep_water_length = G * period_s**2 / (2.0 * math.pi)
    length = deep_water_length
    for _ in range(200):
        next_length = deep_water_length * math.tanh(2.0 * math.pi * depth_m / length)
        if abs(next_length - length) < tol:
            return next_length
        length = next_length
    return length


def signed_angle_difference_deg(angle_deg: float, reference_deg: float) -> float:
    """Return angle-reference in the interval [-180, 180) degrees."""
    return (angle_deg - reference_deg + 180.0) % 360.0 - 180.0


def evaluate_design(data: BreakwaterInputs) -> dict[str, float | str | bool]:
    limit = allowable_wave_height_m(data.harbor_type)
    wavelength = solve_wavelength_m(data.design_wave_period_s, data.water_depth_m)
    k = 2.0 * math.pi / wavelength
    celerity = wavelength / data.design_wave_period_s

    tip_x, tip_y = data.breakwater_tip_xy_m
    quay_x, quay_y = data.quay_wall_point_xy_m
    dx = quay_x - tip_x
    dy = quay_y - tip_y
    distance_to_quay = math.hypot(dx, dy)
    point_angle = math.degrees(math.atan2(dy, dx))
    quay_angle_from_breakwater = signed_angle_difference_deg(
        point_angle, data.breakwater_axis_deg
    )

    kd = data.diffraction_coefficient_from_diagram
    h_quay = kd * data.design_wave_height_m
    kd_required = limit / data.design_wave_height_m
    safety_margin = limit - h_quay
    accepted = h_quay <= limit

    return {
        "allowable_wave_height_m": limit,
        "wavelength_m": wavelength,
        "wave_number_rad_per_m": k,
        "celerity_m_per_s": celerity,
        "distance_tip_to_quay_m": distance_to_quay,
        "r_over_L": distance_to_quay / wavelength,
        "quay_angle_from_breakwater_deg": quay_angle_from_breakwater,
        "quay_wave_height_m": h_quay,
        "required_diffraction_coefficient": kd_required,
        "safety_margin_m": safety_margin,
        "accepted": accepted,
        "decision": "ACCEPTABLE" if accepted else "REVISE LAYOUT",
    }


def new_page() -> Image.Image:
    return Image.new("RGB", PAGE_SIZE, "#FCFBF7")


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def wrapped_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    fnt: ImageFont.ImageFont,
) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        words = paragraph.split()
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if text_size(draw, candidate, fnt)[0] <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: int,
    y: int,
    max_width: int,
    fnt: ImageFont.ImageFont = FONT_BODY,
    fill: str = "#202020",
    line_gap: int = 8,
) -> int:
    line_height = text_size(draw, "Ag", fnt)[1] + line_gap
    for line in wrapped_lines(draw, text, max_width, fnt):
        if line:
            draw.text((x, y), line, font=fnt, fill=fill)
        y += line_height
    return y


def draw_title(
    draw: ImageDraw.ImageDraw,
    title: str,
    subtitle: str | None = None,
    subtitle_font: ImageFont.ImageFont = FONT_SMALL,
) -> int:
    y = MARGIN
    draw.rectangle((0, 0, 22, PAGE_SIZE[1]), fill="#17324D")
    draw.rectangle((MARGIN, y - 12, PAGE_SIZE[0] - MARGIN, y - 6), fill="#C7A35A")
    draw.text((MARGIN, y), title, font=FONT_TITLE, fill="#111111")
    y += 64
    if subtitle:
        draw.text((MARGIN, y), subtitle, font=subtitle_font, fill="#333333")
        y += text_size(draw, subtitle, subtitle_font)[1] + 24
    draw.line((MARGIN, y, PAGE_SIZE[0] - MARGIN, y), fill="#CFC7B8", width=2)
    return y + 28


def draw_section_label(draw: ImageDraw.ImageDraw, text: str, x: int, y: int) -> int:
    draw.text((x, y), text, font=FONT_H2, fill="#111111")
    return y + 36


def decorate_page(page: Image.Image, page_number: int, total_pages: int) -> None:
    draw = ImageDraw.Draw(page)
    footer_y = PAGE_SIZE[1] - 58
    draw.line((MARGIN, footer_y, PAGE_SIZE[0] - MARGIN, footer_y), fill="#D7D0C2", width=2)
    draw.text(
        (MARGIN, footer_y + 16),
        "CE480- HW6 | Breakwater Layout Design Report",
        font=FONT_SMALL,
        fill="#4B4B4B",
    )
    page_text = f"Page {page_number} of {total_pages}"
    text_w, _ = text_size(draw, page_text, FONT_SMALL)
    draw.text(
        (PAGE_SIZE[0] - MARGIN - text_w, footer_y + 16),
        page_text,
        font=FONT_SMALL,
        fill="#4B4B4B",
    )


def draw_table(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    headers: list[str],
    rows: list[list[str]],
    col_fracs: list[float],
    row_font: ImageFont.ImageFont = FONT_SMALL,
    header_font: ImageFont.ImageFont = FONT_SMALL_BOLD,
) -> int:
    col_widths = [int(width * frac) for frac in col_fracs]
    col_widths[-1] = width - sum(col_widths[:-1])
    padding = 12
    line_gap = 5
    x_positions = [x]
    for col_w in col_widths[:-1]:
        x_positions.append(x_positions[-1] + col_w)

    def cell_lines(value: str, col: int, fnt: ImageFont.ImageFont) -> list[str]:
        return wrapped_lines(draw, value, col_widths[col] - 2 * padding, fnt)

    all_rows = [headers] + rows
    fonts = [header_font] + [row_font] * len(rows)
    for row_index, row in enumerate(all_rows):
        fnt = fonts[row_index]
        lines_by_cell = [cell_lines(str(row[col]), col, fnt) for col in range(len(headers))]
        max_lines = max(len(lines) for lines in lines_by_cell)
        line_height = text_size(draw, "Ag", fnt)[1] + line_gap
        row_height = max(42, max_lines * line_height + 2 * padding)
        fill = "#EDEDED" if row_index == 0 else ("#FFFFFF" if row_index % 2 else "#F7F7F7")
        draw.rectangle((x, y, x + width, y + row_height), fill=fill, outline="#B0B0B0")
        cx = x
        for col, col_w in enumerate(col_widths):
            draw.rectangle((cx, y, cx + col_w, y + row_height), outline="#B0B0B0")
            ty = y + padding
            for line in lines_by_cell[col]:
                if line:
                    draw.text((cx + padding, ty), line, font=fnt, fill="#202020")
                ty += line_height
            cx += col_w
        y += row_height
    return y


def page_cover(data: BreakwaterInputs, result: dict[str, float | str | bool]) -> Image.Image:
    page = new_page()
    draw = ImageDraw.Draw(page)
    y = draw_title(draw, data.assignment_code, data.project_name, subtitle_font=FONT_H2)

    draw.rectangle(
        (MARGIN, y, PAGE_SIZE[0] - MARGIN, y + 150),
        fill="#F4EFE4",
        outline="#C7A35A",
        width=2,
    )
    group_y = y + 20
    draw.text((MARGIN + 24, group_y), data.group_name, font=FONT_H2, fill="#111111")
    group_y += 40
    for member in data.group_members:
        draw.text((MARGIN + 24, group_y), member, font=FONT_BODY_BOLD, fill="#202020")
        group_y += 34
    draw.text((MARGIN + 24, group_y + 4), data.course_name, font=FONT_SMALL, fill="#555555")
    y += 184

    if data.example_values:
        status = (
            "This report currently uses example values. Replace USER_INPUTS in the "
            "Python file with the values from HW5 before submitting."
        )
    else:
        status = "This report uses the project values entered in USER_INPUTS."
    y = draw_wrapped(draw, status, MARGIN, y, PAGE_SIZE[0] - 2 * MARGIN, FONT_BODY_BOLD)
    y += 35

    y = draw_section_label(draw, "Objective", MARGIN, y)
    y = draw_wrapped(
        draw,
        "Design/check the breakwater layout so the diffracted wave height at the "
        "quay wall is below the serviceability limit required for the selected "
        "harbor type.",
        MARGIN,
        y,
        PAGE_SIZE[0] - 2 * MARGIN,
    )
    y += 35

    y = draw_section_label(draw, "Homework 6 Workflow", MARGIN, y)
    workflow = (
        "1. Take the 10-year design wave height Hi and corresponding period T from HW5.\n"
        "2. Use the wave angle alpha from HW5, or calculate it by refraction analysis.\n"
        "3. Select the correct diffraction diagram using the angle between the incident wave and the breakwater.\n"
        "4. Read the diffraction coefficient Kd at the quay-wall location.\n"
        "5. Check Hq = Kd Hi against the allowable harbor limit."
    )
    y = draw_wrapped(draw, workflow, MARGIN, y, PAGE_SIZE[0] - 2 * MARGIN)
    y += 35

    y = draw_section_label(draw, "Current Design Check Summary", MARGIN, y)
    summary_rows = [
        ["Incident design wave height, Hi", f"{data.design_wave_height_m:.3f} m"],
        ["Diagram diffraction coefficient, Kd", f"{data.diffraction_coefficient_from_diagram:.3f}"],
        ["Calculated quay-wall wave height, Hq = Kd Hi", f"{float(result['quay_wave_height_m']):.3f} m"],
        ["Allowable wave height, Hallow", f"{float(result['allowable_wave_height_m']):.3f} m"],
        ["Decision", str(result["decision"])],
    ]
    y = draw_table(
        draw,
        MARGIN,
        y,
        PAGE_SIZE[0] - 2 * MARGIN,
        ["Design item", "Value"],
        summary_rows,
        [0.65, 0.35],
        row_font=FONT_BODY,
        header_font=FONT_BODY_BOLD,
    )
    y += 34
    draw_wrapped(
        draw,
        "This summary is automatically updated when the values in USER_INPUTS are changed "
        "and the Python file is run again.",
        MARGIN,
        y,
        PAGE_SIZE[0] - 2 * MARGIN,
        FONT_BODY_BOLD,
    )
    y += 52
    draw.rectangle(
        (MARGIN, y, PAGE_SIZE[0] - MARGIN, y + 230),
        fill="#F4EFE4",
        outline="#C7A35A",
        width=2,
    )
    y += 22
    draw.text((MARGIN + 24, y), "Design equation chain", font=FONT_H2, fill="#111111")
    y += 44
    chain = (
        "HW5 wave condition -> wavelength L -> diagram coordinates r/L and theta -> "
        "diffraction coefficient Kd -> quay-wall height Hq = Kd Hi -> final check "
        "Hq <= Hallow. This is the complete logic used throughout the report."
    )
    draw_wrapped(draw, chain, MARGIN + 24, y, PAGE_SIZE[0] - 2 * MARGIN - 48, FONT_BODY)
    return page


def page_formulas(data: BreakwaterInputs, result: dict[str, float | str | bool]) -> Image.Image:
    page = new_page()
    draw = ImageDraw.Draw(page)
    y = draw_title(draw, "Calculation Method", "Formulas are written where they are used")

    sections = [
        (
            "1. Convert the HW5 wave period into a wavelength",
            "The design wave period T is needed because diffraction diagrams are read "
            "using the non-dimensional distance r/L. First, the deep-water estimate is "
            "L0 = g T^2 / (2 pi). Because the harbor has finite depth, the wavelength "
            "used in the layout check is corrected with the dispersion equation "
            "L = (g T^2 / 2 pi) tanh(2 pi h / L). The script solves this equation "
            "iteratively until L stops changing."
        ),
        (
            "2. Locate the quay-wall point on the diffraction diagram",
            "The quay-wall check point is measured from the breakwater head. If the "
            "breakwater tip is (xt, yt) and the quay point is (xq, yq), the distance is "
            "r = sqrt((xq - xt)^2 + (yq - yt)^2). The value used on the diagram is not "
            "r alone, but r/L. The angular location is also needed, so the script uses "
            "theta = atan2(yq - yt, xq - xt) - beta_BW, where beta_BW is the breakwater "
            "axis direction."
        ),
        (
            "3. Read the diffraction coefficient",
            "The correct diagram is selected from the incident wave angle relative to "
            "the breakwater. At the computed r/L and theta, the diagram gives the "
            "diffraction coefficient Kd. By definition, Kd = Hq / Hi, so it represents "
            "the fraction of the incident design wave height that reaches the quay wall."
        ),
        (
            "4. Calculate the wave height at the quay wall",
            "After Kd is read from the diagram, the quay-wall wave height is calculated "
            "directly as Hq = Kd Hi. This is the main homework result because it tells "
            "how much wave agitation remains after the breakwater shelters the harbor."
        ),
        (
            "5. Check the allowable harbor limit",
            "The design passes when Hq <= Hallow. The assignment gives Hallow = 0.30 m "
            "for marinas and fishery harbors, and Hallow = 0.50 m for container "
            "terminals. The required coefficient can also be written as "
            "Kd,req = Hallow / Hi. Therefore, the layout is acceptable only if the "
            "diagram value satisfies Kd <= Kd,req."
        ),
        (
            "6. Report the safety margin",
            "The safety margin is Margin = Hallow - Hq. A positive value means the "
            "quay-wall wave height is below the limit. A negative value means the "
            "breakwater layout must be revised and the diagram must be read again."
        ),
    ]

    for heading, body in sections:
        y = draw_section_label(draw, heading, MARGIN, y)
        y = draw_wrapped(draw, body, MARGIN, y, PAGE_SIZE[0] - 2 * MARGIN, FONT_BODY)
        y += 24

    y += 10
    draw.rectangle(
        (MARGIN, y, PAGE_SIZE[0] - MARGIN, y + 250),
        fill="#F4EFE4",
        outline="#C7A35A",
        width=2,
    )
    y += 24
    draw.text((MARGIN + 24, y), "Worked use with the current input values", font=FONT_H2, fill="#111111")
    y += 42
    worked = (
        f"For the present example, T = {data.design_wave_period_s:.2f} s and "
        f"h = {data.water_depth_m:.2f} m give L = {float(result['wavelength_m']):.3f} m. "
        f"The distance from the breakwater tip to the quay wall is "
        f"r = {float(result['distance_tip_to_quay_m']):.3f} m, so the diagram coordinate "
        f"is r/L = {float(result['r_over_L']):.3f}. With Kd = "
        f"{data.diffraction_coefficient_from_diagram:.3f}, the protected wave height is "
        f"Hq = Kd Hi = {data.diffraction_coefficient_from_diagram:.3f} x "
        f"{data.design_wave_height_m:.3f} = {float(result['quay_wave_height_m']):.3f} m. "
        f"The required value is Kd,req = Hallow/Hi = "
        f"{float(result['required_diffraction_coefficient']):.3f}."
    )
    draw_wrapped(draw, worked, MARGIN + 24, y, PAGE_SIZE[0] - 2 * MARGIN - 48, FONT_BODY)
    return page


def page_design_basis(data: BreakwaterInputs) -> Image.Image:
    page = new_page()
    draw = ImageDraw.Draw(page)
    y = draw_title(draw, "Design Basis", "What is being checked and why")

    paragraphs = [
        (
            "The breakwater layout is checked as a serviceability problem. The main "
            "question is not whether the structure survives the offshore design wave; "
            "the question is whether the wave agitation reaching the quay wall is small "
            "enough for safe and practical harbor operation."
        ),
        (
            "The incident design condition is taken from HW5: the 10-year design wave "
            "height Hi in front of the breakwater and its corresponding period T. Those "
            "values define the wave length scale, the diagram coordinates, and the "
            "wave height that must be reduced by diffraction."
        ),
        (
            "Wave diffraction occurs because the breakwater blocks part of the incoming "
            "wave front. Near the breakwater head, wave energy bends into the protected "
            "area. The farther the quay wall is inside the shadow zone, and the less "
            "directly exposed it is to the opening, the smaller the diffraction "
            "coefficient Kd should become."
        ),
        (
            f"For a {data.harbor_type} layout, the assignment limit used here is "
            f"Hallow = {allowable_wave_height_m(data.harbor_type):.2f} m. The design "
            "passes only when the calculated quay-wall wave height Hq is less than or "
            "equal to that limit."
        ),
    ]
    for paragraph in paragraphs:
        y = draw_wrapped(draw, paragraph, MARGIN, y, PAGE_SIZE[0] - 2 * MARGIN)
        y += 28

    y = draw_section_label(draw, "Important assumptions", MARGIN, y)
    assumptions = (
        "1. The design wave height, period, and approach angle are taken as already "
        "established by HW5.\n"
        "2. The selected diffraction diagram represents the breakwater-head geometry "
        "well enough for homework-level design.\n"
        "3. The quay-wall check point represents the most critical protected location.\n"
        "4. Reflection, resonance, transmission through the structure, and local berth "
        "geometry are not included unless the instructor asks for them.\n"
        "5. If the diagram value gives Hq above the allowable limit, the layout must be "
        "changed and checked again."
    )
    draw_wrapped(draw, assumptions, MARGIN, y, PAGE_SIZE[0] - 2 * MARGIN, FONT_SMALL)
    return page


def page_wave_theory_notes(data: BreakwaterInputs, result: dict[str, float | str | bool]) -> Image.Image:
    page = new_page()
    draw = ImageDraw.Draw(page)
    y = draw_title(draw, "Wave Theory Notes", "Why wavelength matters in the diffraction diagram")

    l0 = G * data.design_wave_period_s**2 / (2.0 * math.pi)
    l = float(result["wavelength_m"])
    h_over_l = data.water_depth_m / l
    if h_over_l > 0.5:
        depth_class = "deep-water range"
    elif h_over_l < 0.05:
        depth_class = "shallow-water range"
    else:
        depth_class = "intermediate-water range"

    text = (
        "Diffraction diagrams normally use non-dimensional coordinates. That means the "
        "actual plan distance from the breakwater tip to the quay wall is divided by "
        "the local wavelength. A 100 m distance is not equally protected for every wave; "
        "it is more protected for short waves than for long waves."
    )
    y = draw_wrapped(draw, text, MARGIN, y, PAGE_SIZE[0] - 2 * MARGIN)
    y += 35

    rows = [
        ["Deep-water wavelength L0", f"{l0:.3f} m"],
        ["Finite-depth wavelength L", f"{l:.3f} m"],
        ["Water depth h", f"{data.water_depth_m:.3f} m"],
        ["Relative depth h/L", f"{h_over_l:.3f}"],
        ["Depth classification", depth_class],
        ["Wave celerity C", f"{float(result['celerity_m_per_s']):.3f} m/s"],
    ]
    y = draw_table(
        draw,
        MARGIN,
        y,
        PAGE_SIZE[0] - 2 * MARGIN,
        ["Wave parameter", "Value"],
        rows,
        [0.50, 0.50],
        row_font=FONT_BODY,
        header_font=FONT_BODY_BOLD,
    )
    y += 40
    note = (
        "For this homework, the wavelength is mainly used to compute r/L before reading "
        "the diffraction diagram. If the water depth or wave period changes, L changes, "
        "so the same physical layout can correspond to a different point on the diagram."
    )
    draw_wrapped(draw, note, MARGIN, y, PAGE_SIZE[0] - 2 * MARGIN)
    return page


def page_inputs(data: BreakwaterInputs) -> Image.Image:
    page = new_page()
    draw = ImageDraw.Draw(page)
    y = draw_title(draw, "Input Data", "Replace example values with the HW5/project values.")
    rows = [
        ["Project", data.project_name],
        ["Harbor type", data.harbor_type],
        ["Design wave height Hi", f"{data.design_wave_height_m:.3f} m"],
        ["Design wave period T", f"{data.design_wave_period_s:.3f} s"],
        ["Water depth h", f"{data.water_depth_m:.3f} m"],
        ["Incident angle to breakwater", f"{data.incident_angle_to_breakwater_deg:.2f} deg"],
        ["Diffraction coefficient Kd", f"{data.diffraction_coefficient_from_diagram:.3f}"],
        ["Breakwater axis", f"{data.breakwater_axis_deg:.2f} deg"],
        ["Breakwater tip", f"{data.breakwater_tip_xy_m} m"],
        ["Quay-wall check point", f"{data.quay_wall_point_xy_m} m"],
    ]
    draw_table(
        draw,
        MARGIN,
        y,
        PAGE_SIZE[0] - 2 * MARGIN,
        ["Input", "Value"],
        rows,
        [0.42, 0.58],
        row_font=FONT_BODY,
        header_font=FONT_BODY_BOLD,
    )
    return page


def page_results(data: BreakwaterInputs, result: dict[str, float | str | bool]) -> Image.Image:
    page = new_page()
    draw = ImageDraw.Draw(page)
    y = draw_title(draw, "Calculation Results", "Diffraction check at the quay-wall point.")
    rows = [
        ["Wavelength L", f"{float(result['wavelength_m']):.3f} m"],
        ["Wave number k", f"{float(result['wave_number_rad_per_m']):.5f} rad/m"],
        ["Wave celerity C", f"{float(result['celerity_m_per_s']):.3f} m/s"],
        ["Tip-to-quay distance r", f"{float(result['distance_tip_to_quay_m']):.3f} m"],
        ["Dimensionless distance r/L", f"{float(result['r_over_L']):.3f}"],
        ["Quay angle from breakwater", f"{float(result['quay_angle_from_breakwater_deg']):.2f} deg"],
        ["Wave height at quay Hq", f"{float(result['quay_wave_height_m']):.3f} m"],
        ["Allowable wave height", f"{float(result['allowable_wave_height_m']):.3f} m"],
        ["Required Kd or lower", f"{float(result['required_diffraction_coefficient']):.3f}"],
        ["Safety margin", f"{float(result['safety_margin_m']):+.3f} m"],
        ["Decision", str(result["decision"])],
    ]
    y = draw_table(
        draw,
        MARGIN,
        y,
        PAGE_SIZE[0] - 2 * MARGIN,
        ["Quantity", "Result"],
        rows,
        [0.55, 0.45],
        row_font=FONT_BODY,
        header_font=FONT_BODY_BOLD,
    )
    y += 40
    if result["accepted"]:
        note = "The safety margin is positive, so the checked point satisfies the selected harbor limit."
        color = "#1B7F3A"
    else:
        note = "The safety margin is negative, so the layout must be revised and checked again."
        color = "#B00020"
    y = draw_wrapped(draw, note, MARGIN, y, PAGE_SIZE[0] - 2 * MARGIN, FONT_BODY_BOLD, fill=color)
    y += 38

    y = draw_section_label(draw, "Calculation sequence", MARGIN, y)
    sequence = (
        f"1. The finite-depth dispersion equation gives L = "
        f"{float(result['wavelength_m']):.3f} m for T = {data.design_wave_period_s:.2f} s "
        f"and h = {data.water_depth_m:.2f} m.\n"
        f"2. The geometry gives r = {float(result['distance_tip_to_quay_m']):.3f} m, "
        f"therefore r/L = {float(result['r_over_L']):.3f} for the diffraction diagram.\n"
        f"3. The selected diagram gives Kd = {data.diffraction_coefficient_from_diagram:.3f} "
        f"at the quay-wall point.\n"
        f"4. The protected wave height is Hq = Kd Hi = "
        f"{data.diffraction_coefficient_from_diagram:.3f} x {data.design_wave_height_m:.3f} "
        f"= {float(result['quay_wave_height_m']):.3f} m.\n"
        f"5. The margin is Hallow - Hq = {float(result['allowable_wave_height_m']):.3f} - "
        f"{float(result['quay_wave_height_m']):.3f} = {float(result['safety_margin_m']):+.3f} m."
    )
    draw_wrapped(draw, sequence, MARGIN, y, PAGE_SIZE[0] - 2 * MARGIN, FONT_BODY)
    return page


def page_diagram_method() -> Image.Image:
    page = new_page()
    draw = ImageDraw.Draw(page)
    y = draw_title(draw, "Diffraction Diagram Use", "How the diagram value enters the design")
    intro = (
        "Use the incident wave angle relative to the breakwater to choose the correct "
        "diffraction diagram. Then locate the quay-wall point using the dimensionless "
        "radius r/L and its angular position behind the breakwater. The diagram gives "
        "Kd at that point."
    )
    y = draw_wrapped(draw, intro, MARGIN, y, PAGE_SIZE[0] - 2 * MARGIN)
    y += 35
    rows = [
        ["1", "Calculate wavelength L from T and h."],
        ["2", "Measure r from the breakwater tip to the quay-wall check point."],
        ["3", "Calculate r/L."],
        ["4", "Measure angular position of the quay point behind the breakwater."],
        ["5", "Read Kd from the diagram chosen for the incident angle."],
        ["6", "Compute Hq = Kd Hi and compare with the allowable limit."],
    ]
    y = draw_table(
        draw,
        MARGIN,
        y,
        PAGE_SIZE[0] - 2 * MARGIN,
        ["Step", "Action"],
        rows,
        [0.12, 0.88],
        row_font=FONT_BODY,
        header_font=FONT_BODY_BOLD,
    )
    y += 40
    note = (
        "Important: this script does not replace the required diffraction diagram. "
        "It records the diagram coefficient and completes the design check consistently."
    )
    draw_wrapped(draw, note, MARGIN, y, PAGE_SIZE[0] - 2 * MARGIN, FONT_BODY_BOLD)
    return page


def map_point(
    x: float,
    y: float,
    bounds: tuple[float, float, float, float],
    plot_box: tuple[int, int, int, int],
) -> tuple[int, int]:
    min_x, max_x, min_y, max_y = bounds
    left, top, right, bottom = plot_box
    px = left + (x - min_x) / (max_x - min_x) * (right - left)
    py = bottom - (y - min_y) / (max_y - min_y) * (bottom - top)
    return int(px), int(py)


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str,
    width: int = 5,
    head: int = 18,
) -> None:
    draw.line((start[0], start[1], end[0], end[1]), fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    left = (
        end[0] - head * math.cos(angle - math.pi / 6),
        end[1] - head * math.sin(angle - math.pi / 6),
    )
    right = (
        end[0] - head * math.cos(angle + math.pi / 6),
        end[1] - head * math.sin(angle + math.pi / 6),
    )
    draw.polygon([end, (int(left[0]), int(left[1])), (int(right[0]), int(right[1]))], fill=color)


def page_layout(data: BreakwaterInputs, result: dict[str, float | str | bool]) -> Image.Image:
    page = new_page()
    draw = ImageDraw.Draw(page)
    y = draw_title(draw, "Plan Schematic", "Local layout coordinates used for the check")

    plot_box = (MARGIN + 45, y + 45, PAGE_SIZE[0] - MARGIN - 45, PAGE_SIZE[1] - 245)
    left, top, right, bottom = plot_box
    draw.rectangle(plot_box, outline="#999999", width=2)

    tip = data.breakwater_tip_xy_m
    quay = data.quay_wall_point_xy_m
    axis_rad = math.radians(data.breakwater_axis_deg)
    axis_vector = (math.cos(axis_rad), math.sin(axis_rad))
    land_end = (tip[0] - 120.0 * axis_vector[0], tip[1] - 120.0 * axis_vector[1])
    wave_rad = math.radians(data.breakwater_axis_deg + data.incident_angle_to_breakwater_deg + 180.0)
    arrow_start = (tip[0] + 95.0 * math.cos(wave_rad), tip[1] + 95.0 * math.sin(wave_rad))

    xs = [tip[0], quay[0], land_end[0], arrow_start[0]]
    ys = [tip[1], quay[1], land_end[1], arrow_start[1]]
    min_x, max_x = min(xs) - 35, max(xs) + 35
    min_y, max_y = min(ys) - 35, max(ys) + 35
    if max_x - min_x < 1:
        max_x += 1
    if max_y - min_y < 1:
        max_y += 1
    bounds = (min_x, max_x, min_y, max_y)

    # Grid
    for i in range(6):
        gx = left + i * (right - left) / 5
        gy = top + i * (bottom - top) / 5
        draw.line((int(gx), top, int(gx), bottom), fill="#E1E1E1", width=1)
        draw.line((left, int(gy), right, int(gy)), fill="#E1E1E1", width=1)

    p_land = map_point(*land_end, bounds, plot_box)
    p_tip = map_point(*tip, bounds, plot_box)
    p_quay = map_point(*quay, bounds, plot_box)
    p_wave_start = map_point(*arrow_start, bounds, plot_box)

    draw.line((*p_land, *p_tip), fill="#303030", width=14)
    draw.ellipse((p_tip[0] - 9, p_tip[1] - 9, p_tip[0] + 9, p_tip[1] + 9), fill="#303030")
    draw.line((*p_tip, *p_quay), fill="#666666", width=3)
    draw.ellipse((p_quay[0] - 11, p_quay[1] - 11, p_quay[0] + 11, p_quay[1] + 11), fill="#1F77B4")
    wave_mid = (
        int(p_wave_start[0] + 0.62 * (p_tip[0] - p_wave_start[0])),
        int(p_wave_start[1] + 0.62 * (p_tip[1] - p_wave_start[1])),
    )
    draw_arrow(draw, p_wave_start, wave_mid, "#D62728", width=6, head=24)

    label_x, label_y = left + 24, bottom + 25
    legend = (
        "Black line: breakwater | Blue point: quay-wall check point | "
        "Red arrow: incident waves | Gray line: r from tip to quay"
    )
    draw_wrapped(draw, legend, label_x, label_y, right - left - 48, FONT_SMALL)

    values = (
        f"r = {float(result['distance_tip_to_quay_m']):.1f} m\n"
        f"L = {float(result['wavelength_m']):.1f} m\n"
        f"r/L = {float(result['r_over_L']):.2f}\n"
        f"angle to quay = {float(result['quay_angle_from_breakwater_deg']):.1f} deg"
    )
    box_x, box_y = right - 315, top + 24
    draw.rectangle((box_x - 14, box_y - 12, right - 18, box_y + 132), fill="white", outline="#AAAAAA")
    draw_wrapped(draw, values, box_x, box_y, 270, FONT_SMALL)
    return page


def page_sensitivity(data: BreakwaterInputs, result: dict[str, float | str | bool]) -> Image.Image:
    page = new_page()
    draw = ImageDraw.Draw(page)
    y = draw_title(draw, "Wave Height Sensitivity", "Hq as a function of the diffraction coefficient")

    plot = (MARGIN + 80, y + 55, PAGE_SIZE[0] - MARGIN - 40, PAGE_SIZE[1] - 265)
    left, top, right, bottom = plot
    max_h = max(data.design_wave_height_m, float(result["allowable_wave_height_m"])) * 1.08

    def sx(kd: float) -> int:
        return int(left + kd * (right - left))

    def sy(h: float) -> int:
        return int(bottom - h / max_h * (bottom - top))

    draw.rectangle(plot, outline="#999999", width=2)
    for i in range(6):
        kd = i / 5
        px = sx(kd)
        draw.line((px, top, px, bottom), fill="#E1E1E1", width=1)
        draw.text((px - 14, bottom + 12), f"{kd:.1f}", font=FONT_SMALL, fill="#333333")
    for i in range(6):
        h = max_h * i / 5
        py = sy(h)
        draw.line((left, py, right, py), fill="#E1E1E1", width=1)
        draw.text((left - 70, py - 10), f"{h:.2f}", font=FONT_SMALL, fill="#333333")

    points = [(sx(kd / 100), sy((kd / 100) * data.design_wave_height_m)) for kd in range(101)]
    draw.line(points, fill="#1F77B4", width=5)

    limit = float(result["allowable_wave_height_m"])
    kd_required = float(result["required_diffraction_coefficient"])
    kd_current = data.diffraction_coefficient_from_diagram
    h_current = float(result["quay_wave_height_m"])

    draw.line((left, sy(limit), right, sy(limit)), fill="#D62728", width=4)
    draw.line((sx(min(kd_required, 1.0)), top, sx(min(kd_required, 1.0)), bottom), fill="#2CA02C", width=4)
    draw.line((sx(min(kd_current, 1.0)), top, sx(min(kd_current, 1.0)), bottom), fill="#111111", width=3)
    draw.ellipse(
        (
            sx(min(kd_current, 1.0)) - 8,
            sy(h_current) - 8,
            sx(min(kd_current, 1.0)) + 8,
            sy(h_current) + 8,
        ),
        fill="#111111",
    )

    draw.text((left, bottom + 55), "Diffraction coefficient, Kd", font=FONT_SMALL_BOLD, fill="#111111")
    draw.text((left - 75, top - 35), "Hq (m)", font=FONT_SMALL_BOLD, fill="#111111")
    legend_y = bottom + 95
    draw.text((left, legend_y), "Blue: Hq = Kd Hi", font=FONT_SMALL, fill="#1F77B4")
    draw.text((left + 260, legend_y), "Red: allowable limit", font=FONT_SMALL, fill="#D62728")
    draw.text((left + 540, legend_y), "Green: required Kd", font=FONT_SMALL, fill="#2CA02C")
    draw.text((left + 820, legend_y), "Black: diagram Kd", font=FONT_SMALL, fill="#111111")

    note = (
        f"To satisfy the criterion, the final layout must give Kd <= {kd_required:.3f}. "
        "If the diagram value is larger than this, revise the layout and read a new "
        "diffraction coefficient."
    )
    draw_wrapped(draw, note, MARGIN, PAGE_SIZE[1] - 130, PAGE_SIZE[0] - 2 * MARGIN, FONT_BODY)
    return page


def page_required_kd_curve(data: BreakwaterInputs, result: dict[str, float | str | bool]) -> Image.Image:
    page = new_page()
    draw = ImageDraw.Draw(page)
    y = draw_title(draw, "Required Diffraction Reduction", "Maximum Kd allowed for different design wave heights")

    plot = (MARGIN + 85, y + 55, PAGE_SIZE[0] - MARGIN - 40, PAGE_SIZE[1] - 280)
    left, top, right, bottom = plot
    limit = float(result["allowable_wave_height_m"])
    current_hi = data.design_wave_height_m
    max_hi = max(3.0, current_hi * 2.5, limit * 4.0)

    def sx(hi: float) -> int:
        return int(left + hi / max_hi * (right - left))

    def sy(kd: float) -> int:
        return int(bottom - kd / 1.2 * (bottom - top))

    draw.rectangle(plot, outline="#999999", width=2)
    for i in range(6):
        hi = max_hi * i / 5
        px = sx(hi)
        draw.line((px, top, px, bottom), fill="#E1E1E1", width=1)
        draw.text((px - 20, bottom + 12), f"{hi:.1f}", font=FONT_SMALL, fill="#333333")
    for i in range(7):
        kd = 1.2 * i / 6
        py = sy(kd)
        draw.line((left, py, right, py), fill="#E1E1E1", width=1)
        draw.text((left - 60, py - 10), f"{kd:.1f}", font=FONT_SMALL, fill="#333333")

    points = []
    for i in range(1, 301):
        hi = max_hi * i / 300
        kd_req = min(limit / hi, 1.2)
        points.append((sx(hi), sy(kd_req)))
    draw.line(points, fill="#1F77B4", width=5)

    kd_current = data.diffraction_coefficient_from_diagram
    kd_required = float(result["required_diffraction_coefficient"])
    p_current = (sx(min(current_hi, max_hi)), sy(min(kd_current, 1.2)))
    p_required = (sx(min(current_hi, max_hi)), sy(min(kd_required, 1.2)))
    draw.line((p_required[0], top, p_required[0], bottom), fill="#555555", width=3)
    draw.ellipse((p_current[0] - 9, p_current[1] - 9, p_current[0] + 9, p_current[1] + 9), fill="#111111")
    draw.ellipse((p_required[0] - 8, p_required[1] - 8, p_required[0] + 8, p_required[1] + 8), fill="#2CA02C")

    draw.text((left, bottom + 55), "Design wave height, Hi (m)", font=FONT_SMALL_BOLD, fill="#111111")
    draw.text((left - 75, top - 35), "Kd,req", font=FONT_SMALL_BOLD, fill="#111111")
    draw.text((left, bottom + 95), "Blue curve: Kd,req = Hallow / Hi", font=FONT_SMALL, fill="#1F77B4")
    draw.text((left + 360, bottom + 95), "Black point: diagram Kd at current Hi", font=FONT_SMALL, fill="#111111")
    draw.text((left + 760, bottom + 95), "Green point: required Kd", font=FONT_SMALL, fill="#2CA02C")

    note = (
        "This figure shows why larger design waves require stronger sheltering. If Hi "
        "increases while the allowable harbor agitation stays fixed, the maximum "
        "acceptable Kd becomes smaller."
    )
    draw_wrapped(draw, note, MARGIN, PAGE_SIZE[1] - 130, PAGE_SIZE[0] - 2 * MARGIN, FONT_BODY)
    return page


def page_pass_fail_map(data: BreakwaterInputs, result: dict[str, float | str | bool]) -> Image.Image:
    page = new_page()
    draw = ImageDraw.Draw(page)
    y = draw_title(draw, "Pass/Fail Design Envelope", "Combination of Hi and Kd that satisfies the harbor limit")

    plot = (MARGIN + 85, y + 55, PAGE_SIZE[0] - MARGIN - 40, PAGE_SIZE[1] - 280)
    left, top, right, bottom = plot
    limit = float(result["allowable_wave_height_m"])
    max_hi = max(3.0, data.design_wave_height_m * 2.5, limit * 4.0)
    max_kd = 1.0

    def sx(hi: float) -> int:
        return int(left + hi / max_hi * (right - left))

    def sy(kd: float) -> int:
        return int(bottom - kd / max_kd * (bottom - top))

    cols = 55
    rows = 36
    for i in range(cols):
        hi0 = max_hi * i / cols
        hi1 = max_hi * (i + 1) / cols
        for j in range(rows):
            kd0 = max_kd * j / rows
            kd1 = max_kd * (j + 1) / rows
            hi_mid = 0.5 * (hi0 + hi1)
            kd_mid = 0.5 * (kd0 + kd1)
            color = "#DDF1DF" if hi_mid * kd_mid <= limit else "#F6D6D6"
            draw.rectangle((sx(hi0), sy(kd1), sx(hi1), sy(kd0)), fill=color, outline=color)

    draw.rectangle(plot, outline="#999999", width=2)
    for i in range(6):
        hi = max_hi * i / 5
        px = sx(hi)
        draw.line((px, top, px, bottom), fill="#FFFFFF", width=1)
        draw.text((px - 20, bottom + 12), f"{hi:.1f}", font=FONT_SMALL, fill="#333333")
    for i in range(6):
        kd = max_kd * i / 5
        py = sy(kd)
        draw.line((left, py, right, py), fill="#FFFFFF", width=1)
        draw.text((left - 60, py - 10), f"{kd:.1f}", font=FONT_SMALL, fill="#333333")

    boundary = []
    for i in range(1, 301):
        hi = max_hi * i / 300
        kd = min(limit / hi, max_kd)
        boundary.append((sx(hi), sy(kd)))
    draw.line(boundary, fill="#111111", width=5)

    current = (sx(min(data.design_wave_height_m, max_hi)), sy(min(data.diffraction_coefficient_from_diagram, max_kd)))
    draw.ellipse((current[0] - 10, current[1] - 10, current[0] + 10, current[1] + 10), fill="#1F77B4")
    draw.text((current[0] + 12, current[1] - 26), "current check", font=FONT_SMALL_BOLD, fill="#1F77B4")

    draw.text((left, bottom + 55), "Design wave height, Hi (m)", font=FONT_SMALL_BOLD, fill="#111111")
    draw.text((left - 70, top - 35), "Kd", font=FONT_SMALL_BOLD, fill="#111111")
    draw.text((left, bottom + 95), "Green area: Hq <= Hallow", font=FONT_SMALL, fill="#1B7F3A")
    draw.text((left + 310, bottom + 95), "Red area: Hq > Hallow", font=FONT_SMALL, fill="#B00020")
    draw.text((left + 600, bottom + 95), "Black boundary: Hq = Hallow", font=FONT_SMALL, fill="#111111")

    note = (
        "The layout passes only inside the green region. Moving downward means reducing "
        "Kd by improving the layout. Moving left means the design wave height is smaller, "
        "which is not a layout change and should only come from the HW5 design condition."
    )
    draw_wrapped(draw, note, MARGIN, PAGE_SIZE[1] - 130, PAGE_SIZE[0] - 2 * MARGIN, FONT_BODY)
    return page


def page_wavelength_depth_curve(data: BreakwaterInputs, result: dict[str, float | str | bool]) -> Image.Image:
    page = new_page()
    draw = ImageDraw.Draw(page)
    y = draw_title(draw, "Wavelength Versus Depth", "Effect of water depth for the selected wave period")

    plot = (MARGIN + 85, y + 55, PAGE_SIZE[0] - MARGIN - 40, PAGE_SIZE[1] - 280)
    left, top, right, bottom = plot
    max_depth = max(20.0, data.water_depth_m * 2.5)
    depths = [max_depth * i / 240 for i in range(1, 241)]
    lengths = [solve_wavelength_m(data.design_wave_period_s, h) for h in depths]
    max_l = max(lengths) * 1.08

    def sx(depth: float) -> int:
        return int(left + depth / max_depth * (right - left))

    def sy(length: float) -> int:
        return int(bottom - length / max_l * (bottom - top))

    draw.rectangle(plot, outline="#999999", width=2)
    for i in range(6):
        depth = max_depth * i / 5
        px = sx(depth)
        draw.line((px, top, px, bottom), fill="#E1E1E1", width=1)
        draw.text((px - 20, bottom + 12), f"{depth:.0f}", font=FONT_SMALL, fill="#333333")
    for i in range(6):
        length = max_l * i / 5
        py = sy(length)
        draw.line((left, py, right, py), fill="#E1E1E1", width=1)
        draw.text((left - 70, py - 10), f"{length:.0f}", font=FONT_SMALL, fill="#333333")

    points = [(sx(depth), sy(length)) for depth, length in zip(depths, lengths)]
    draw.line(points, fill="#1F77B4", width=5)

    current = (sx(min(data.water_depth_m, max_depth)), sy(float(result["wavelength_m"])))
    draw.line((current[0], top, current[0], bottom), fill="#555555", width=3)
    draw.ellipse((current[0] - 9, current[1] - 9, current[0] + 9, current[1] + 9), fill="#111111")
    draw.text((current[0] + 14, current[1] - 25), "current depth", font=FONT_SMALL_BOLD, fill="#111111")

    draw.text((left, bottom + 55), "Water depth, h (m)", font=FONT_SMALL_BOLD, fill="#111111")
    draw.text((left - 78, top - 35), "L (m)", font=FONT_SMALL_BOLD, fill="#111111")
    draw.text((left, bottom + 95), f"Period held constant: T = {data.design_wave_period_s:.2f} s", font=FONT_SMALL, fill="#111111")

    note = (
        "This figure explains why the depth used in the layout check matters. A different "
        "depth gives a different wavelength, which changes r/L and therefore the point "
        "read from the diffraction diagram."
    )
    draw_wrapped(draw, note, MARGIN, PAGE_SIZE[1] - 130, PAGE_SIZE[0] - 2 * MARGIN, FONT_BODY)
    return page


def page_conceptual_diffraction() -> Image.Image:
    page = new_page()
    draw = ImageDraw.Draw(page)
    y = draw_title(draw, "Conceptual Diffraction Figure", "Wave bending around the breakwater head")

    plot = (MARGIN + 30, y + 20, PAGE_SIZE[0] - MARGIN - 30, PAGE_SIZE[1] - 265)
    left, top, right, bottom = plot
    draw.rectangle(plot, outline="#999999", width=2)
    water_fill = "#EAF5FB"
    draw.rectangle((left + 2, top + 2, right - 2, bottom - 2), fill=water_fill)

    bw_x = left + 390
    bw_y0 = top + 85
    bw_y1 = bottom - 90
    tip = (bw_x, bw_y0)
    draw.line((bw_x, bw_y0, bw_x, bw_y1), fill="#303030", width=18)
    draw.ellipse((bw_x - 12, bw_y0 - 12, bw_x + 12, bw_y0 + 12), fill="#303030")

    # Incoming parallel wave crests.
    for offset in range(-180, 540, 90):
        x0 = left + 40 + offset
        draw.line((x0, top + 70, x0 + 220, top + 260), fill="#1F77B4", width=4)
    draw_arrow(draw, (left + 110, top + 350), (left + 300, top + 520), "#D62728", width=7, head=28)
    draw.text((left + 65, top + 535), "incident waves", font=FONT_SMALL_BOLD, fill="#D62728")

    # Diffracted crests in the sheltered area.
    for radius in [95, 170, 245, 320, 395]:
        bbox = (tip[0] - radius, tip[1] - radius, tip[0] + radius, tip[1] + radius)
        draw.arc(bbox, start=0, end=105, fill="#1F77B4", width=4)
    draw.text((bw_x + 105, top + 195), "diffracted wave crests", font=FONT_SMALL_BOLD, fill="#1F77B4")

    # Shadow and quay wall.
    shadow = [(bw_x + 15, bw_y0 + 20), (right - 70, top + 185), (right - 70, bottom - 80), (bw_x + 15, bw_y1)]
    draw.polygon(shadow, fill="#DDF1DF")
    for radius in [95, 170, 245, 320, 395]:
        bbox = (tip[0] - radius, tip[1] - radius, tip[0] + radius, tip[1] + radius)
        draw.arc(bbox, start=0, end=105, fill="#1F77B4", width=4)
    draw.line((right - 190, bottom - 280, right - 190, bottom - 100), fill="#8C564B", width=10)
    draw.text((right - 250, bottom - 80), "quay wall", font=FONT_SMALL_BOLD, fill="#8C564B")
    draw.text((bw_x + 40, bottom - 180), "protected shadow zone", font=FONT_SMALL_BOLD, fill="#1B7F3A")
    draw.text((bw_x - 120, bw_y1 + 28), "breakwater", font=FONT_SMALL_BOLD, fill="#303030")

    explanation = (
        "This conceptual figure shows the physical reason for using Kd. The breakwater "
        "blocks the direct wave front, but energy bends around the head. The diagram "
        "coefficient Kd estimates how much of the incident wave height reaches the "
        "selected quay-wall point."
    )
    draw_wrapped(draw, explanation, MARGIN, PAGE_SIZE[1] - 135, PAGE_SIZE[0] - 2 * MARGIN, FONT_BODY)
    return page


def page_result_interpretation(data: BreakwaterInputs, result: dict[str, float | str | bool]) -> Image.Image:
    page = new_page()
    draw = ImageDraw.Draw(page)
    y = draw_title(draw, "Result Interpretation", "Engineering meaning of the calculated values")

    hq = float(result["quay_wave_height_m"])
    limit = float(result["allowable_wave_height_m"])
    margin = float(result["safety_margin_m"])
    kd_req = float(result["required_diffraction_coefficient"])
    kd = data.diffraction_coefficient_from_diagram
    decision = str(result["decision"])

    summary = (
        f"With the current input values, the diagram coefficient is Kd = {kd:.3f}. "
        f"The calculated quay-wall wave height is Hq = {hq:.3f} m, while the allowable "
        f"limit is Hallow = {limit:.3f} m. The resulting safety margin is "
        f"{margin:+.3f} m, so the decision is: {decision}."
    )
    y = draw_wrapped(draw, summary, MARGIN, y, PAGE_SIZE[0] - 2 * MARGIN, FONT_BODY_BOLD)
    y += 40

    y = draw_section_label(draw, "What the numbers mean", MARGIN, y)
    interpretation = (
        f"The required coefficient is Kd,req = {kd_req:.3f}. Any layout and diagram "
        f"reading with Kd less than or equal to {kd_req:.3f} satisfies the wave-height "
        "criterion for the chosen harbor type. A lower Kd means stronger sheltering. "
        "A higher Kd means the quay wall receives too much wave energy."
    )
    y = draw_wrapped(draw, interpretation, MARGIN, y, PAGE_SIZE[0] - 2 * MARGIN)
    y += 35

    y = draw_section_label(draw, "How to improve the design if necessary", MARGIN, y)
    design_logic = (
        "The most direct way to improve the result is to move the quay wall farther "
        "inside the protected region or increase the breakwater projection so the "
        "critical quay point lies on a lower Kd contour. Rotating the breakwater or "
        "narrowing the entrance can also reduce direct wave penetration. Each layout "
        "change should be followed by a new diagram reading, because the value of Kd "
        "depends on both angle and non-dimensional position."
    )
    y = draw_wrapped(draw, design_logic, MARGIN, y, PAGE_SIZE[0] - 2 * MARGIN)
    y += 35

    y = draw_section_label(draw, "Submission note", MARGIN, y)
    note = (
        "Before submission, replace the example inputs with your HW5/project values "
        "and attach or cite the diffraction diagram used to read Kd if your instructor "
        "requires showing the graphical reading."
    )
    draw_wrapped(draw, note, MARGIN, y, PAGE_SIZE[0] - 2 * MARGIN)
    return page


def page_revision_rule(result: dict[str, float | str | bool]) -> Image.Image:
    page = new_page()
    draw = ImageDraw.Draw(page)
    y = draw_title(draw, "If the Layout Fails", "Revision rule")
    y = draw_wrapped(
        draw,
        f"For this input set, the layout must satisfy Kd <= "
        f"{float(result['required_diffraction_coefficient']):.3f}. If the "
        "diffraction diagram gives a larger value, the wave height at the quay wall "
        "is too large.",
        MARGIN,
        y,
        PAGE_SIZE[0] - 2 * MARGIN,
        FONT_BODY_BOLD,
    )
    y += 40
    y = draw_section_label(draw, "Possible layout revisions", MARGIN, y)
    revisions = (
        "1. Increase the breakwater length so the quay wall is deeper inside the shadow zone.\n"
        "2. Rotate or reposition the breakwater so incident waves have less direct access to the quay.\n"
        "3. Reduce or reorient the harbor entrance opening.\n"
        "4. Add a secondary or return breakwater if a single breakwater cannot satisfy the limit.\n"
        "5. Move the quay-wall check point or layout farther from the diffraction tip, if the site plan allows it.\n\n"
        "After each revision, select/read the new diffraction coefficient from the proper diagram and repeat the check."
    )
    y = draw_wrapped(draw, revisions, MARGIN, y, PAGE_SIZE[0] - 2 * MARGIN)
    y += 42

    y = draw_section_label(draw, "Revision effect on the calculation", MARGIN, y)
    rows = [
        ["Longer breakwater", "Usually moves the quay point deeper into lower Kd contours."],
        ["Different orientation", "Changes the incident angle and may require a different diagram."],
        ["Narrower entrance", "Reduces direct exposure and can lower the selected Kd value."],
        ["Secondary breakwater", "Creates another sheltering boundary when one arm is not enough."],
        ["Relocated quay", "Changes r and theta, so the diagram reading changes."],
    ]
    y = draw_table(
        draw,
        MARGIN,
        y,
        PAGE_SIZE[0] - 2 * MARGIN,
        ["Revision", "Expected calculation effect"],
        rows,
        [0.34, 0.66],
        row_font=FONT_BODY,
        header_font=FONT_BODY_BOLD,
    )
    y += 34
    draw_wrapped(
        draw,
        "The revised design should be accepted only after the new diagram reading gives "
        "Hq = Kd Hi less than or equal to the allowable wave height.",
        MARGIN,
        y,
        PAGE_SIZE[0] - 2 * MARGIN,
        FONT_BODY_BOLD,
    )
    return page


def generate_pdf_report(
    data: BreakwaterInputs,
    result: dict[str, float | str | bool],
    output_pdf: Path = OUTPUT_DIR / "HW6_Breakwater_Report.pdf",
) -> Path:
    pages = [
        page_cover(data, result),
        page_design_basis(data),
        page_formulas(data, result),
        page_wave_theory_notes(data, result),
        page_inputs(data),
        page_results(data, result),
        page_result_interpretation(data, result),
        page_conceptual_diffraction(),
        page_diagram_method(),
        page_sensitivity(data, result),
        page_required_kd_curve(data, result),
        page_pass_fail_map(data, result),
        page_wavelength_depth_curve(data, result),
        page_layout(data, result),
        page_revision_rule(result),
    ]
    for index, page in enumerate(pages, start=1):
        decorate_page(page, index, len(pages))
    pages[0].save(
        output_pdf,
        "PDF",
        resolution=150.0,
        save_all=True,
        append_images=pages[1:],
    )
    return output_pdf


def main() -> None:
    result = evaluate_design(USER_INPUTS)
    pdf_path = generate_pdf_report(USER_INPUTS, result)

    print("HW6 breakwater diffraction check")
    print(f"Design wave height Hi: {USER_INPUTS.design_wave_height_m:.3f} m")
    print(f"Diffraction coefficient Kd: {USER_INPUTS.diffraction_coefficient_from_diagram:.3f}")
    print(f"Wave height at quay Hq: {float(result['quay_wave_height_m']):.3f} m")
    print(f"Allowable wave height: {float(result['allowable_wave_height_m']):.3f} m")
    print(f"Required Kd or lower: {float(result['required_diffraction_coefficient']):.3f}")
    print(f"Decision: {result['decision']}")
    print(f"PDF report saved to: {pdf_path}")


if __name__ == "__main__":
    main()
