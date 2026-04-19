"""
PDF export for RouteIQ itineraries.

Generates a branded, multi-page PDF with:
 - RouteIQ logo + name in the header of every page
 - Page number + proprietary notice in the footer of every page
 - Cover page with itinerary summary and key metrics
 - Stop-sequence table (paginated automatically)
 - Warnings & optimisation notes page (when present)
 - Closing brand page describing RouteIQ
"""

import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
    PageBreak,
)
from reportlab.platypus.flowables import HRFlowable
from reportlab.graphics.shapes import Drawing, Rect, Circle, Line, Polygon
from reportlab.graphics import renderPDF

# ── Brand colours (used in header/footer canvas drawing) ─────────────────────
GOLD        = colors.HexColor("#D4A843")
TEAL        = colors.HexColor("#2EA4A4")
SAGE        = colors.HexColor("#3FB950")
MIDNIGHT    = colors.HexColor("#0D1117")
CHARCOAL    = colors.HexColor("#161B22")
STEEL       = colors.HexColor("#30363D")
INK         = colors.HexColor("#8B949E")
WHITE_TEXT  = colors.HexColor("#F0F6FC")
BODY_TEXT   = colors.HexColor("#C9D1D9")
VIOLET      = colors.HexColor("#8957E5")
CRIMSON     = colors.HexColor("#E74C3C")

# ── PDF content colours (dark text on white paper) ────────────────────────────
PDF_TITLE   = colors.HexColor("#0D1117")
PDF_BODY    = colors.HexColor("#2D3748")
PDF_MUTED   = colors.HexColor("#666666")
PDF_ROW_ALT = colors.HexColor("#F5F5F5")
PDF_BORDER  = colors.HexColor("#CCCCCC")

PAGE_W, PAGE_H = A4          # 210 × 297 mm
MARGIN         = 18 * mm
HEADER_H       = 16 * mm
FOOTER_H       = 12 * mm

CONTENT_W = PAGE_W - 2 * MARGIN
CONTENT_Y_TOP    = PAGE_H - MARGIN - HEADER_H - 4 * mm
CONTENT_Y_BOTTOM = MARGIN + FOOTER_H + 4 * mm
CONTENT_H = CONTENT_Y_TOP - CONTENT_Y_BOTTOM


# ── Logo drawing ──────────────────────────────────────────────────────────────
def _make_logo(width=70, height=20):
    """Return a reportlab Drawing of the RouteIQ logotype."""
    d = Drawing(width, height)

    # Gold hexagon-ish map-pin background
    cx, cy, r = 9, 10, 8
    pts = []
    import math
    for i in range(6):
        angle = math.radians(60 * i - 30)
        pts += [cx + r * math.cos(angle), cy + r * math.sin(angle)]
    pin = Polygon(pts, fillColor=GOLD, strokeColor=colors.white, strokeWidth=0.5)
    d.add(pin)

    # White "R" inside the hexagon
    from reportlab.graphics.shapes import String
    d.add(String(cx - 3.5, cy - 4, "R",
                 fontName="Helvetica-Bold", fontSize=10, fillColor=colors.white))

    # "RouteIQ" text to the right
    d.add(String(21, cy - 5.5, "RouteIQ",
                 fontName="Helvetica-Bold", fontSize=13, fillColor=GOLD))
    return d


# ── Page callbacks ─────────────────────────────────────────────────────────────
def _draw_header(canvas, doc):
    canvas.saveState()

    # Background bar
    canvas.setFillColor(MIDNIGHT)
    canvas.rect(0, PAGE_H - MARGIN - HEADER_H, PAGE_W, HEADER_H + MARGIN, fill=1, stroke=0)

    # Gold bottom border of header
    canvas.setStrokeColor(GOLD)
    canvas.setLineWidth(1)
    canvas.line(MARGIN, PAGE_H - MARGIN - HEADER_H, PAGE_W - MARGIN, PAGE_H - MARGIN - HEADER_H)

    # Logo
    logo = _make_logo(70, 18)
    renderPDF.draw(logo, canvas, MARGIN, PAGE_H - MARGIN - HEADER_H + 1)

    # Doc subtitle on the right
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(INK)
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - MARGIN - HEADER_H + 7,
                           "Optimised Route Itinerary")
    canvas.restoreState()


def _draw_footer(canvas, doc):
    canvas.saveState()

    y_line = MARGIN + FOOTER_H - 2
    canvas.setStrokeColor(STEEL)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, y_line, PAGE_W - MARGIN, y_line)

    # Left: proprietary notice — two lines to avoid overflow
    canvas.setFont("Helvetica", 6)
    canvas.setFillColor(INK)
    canvas.drawString(MARGIN, MARGIN + 6.5,
                      "\u00a9 RouteIQ \u2014 Proprietary & Confidential. "
                      "Exclusive property of RouteIQ.")
    canvas.drawString(MARGIN, MARGIN + 1,
                      "Unauthorised reproduction, distribution, or use without "
                      "prior written permission is strictly prohibited.")

    # Right: page number (aligned to bottom notice line)
    canvas.setFont("Helvetica-Bold", 7)
    canvas.setFillColor(GOLD)
    canvas.drawRightString(PAGE_W - MARGIN, MARGIN + 1,
                           f"Page {doc.page}")

    canvas.restoreState()


def _draw_all(canvas, doc):
    _draw_header(canvas, doc)
    _draw_footer(canvas, doc)


# ── Styles ────────────────────────────────────────────────────────────────────
def _build_styles():
    base = getSampleStyleSheet()

    def s(name, **kw):
        return ParagraphStyle(name, **kw)

    return {
        "cover_title": s("ct", fontName="Helvetica-Bold", fontSize=20,
                         textColor=PDF_TITLE, leading=26, alignment=TA_CENTER),
        "cover_sub":   s("cs", fontName="Helvetica", fontSize=10,
                         textColor=PDF_MUTED, leading=14, alignment=TA_CENTER),
        "section":     s("sec", fontName="Helvetica-Bold", fontSize=12,
                         textColor=GOLD, leading=16, spaceBefore=8),
        "body":        s("bd", fontName="Helvetica", fontSize=8.5,
                         textColor=PDF_BODY, leading=12),
        "body_small":  s("bds", fontName="Helvetica", fontSize=7.5,
                         textColor=PDF_MUTED, leading=10),
        "metric_val":  s("mv", fontName="Helvetica-Bold", fontSize=16,
                         textColor=GOLD, leading=20, alignment=TA_CENTER),
        "metric_lbl":  s("ml", fontName="Helvetica", fontSize=7,
                         textColor=PDF_MUTED, leading=10, alignment=TA_CENTER),
        "th":          s("th", fontName="Helvetica-Bold", fontSize=7.5,
                         textColor=WHITE_TEXT, leading=10, alignment=TA_CENTER),
        "td":          s("td", fontName="Helvetica", fontSize=7,
                         textColor=PDF_BODY, leading=9),
        "td_c":        s("tdc", fontName="Helvetica", fontSize=7,
                         textColor=PDF_BODY, leading=9, alignment=TA_CENTER),
        "about_title": s("at", fontName="Helvetica-Bold", fontSize=15,
                         textColor=PDF_TITLE, leading=20),
        "about_body":  s("ab", fontName="Helvetica", fontSize=9,
                         textColor=PDF_BODY, leading=14),
        "about_small": s("abs", fontName="Helvetica", fontSize=8,
                         textColor=PDF_MUTED, leading=12),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────
_STOP_TYPE_COLORS = {
    "Delivery":  GOLD,
    "Pickup":    SAGE,
    "Meeting":   TEAL,
    "Warehouse": VIOLET,
    "Customs":   CRIMSON,
    "Rest":      INK,
}
_PRIORITY_COLORS = {
    "High":   CRIMSON,
    "Medium": GOLD,
    "Low":    SAGE,
}
_RISK_COLORS = {
    "High":   CRIMSON,
    "Medium": GOLD,
    "Low":    TEAL,
}


def _colored_pill(text, color, st):
    hex_c = color.hexval() if hasattr(color, 'hexval') else str(color)
    return Paragraph(
        f'<font color="{hex_c}"><b>{text}</b></font>', st["td_c"]
    )


def _divider():
    return HRFlowable(width="100%", thickness=0.5, color=STEEL,
                      spaceAfter=4, spaceBefore=4)


# ── Cover page ────────────────────────────────────────────────────────────────
def _cover_section(itin, constraints, st):
    elems = []
    elems.append(Spacer(1, 10 * mm))

    title = itin.get("itinerary_title", "Optimised Route Itinerary")
    elems.append(Paragraph(title, st["cover_title"]))
    elems.append(Spacer(1, 3 * mm))

    gen_date = datetime.now().strftime("%d %B %Y, %H:%M")
    elems.append(Paragraph(f"Generated: {gen_date}", st["cover_sub"]))
    elems.append(Spacer(1, 8 * mm))
    elems.append(_divider())
    elems.append(Spacer(1, 4 * mm))

    # ── Summary grid ────────────────────────────────────────────────────────
    def kv(label, val):
        return [Paragraph(label, st["body_small"]), Paragraph(str(val), st["body"])]

    driver   = itin.get("driver", constraints.get("driver_name", "—") if constraints else "—")
    vehicle  = itin.get("vehicle", constraints.get("vehicle_type", "—") if constraints else "—")
    mode     = itin.get("transport_mode", constraints.get("transport_mode", "Road") if constraints else "Road")
    date     = itin.get("date", "—")

    summary_data = [
        kv("Driver",          driver),
        kv("Vehicle",         vehicle),
        kv("Transport Mode",  mode),
        kv("Route Date",      date),
    ]
    src = itin.get("routing_source", "")
    routing_label = "OSRM (real road network)" if src == "osrm" else "Haversine estimate"
    summary_data.append(kv("Distance Source", routing_label))

    summary_table = Table(
        summary_data,
        colWidths=[CONTENT_W * 0.35, CONTENT_W * 0.65],
    )
    summary_table.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [PDF_ROW_ALT, colors.white]),
        ("BOX",         (0, 0), (-1, -1), 0.5, PDF_BORDER),
        ("INNERGRID",   (0, 0), (-1, -1), 0.25, PDF_BORDER),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elems.append(summary_table)
    elems.append(Spacer(1, 6 * mm))
    elems.append(_divider())
    elems.append(Spacer(1, 4 * mm))

    # ── Key metrics ─────────────────────────────────────────────────────────
    elems.append(Paragraph("Key Metrics", st["section"]))
    elems.append(Spacer(1, 2 * mm))

    metrics = [
        (f"{itin.get('total_distance_km', '—')} km",  "Total Distance"),
        (f"{itin.get('total_duration_min', '—')} min", "Total Duration"),
        (f"{itin.get('efficiency_score', '—')}/100",   "Efficiency Score"),
        (f"{itin.get('on_time_probability', '—')}%",   "On-Time Probability"),
    ]
    metric_rows = [
        [Paragraph(v, st["metric_val"]) for v, _ in metrics],
        [Paragraph(l, st["metric_lbl"]) for _, l in metrics],
    ]
    col_w = CONTENT_W / 4
    metric_table = Table(metric_rows, colWidths=[col_w] * 4, rowHeights=[22, 12])
    metric_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), PDF_ROW_ALT),
        ("BOX",          (0, 0), (-1, -1), 0.5, PDF_BORDER),
        ("INNERGRID",    (0, 0), (-1, -1), 0.25, PDF_BORDER),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elems.append(metric_table)
    elems.append(Spacer(1, 4 * mm))

    n_stops = len(itin.get("stops", []))
    elems.append(Paragraph(
        f"This itinerary covers <b>{n_stops} stops</b>. "
        "Full stop sequence is detailed on the following pages.",
        st["body"]
    ))
    return elems


# ── Stop sequence table ───────────────────────────────────────────────────────
def _stops_section(itin, st):
    elems = []
    stops = sorted(itin.get("stops", []), key=lambda s: s.get("sequence", 0))
    if not stops:
        return elems

    elems.append(Paragraph("Stop Sequence", st["section"]))
    elems.append(Spacer(1, 2 * mm))

    # Column config: (header, width_fraction, key_or_func)
    col_defs = [
        ("#",          0.040, lambda s: str(s.get("sequence", ""))),
        ("Location",   0.210, lambda s: s.get("location_name", "—")),
        ("Type",       0.085, lambda s: s.get("stop_type", "—")),
        ("Priority",   0.075, lambda s: s.get("priority", "—")),
        ("Arrival",    0.075, lambda s: s.get("arrival_time", "—")),
        ("Departure",  0.075, lambda s: s.get("departure_time", "—")),
        ("Svc (min)",  0.065, lambda s: str(s.get("service_duration_min", "—"))),
        ("Travel (m)", 0.065, lambda s: str(s.get("travel_time_from_prev_min", "—"))),
        ("Dist (km)",  0.065, lambda s: str(s.get("distance_from_prev_km", "—"))),
        ("Risk",       0.070, lambda s: s.get("risk_flag", "—") or "—"),
        ("Notes",      0.170, lambda s: (s.get("notes", "") or "")[:60]),
    ]

    col_widths = [CONTENT_W * f for _, f, _ in col_defs]
    header_row = [Paragraph(h, st["th"]) for h, _, _ in col_defs]

    rows = [header_row]
    style_cmds = [
        ("BACKGROUND",   (0, 0), (-1, 0),  MIDNIGHT),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, PDF_ROW_ALT]),
        ("BOX",          (0, 0), (-1, -1), 0.5, PDF_BORDER),
        ("INNERGRID",    (0, 0), (-1, -1), 0.25, PDF_BORDER),
        ("TOPPADDING",   (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
        ("LEFTPADDING",  (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",        (0, 0), (0, -1),  "CENTER"),
        ("ALIGN",        (4, 0), (9, -1),  "CENTER"),
    ]

    for row_i, stop in enumerate(stops, start=1):
        cells = []
        for col_i, (_, _, fn) in enumerate(col_defs):
            val = fn(stop)
            if col_defs[col_i][0] == "Type":
                c = _STOP_TYPE_COLORS.get(val, INK)
                cells.append(_colored_pill(val, c, st))
            elif col_defs[col_i][0] == "Priority":
                c = _PRIORITY_COLORS.get(val, INK)
                cells.append(_colored_pill(val, c, st))
            elif col_defs[col_i][0] == "Risk":
                c = _RISK_COLORS.get(val, INK)
                cells.append(_colored_pill(val if val != "None" else "OK", c, st))
            else:
                cells.append(Paragraph(val, st["td_c"] if col_i in (0, 4, 5, 6, 7, 8) else st["td"]))
        rows.append(cells)

        risk = stop.get("risk_flag", "None") or "None"
        if risk in ("High", "Medium"):
            rc = CRIMSON if risk == "High" else GOLD
            style_cmds.append(("LEFTPADDING",  (1, row_i), (1, row_i), 2))
            style_cmds.append(("LINEAFTER",    (0, row_i), (0, row_i), 2, rc))

    tbl = Table(rows, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle(style_cmds))
    elems.append(tbl)
    return elems


# ── Warnings & notes page ─────────────────────────────────────────────────────
def _notes_section(itin, st):
    warnings = itin.get("warnings", [])
    notes    = itin.get("optimization_notes", "")
    if not warnings and not notes:
        return []

    elems = [Spacer(1, 4 * mm)]
    elems.append(Paragraph("Warnings & Optimisation Notes", st["section"]))
    elems.append(_divider())

    if warnings:
        elems.append(Spacer(1, 2 * mm))
        elems.append(Paragraph("Warnings", st["body"]))
        elems.append(Spacer(1, 1 * mm))
        for w in warnings:
            elems.append(Paragraph(f"\u2022 {w}", st["body_small"]))
        elems.append(Spacer(1, 3 * mm))

    if notes:
        elems.append(Paragraph("Optimisation Notes", st["body"]))
        elems.append(Spacer(1, 1 * mm))
        elems.append(Paragraph(notes, st["body_small"]))

    return elems


# ── Vehicle efficiency lookup ─────────────────────────────────────────────────
_VEHICLE_EFF = {
    "Truck":              ("4–5 km/L",   "₹18–22/km"),
    "Van":                ("10–12 km/L", "₹8–10/km"),
    "Tempo":              ("7–9 km/L",   "₹10–13/km"),
    "Car":                ("14–18 km/L", "₹6–8/km"),
    "Motorcycle":         ("40–50 km/L", "₹2–3/km"),
    "Refrigerated Truck": ("3–4 km/L",   "₹22–28/km"),
}


# ── Optimised route path section ──────────────────────────────────────────────
def _route_path_section(itin, st):
    """Leg-by-leg route table. Shown only when stops carry distance data."""
    stops = sorted(itin.get("stops", []), key=lambda s: s.get("sequence", 0))
    if len(stops) < 2:
        return []

    is_opt = bool(str(itin.get("optimization_notes", "")).strip())
    elems  = [Spacer(1, 4 * mm)]
    elems.append(Paragraph(
        "Optimised Route Sequence" if is_opt else "Route Sequence", st["section"]
    ))
    if is_opt:
        elems.append(Paragraph(
            "Stops have been reordered by the AI optimiser to minimise total distance.",
            st["body_small"]
        ))
    elems.append(Spacer(1, 2 * mm))
    elems.append(_divider())
    elems.append(Spacer(1, 2 * mm))

    cw = CONTENT_W
    col_widths = [cw*0.06, cw*0.30, cw*0.28, cw*0.13, cw*0.13, cw*0.10]
    header = [
        Paragraph("#",           st["th"]),
        Paragraph("From",        st["th"]),
        Paragraph("To",          st["th"]),
        Paragraph("Dist (km)",   st["th"]),
        Paragraph("Travel (min)",st["th"]),
        Paragraph("Type",        st["th"]),
    ]
    rows = [header]
    for i, stop in enumerate(stops):
        from_name = "— Start —" if i == 0 else stops[i - 1].get("location_name", "—")
        dist  = stop.get("distance_from_prev_km", 0)
        trvl  = stop.get("travel_time_from_prev_min", 0)
        stype = stop.get("stop_type", "—")
        c     = _STOP_TYPE_COLORS.get(stype, INK)
        rows.append([
            Paragraph(str(stop.get("sequence", i + 1)), st["td_c"]),
            Paragraph(from_name, st["td"]),
            Paragraph(stop.get("location_name", "—"), st["td"]),
            Paragraph(str(dist)  if dist  else "—", st["td_c"]),
            Paragraph(str(trvl)  if trvl  else "—", st["td_c"]),
            _colored_pill(stype[:4], c, st),
        ])

    tbl = Table(rows, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  MIDNIGHT),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, PDF_ROW_ALT]),
        ("BOX",           (0, 0), (-1, -1), 0.5, PDF_BORDER),
        ("INNERGRID",     (0, 0), (-1, -1), 0.25, PDF_BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING",   (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",         (0, 0), (0, -1),  "CENTER"),
        ("ALIGN",         (3, 0), (5, -1),  "CENTER"),
    ]))
    elems.append(tbl)
    return elems


# ── Fuel & cost statistics section ────────────────────────────────────────────
def _fuel_section(itin, constraints, st):
    total_dist   = float(itin.get("total_distance_km",    0) or 0)
    fuel_cost    = float(itin.get("estimated_fuel_cost_inr", 0) or 0)
    duration_min = int(  itin.get("total_duration_min",   0) or 0)
    vehicle      = (itin.get("vehicle")
                    or (constraints.get("vehicle_type") if constraints else None)
                    or "—")
    routing_src  = itin.get("routing_source", "—")
    efficiency   = itin.get("efficiency_score", "—")

    cost_per_km   = f"₹{fuel_cost / total_dist:.1f}" if total_dist > 0 else "—"
    fuel_cost_fmt = f"₹{fuel_cost:,.0f}"             if fuel_cost    else "—"
    dist_fmt      = f"{total_dist:.1f} km"            if total_dist   else "—"
    dur_fmt       = (f"{duration_min // 60}h {duration_min % 60}m"
                     if duration_min else "—")

    eff_info      = _VEHICLE_EFF.get(vehicle, ("—", "—"))
    routing_label = ("OSRM (real road network)"
                     if routing_src == "osrm" else "Haversine straight-line estimate")

    elems = [Spacer(1, 4 * mm)]
    elems.append(Paragraph("Fuel & Cost Analysis", st["section"]))
    elems.append(_divider())
    elems.append(Spacer(1, 2 * mm))

    def kv(label, val, note=""):
        return [
            Paragraph(label, st["body_small"]),
            Paragraph(val,   st["body"]),
            Paragraph(note,  st["body_small"]),
        ]

    data = [
        kv("Total Route Distance",     dist_fmt),
        kv("Total Drive Time",         dur_fmt),
        kv("Estimated Fuel Cost",      fuel_cost_fmt,
           "Based on vehicle type × city fuel price"),
        kv("Cost per km",              cost_per_km),
        kv("Vehicle",                  vehicle),
        kv("Typical Fuel Efficiency",  eff_info[0]),
        kv("Typical Running Cost",     eff_info[1]),
        kv("Distance Source",          routing_label),
        kv("Route Efficiency Score",   f"{efficiency}/100" if efficiency != "—" else "—"),
    ]

    tbl = Table(
        data,
        colWidths=[CONTENT_W * 0.34, CONTENT_W * 0.32, CONTENT_W * 0.34],
    )
    tbl.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [PDF_ROW_ALT, colors.white]),
        ("BOX",            (0, 0), (-1, -1), 0.5, PDF_BORDER),
        ("INNERGRID",      (0, 0), (-1, -1), 0.25, PDF_BORDER),
        ("LEFTPADDING",    (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",   (0, 0), (-1, -1), 8),
        ("TOPPADDING",     (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elems.append(tbl)
    return elems


# ── About RouteIQ (last page) ─────────────────────────────────────────────────
_ABOUT_TEXT = """\
<b>RouteIQ</b> is an AI-powered logistics and route optimisation platform purpose-built \
for modern delivery and field-operations teams. By combining machine-learning algorithms \
with real-world road-network data, RouteIQ transforms a list of stops into a fully \
optimised itinerary — minimising total distance, honouring time windows, and flagging \
potential risks before a driver ever leaves the depot.

<b>Core Capabilities</b>

\u2022 <b>Intelligent Route Planning</b> \u2014 Nearest-neighbour and AI-enhanced optimisation \
produces efficient sequences from raw stop lists, with real road distances sourced live \
from the OSRM routing engine.

\u2022 <b>Multi-modal Support</b> \u2014 Road, Rail, Air, and Sea transport modes are each \
understood by the planning engine, enabling seamless multi-leg itinerary generation.

\u2022 <b>Constraint Enforcement</b> \u2014 Driver working hours, vehicle capacity limits, \
delivery time windows, and priority levels are all respected during optimisation, with \
any violations surfaced clearly in the output.

\u2022 <b>Live Risk Detection</b> \u2014 Each stop is automatically assessed for schedule \
risk (tight windows, high-priority cargo, known delay patterns), giving dispatchers an \
at-a-glance risk dashboard.

\u2022 <b>Animated Route Maps</b> \u2014 Interactive, browser-rendered maps animate the \
vehicle along the optimised path, providing an intuitive visual review before dispatch.

\u2022 <b>Weather & Fuel Intelligence</b> \u2014 Per-stop weather forecasts at expected \
arrival times and fuel-cost projections help planners anticipate delays and budget \
accurately.

\u2022 <b>Natural Language Input</b> \u2014 Dispatchers can describe a route in plain English \
(e.g.\u00a0"deliver to Andheri by 10\u202fam, high priority") and the AI engine parses stops, \
times, and constraints automatically.

\u2022 <b>Cluster Analysis</b> \u2014 K-Means clustering groups stops geographically, enabling \
smart territory assignment and multi-vehicle load balancing.

RouteIQ is designed to be both powerful enough for enterprise logistics teams and \
accessible enough for independent operators, all through a clean, dark-themed web \
interface that runs entirely in the browser.\
"""

_DISCLAIMER = (
    "The information contained in this PDF was generated automatically by RouteIQ "
    "based on the data provided at the time of generation. RouteIQ makes no warranty, "
    "express or implied, regarding the accuracy of real-time road conditions, travel "
    "times, or fuel estimates. Always verify critical logistics decisions independently."
)


def _about_section(st):
    elems = [PageBreak()]
    elems.append(Spacer(1, 6 * mm))

    logo = _make_logo(100, 28)
    from reportlab.platypus import Image as RLImage
    from reportlab.platypus.flowables import Flowable

    class _LogoFlowable(Flowable):
        def wrap(self, aw, ah): return 100, 28
        def draw(self):
            renderPDF.draw(_make_logo(100, 28), self.canv, 0, 0)

    elems.append(_LogoFlowable())
    elems.append(Spacer(1, 4 * mm))
    elems.append(Paragraph("About RouteIQ", st["about_title"]))
    elems.append(Spacer(1, 2 * mm))
    elems.append(_divider())
    elems.append(Spacer(1, 3 * mm))

    for para in _ABOUT_TEXT.strip().split("\n\n"):
        elems.append(Paragraph(para.strip(), st["about_body"]))
        elems.append(Spacer(1, 3 * mm))

    elems.append(_divider())
    elems.append(Spacer(1, 2 * mm))
    elems.append(Paragraph(_DISCLAIMER, st["about_small"]))
    return elems


# ── Public API ────────────────────────────────────────────────────────────────
def generate_itinerary_pdf(itin: dict, constraints: dict = None) -> bytes:
    """
    Build a branded RouteIQ PDF from an itinerary dict.

    Returns the PDF as raw bytes, ready for st.download_button.
    """
    buf = io.BytesIO()

    frame = Frame(
        MARGIN, CONTENT_Y_BOTTOM,
        CONTENT_W, CONTENT_H,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
        id="main",
    )
    template = PageTemplate(id="main", frames=[frame], onPage=_draw_all)
    doc = BaseDocTemplate(
        buf,
        pagesize=A4,
        pageTemplates=[template],
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN + HEADER_H + 4 * mm,
        bottomMargin=MARGIN + FOOTER_H + 4 * mm,
        title=itin.get("itinerary_title", "RouteIQ Itinerary"),
        author="RouteIQ",
        subject="Optimised Route Itinerary",
        creator="RouteIQ Platform",
    )

    st = _build_styles()
    story = []

    story += _cover_section(itin, constraints or {}, st)
    story.append(PageBreak())
    story += _stops_section(itin, st)
    story += _route_path_section(itin, st)
    story += _fuel_section(itin, constraints or {}, st)
    story += _notes_section(itin, st)
    story += _about_section(st)

    doc.build(story)
    return buf.getvalue()
