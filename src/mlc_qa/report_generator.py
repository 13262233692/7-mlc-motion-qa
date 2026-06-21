"""
PDF Report Generator for MLC Motion QA.

Generates professional PDF reports with QA results, including:
- Summary statistics
- Pass/fail status
- Control point analysis
- Leaf error details
- Fraction trend analysis (optional)
- Graphical representation
"""
import io
from datetime import datetime
from typing import Optional, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    ListFlowable,
    ListItem,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.graphics.shapes import Drawing, Line, PolyLine, String, Rect
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.widgets.markers import makeMarker

from mlc_qa import models
from mlc_qa.config import MAX_LEAF_DEVIATION_THRESHOLD, CONTROL_POINT_PASS_THRESHOLD


class ReportGenerator:
    """PDF report generator for QA results."""

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a5276'),
            alignment=TA_CENTER,
            spaceAfter=12,
        ))
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2c3e50'),
            spaceBefore=12,
            spaceAfter=8,
        ))
        self.styles.add(ParagraphStyle(
            name='SubSectionHeader',
            parent=self.styles['Heading3'],
            fontSize=13,
            textColor=colors.HexColor('#34495e'),
            spaceBefore=8,
            spaceAfter=6,
        ))
        self.styles.add(ParagraphStyle(
            name='PassStatus',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=colors.green,
            alignment=TA_CENTER,
            spaceAfter=6,
        ))
        self.styles.add(ParagraphStyle(
            name='FailStatus',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=colors.red,
            alignment=TA_CENTER,
            spaceAfter=6,
        ))
        self.styles.add(ParagraphStyle(
            name='InfoText',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#566573'),
        ))
        self.styles.add(ParagraphStyle(
            name='WarningText',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.orange,
        ))

    def generate_report(
        self,
        qa_result: models.QAResult,
        leaf_samples: Optional[list] = None,
        fraction_summaries: Optional[List[models.FractionQASummary]] = None,
    ) -> bytes:
        """
        Generate a PDF report for a QA result.

        Args:
            qa_result: QAResult database object.
            leaf_samples: List of LeafErrorSample objects.
            fraction_summaries: Optional list of FractionQASummary for trend chart.

        Returns:
            PDF content as bytes.
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
            title=f"MLC Motion QA Report - {qa_result.id}",
        )

        elements = []
        self._add_header(elements, qa_result)
        self._add_summary(elements, qa_result)
        self._add_detailed_metrics(elements, qa_result)

        if leaf_samples:
            self._add_leaf_errors(elements, leaf_samples)

        if fraction_summaries:
            self._add_trend_chart(elements, fraction_summaries)

        self._add_analysis_notes(elements, qa_result)
        self._add_footer(elements)

        doc.build(elements)
        buffer.seek(0)
        return buffer.getvalue()

    def _add_header(self, elements: list, qa_result: models.QAResult):
        """Add report header."""
        elements.append(Paragraph(
            "Multi-Leaf Collimator Motion QA Report",
            self.styles['ReportTitle']
        ))
        elements.append(Spacer(1, 0.1 * inch))

        header_data = [
            ["Report Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["QA Date:", qa_result.qa_date.strftime("%Y-%m-%d %H:%M:%S") if qa_result.qa_date else "N/A"],
            ["Log File:", qa_result.log_filename or "N/A"],
            ["Beam:", qa_result.beam.beam_name if qa_result.beam else "N/A"],
            ["Plan UID:", qa_result.plan.plan_uid if qa_result.plan else "N/A"],
        ]

        header_table = Table(header_data, colWidths=[1.5 * inch, 4.5 * inch])
        header_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#566573')),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#2c3e50')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(header_table)
        elements.append(Spacer(1, 0.2 * inch))

        status_text = "PASS" if qa_result.overall_pass else "FAIL"
        status_style = self.styles['PassStatus'] if qa_result.overall_pass else self.styles['FailStatus']
        elements.append(Paragraph(f"Overall Status: {status_text}", status_style))
        elements.append(Spacer(1, 0.15 * inch))

    def _add_summary(self, elements: list, qa_result: models.QAResult):
        """Add summary statistics section."""
        elements.append(Paragraph("Summary Statistics", self.styles['SectionHeader']))

        def format_value(value, unit="", decimals=3):
            if value is None:
                return "N/A"
            return f"{value:.{decimals}f} {unit}".strip()

        pass_rate_color = colors.green if qa_result.control_point_pass_rate_pct >= CONTROL_POINT_PASS_THRESHOLD else colors.red
        max_dev_color = colors.green if qa_result.max_leaf_deviation_mm <= MAX_LEAF_DEVIATION_THRESHOLD else colors.red

        summary_data = [
            [
                "Metric",
                "Value",
                "Threshold",
                "Status"
            ],
            [
                "Max Leaf Deviation",
                format_value(qa_result.max_leaf_deviation_mm, "mm"),
                format_value(MAX_LEAF_DEVIATION_THRESHOLD, "mm"),
                Paragraph(
                    "✓ PASS" if qa_result.max_leaf_deviation_mm <= MAX_LEAF_DEVIATION_THRESHOLD else "✗ FAIL",
                    ParagraphStyle('tmp', textColor=max_dev_color, fontSize=10)
                ),
            ],
            [
                "Mean Leaf Deviation",
                format_value(qa_result.mean_leaf_deviation_mm, "mm"),
                "-",
                "-"
            ],
            [
                "RMSE",
                format_value(qa_result.rmse_mm, "mm"),
                "-",
                "-"
            ],
            [
                "Dose Rate Deviation",
                format_value(qa_result.dose_rate_deviation_pct, "%", 2),
                "-",
                "-"
            ],
            [
                "Control Point Pass Rate",
                format_value(qa_result.control_point_pass_rate_pct, "%", 2),
                format_value(CONTROL_POINT_PASS_THRESHOLD, "%", 2),
                Paragraph(
                    "✓ PASS" if qa_result.control_point_pass_rate_pct >= CONTROL_POINT_PASS_THRESHOLD else "✗ FAIL",
                    ParagraphStyle('tmp', textColor=pass_rate_color, fontSize=10)
                ),
            ],
        ]

        summary_table = Table(summary_data, colWidths=[1.8 * inch, 1.5 * inch, 1.2 * inch, 1.5 * inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a5276')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7')),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(summary_table)
        elements.append(Spacer(1, 0.2 * inch))

    def _add_detailed_metrics(self, elements: list, qa_result: models.QAResult):
        """Add detailed metrics section."""
        elements.append(Paragraph("Detailed Information", self.styles['SectionHeader']))

        details_data = [
            ["Number of Control Points:", str(qa_result.num_control_points or "N/A")],
            ["Failed Control Points:", str(qa_result.num_failed_control_points or "N/A")],
            ["Number of Leaves per Bank:", str(qa_result.num_leaves or "N/A")],
            ["Gantry Angle Range:", qa_result.gantry_angle_range or "N/A"],
        ]

        details_table = Table(details_data, colWidths=[2.5 * inch, 4 * inch])
        details_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#566573')),
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8f9fa')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(details_table)
        elements.append(Spacer(1, 0.2 * inch))

    def _add_leaf_errors(self, elements: list, leaf_samples: list):
        """Add top leaf errors table."""
        elements.append(Paragraph("Top Leaf Errors (by deviation)", self.styles['SectionHeader']))

        if not leaf_samples:
            elements.append(Paragraph("No leaf error samples available.", self.styles['InfoText']))
            return

        header = ["CP", "Leaf", "Bank", "Planned (mm)", "Actual (mm)", "Deviation (mm)"]
        table_data = [header]

        for sample in leaf_samples[:20]:
            dev_color = colors.red if abs(sample.deviation_mm) > MAX_LEAF_DEVIATION_THRESHOLD else colors.HexColor('#2c3e50')
            row = [
                str(sample.control_point_index),
                str(sample.leaf_index),
                sample.bank,
                f"{sample.planned_position_mm:.3f}",
                f"{sample.actual_position_mm:.3f}",
                Paragraph(
                    f"{sample.deviation_mm:.3f}",
                    ParagraphStyle('dev', textColor=dev_color, fontSize=9)
                ),
            ]
            table_data.append(row)

        leaf_table = Table(table_data, colWidths=[0.6 * inch, 0.6 * inch, 0.6 * inch, 1.2 * inch, 1.2 * inch, 1.3 * inch])
        leaf_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(leaf_table)
        elements.append(Spacer(1, 0.1 * inch))
        elements.append(Paragraph(
            f"Showing top {min(20, len(leaf_samples))} leaf errors out of {len(leaf_samples)} samples.",
            self.styles['InfoText']
        ))

    def _add_trend_chart(self, elements: list, summaries: List[models.FractionQASummary]):
        """Add fraction trend chart and summary."""
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Paragraph("Fraction Trend Analysis", self.styles['SectionHeader']))

        if len(summaries) < 2:
            elements.append(Paragraph(
                "Insufficient fraction data for trend analysis (need at least 2 fractions).",
                self.styles['InfoText']
            ))
            return

        drawing = Drawing(6.5 * inch, 3 * inch)

        chart_x = 0.8 * inch
        chart_y = 0.5 * inch
        chart_w = 5.5 * inch
        chart_h = 2.2 * inch

        valid_summaries = [s for s in summaries if s.rmse_mm is not None]
        if len(valid_summaries) < 2:
            elements.append(Paragraph(
                "Insufficient valid data for trend chart.",
                self.styles['InfoText']
            ))
            return

        frac_nums = [s.fraction_number for s in valid_summaries]
        rmse_vals = [s.rmse_mm for s in valid_summaries]
        max_dev_vals = [s.max_leaf_deviation_mm for s in valid_summaries]

        min_x, max_x = min(frac_nums), max(frac_nums)
        all_vals = rmse_vals + max_dev_vals
        min_y, max_y = 0, max(all_vals) * 1.15

        if max_y == 0:
            max_y = 1.0

        x_range = max_x - min_x if max_x != min_x else 1
        y_range = max_y - min_y if max_y != min_y else 1

        def scale_x(x_val):
            return chart_x + (x_val - min_x) / x_range * chart_w

        def scale_y(y_val):
            return chart_y + (y_val - min_y) / y_range * chart_h

        drawing.add(Rect(
            chart_x, chart_y, chart_w, chart_h,
            fillColor=colors.HexColor('#f8f9fa'),
            strokeColor=colors.HexColor('#bdc3c7'),
            strokeWidth=0.5,
        ))

        for i in range(5):
            y_val = min_y + (y_range * i / 4)
            y_pos = scale_y(y_val)
            drawing.add(Line(
                chart_x, y_pos, chart_x + chart_w, y_pos,
                strokeColor=colors.HexColor('#e0e0e0'),
                strokeWidth=0.5,
            ))
            drawing.add(String(
                chart_x - 5, y_pos - 4, f"{y_val:.2f}",
                fontSize=7,
                fillColor=colors.HexColor('#7f8c8d'),
                textAnchor='end',
            ))

        for i, fx in enumerate(frac_nums):
            x_pos = scale_x(fx)
            drawing.add(String(
                x_pos, chart_y - 12, f"F{fx}",
                fontSize=7,
                fillColor=colors.HexColor('#7f8c8d'),
                textAnchor='middle',
            ))

        rmse_points = [(scale_x(f), scale_y(v)) for f, v in zip(frac_nums, rmse_vals)]
        rmse_flat = []
        for x, y in rmse_points:
            rmse_flat.extend([x, y])

        fill_flat_xs = rmse_xs = [p[0] for p in rmse_points]
        fill_flat_ys = rmse_ys = [p[1] for p in rmse_points]
        fill_points = rmse_flat[:]
        for x in reversed(rmse_xs):
            fill_points.extend([x, chart_y])

        drawing.add(PolyLine(
            fill_points,
            fillColor=colors.HexColor('#3498db'),
            fillOpacity=0.15,
            strokeColor=None,
        ))
        drawing.add(PolyLine(
            rmse_flat,
            strokeColor=colors.HexColor('#2980b9'),
            strokeWidth=2,
        ))
        for x, y in rmse_points:
            from reportlab.graphics.shapes import Circle
            drawing.add(Circle(x, y, 3, fillColor=colors.HexColor('#2980b9'), strokeColor=None))
            drawing.add(String(x, y + 8, f"{y_range * (y - chart_y) / chart_h + min_y:.2f}",
                fontSize=6, fillColor=colors.HexColor('#2c3e50'), textAnchor='middle'))

        max_dev_points = [(scale_x(f), scale_y(v)) for f, v in zip(frac_nums, max_dev_vals)]
        max_dev_flat = []
        for x, y in max_dev_points:
            max_dev_flat.extend([x, y])

        drawing.add(PolyLine(
            max_dev_flat,
            strokeColor=colors.HexColor('#e74c3c'),
            strokeWidth=1.5,
            strokeDashArray=[4, 2],
        ))
        for x, y in max_dev_points:
            from reportlab.graphics.shapes import Circle
            drawing.add(Circle(x, y, 3, fillColor=colors.white, strokeColor=colors.HexColor('#e74c3c'), strokeWidth=1.5))

        legend_y = chart_y + chart_h + 8
        drawing.add(Rect(chart_x, legend_y, 10, 10,
            fillColor=colors.HexColor('#2980b9')))
        drawing.add(String(chart_x + 14, legend_y + 2, "RMSE (mm)",
            fontSize=8, fillColor=colors.HexColor('#2c3e50')))
        drawing.add(Rect(chart_x + 90, legend_y, 10, 10,
            fillColor=colors.HexColor('#e74c3c')))
        drawing.add(String(chart_x + 104, legend_y + 2, "Max Deviation (mm)",
            fontSize=8, fillColor=colors.HexColor('#2c3e50')))

        elements.append(drawing)
        elements.append(Spacer(1, 0.1 * inch))

        trend_data = [
            ["Fraction", "RMSE (mm)", "Max Dev (mm)", "Pass Rate (%)", "Version"],
        ]
        for s in valid_summaries[:10]:
            trend_data.append([
                str(s.fraction_number),
                f"{s.rmse_mm:.3f}" if s.rmse_mm else "N/A",
                f"{s.max_leaf_deviation_mm:.3f}" if s.max_leaf_deviation_mm else "N/A",
                f"{s.overall_pass_rate_pct:.1f}%" if s.overall_pass_rate_pct is not None else "N/A",
                str(s.plan_version),
            ])

        trend_table = Table(trend_data, colWidths=[0.8 * inch, 1.2 * inch, 1.3 * inch, 1.2 * inch, 0.8 * inch])
        trend_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        elements.append(trend_table)

        if len(valid_summaries) > 10:
            elements.append(Spacer(1, 0.05 * inch))
            elements.append(Paragraph(
                f"... showing 10 of {len(valid_summaries)} fractions",
                self.styles['InfoText']
            ))

    def _add_analysis_notes(self, elements: list, qa_result: models.QAResult):
        """Add analysis notes section."""
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Paragraph("Analysis Notes", self.styles['SectionHeader']))

        notes_items = []

        if qa_result.max_leaf_deviation_mm > MAX_LEAF_DEVIATION_THRESHOLD:
            notes_items.append(ListItem(Paragraph(
                f"<b>Warning:</b> Maximum leaf deviation ({qa_result.max_leaf_deviation_mm:.3f} mm) "
                f"exceeds threshold ({MAX_LEAF_DEVIATION_THRESHOLD} mm).",
                self.styles['WarningText']
            )))

        if qa_result.control_point_pass_rate_pct < CONTROL_POINT_PASS_THRESHOLD:
            notes_items.append(ListItem(Paragraph(
                f"<b>Warning:</b> Control point pass rate ({qa_result.control_point_pass_rate_pct:.2f}%) "
                f"is below threshold ({CONTROL_POINT_PASS_THRESHOLD}%).",
                self.styles['WarningText']
            )))

        if qa_result.num_failed_control_points and qa_result.num_failed_control_points > 0:
            notes_items.append(ListItem(Paragraph(
                f"{qa_result.num_failed_control_points} control points failed the tolerance criteria.",
                self.styles['InfoText']
            )))

        if qa_result.notes:
            notes_items.append(ListItem(Paragraph(
                f"Additional notes: {qa_result.notes}",
                self.styles['InfoText']
            )))

        if not notes_items:
            notes_items.append(ListItem(Paragraph(
                "All metrics are within acceptable tolerance limits.",
                self.styles['InfoText']
            )))

        elements.append(ListFlowable(notes_items, bulletType="bullet", start="•"))

    def _add_footer(self, elements: list):
        """Add report footer."""
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph(
            "--- End of Report ---",
            ParagraphStyle(
                'Footer',
                parent=self.styles['Normal'],
                alignment=TA_CENTER,
                textColor=colors.HexColor('#95a5a6'),
                fontSize=9,
            )
        ))


def generate_qa_report_pdf(
    qa_result: models.QAResult,
    leaf_samples: Optional[list] = None,
    fraction_summaries: Optional[List[models.FractionQASummary]] = None,
) -> bytes:
    """
    Convenience function to generate a PDF report.

    Args:
        qa_result: QAResult database object.
        leaf_samples: List of LeafErrorSample objects.
        fraction_summaries: Optional list of FractionQASummary for trend chart.

    Returns:
        PDF content as bytes.
    """
    generator = ReportGenerator()
    return generator.generate_report(qa_result, leaf_samples, fraction_summaries)
