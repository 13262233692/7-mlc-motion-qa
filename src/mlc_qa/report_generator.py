"""
PDF Report Generator for MLC Motion QA.

Generates professional PDF reports with QA results, including:
- Summary statistics
- Pass/fail status
- Control point analysis
- Leaf error details
- Graphical representation (ASCII fallback)
"""
import io
from datetime import datetime
from typing import Optional

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
    ) -> bytes:
        """
        Generate a PDF report for a QA result.

        Args:
            qa_result: QAResult database object.
            leaf_samples: List of LeafErrorSample objects.

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
) -> bytes:
    """
    Convenience function to generate a PDF report.

    Args:
        qa_result: QAResult database object.
        leaf_samples: List of LeafErrorSample objects.

    Returns:
        PDF content as bytes.
    """
    generator = ReportGenerator()
    return generator.generate_report(qa_result, leaf_samples)
