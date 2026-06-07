import os
import sys
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_elements(num_pages)
            super().showPage()
        super().save()

    def draw_page_elements(self, page_count):
        self.saveState()
        left_margin = 36
        right_margin = 792 - 36
        top_y = 612 - 36
        line_top_y = 612 - 42
        bottom_y = 25
        line_bottom_y = 35

        # --- Header ---
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#2B6CB0"))
        self.drawString(left_margin, top_y, "SUITE2P CELL CLASSIFICATION")
        
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))
        self.drawRightString(right_margin, top_y, "ACTIVE FEATURE REFERENCE GUIDE")
        
        # Header line
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.75)
        self.line(left_margin, line_top_y, right_margin, line_top_y)

        # --- Footer ---
        self.line(left_margin, line_bottom_y, right_margin, line_bottom_y)
        
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#4A5568"))
        self.drawString(left_margin, bottom_y, "Suite2p ML Active Learning Release")
        
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(right_margin, bottom_y, page_text)
        
        self.restoreState()

def build_pdf(filename="suite2p_features_reference.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=landscape(letter),
        leftMargin=36,
        rightMargin=36,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=12
    )
    
    th_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=11,
        textColor=colors.white
    )
    
    td_name_style = ParagraphStyle(
        'TableCellName',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#2B6CB0")
    )
    
    td_cat_style = ParagraphStyle(
        'TableCellCategory',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#4A5568")
    )
    
    td_calc_style = ParagraphStyle(
        'TableCellCalc',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=7.5,
        leading=9,
        textColor=colors.HexColor("#2D3748")
    )
    
    td_utility_style = ParagraphStyle(
        'TableCellUtility',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10
    )
    
    td_comment_style = ParagraphStyle(
        'TableCellComment',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#2D3748")
    )

    story = []
    story.append(Paragraph("Suite2p Cell Classification — Feature Reference Guide", title_style))
    story.append(Paragraph("Detailed explanation of all spatial, temporal, and biological features used by the classifiers (Continuous, No-Idx, and 3-Bin-Idx configurations).", subtitle_style))
    story.append(Spacer(1, 4))

    features_raw = [
        # --- SPATIAL FEATURES ---
        {
            "name": "number_of_bright_pixels",
            "category": "Spatial",
            "calc": "sum(lam > 0.1 * max(lam))",
            "utility": "<font color='#38A169'><b>Highly Useful</b></font>",
            "comments": "Measures spatial footprint compactness. Real cell somas are highly concentrated and have few pixels with high weights (lam). Blood vessels or out-of-focus background are diffuse and yield much larger bright pixel counts."
        },
        {
            "name": "solidity",
            "category": "Spatial",
            "calc": "s.get('solidity', 1.0)",
            "utility": "<font color='#3182CE'>Useful</font>",
            "comments": "Ratio of the ROI area to its convex hull. Neuronal somas are convex (solidity near 1.0). Elongated, fragmented, or hollow footprints represent artifacts and yield low solidity."
        },
        {
            "name": "mrs",
            "category": "Spatial",
            "calc": "s.get('mrs', 0)",
            "utility": "<font color='#4A5568'>Moderate</font>",
            "comments": "Mean square distance of pixels to the center of mass. Represents compactness. Compact circular structures have lower Mrs values."
        },
        {
            "name": "roi_idx_norm",
            "category": "Spatial (Continuous)",
            "calc": "idx / n_rois",
            "utility": "<font color='#DD6B20'><b>Overfits</b></font>",
            "comments": "Continuous normalized sorting rank. Represents Suite2p's internal cell probability order. Highly predictive but prone to severe overfitting on noisy runs."
        },
        {
            "name": "roi_idx_norm_3bin",
            "category": "Spatial (Binned)",
            "calc": "digitize(idx / n_rois, [0.1, 0.4])",
            "utility": "<font color='#38A169'><b>Highly Useful</b></font>",
            "comments": "Coarse discretized sorting rank (0, 1, or 2). Limits index feature power (plunging importance rank to #20), forcing the model to prioritize biology while retaining a weak prior."
        },
        
        # --- TRACE STATISTICS ---
        {
            "name": "skew_f",
            "category": "Trace Stats",
            "calc": "skew(F)",
            "utility": "<font color='#38A169'><b>Highly Useful</b></font>",
            "comments": "Skewness of raw trace. Real calcium activity has asymmetric upward transients and slow decays (high positive skew). Background noise is symmetric (skew near 0)."
        },
        {
            "name": "std_f",
            "category": "Trace Stats",
            "calc": "std(F)",
            "utility": "<font color='#38A169'><b>Highly Useful</b></font>",
            "comments": "Standard deviation of raw trace. Measures absolute variance. Active somas show high variance; inactive ROIs or pure noise show very low variance."
        },
        {
            "name": "max_to_mean_f",
            "category": "Trace Stats",
            "calc": "max(F) / mean(F)",
            "utility": "<font color='#3182CE'>Useful</font>",
            "comments": "Maximum to mean ratio of raw trace. Helps identify cells with sparse, high-amplitude calcium transients."
        },
        {
            "name": "cv_f",
            "category": "Trace Stats",
            "calc": "std(F) / mean(F)",
            "utility": "<font color='#4A5568'>Moderate</font>",
            "comments": "Coefficient of variation of raw trace. Normalized variance metric."
        },
        {
            "name": "skew_fneu",
            "category": "Trace Stats",
            "calc": "skew(Fneu)",
            "utility": "<font color='#4A5568'>Moderate</font>",
            "comments": "Skewness of the neuropil background. Captures out-of-focus background transient patterns."
        },
        {
            "name": "corr_f_fneu",
            "category": "Trace Stats",
            "calc": "corrcoef(F, Fneu)[0, 1]",
            "utility": "<font color='#E53E3E'><b>CRITICAL</b></font>",
            "comments": "<b>Pearson correlation between raw trace and neuropil background.</b> True neurons fire independently from background (low/negative correlation). High correlation (~1.0) indicates background contamination/false detection."
        },
        {
            "name": "skew_fcorr",
            "category": "Trace Stats",
            "calc": "skew(F - 0.7 * Fneu)",
            "utility": "<font color='#38A169'><b>Highly Useful</b></font>",
            "comments": "Skewness of neuropil-corrected trace. Extremely clean marker of true, background-corrected calcium transients."
        },
        {
            "name": "std_fcorr",
            "category": "Trace Stats",
            "calc": "std(F - 0.7 * Fneu)",
            "utility": "<font color='#38A169'><b>Highly Useful</b></font>",
            "comments": "Standard deviation of corrected trace. Represents pure calcium transient amplitude."
        },

        # --- TRACE PERCENTILES ---
        {
            "name": "q10, q25, q50, q75",
            "category": "Percentiles",
            "calc": "quantile(Fcorr, p)",
            "utility": "<font color='#4A5568'>Moderate</font>",
            "comments": "Lower-to-middle percentiles mapping the baseline quiet state of the corrected trace."
        },
        {
            "name": "q90, q95, q99",
            "category": "Percentiles",
            "calc": "quantile(Fcorr, p)",
            "utility": "<font color='#38A169'><b>Highly Useful</b></font>",
            "comments": "Upper percentiles capturing calcium transient firing events. Strong separator of active cells from background noise."
        },

        # --- BIOLOGICAL DYNAMICS ---
        {
            "name": "avg_asym",
            "category": "Dynamics",
            "calc": "mean(decay / rise)",
            "utility": "<font color='#38A169'><b>Highly Useful</b></font>",
            "comments": "Average transient peak asymmetry. Real calcium binding/rise is fast, while decay is slow (asymmetry > 1.5). White noise fluctuations are symmetric (asymmetry ~1.0)."
        },
        {
            "name": "max_asym",
            "category": "Dynamics",
            "calc": "max(decay / rise)",
            "utility": "<font color='#3182CE'>Useful</font>",
            "comments": "Maximum peak asymmetry. Detects sparsely active cells."
        },
        {
            "name": "max_width",
            "category": "Dynamics",
            "calc": "max(peak_widths)",
            "utility": "<font color='#3182CE'>Useful</font>",
            "comments": "Maximum width at half-maximum of detected peaks. Filters out high-frequency noise spikes (width 1-2 frames)."
        },

        # --- AMPLITUDES ---
        {
            "name": "range_fcorr",
            "category": "Amplitudes",
            "calc": "max(Fcorr) - min(Fcorr)",
            "utility": "<font color='#3182CE'>Useful</font>",
            "comments": "Total peak-to-peak amplitude of background-corrected trace."
        },
        {
            "name": "range_f",
            "category": "Amplitudes",
            "calc": "max(F) - min(F)",
            "utility": "<font color='#4A5568'>Moderate</font>",
            "comments": "Total peak-to-peak amplitude of raw trace."
        },

        # --- NEW BIOLOGICAL FEATURES ---
        {
            "name": "snr",
            "category": "New Biological",
            "calc": "std(Fcorr) / std(diff(Fcorr))",
            "utility": "<font color='#38A169'><b>Highly Useful</b></font>",
            "comments": "Temporal signal-to-noise ratio. Real neural signals decay slowly and are temporally smooth. Random noise fluctuates rapidly, yielding low SNR."
        },
        {
            "name": "activity_ratio",
            "category": "New Biological",
            "calc": "std(Fcorr > med) / std(Fcorr <= med)",
            "utility": "<font color='#38A169'><b>Highly Useful</b></font>",
            "comments": "Ratio of variance above baseline median to variance below. Cleanly captures positive-only calcium events compared to symmetric noise."
        },
        {
            "name": "peak_density",
            "category": "New Biological",
            "calc": "len(peaks_2sig) / len(Fcorr)",
            "utility": "<font color='#3182CE'>Useful</font>",
            "comments": "Fires-per-frame density, counting peaks exceeding 2 standard deviations. Helps distinguish active firing somas from inactive/noise ROIs."
        }
    ]

    table_data = [[
        Paragraph("Feature Name", th_style),
        Paragraph("Category", th_style),
        Paragraph("Calculation / Formula", th_style),
        Paragraph("Utility & Importance", th_style),
        Paragraph("Meaning & Comments", th_style)
    ]]

    for f in features_raw:
        table_data.append([
            Paragraph(f["name"], td_name_style),
            Paragraph(f["category"], td_cat_style),
            Paragraph(f["calc"], td_calc_style),
            Paragraph(f["utility"], td_utility_style),
            Paragraph(f["comments"], td_comment_style)
        ])

    col_widths = [115, 80, 130, 95, 300]
    t = Table(table_data, colWidths=col_widths, repeatRows=1)
    
    t_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 5),
        ('TOPPADDING', (0, 0), (-1, 0), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
    ])
    
    for idx in range(1, len(table_data)):
        bg_color = colors.HexColor("#F7FAFC") if idx % 2 == 0 else colors.white
        t_style.add('BACKGROUND', (0, idx), (-1, idx), bg_color)
        t_style.add('TOPPADDING', (0, idx), (-1, idx), 4)
        t_style.add('BOTTOMPADDING', (0, idx), (-1, idx), 4)
        t_style.add('LEFTPADDING', (0, idx), (-1, idx), 4)
        t_style.add('RIGHTPADDING', (0, idx), (-1, idx), 4)

    t.setStyle(t_style)
    story.append(t)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF: {filename}")

if __name__ == "__main__":
    output_pdf = "suite2p_features_reference.pdf"
    if len(sys.argv) > 1:
        output_pdf = sys.argv[1]
    build_pdf(output_pdf)
