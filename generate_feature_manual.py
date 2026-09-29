import os
import sys
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to draw 'Page X of Y' page numbers and professional headers/footers
    on all pages except the first page (cover).
    """
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
            if self._pageNumber > 1:
                self.draw_decorations(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#1A365D")) # Deep Navy
        
        # Header text
        self.drawString(54, 750, "SUITE2P AI CLASSIFIER - FEATURE REFERENCE GUIDE")
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))
        self.drawRightString(558, 750, "PHD SUPERVISOR REFERENCE MANUAL")
        
        # Header Line
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.75)
        self.line(54, 742, 558, 742)
        
        # Footer text
        self.drawString(54, 40, "Confidential - PhD Research & Validation")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 40, page_text)
        
        # Footer Line
        self.line(54, 52, 558, 52)
        self.restoreState()


def build_pdf(filename):
    # Setup document
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=72,
        bottomMargin=72
    )

    # Styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=26,
        leading=32,
        textColor=colors.HexColor("#1A365D"),
        alignment=1, # Center
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#4A5568"),
        alignment=1, # Center
        spaceAfter=40
    )
    
    meta_style = ParagraphStyle(
        'CoverMeta',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#718096"),
        alignment=1, # Center
        spaceAfter=60
    )
    
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1A365D"),
        spaceBefore=15,
        spaceAfter=10,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#2B6CB0"), # Teal-Blue
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=8
    )

    bullet_style = ParagraphStyle(
        'BulletCustom',
        parent=styles['Bullet'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#2D3748"),
        bulletIndent=10,
        leftIndent=20,
        spaceAfter=4
    )

    formula_style = ParagraphStyle(
        'FormulaText',
        parent=styles['Code'],
        fontName='Courier',
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#C53030"), # Reddish
        backColor=colors.HexColor("#F7FAFC"),
        borderColor=colors.HexColor("#E2E8F0"),
        borderWidth=0.5,
        borderPadding=6,
        spaceAfter=8,
        spaceBefore=4
    )

    story = []

    # =========================================================================
    # COVER PAGE
    # =========================================================================
    story.append(Spacer(1, 100))
    story.append(Paragraph("Suite2p AI Classifier", title_style))
    story.append(Paragraph("Model Features Reference Manual & Biological Context", subtitle_style))
    story.append(Spacer(1, 20))
    
    # Table for metadata
    meta_data = [
        [Paragraph("<b>Prepared For:</b>", body_style), Paragraph("PhD Supervisor Meeting", body_style)],
        [Paragraph("<b>Project:</b>", body_style), Paragraph("Suite2p ROI Classification and Active Learning", body_style)],
        [Paragraph("<b>Date:</b>", body_style), Paragraph("June 24, 2026", body_style)],
        [Paragraph("<b>Environment:</b>", body_style), Paragraph("LightGBM Pipeline (Regular / No-Index Presets)", body_style)],
    ]
    meta_table = Table(meta_data, colWidths=[120, 250])
    meta_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(KeepTogether([meta_table]))
    
    story.append(Spacer(1, 100))
    story.append(Paragraph("This document contains an exhaustive breakdown of every shape, spatial, signal-to-noise, temporal, and biological feature utilized in the Suite2p cell classification models. Formulas, physiological reasoning, numerical implementations, and failure modes are detailed below.", meta_style))
    story.append(PageBreak())

    # =========================================================================
    # INTRODUCTION
    # =========================================================================
    story.append(Paragraph("1. Overview of the Classification Pipeline", h1_style))
    story.append(Paragraph(
        "Automated curation of calcium imaging regions of interest (ROIs) is a critical bottleneck in "
        "two-photon data processing. Suite2p extracts candidates based on spatial and temporal correlation, "
        "but manual sorting remains noisy and time-consuming. The <b>Suite2p AI Classifier</b> provides "
        "machine learning models that evaluate shape-invariant morphological signatures and trace kinetics "
        "to distinguish real neuronal somata from neuropil debris, processes, and optical noise.",
        body_style
    ))
    story.append(Paragraph(
        "Two main configurations are maintained to address different experiment pipelines:",
        body_style
    ))
    story.append(Paragraph(
        "<b>1. Regular Model (25 features)</b>: Integrates continuous spatial index rankings "
        "(sorting coordinates) to leverage Suite2p's default sorting order. Useful when the default "
        "classifier provides a reliable relative sorting, but can introduce spatial and pipeline biases.",
        bullet_style
    ))
    story.append(Paragraph(
        "<b>2. No-Index Model (27 features)</b>: Free of spatial indexing bias. Relies solely on "
        "biological and numerical signal shapes. Introduces advanced temporal SNR, peak densities, and "
        "activity metrics, ensuring high generalizability across rigs and configurations.",
        bullet_style
    ))
    story.append(Spacer(1, 15))

    # =========================================================================
    # DETAILED FEATURE REFERENCE
    # =========================================================================
    story.append(Paragraph("2. Feature-by-Feature Detailed Directory", h1_style))
    
    features = [
        # --- solidity ---
        {
            "name": "solidity",
            "cat": "Morphology & Spatial Shape",
            "formula": "Solidity = Area(ROI) / Area(Convex Hull)",
            "bio": "Real neuron somata are roughly spherical or ellipsoidal in 2D projection, yielding highly convex shapes with smooth, closed boundaries. Debris, dendritic segments, or crossing axons form highly concave, star-shaped, or irregular ROIs.",
            "num": "Calculated by Suite2p. Compares the number of pixels in the ROI to the pixels enclosed by its convex envelope.",
            "pros": "Extremely robust at rejecting overlapping ROIs, dendrites, and diffuse background noise.",
            "cons": "Out-of-focus round debris or localized bright noise blobs can also have a solidity near 1.0, requiring trace-based validation."
        },
        # --- mrs ---
        {
            "name": "mrs",
            "cat": "Morphology & Spatial Shape",
            "formula": "Mrs = Standard Deviation of Centroid-to-Boundary Radii",
            "bio": "Measures the radial symmetry and roundness of the ROI. A perfect circle has Mrs = 0. Real somata exhibit low Mrs (high roundness). Elongated dendritic processes, motion-sheared shapes, and multi-peak artifacts have high Mrs.",
            "num": "Calculated by Suite2p during contour extraction and saved as 'mrs' in stat.npy.",
            "pros": "Strong indicator of ellipsoidal somatic structure. Helps prune highly elongated shapes.",
            "cons": "Small cells (e.g. interneurons or deep cells) with few pixels are susceptible to discretization noise, inflating Mrs artificially."
        },
        # --- compact ---
        {
            "name": "compact",
            "cat": "Morphology & Spatial Shape",
            "formula": "Compactness = Perimeter² / (4π × Area)",
            "bio": "Another metric of circularity. Somatic ROIs are compact and close to 1.0. High values indicate complex, irregular, or branch-like boundaries.",
            "num": "Saved in stat.npy as 'compact'. Reflects boundary complexity.",
            "pros": "Quantifies spatial irregularity; excellent for identifying dendrites.",
            "cons": "Correlates strongly with Mrs and solidity, which can lead to feature redundancy."
        },
        # --- aspect_ratio ---
        {
            "name": "aspect_ratio",
            "cat": "Morphology & Spatial Shape",
            "formula": "Aspect Ratio = Major Axis / Minor Axis (of fitted ellipse)",
            "bio": "Real somata are slightly oval or round (Aspect Ratio ≈ 1.0 - 1.5). Dendrites, blood vessels, and motion-blur artifacts have highly elongated geometries (Aspect Ratio > 2.0).",
            "num": "Calculated from the principal moments of the ROI coordinate matrix.",
            "pros": "Directly filters out linear structures (apical dendrites, blood vessels).",
            "cons": "Neurons imaged at steep angles or near the edge of the scan field may appear moderately elongated."
        },
        # --- number_of_bright_pixels ---
        {
            "name": "number_of_bright_pixels",
            "cat": "Morphology & Spatial Shape",
            "formula": "Bright Pixels = Count( λ_i > 0.1 × max(λ) )",
            "bio": "In a real neuron, fluorescence is concentrated in a bright cytoplasmic shell around the nucleus. The spatial weights (λ) of the pixels should have a dense, highly weighted core.",
            "num": "Calculates the number of pixels in stat['lam'] that exceed 10% of the maximum weight.",
            "pros": "Identifies the core mass of the cell, filtering out diffuse neuropil blocks.",
            "cons": "Sensitive to absolute photon counts and SNR. Dim cells with few photons can have noisy core estimations."
        },
        # --- bright_pixels_ratio ---
        {
            "name": "bright_pixels_ratio",
            "cat": "Morphology & Spatial Shape",
            "formula": "Bright Pixels Ratio = Number of Bright Pixels / npix",
            "bio": "Eliminates absolute size dependencies. Real cells show a compact bright core, meaning a large proportion of their pixels are bright (ratio of 0.4 - 0.8). Diffuse ROIs or background noise have low ratios.",
            "num": "Divides the number_of_bright_pixels by the total pixel count (npix).",
            "pros": "Size-invariant. Distinguishes small, compact cells from large, diffuse neuropil patches.",
            "cons": "Very small ROIs (e.g. <15 pixels) can have extreme ratios due to small-denominator effects."
        },
        # --- npix ---
        {
            "name": "npix",
            "cat": "Morphology & Spatial Shape",
            "formula": "Npix = Total number of pixels in the ROI",
            "bio": "Represents the physical cross-sectional size of the somatic region.",
            "num": "Extracted directly from stat['npix'].",
            "pros": "Guards against extremely small (sub-cellular) or extremely large (multi-cell/neuropil) ROIs.",
            "cons": "Highly dependent on optical zoom, frame size, and resolution. Can cause overfitting if magnification changes."
        },
        # --- roi_idx_norm ---
        {
            "name": "roi_idx_norm",
            "cat": "Spatial Rank Index",
            "formula": "ROI Index Norm = i / N_cells",
            "bio": "Suite2p outputs ROIs in order of its internal classifier score. ROIs with low indices are highly confident cells under Suite2p's default model.",
            "num": "The 0-indexed position of the ROI divided by the total number of candidate ROIs.",
            "pros": "Captures the relative confidence of the baseline Suite2p classifier, acting as a strong prior.",
            "cons": "Creates a dependency on the Suite2p classifier. If Suite2p misclassifies a session, this index propagates the bias."
        },
        # --- skew_f ---
        {
            "name": "skew_f",
            "cat": "Trace Statistics",
            "formula": "Skew(F) = E[(F - μ_F)³] / σ_F³",
            "bio": "Active calcium indicator signals are asymmetric: long periods near baseline, interrupted by sharp, upward spikes. This generates positive skewness. Flat noise or symmetric baseline fluctuations have a skewness near 0.",
            "num": "Calculates the third standardized moment of the raw fluorescence trace F.",
            "pros": "Very powerful biological marker. Rejects quiet background noise and symmetrical laser power oscillations.",
            "cons": "High-amplitude artifacts (e.g. motion artifacts, laser power fluctuations) can cause massive positive or negative skews."
        },
        # --- std_f ---
        {
            "name": "std_f",
            "cat": "Trace Statistics",
            "formula": "Std(F) = σ_F (in Regular)   or   σ_F / σ_ΔF (in No-Index)",
            "bio": "Measures trace variance. Real cells with transients have high variance. In the No-Index model, it is divided by the standard deviation of first differences (σ_ΔF) to normalize against high-frequency noise.",
            "num": "Standard deviation of trace F. Normalized by std(diff(F)) in No-Index.",
            "pros": "Distinguishes active cells from inactive background. Normalization provides scale-invariance.",
            "cons": "Uncorrected neuropil contamination or regional baseline drift can inflate this value."
        },
        # --- max_to_mean_f ---
        {
            "name": "max_to_mean_f",
            "cat": "Trace Statistics",
            "formula": "Max-to-Mean = max(F) / mean(F)",
            "bio": "Active neurons show large, transient upward spikes that far exceed their average baseline fluorescence level.",
            "num": "Maximum value of trace F divided by the mean value of trace F.",
            "pros": "Identifies cells with sparse, high-amplitude bursts.",
            "cons": "If the baseline offset (mean) is high due to background excitation, this ratio is heavily compressed."
        },
        # --- cv_f ---
        {
            "name": "cv_f",
            "cat": "Trace Statistics",
            "formula": "CV(F) = σ_F / μ_F",
            "bio": "The coefficient of variation. Measures trace variability relative to the mean brightness.",
            "num": "Standard deviation divided by mean of trace F.",
            "pros": "Provides a normalized measure of signal fluctuation.",
            "cons": "Sensitive to absolute baseline fluorescence offsets."
        },
        # --- skew_fneu ---
        {
            "name": "skew_fneu",
            "cat": "Trace Statistics",
            "formula": "Skew(Fneu) = E[(Fneu - μ_fneu)³] / σ_fneu³",
            "bio": "The neuropil trace represents average out-of-focus background light. Neuropil signals are typically diffuse and symmetric (low skewness). High positive skew in the neuropil indicates a nearby highly active cell bleeding in.",
            "num": "Skewness of the neuropil trace Fneu.",
            "pros": "Helps detect when nearby active cells are contaminating the local background.",
            "cons": "Highly synchronized population activity can cause regional neuropil skewness, confusing the model."
        },
        # --- corr_f_fneu ---
        {
            "name": "corr_f_fneu",
            "cat": "Trace Statistics",
            "formula": "Corr(F, Fneu) = Pearson_r(F, Fneu)",
            "bio": "If an ROI's signal is just out-of-focus neuropil contamination (not a real cell), F and Fneu will be highly correlated. Real cells have independent somatic activity, lowering their correlation.",
            "num": "Pearson correlation coefficient between traces F and Fneu.",
            "pros": "Excellent for pruning non-cells that are completely contaminated by local neuropil background.",
            "cons": "In densely packed tissues, a real cell might fire in sync with surrounding neuropil, leading to high correlation."
        },
        # --- skew_fcorr ---
        {
            "name": "skew_fcorr",
            "cat": "Trace Statistics",
            "formula": "Skew(Fcorr) = Skew(F - 0.7 × Fneu)",
            "bio": "The neuropil-corrected trace (Fcorr) contains the true cell signal. Real active neurons must show strong positive skewness in Fcorr.",
            "num": "Skewness of the corrected trace Fcorr.",
            "pros": "The primary temporal indicator of biological action. Rejects artifacts that disappear after subtraction.",
            "cons": "If the 0.7 subtraction coefficient is too high or low, it can introduce artifacts or suppress real signals."
        },
        # --- std_fcorr ---
        {
            "name": "std_fcorr",
            "cat": "Trace Statistics",
            "formula": "Std(Fcorr) = σ_Fcorr (Regular)   or   σ_Fcorr / σ_ΔFcorr (No-Index)",
            "bio": "Trace variability after neuropil correction. In No-Index, normalization by first differences removes dependence on absolute fluorescence scales and rig noise levels.",
            "num": "Standard deviation of corrected trace. Normalized by std(diff(Fcorr)) in No-Index.",
            "pros": "Captures true cell signal amplitude relative to baseline noise.",
            "cons": "Can be inflated by subtraction noise if Fneu is extremely noisy."
        },
        # --- quantiles ---
        {
            "name": "q10, q25, q50, q75, q90, q95, q99",
            "cat": "Trace Quantiles",
            "formula": "Normalized Q_p = (Q_p - Median) / σ_ΔFcorr",
            "bio": "Real GCaMP traces spend most of their time at baseline (lower quantiles are tightly clustered). Active calcium transients pull the upper quantiles (q90, q95, q99) far above the median, creating a characteristic distribution shape.",
            "num": "Percentiles of corrected trace Fcorr. Normalized by subtracting median and dividing by first-difference std in No-Index.",
            "pros": "Quantifies the entire distribution shape, distinguishing brief transients from continuous background shifts.",
            "cons": "Susceptible to slow baseline drift if the imaging field drifts during long acquisitions."
        },
        # --- avg_asym & max_asym ---
        {
            "name": "avg_asym, max_asym",
            "cat": "Calcium kinetics & Peak Dynamics",
            "formula": "Asymmetry = Decay Width / Rise Width (at half-height)",
            "bio": "GCaMP calcium indicators have a very fast rise time (calcium influx) and a much slower exponential decay (calcium extrusion). Biological peaks are highly asymmetric (Asymmetry > 2.0). Noise fluctuations are symmetric (Asymmetry ≈ 1.0).",
            "num": "Detects peaks (height > mean + 2σ, distance > 10). Computes rise and decay width at half-height using peak_widths.",
            "pros": "A gold-standard biological classifier. Strongly rejects symmetric optical or mechanical noise.",
            "cons": "Requires active firing. Silent or quiet cells have no peaks, defaulting asymmetry to 1.0 (noise-like)."
        },
        # --- max_width ---
        {
            "name": "max_width",
            "cat": "Calcium kinetics & Peak Dynamics",
            "formula": "Max Width = max(Peak Width at half-height) in frames",
            "bio": "GCaMP calcium transients have a characteristic physiological duration (typically 0.5s to 3s). High-frequency noise spikes are narrow (1-2 frames).",
            "num": "Maximum width at half-height among detected peaks.",
            "pros": "Filters out high-frequency electrical or photon noise.",
            "cons": "Slow mechanical motion artifacts can create wide peaks that mimic biological transients."
        },
        # --- range_fcorr & range_f ---
        {
            "name": "range_fcorr, range_f",
            "cat": "Trace Amplitudes",
            "formula": "Range = (max - min) / σ_Δ",
            "bio": "The total peak-to-peak amplitude. Active cells show large fluctuations compared to baseline noise.",
            "num": "Difference between max and min values of the trace, normalized by first-difference std in No-Index.",
            "pros": "Captures the maximum transient magnitude.",
            "cons": "Single high-amplitude artifacts (e.g. dust particles crossing the laser path) will dominate the range."
        },
        # --- snr ---
        {
            "name": "snr",
            "cat": "Biological & Signal Quality",
            "formula": "SNR = σ_Fcorr / σ_ΔFcorr",
            "bio": "Real calcium transients are slow biological signals, whereas noise is fast. By taking the ratio of trace standard deviation to first-difference standard deviation (which isolates fast noise), we obtain a pure temporal SNR.",
            "num": "Ratio of corrected trace standard deviation to the standard deviation of its first difference.",
            "pros": "Provides a clean, unit-less signal quality metric that is independent of absolute brightness.",
            "cons": "Highly active cells with very frequent, fast-firing transients can have slightly elevated first differences."
        },
        # --- activity_ratio ---
        {
            "name": "activity_ratio",
            "cat": "Biological & Signal Quality",
            "formula": "Activity Ratio = σ(Fcorr > Median) / σ(Fcorr ≤ Median)",
            "bio": "Calcium transients are strictly positive. Therefore, the variance of the signal above the median (which contains transients) should be much higher than the variance below the median (which represents baseline noise).",
            "num": "Calculates standard deviation of values above median, divided by standard deviation of values below/equal to median.",
            "pros": "Highly sensitive to positive excursions. Noise has a symmetric ratio ≈ 1.0.",
            "cons": "Slow baseline drift or asymmetric artifacts can skew this ratio."
        },
        # --- peak_density ---
        {
            "name": "peak_density",
            "cat": "Biological & Signal Quality",
            "formula": "Peak Density = Number of Detected Peaks / Total Frames",
            "bio": "Refers to the firing rate. Real cells fire occasionally, whereas very noisy ROIs or blood vessels have false peaks (high density) or none (zero density).",
            "num": "Count of detected peaks divided by trace length.",
            "pros": "Captures active firing frequency.",
            "cons": "Highly silent cells have zero peak density, making them look similar to flat noise."
        }
    ]

    for feat in features:
        # Build KeepTogether block for each feature to avoid breaking across pages
        feat_story = []
        feat_story.append(Paragraph(f"Feature: {feat['name']}", h2_style))
        
        # Details Table
        details = [
            [Paragraph("<b>Category:</b>", body_style), Paragraph(feat['cat'], body_style)],
            [Paragraph("<b>Exact Formula:</b>", body_style), Paragraph(feat['formula'], formula_style)],
            [Paragraph("<b>Biological Context:</b>", body_style), Paragraph(feat['bio'], body_style)],
            [Paragraph("<b>Implementation:</b>", body_style), Paragraph(feat['num'], body_style)],
            [Paragraph("<b>Why it Helps:</b>", body_style), Paragraph(feat['pros'], body_style)],
            [Paragraph("<b>Downsides & Risks:</b>", body_style), Paragraph(feat['cons'], body_style)],
        ]
        
        t = Table(details, colWidths=[110, 394])
        t.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('LINEBELOW', (0,0), (-1,-2), 0.5, colors.HexColor("#EDF2F7")),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
        ]))
        
        feat_story.append(t)
        feat_story.append(Spacer(1, 15))
        story.append(KeepTogether(feat_story))

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF: {filename}")


if __name__ == "__main__":
    output_pdf = "suite2p_features_reference.pdf"
    build_pdf(output_pdf)
