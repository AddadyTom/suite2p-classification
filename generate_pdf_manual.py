import os
import sys
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

class SimpleNumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to draw simple page numbers at the bottom center of all pages.
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
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#718096"))
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawCentredString(306, 36, page_text)
        self.restoreState()


def build_pdf(filename):
    # Setup document (smaller top/bottom margins since no header/footers)
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    # Styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=15
    )
    
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#2B6CB0"), # Teal-Blue
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9,
        leading=12.5,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=6
    )

    formula_style = ParagraphStyle(
        'FormulaText',
        parent=styles['Code'],
        fontName='Courier',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#C53030"), # Reddish
        backColor=colors.HexColor("#F7FAFC"),
        borderColor=colors.HexColor("#E2E8F0"),
        borderWidth=0.5,
        borderPadding=5,
        spaceAfter=6,
        spaceBefore=4
    )

    story = []

    story.append(Paragraph("Suite2p Classifier - Feature Reference Guide", title_style))
    story.append(Paragraph(
        "Below is the complete reference directory for the features used in the Suite2p classification models. "
        "Quantiles, asymmetry metrics, range statistics, smoothness ratios, and active fraction parameters "
        "are grouped together as they represent unified concept metrics.",
        body_style
    ))
    story.append(Spacer(1, 10))

    features = [
        # --- 1. npix ---
        {
            "name": "npix",
            "cat": "Spatial & Morphology",
            "formula": "npix = Sum_{p in ROI} 1",
            "bio": "Reflects the cross-sectional physical area of the ROI. Neuronal somas imaged under typical magnification (e.g., 16x or 20x objective, 1-2x zoom) fall within a specific size envelope (typically 50 to 300 pixels).",
            "num": "Extracted directly from stat['npix']. Counts the total number of pixels assigned to the ROI.",
            "pros": "Highly effective at filtering out extremely small noise fragments (sub-somatic debris) or very large artifacts (dendritic trunks, blood vessels, or multi-cell merges).",
            "cons": "Extremely dependent on magnification, frame size, pixel size, and optical zoom. A model trained on a specific zoom level may misclassify valid cells if applied to data collected at different optical settings."
        },
        # --- 2. solidity ---
        {
            "name": "solidity",
            "cat": "Spatial & Morphology",
            "formula": "solidity = Area(ROI) / Area(Convex Hull)",
            "bio": "Neuronal somas are generally compact, round, or ellipsoidal shapes, which are highly convex. Debris, overlapping processes, or crossing axons form irregular, concave, or branched structures with lower solidity.",
            "num": "Calculated by Suite2p and stored in stat['solidity']. Measures the ratio of ROI pixels to the pixels enclosed by its minimum bounding convex polygon.",
            "pros": "Superb at identifying and rejecting branching dendrites, axons, and complex background noise.",
            "cons": "Out-of-focus round debris or localized laser flare hotspots can have a solidity close to 1.0. Conversely, true cells with bright proximal dendrites included in the ROI may have reduced solidity and get falsely rejected."
        },
        # --- 3. mrs ---
        {
            "name": "mrs",
            "cat": "Spatial & Morphology",
            "formula": "mrs = std( { ||p_i - centroid|| } for p_i in boundary )",
            "bio": "Mean Square Radius variation. Measures the radial symmetry and circularity of the ROI boundary. A perfect circle has Mrs = 0. Real somata exhibit low Mrs, whereas irregular, elongated, or multi-center ROIs have high Mrs.",
            "num": "Calculated by Suite2p during boundary contour extraction and stored in stat['mrs'].",
            "pros": "Strong morphological metric to enforce circularity and prune highly asymmetric or multi-peak shapes.",
            "cons": "Small cells (e.g. interneurons or deep cells) containing very few pixels are susceptible to pixelation/discretization noise, which inflates Mrs artificially and leads to false negatives."
        },
        # --- 4. compact ---
        {
            "name": "compact",
            "cat": "Spatial & Morphology",
            "formula": "compact = perimeter^2 / (4 * pi * Area)",
            "bio": "Morphological compactness or circularity. A perfect circle has a compactness of 1.0. Complex, jagged, or branch-like boundaries have significantly higher values.",
            "num": "Saved in stat['compact']. Measures boundary complexity based on pixel perimeter and area.",
            "pros": "Excellent at identifying and pruning complex dendritic boundaries and merged doublets.",
            "cons": "Highly redundant with Mrs and solidity, which can increase model complexity without adding unique variance if not regularized."
        },
        # --- 5. aspect_ratio ---
        {
            "name": "aspect_ratio",
            "cat": "Spatial & Morphology",
            "formula": "aspect_ratio = Major Axis / Minor Axis (of fitted ellipse)",
            "bio": "Neuronal somata are round or slightly oval (Aspect Ratio between 1.0 and 1.5). Dendrites, blood vessels, and motion-shearing artifacts are highly elongated (Aspect Ratio > 2.0).",
            "num": "Calculated from the principal moments of the ROI coordinate matrix and stored in stat['aspect_ratio'].",
            "pros": "Directly filters out linear, elongated non-somatic structures (blood vessels, apical dendrites, and motion streaking).",
            "cons": "Cells imaged at steep angles near the edges of the field of view or compressed by nearby tissue may appear moderately elongated, risking false rejection."
        },
        # --- 6. radius ---
        {
            "name": "radius",
            "cat": "Spatial & Morphology",
            "formula": "radius = sqrt(a * b / 2) (where a, b are fitted ellipse semi-axes)",
            "bio": "Represents the equivalent physical radius of the cell soma based on the fitted ellipse parameters.",
            "num": "Extracted from stat['radius'] which is computed by fitting an ellipse to the ROI coordinates.",
            "pros": "Provides a clean, interpretable physical scale constraint for the cellular soma.",
            "cons": "Fails to accurately represent cells with irregular or highly asymmetric non-elliptical boundaries."
        },
        # --- 7. number_of_bright_pixels ---
        {
            "name": "number_of_bright_pixels",
            "cat": "Spatial & Morphology",
            "formula": "nbright = count( lambda_i > 0.1 * max(lambda) )",
            "bio": "In genuine neurons, GCaMP expression is concentrated in a bright cytoplasmic shell around the nucleus. The spatial weights (lambda) of the pixels should have a dense, highly weighted core rather than a diffuse or flat distribution.",
            "num": "Calculated from the spatial pixel weights in stat['lam'] by counting those exceeding 10% of the maximum weight.",
            "pros": "Highly effective at identifying the dense core mass of a cell, filtering out diffuse neuropil or weak background patches. Ranked #1 in SHAP importance.",
            "cons": "Vulnerable to low signal-to-noise ratio. In dim cells with few photons, the spatial weights can be extremely noisy, leading to unstable core mass estimation."
        },
        # --- 8. bright_pixels_ratio ---
        {
            "name": "bright_pixels_ratio",
            "cat": "Spatial & Morphology",
            "formula": "bright_pixels_ratio = number_of_bright_pixels / npix",
            "bio": "Size-normalized metric of cytoplasmic signal concentration. Real cells show a compact bright core, meaning a large proportion of their total pixels are bright (ratio of 0.4 - 0.8). Diffuse ROIs or background noise have low ratios.",
            "num": "Divides the number of bright pixels by the total pixel count (npix).",
            "pros": "Size-invariant. Excellent for distinguishing small, compact cells from large, diffuse neuropil patches.",
            "cons": "Very small ROIs can have extreme, noisy ratios due to small-denominator effects."
        },
        # --- 9. area_to_radius_sq ---
        {
            "name": "area_to_radius_sq",
            "cat": "Spatial & Morphology",
            "formula": "area_to_radius_sq = npix / max(radius^2, 1e-6)",
            "bio": "For a perfect circle, Area = pi * radius^2, so this ratio is pi (~3.14). Significant deviations from this ratio indicate irregular, elongated, or fragmented geometries.",
            "num": "Computed by dividing npix by the squared radius (with a 1e-6 safety denominator).",
            "pros": "Quantifies deviations from ideal circular geometry using two separate measurements (pixel count vs fitted ellipse radius).",
            "cons": "Highly sensitive to radius estimation errors, which are common for highly irregular or fragmented shapes."
        },
        # --- 10. bright_pixels_to_radius_sq ---
        {
            "name": "bright_pixels_to_radius_sq",
            "cat": "Spatial & Morphology",
            "formula": "bright_pixels_to_radius_sq = number_of_bright_pixels / max(radius^2, 1e-6)",
            "bio": "Normalizes the size of the active cell core by the squared radius of the fitted ellipse. Real cells show a high density of bright pixels in their central core.",
            "num": "Computed by dividing number_of_bright_pixels by the squared radius.",
            "pros": "Ensures that the bright core size scales properly with the overall fitted size of the cell.",
            "cons": "Fails for hollow cells (e.g. ring-shaped labeling where the center is dark) or cells with highly non-uniform GCaMP expression."
        },
        # --- 11. roi_idx_norm ---
        {
            "name": "roi_idx_norm",
            "cat": "Spatial Rank Index",
            "formula": "roi_idx_norm = i / N_cells",
            "bio": "Suite2p outputs ROIs in order of its internal classifier score. Lower indices represent high confidence under Suite2p's default model. Normalizing it provides a standard prior across sessions.",
            "num": "The 0-indexed position of the ROI in the output array, divided by the total number of candidate ROIs.",
            "pros": "Acts as a strong, continuous prior by leveraging Suite2p's default classification ranking.",
            "cons": "Highly pipeline-dependent. It propagates any systematic sorting bias from Suite2p's default classifier. If the default classifier performs poorly on a session, this index will mislead the AI model."
        },
        # --- 12. skew_f ---
        {
            "name": "skew_f",
            "cat": "Raw Trace Statistics",
            "formula": "skew_f = E[(F - mean(F))^3] / std(F)^3",
            "bio": "Active calcium indicator signals are highly asymmetric: long periods near baseline, interrupted by sharp, upward spikes. This generates positive skewness. Symmetrical noise has a skewness near 0.",
            "num": "Calculates the third standardized moment of the raw fluorescence trace F.",
            "pros": "A very powerful biological marker. Rejects quiet background noise and symmetrical laser power oscillations.",
            "cons": "Extremely vulnerable to positive-going non-biological artifacts, such as movement artifacts, laser power spikes, or dust particles passing through the light path."
        },
        # --- 13. std_f ---
        {
            "name": "std_f",
            "cat": "Raw Trace Statistics",
            "formula": "Regular: std(F)\nNo-Index: std(F) / std(diff(F))",
            "bio": "Measures total trace variability. Active cells with calcium transients have high variance. In the No-Index model, it is normalized by the standard deviation of its first difference to remove rig-specific noise scaling.",
            "num": "Standard deviation of trace F. Normalized by std(diff(F)) in No-Index configurations.",
            "pros": "Effective at distinguishing active cells from inactive background. Normalization provides scale-invariance across different laser power levels.",
            "cons": "Highly sensitive to slow baseline drifts, photobleaching, or z-drift, which inflate trace variance without reflecting biological transients."
        },
        # --- 14. max_to_mean_f ---
        {
            "name": "max_to_mean_f",
            "cat": "Raw Trace Statistics",
            "formula": "max_to_mean_f = max(F) / mean(F)",
            "bio": "Active neurons show large, transient upward spikes that far exceed their average baseline fluorescence level.",
            "num": "Maximum value of trace F divided by the mean value of trace F (with a safety threshold mean >= 1e-10).",
            "pros": "Identifies cells with sparse, high-amplitude bursts.",
            "cons": "If the baseline offset (mean) is high due to deep imaging or strong neuropil background, this ratio is heavily compressed."
        },
        # --- 15. cv_f ---
        {
            "name": "cv_f",
            "cat": "Raw Trace Statistics",
            "formula": "cv_f = std(F) / mean(F)",
            "bio": "The coefficient of variation. Measures trace variability relative to the mean brightness.",
            "num": "Standard deviation divided by mean of trace F.",
            "pros": "Provides a normalized measure of signal fluctuation.",
            "cons": "Highly sensitive to absolute baseline fluorescence offsets. division by small numbers can cause instability when mean fluorescence is close to zero."
        },
        # --- 16. skew_fneu ---
        {
            "name": "skew_fneu",
            "cat": "Neuropil Statistics",
            "formula": "skew_fneu = E[(Fneu - mean(Fneu))^3] / std(Fneu)^3",
            "bio": "The neuropil trace represents average out-of-focus background light. Neuropil signals are typically diffuse and symmetric (low skewness). High positive skew in the neuropil indicates a nearby highly active cell bleeding in.",
            "num": "Skewness of the neuropil trace Fneu.",
            "pros": "Helps detect when nearby active cells are contaminating the local background.",
            "cons": "Highly synchronized population activity can cause regional neuropil skewness, confusing the model."
        },
        # --- 17. corr_f_fneu ---
        {
            "name": "corr_f_fneu",
            "cat": "Neuropil Statistics",
            "formula": "corr_f_fneu = Pearson_r(F, Fneu)",
            "bio": "If an ROI's signal is just out-of-focus neuropil contamination (not a real cell), F and Fneu will be highly correlated. Real cells have independent somatic activity, lowering their correlation.",
            "num": "Pearson correlation coefficient between traces F and Fneu.",
            "pros": "Excellent for pruning non-cells that are completely contaminated by local neuropil background.",
            "cons": "In densely packed tissues, a real cell might fire in sync with surrounding neuropil, leading to high correlation."
        },
        # --- 18. skew_fcorr ---
        {
            "name": "skew_fcorr",
            "cat": "Neuropil-Corrected Trace Statistics",
            "formula": "skew_fcorr = skew(F - 0.7 * Fneu)",
            "bio": "The neuropil-corrected trace (Fcorr) contains the true cell signal. Real active neurons must show strong positive skewness in Fcorr.",
            "num": "Skewness of the corrected trace Fcorr.",
            "pros": "The primary temporal indicator of biological action. Rejects artifacts that disappear after subtraction.",
            "cons": "If the 0.7 subtraction coefficient is too high or low, it can introduce artifacts or suppress real signals."
        },
        # --- 19. std_fcorr ---
        {
            "name": "std_fcorr",
            "cat": "Neuropil-Corrected Trace Statistics",
            "formula": "Regular: std(Fcorr)\nNo-Index: std(Fcorr) / std(diff(Fcorr))",
            "bio": "Trace variability after neuropil correction. In No-Index, normalization by first differences removes dependence on absolute fluorescence scales and rig noise levels.",
            "num": "Standard deviation of corrected trace. Normalized by std(diff(Fcorr)) in No-Index.",
            "pros": "Captures true cell signal amplitude relative to baseline noise.",
            "cons": "Can be inflated by subtraction noise if Fneu is extremely noisy."
        },
        # --- 20. quantiles (q10, q25, q50, q75, q90, q95, q99) ---
        {
            "name": "q10, q25, q50, q75, q90, q95, q99",
            "cat": "Trace Quantiles",
            "formula": "Regular: Percentile_p(Fcorr)\nNo-Index: (Percentile_p(Fcorr) - median(Fcorr)) / std(diff(Fcorr))\n[q50 in No-Index is: median(Fcorr) / std(diff(Fcorr))]",
            "bio": "Real GCaMP traces spend most of their time at baseline (lower quantiles are tightly clustered). Active calcium transients pull the upper quantiles (q90, q95, q99) far above the median, creating a characteristic distribution shape.",
            "num": "Percentiles of corrected trace Fcorr. Normalized by subtracting median and dividing by first-difference std in No-Index.",
            "pros": "Quantifies the entire distribution shape, distinguishing brief transients from continuous background shifts. Ranked highly in SHAP.",
            "cons": "For inactive cells, upper quantiles just reflect the tail of the baseline noise. If noise is non-Gaussian (e.g. shot noise), these quantiles can mimic active traces."
        },
        # --- 21. range_fcorr / range_f ---
        {
            "name": "range_fcorr, range_f",
            "cat": "Trace Amplitudes",
            "formula": "Regular: max(x) - min(x)\nNo-Index: (max(x) - min(x)) / std(diff(x))",
            "bio": "Captures the maximum dynamic range of the raw (F) or corrected (Fcorr) trace relative to the baseline and high-frequency noise. Active cells display large transient ranges.",
            "num": "Peak-to-peak range, normalized by the standard deviation of its first difference in the No-Index model.",
            "pros": "Highlights the absolute maximum signal excursion, useful for picking up sparse, high-amplitude bursts.",
            "cons": "Extremely vulnerable to single outlier spikes, hot pixels, laser glitches, or motion artifacts that happen once in a session."
        },
        # --- 22. snr ---
        {
            "name": "snr",
            "cat": "Biological & Signal Quality",
            "formula": "snr = std(Fcorr) / std(diff(Fcorr))",
            "bio": "Disentangles the slow biological signal variance from fast high-frequency measurement noise. A high SNR is a strong, unit-less indicator of cell activity.",
            "num": "Ratio of corrected trace standard deviation to first-difference standard deviation.",
            "pros": "Provides a clean, unit-less signal quality metric that is independent of absolute brightness.",
            "cons": "Silent or inactive cells with stable baselines have low SNR and may be rejected, even if they are structurally perfect."
        },
        # --- 23. activity_ratio ---
        {
            "name": "activity_ratio",
            "cat": "Biological & Signal Quality",
            "formula": "activity_ratio = std(Fcorr > median) / std(Fcorr <= median)",
            "bio": "Since calcium transients are positive, the variance of the signal above the median (transients) is much larger than below (baseline noise). Noise is symmetric, yielding a ratio ~ 1.0.",
            "num": "Standard deviation of trace values above the median divided by standard deviation of values below/equal to the median.",
            "pros": "Highly sensitive to positive excursions. Noise has a symmetric ratio ~ 1.0.",
            "cons": "Slow baseline drifts or asymmetric artifacts (like laser power decay) can distort this ratio."
        },
        # --- 24. peak_density ---
        {
            "name": "peak_density",
            "cat": "Biological & Signal Quality",
            "formula": "peak_density = count(peaks) / T",
            "bio": "Directly quantifies the firing rate of the cell. Real cells fire occasionally, whereas very noisy ROIs or blood vessels have false peaks (high density) or none (zero density).",
            "num": "Number of peaks exceeding mean + 2*std with minimum distance of 10 frames, divided by total frames T.",
            "pros": "Captures active firing frequency.",
            "cons": "Highly silent cells have zero peak density, making them look similar to flat noise. Noisy traces can inflate the density with false peaks."
        },
        # --- 25. avg_asym / max_asym ---
        {
            "name": "avg_asym, max_asym",
            "cat": "Calcium Kinetics & Peak Dynamics",
            "formula": "Asymmetry = decay_width / rise_width (at half-height)",
            "bio": "GCaMP calcium indicators have rapid binding kinetics (fast rise) and slow dissociation kinetics (slow decay). Biological peaks are highly asymmetric (Asymmetry >= 2.0). Noise fluctuations are symmetric (Asymmetry ~ 1.0).",
            "num": "Detects peaks and calculates decay/rise width at half-height using peak_widths. Computes average and maximum asymmetry.",
            "pros": "Excellent biological classifier. Strongly rejects symmetric optical, electrical, or mechanical noise.",
            "cons": "Requires active firing. Silent cells default to 1.0, making active but quiet cells look like noise."
        },
        # --- 26. max_width ---
        {
            "name": "max_width",
            "cat": "Calcium Kinetics & Peak Dynamics",
            "formula": "max_width = max(decay_width + rise_width) (at half-height)",
            "bio": "Real GCaMP transients last hundreds of milliseconds to seconds. High-frequency noise is very narrow (1-2 frames). This feature separates biological transients from electronic noise.",
            "num": "Maximum full-width at half-maximum (FWHM) in frames among detected peaks.",
            "pros": "Filters out high-frequency electrical or photon noise.",
            "cons": "Slow mechanical motion artifacts can create wide peaks that mimic biological transients."
        },
        # --- 27. skew_diff_fcorr ---
        {
            "name": "skew_diff_fcorr",
            "cat": "Neuropil-Corrected Trace Statistics",
            "formula": "skew_diff_fcorr = skew(diff(Fcorr))",
            "bio": "The first derivative of a calcium transient has a large positive spike (rise phase) and a small, prolonged negative baseline (decay phase). This yields positive skewness in the derivative.",
            "num": "Skewness of the first difference of the corrected trace.",
            "pros": "Captures the sharp rise signature of biological transients.",
            "cons": "High-frequency upward noise spikes can also inflate the skewness of the first difference."
        },
        # --- 28. smoothness_ratio_w5 / smoothness_ratio_w11 ---
        {
            "name": "smoothness_ratio_w5, smoothness_ratio_w11",
            "cat": "Advanced Temporal & Smoothing",
            "formula": "smoothness_ratio = std(F_smooth) / std(Fcorr)\n[where F_smooth is smoothed with window size W=5 or W=11]",
            "bio": "Real calcium transients are slow and low-frequency, so smoothing them does not significantly reduce their variance. High-frequency noise is smoothed out, drastically reducing its variance. A ratio near 1.0 indicates a slow, smooth signal (real cell), while a low ratio indicates high-frequency noise.",
            "num": "Ratio of smoothed standard deviation to raw standard deviation using uniform_filter1d.",
            "pros": "Highly effective at identifying and rejecting high-frequency noise traces.",
            "cons": "Highly active cells with very fast transients might see some reduction in variance, reducing the ratio."
        },
        # --- 29. fraction_active_k3 / fraction_active_k5 ---
        {
            "name": "fraction_active_k3, fraction_active_k5",
            "cat": "Advanced Temporal & Smoothing",
            "formula": "fraction_active = count(Fcorr > median + k * std_below_median) / T\n[where k=3 or k=5]",
            "bio": "Quantifies the proportion of time a cell spends in an active state. Using the below-median standard deviation provides a robust, transient-free estimate of baseline noise.",
            "num": "Fraction of frames exceeding the median by more than k baseline standard deviations.",
            "pros": "Robustly measures active firing time relative to baseline noise.",
            "cons": "Very quiet cells will have a fraction near 0, while extremely active cells might shift the median and lead to underestimation of active frames."
        },
        # --- 30. q99_smooth_ratio ---
        {
            "name": "q99_smooth_ratio",
            "cat": "Advanced Temporal & Smoothing",
            "formula": "q99_smooth_ratio = (P99(Fcorr) - median(Fcorr)) / (P99(F_smooth_w5) - median(F_smooth_w5))",
            "bio": "High-frequency spikes (noise) are heavily suppressed by smoothing, yielding a high ratio (>>1.0). Slow biological transients are relatively unaffected by a small smoothing window, yielding a ratio close to 1.0.",
            "num": "The ratio of the 99th percentile (above median) of the raw trace to that of the 5-frame smoothed trace.",
            "pros": "Highlights high-frequency noise spikes that attempt to mimic calcium transients.",
            "cons": "Very fast, transient single-action-potential signals might be slightly smoothed, raising the ratio and leading to false rejection."
        },
        # --- 31. skew_fcorr_smooth_w5 / skew_fcorr_smooth_w11 ---
        {
            "name": "skew_fcorr_smooth_w5, skew_fcorr_smooth_w11",
            "cat": "Advanced Temporal & Smoothing",
            "formula": "skew_fcorr_smooth = skew(F_smooth) [window size W=5 or W=11]",
            "bio": "Smoothing removes high-frequency noise spikes that could artificially inflate or deflate the skewness, leaving a cleaner estimate of the biological signal's asymmetry.",
            "num": "Skewness of the 5-frame or 11-frame smoothed corrected trace.",
            "pros": "Removes high-frequency noise spikes for a cleaner estimate of biological signal asymmetry.",
            "cons": "Can slightly reduce the skewness of very brief, genuine transients."
        },
        # --- 32. peak_to_q99_ratio / peak_to_q95_ratio ---
        {
            "name": "peak_to_q99_ratio, peak_to_q95_ratio",
            "cat": "Advanced Temporal & Smoothing",
            "formula": "peak_to_quantile = (max(Fcorr) - median(Fcorr)) / (Percentile_p(Fcorr) - median(Fcorr))\n[where p=99 or p=95]",
            "bio": "Distinguishes single brief artifacts/hot pixels (where max >> P99, ratio >> 1.0) from sustained calcium transients (where P99 is close to max, ratio is close to 1.0).",
            "num": "The maximum trace amplitude above median divided by the 99th or 95th percentile above median.",
            "pros": "Superb at identifying single hot pixels or brief laser glitch spikes.",
            "cons": "Very brief, valid single-action-potential transients might have high ratios and be misclassified."
        }
    ]

    for feat in features:
        # Build KeepTogether block for each feature to avoid breaking across pages
        feat_story = []
        feat_story.append(Paragraph(f"Feature Group: {feat['name']}", h2_style))
        
        # Details Table
        details = [
            [Paragraph("<b>Category:</b>", body_style), Paragraph(feat['cat'], body_style)],
            [Paragraph("<b>Formula:</b>", body_style), Paragraph(feat['formula'].replace('\n', '<br/>'), formula_style)],
            [Paragraph("<b>Biological Context:</b>", body_style), Paragraph(feat['bio'], body_style)],
            [Paragraph("<b>Implementation:</b>", body_style), Paragraph(feat['num'], body_style)],
            [Paragraph("<b>Why it Helps:</b>", body_style), Paragraph(feat['pros'], body_style)],
            [Paragraph("<b>Downside & Risks:</b>", body_style), Paragraph(feat['cons'], body_style)],
        ]
        
        t = Table(details, colWidths=[110, 394])
        t.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('LINEBELOW', (0,0), (-1,-2), 0.5, colors.HexColor("#EDF2F7")),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
        ]))
        
        feat_story.append(t)
        feat_story.append(Spacer(1, 10))
        story.append(KeepTogether(feat_story))

    # Build document
    doc.build(story, canvasmaker=SimpleNumberedCanvas)
    print(f"Successfully generated PDF: {filename}")


if __name__ == "__main__":
    output_pdf = "suite2p_features_reference.pdf"
    build_pdf(output_pdf)
