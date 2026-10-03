const pptxgen = require("pptxgenjs");
const { applyTheme } = require("/home/tomer/.config/Claude/local-agent-mode-sessions/skills-plugin/773281d4-95ad-4fe6-87a3-01b90b7a5705/104ed2f4-28ec-4882-8145-cc5d698cd8e6/skills/pptx/scripts/apply_theme.js");

const OUT = process.argv[2] || "suite2p_roi_classification_summary.pptx";
const CROPS = __dirname + "/crops.png"; // generated from Stav17 crops (scripts/prepare_cnn_crops.py)

const THEME = {
  name: "Calcium Imaging",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B2631", lt1: "FFFFFF", dk2: "0E3B43", lt2: "EEF4F3",
    accent1: "12A37D", accent2: "0E6E7E", accent3: "E39B2F",
    accent4: "6B7B88", accent5: "C44536", accent6: "A8D5C4",
    hlink: "0E6E7E", folHlink: "6B7B88",
  },
};
const H = THEME.colors; // hex, for chart options that only take hex

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.title = "Suite2p ROI classification: image features and honest evaluation";
pres.author = "Claude";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

const W = 13.333, M = 0.6;
const FOOTER = "Suite2p ROI classification  ·  branch feat/image-features  ·  F1 on held-out sessions";

// ---------- layouts ----------
pres.defineSlideMaster({
  title: "Title dark",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 2.0, w: W - 2 * M, h: 1.9,
      fontFace: THEME.headFontFace, fontSize: 44, bold: true, color: C.background1, valign: "bottom", margin: 0 },
      text: "Title" } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 4.15, w: W - 2 * M, h: 1.4,
      fontFace: THEME.bodyFontFace, fontSize: 20, color: C.accent6, valign: "top", margin: 0 },
      text: "Subtitle" } },
  ],
});
pres.defineSlideMaster({
  title: "Content",
  background: { color: C.background1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.35, w: W - 2 * M, h: 0.85,
      fontFace: THEME.headFontFace, fontSize: 30, bold: true, color: C.text2, valign: "middle", margin: 0 },
      text: "Title" } },
    { text: { text: FOOTER, options: { x: M, y: 7.0, w: 10, h: 0.3, fontSize: 10, color: C.accent4, margin: 0 } } },
  ],
  slideNumber: { x: W - M - 0.5, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: C.accent4, align: "right" },
});
pres.defineSlideMaster({
  title: "Closing dark",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.45, w: W - 2 * M, h: 0.9,
      fontFace: THEME.headFontFace, fontSize: 32, bold: true, color: C.background1, valign: "middle", margin: 0 },
      text: "Title" } },
  ],
  slideNumber: { x: W - M - 0.5, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: C.accent6, align: "right" },
});

// ---------- helpers ----------
const chartBase = () => ({
  catAxisLabelColor: H.dk1, valAxisLabelColor: H.accent4,
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt",
  titleFontFace: "+mn-lt", legendFontFace: "+mn-lt",
  catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, dataLabelFontSize: 11,
  valGridLine: { color: "DDE5E4", size: 0.75 }, catGridLine: { style: "none" },
  dataLabelColor: H.dk1,
});

function card(slide, name, x, y, w, h, fill) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    objectName: name, x, y, w, h, rectRadius: 0.12, fill: { color: fill || C.background2 }, line: { type: "none" },
  });
}
function numberDot(slide, name, n, x, y, d, fill) {
  slide.addShape(pres.shapes.OVAL, { objectName: name + " dot", x, y, w: d, h: d, fill: { color: fill || C.accent1 }, line: { type: "none" } });
  slide.addText(String(n), { objectName: name + " num", x, y, w: d, h: d, align: "center", valign: "middle",
    fontSize: Math.round(d * 30), bold: true, color: C.background1, margin: 0, isTextBox: true });
}
function para(text, opts) { return { text, options: Object.assign({ breakLine: true }, opts || {}) }; }

// =====================================================================
pres.addSection({ title: "Overview" });

// 1. Title
{
  const s = pres.addSlide({ masterName: "Title dark", sectionTitle: "Overview" });
  s.addText("Better Suite2p cell classification with image features", { placeholder: "title" });
  s.addText("Honest evaluation, new features, neural networks, and what the labels tell us  ·  October 2026",
    { placeholder: "body" });
  s.addNotes("Summary of the work on branch feat/image-features: fixing the evaluation, adding image features from ops.npy, " +
    "testing CNNs and learned embeddings, a model-family comparison, and a hold-out experiment on labelling style.");
}

// 2. Bottom line
{
  const s = pres.addSlide({ masterName: "Content", sectionTitle: "Overview" });
  s.addText("Image features lift F1 from 0.827 to 0.853", { placeholder: "title" });
  const stats = [
    ["0.827", "Current 27-feature model", "honest F1, threshold tuned on training sessions only", C.accent4],
    ["0.853", "+ 17 image features", "ready as the 'image' preset in apply_AI.py", C.accent1],
    ["0.861", "+ CNN embedding", "best overall, but needs a CNN at inference", C.accent2],
  ];
  const cw = 3.85, gap = 0.3, y0 = 1.55;
  stats.forEach(([num, head, sub, col], i) => {
    const x = M + i * (cw + gap);
    card(s, `stat card ${i + 1}`, x, y0, cw, 2.6);
    s.addText(num, { objectName: `stat ${i + 1} number`, x: x + 0.3, y: y0 + 0.2, w: cw - 0.6, h: 1.1,
      fontSize: 60, bold: true, color: col, fontFace: THEME.headFontFace, margin: 0, isTextBox: true });
    s.addText([para(head, { bold: true, fontSize: 18, color: C.text1 }), { text: sub, options: { fontSize: 14, color: C.accent4 } }],
      { objectName: `stat ${i + 1} text`, x: x + 0.3, y: y0 + 1.35, w: cw - 0.6, h: 1.1, valign: "top", margin: 0, isTextBox: true });
  });
  const take = [
    ["The gain comes from images, not traces or model choice", "Hand-made trace features beat every learned trace embedding; LightGBM, XGBoost and CatBoost tie within ±0.003"],
    ["The hardest sessions are a labelling-style problem", "Inbar3, inbar1 and Stav14 rank well (0.82–0.88 F1 at their own best threshold) but were labelled more or less strictly"],
  ];
  take.forEach(([h, b], i) => {
    const y = 4.55 + i * 1.1;
    numberDot(s, `takeaway ${i + 1}`, i + 1, M, y + 0.05, 0.5);
    s.addText([para(h, { bold: true, fontSize: 17, color: C.text2 }), { text: b, options: { fontSize: 14, color: C.text1 } }],
      { objectName: `takeaway ${i + 1} text`, x: M + 0.75, y, w: W - 2 * M - 0.75, h: 0.95, valign: "top", margin: 0, isTextBox: true });
  });
  s.addNotes("All numbers are F1 on sessions the model never saw (GroupKFold by session). 0.827 and 0.853 are means over 15 folds " +
    "(5 folds x 3 repeats); 0.861 is on the 5 repeat-0 folds, where the image-feature model scores 0.853 as well.");
}

// =====================================================================
pres.addSection({ title: "Data and evaluation" });

// 3. Data
{
  const s = pres.addSlide({ masterName: "Content", sectionTitle: "Data and evaluation" });
  s.addText("29 clean sessions, 84,583 ROIs, trusted labels", { placeholder: "title" });
  const kp = [["29", "sessions"], ["84,583", "ROIs"], ["17.6%", "labelled cells"]];
  kp.forEach(([n, l], i) => {
    const y = 1.55 + i * 1.6;
    card(s, `key figure ${i + 1}`, M, y, 3.3, 1.35);
    s.addText([para(n, { fontSize: 40, bold: true, color: C.accent2, fontFace: THEME.headFontFace }), { text: l, options: { fontSize: 15, color: C.accent4 } }],
      { objectName: `key figure ${i + 1} text`, x: M + 0.3, y: y + 0.12, w: 2.8, h: 1.15, valign: "middle", margin: 0, isTextBox: true });
  });
  const x2 = M + 3.7, w2 = W - M - x2;
  card(s, "excluded card", x2, 1.55, w2, 2.6);
  s.addText([
    para("Left out, and why", { bold: true, fontSize: 18, color: C.text2, paraSpaceAfter: 6 }),
    para("Yael sessions: known bad data", { bullet: true, fontSize: 15 }),
    para("stav22: held-out session in the existing training script", { bullet: true, fontSize: 15 }),
    para("Stav3/21: 70% labelled cells vs 5–28% elsewhere, looks uncurated", { bullet: true, fontSize: 15 }),
    { text: "Stav1: byte-identical copy of Stav5 (same traces, ROIs, labels) with an ops.npy from another recording", options: { bullet: true, fontSize: 15 } },
  ], { objectName: "excluded text", x: x2 + 0.3, y: 1.7, w: w2 - 0.6, h: 2.35, valign: "top", margin: 0, color: C.text1, isTextBox: true });
  card(s, "labels card", x2, 4.4, w2, 2.3);
  s.addText([
    para("Label bugs fixed", { bold: true, fontSize: 18, color: C.text2, paraSpaceAfter: 6 }),
    para("Priority now: iscell_final → iscell_backup_before_AI → iscell", { bullet: true, fontSize: 15 }),
    para("Old train_model.py never read iscell_final, so final-only sessions were skipped", { bullet: true, fontSize: 15 }),
    { text: "For Inbar6 and inbar4 it trained on an iscell.npy that apply_AI.py had overwritten (8–9% of ROIs differ)", options: { bullet: true, fontSize: 15 } },
  ], { objectName: "labels text", x: x2 + 0.3, y: 4.55, w: w2 - 0.6, h: 2.05, valign: "top", margin: 0, color: C.text1, isTextBox: true });
  s.addNotes("Stav1 was detected because its image features made no sense: a label-free alignment check between ops.npy and stat.npy " +
    "scores 0.05 for Stav1 and 0.27 or more for every other session. Its traces, ROIs and labels turned out to be byte-identical to Stav5.");
}

// 4. Evaluation
{
  const s = pres.addSlide({ masterName: "Content", sectionTitle: "Data and evaluation" });
  s.addText("Honest evaluation: the test sessions never touch tuning", { placeholder: "title" });
  const steps = [
    ["Split by session", "GroupKFold, 5 folds × 3 shuffled repeats: every session is tested by a model that never saw it"],
    ["Inner CV on training sessions", "A second session-grouped CV inside the training folds only"],
    ["Pick trees and threshold", "Early-stopping round and the F1-optimal threshold come from the inner CV"],
    ["Score the held-out sessions", "Precision, recall and F1 per fold; gains are paired fold by fold"],
  ];
  const bw = 2.75, gap = 0.37, y = 1.6;
  steps.forEach(([h, b], i) => {
    const x = M + i * (bw + gap);
    card(s, `step ${i + 1} card`, x, y, bw, 2.75);
    numberDot(s, `step ${i + 1}`, i + 1, x + 0.25, y + 0.25, 0.55, C.accent2);
    s.addText([para(h, { bold: true, fontSize: 16, color: C.text2, paraSpaceAfter: 4 }), { text: b, options: { fontSize: 14, color: C.text1 } }],
      { objectName: `step ${i + 1} text`, x: x + 0.25, y: y + 0.95, w: bw - 0.5, h: 1.7, valign: "top", margin: 0, isTextBox: true });
    if (i < steps.length - 1) {
      s.addShape(pres.shapes.RIGHT_ARROW, { objectName: `arrow ${i + 1}`, x: x + bw + 0.06, y: y + 1.2, w: 0.25, h: 0.35,
        fill: { color: C.accent4 }, line: { type: "none" } });
    }
  });
  card(s, "old vs new card", M, 4.75, W - 2 * M, 1.95, C.background2);
  s.addText([
    para("Before: the old script tuned early stopping and the threshold on the test fold itself", { fontSize: 16, bold: true, color: C.text2, paraSpaceAfter: 6 }),
    para("Same 27 features, same data: tuned on the test fold 0.832  →  honest 0.827", { bullet: true, fontSize: 15 }),
    { text: "The README's 0.856 came from a different session set and labels, so it is not comparable", options: { bullet: true, fontSize: 15 } },
  ], { objectName: "old vs new text", x: M + 0.3, y: 4.9, w: W - 2 * M - 0.6, h: 1.7, valign: "top", margin: 0, color: C.text1, isTextBox: true });
  s.addNotes("The honest baseline is what every later gain is measured against. ΔF1 values in later slides are paired per fold, " +
    "which removes most of the fold-to-fold variance between sessions.");
}

// =====================================================================
pres.addSection({ title: "Image features" });

// 5. What are image features
{
  const s = pres.addSlide({ masterName: "Content", sectionTitle: "Image features" });
  s.addText("Image features compare each mask with the images", { placeholder: "title" });
  const imgH = 5.55, imgW = imgH * 1011 / 1320;
  s.addImage({ path: CROPS, x: M, y: 1.3, w: imgW, h: imgH, objectName: "example crops",
    altText: "Example ROI crops from Stav17: two labelled cells and two non-cells, each shown in meanImg, max_proj and Vcorr with the ROI mask outlined" });
  const x2 = M + imgW + 0.5, w2 = W - M - x2;
  s.addText("Your morphology features only see the mask's shape (stat.npy). These 17 features look at what the images show inside and around it (ops.npy).",
    { objectName: "intro text", x: x2, y: 1.35, w: w2, h: 0.9, fontSize: 15, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  const rows = [
    ["Brighter than its surroundings?", "Mask vs a 2–7 px ring on meanImg, meanImgE, max_proj and Vcorr; contrast vs Suite2p's neuropil mask (8 features)"],
    ["Does the mask match the blob?", "Correlation between the mask weights (lam) and the image, inside the mask and over mask + ring (8 features)"],
    ["Can the images be trusted?", "Label-free ops/stat alignment check; failing sessions fall back to the regular model"],
  ];
  rows.forEach(([h, b], i) => {
    const y = 2.45 + i * 1.45;
    numberDot(s, `question ${i + 1}`, i + 1, x2, y + 0.05, 0.5);
    s.addText([para(h, { bold: true, fontSize: 17, color: C.text2 }), { text: b, options: { fontSize: 14, color: C.text1 } }],
      { objectName: `question ${i + 1} text`, x: x2 + 0.75, y, w: w2 - 0.75, h: 1.2, valign: "top", margin: 0, isTextBox: true });
  });
  s.addText("Illustrative examples from Stav17. Green: labelled cell; red: labelled not-cell.",
    { objectName: "crops caption", x: x2, y: 6.6, w: w2, h: 0.3, fontSize: 11, italic: true, color: C.accent4, margin: 0, isTextBox: true });
  s.addNotes("max_proj and Vcorr are cropped by Suite2p to yrange/xrange; they are placed back at that offset rather than resized. " +
    "Vcorr is the local correlation map, so it carries activity information even though it is an image.");
}

// 6. Results by feature group
{
  const s = pres.addSlide({ masterName: "Content", sectionTitle: "Image features" });
  s.addText("Image features add +0.026 F1, better in all 15 folds", { placeholder: "title" });
  const labels = ["Baseline (27)", "+ trace-norm", "Trace-norm replaces raw q/range", "+ ring contrast", "+ lam correlation", "+ all image features"];
  const vals = [0.827, 0.828, 0.826, 0.843, 0.850, 0.853];
  s.addChart(pres.charts.BAR, [{ name: "F1", labels, values: vals }], Object.assign(chartBase(), {
    objectName: "F1 by feature set chart", x: M, y: 1.35, w: 7.6, h: 5.45, barDir: "bar",
    chartColors: [H.accent4, H.accent4, H.accent4, H.accent1, H.accent1, H.accent2],
    valAxisMinVal: 0.80, valAxisMaxVal: 0.86, valAxisMajorUnit: 0.01, valAxisLabelFormatCode: "0.00",
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", showLegend: false,
    catAxisOrientation: "maxMin", barGapWidthPct: 45,
    showTitle: true, title: "F1, mean of 15 session folds", titleFontSize: 13, titleColor: H.accent4,
  }));
  const x2 = M + 8.0, w2 = W - M - x2;
  card(s, "commentary card", x2, 1.35, w2, 5.45);
  s.addText([
    para("What it means", { bold: true, fontSize: 18, color: C.text2, paraSpaceAfter: 8 }),
    para("lam correlation carries most of it: +0.023 alone", { bullet: true, fontSize: 15, paraSpaceAfter: 6 }),
    para("Ring contrast alone: +0.015", { bullet: true, fontSize: 15, paraSpaceAfter: 6 }),
    para("Intensity-normalized trace features (dF/F, q99 / noise): no gain; the existing ones are already noise-scaled", { bullet: true, fontSize: 15, paraSpaceAfter: 6 }),
    { text: "27 of 29 sessions improve", options: { bullet: true, fontSize: 15 } },
  ], { objectName: "commentary text", x: x2 + 0.3, y: 1.55, w: w2 - 0.6, h: 5.05, valign: "top", margin: 0, color: C.text1, isTextBox: true });
  s.addNotes("Paired ΔF1 vs baseline: trace-norm +0.001 (8/15 folds up), replace −0.001 (7/15), ring contrast +0.015 (14/15), " +
    "lam correlation +0.023 (15/15), all image features +0.026 ± 0.008 (15/15).");
}

// 7. Hardest sessions
{
  const s = pres.addSlide({ masterName: "Content", sectionTitle: "Image features" });
  s.addText("The biggest gains are on the hardest sessions", { placeholder: "title" });
  const sess = ["Inbar12", "Inbar3", "Stav14", "inbar1", "Inbar9", "Inbar11", "Inbar2", "Inbar10", "Inbar5", "Stav8"];
  const base = [0.685, 0.693, 0.746, 0.755, 0.763, 0.767, 0.780, 0.786, 0.787, 0.807];
  const img = [0.805, 0.742, 0.761, 0.805, 0.799, 0.825, 0.822, 0.813, 0.812, 0.852];
  s.addChart(pres.charts.BAR, [
    { name: "Baseline (27)", labels: sess, values: base },
    { name: "+ image features", labels: sess, values: img },
  ], Object.assign(chartBase(), {
    objectName: "per-session F1 chart", x: M, y: 1.35, w: W - 2 * M, h: 4.6, barDir: "col", barGrouping: "clustered",
    chartColors: [H.accent4, H.accent1], valAxisMinVal: 0.6, valAxisMaxVal: 0.9, valAxisMajorUnit: 0.05, valAxisLabelFormatCode: "0.00",
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", dataLabelFontSize: 10,
    showLegend: true, legendPos: "t", legendFontSize: 12, legendColor: H.dk1, barGapWidthPct: 60,
  }));
  s.addText([
    { text: "The 10 lowest-scoring sessions under the current model. ", options: { bold: true, color: C.text2 } },
    { text: "Inbar12 goes from worst (0.685) to 0.805. Inbar3 and Stav14 stay hardest; the labelling slide explains why.", options: { color: C.text1 } },
  ], { objectName: "per-session caption", x: M, y: 6.15, w: W - 2 * M, h: 0.7, fontSize: 15, margin: 0, valign: "top", isTextBox: true });
  s.addNotes("Per-session F1 is the mean over the 3 CV repeats. Only 2 of 29 sessions drop, by less than 0.005 (Stav16, Stav20).");
}

// 8. SHAP
{
  const s = pres.addSlide({ masterName: "Content", sectionTitle: "Image features" });
  s.addText("What the new model relies on most", { placeholder: "title" });
  const feats = ["max_proj_lam_corr", "mrs", "Vcorr_in_mean", "skew_fcorr", "std_fcorr", "meanImg_neuropil_contrast",
    "std_f", "corr_f_fneu", "radius", "q999_over_noise", "compact", "range_ratio_f_fneu"];
  const shap = [1.081, 0.869, 0.517, 0.422, 0.330, 0.285, 0.270, 0.248, 0.236, 0.232, 0.221, 0.207];
  const isNew = [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0];
  s.addChart(pres.charts.BAR, [
    { name: "New feature", labels: feats, values: shap.map((v, i) => (isNew[i] ? v : 0)) },
    { name: "Existing feature", labels: feats, values: shap.map((v, i) => (isNew[i] ? 0 : v)) },
  ], Object.assign(chartBase(), {
    objectName: "SHAP chart", x: M, y: 1.35, w: 7.9, h: 5.45, barDir: "bar", barGrouping: "stacked",
    chartColors: [H.accent1, H.accent4], catAxisOrientation: "maxMin", valAxisLabelFormatCode: "0.0",
    showValue: false, showLegend: true, legendPos: "b", legendFontSize: 12, legendColor: H.dk1, barGapWidthPct: 40,
    showTitle: true, title: "Mean |SHAP|, held-out folds", titleFontSize: 13, titleColor: H.accent4, catAxisLabelFontSize: 11,
  }));
  const x2 = M + 8.3, w2 = W - M - x2;
  card(s, "SHAP commentary card", x2, 1.35, w2, 5.45);
  s.addText([
    para("Reading it", { bold: true, fontSize: 18, color: C.text2, paraSpaceAfter: 8 }),
    para("#1 max_proj_lam_corr: does the mask sit on something that lit up?", { bullet: true, fontSize: 15, paraSpaceAfter: 6 }),
    para("3 of the top 6 are new image features", { bullet: true, fontSize: 15, paraSpaceAfter: 6 }),
    { text: "Your morphology (mrs, radius, compact) and trace features (skew, std) stay important; image features add to them rather than replace them", options: { bullet: true, fontSize: 15 } },
  ], { objectName: "SHAP commentary text", x: x2 + 0.3, y: 1.55, w: w2 - 0.6, h: 5.05, valign: "top", margin: 0, color: C.text1, isTextBox: true });
  s.addNotes("SHAP from the model with normalized trace features plus image features, averaged over held-out folds.");
}

// =====================================================================
pres.addSection({ title: "Neural networks and alternatives" });

// 9. Neural networks
{
  const s = pres.addSlide({ masterName: "Content", sectionTitle: "Neural networks and alternatives" });
  s.addText("A CNN matches the image features, adds 0.007", { placeholder: "title" });
  const labels = ["LightGBM, 27 features", "LightGBM + image features", "CNN alone (images only)", "Fusion net (images + features)", "LightGBM + image + CNN score", "3-booster average + CNN embedding"];
  const vals = [0.828, 0.853, 0.850, 0.858, 0.859, 0.861];
  s.addChart(pres.charts.BAR, [{ name: "F1", labels, values: vals }], Object.assign(chartBase(), {
    objectName: "neural models chart", x: M, y: 1.35, w: 7.6, h: 5.45, barDir: "bar",
    chartColors: [H.accent4, H.accent1, H.accent2, H.accent2, H.accent2, H.accent2],
    valAxisMinVal: 0.80, valAxisMaxVal: 0.87, valAxisMajorUnit: 0.01, valAxisLabelFormatCode: "0.00",
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", showLegend: false, catAxisOrientation: "maxMin",
    barGapWidthPct: 45, showTitle: true, title: "F1, mean of 5 session folds", titleFontSize: 13, titleColor: H.accent4,
  }));
  const x2 = M + 8.0, w2 = W - M - x2;
  card(s, "neural commentary card", x2, 1.35, w2, 5.45);
  s.addText([
    para("What it means", { bold: true, fontSize: 18, color: C.text2, paraSpaceAfter: 8 }),
    para("The CNN learns about what the 17 image features already capture", { bullet: true, fontSize: 15, paraSpaceAfter: 6 }),
    para("Training images and features jointly (fusion) is the best single model, 0.858", { bullet: true, fontSize: 15, paraSpaceAfter: 6 }),
    para("Embedding or score on top: +0.007–0.008, a tie between the two", { bullet: true, fontSize: 15, paraSpaceAfter: 6 }),
    { text: "Cost: crops and a PyTorch model at inference, and a less explainable decision", options: { bullet: true, fontSize: 15 } },
  ], { objectName: "neural commentary text", x: x2 + 0.3, y: 1.55, w: w2 - 0.6, h: 5.05, valign: "top", margin: 0, color: C.text1, isTextBox: true });
  s.addNotes("CNN input: 32x32 crops of meanImg, max_proj, Vcorr plus the lam mask. CNN scores are nested out-of-fold; embeddings come " +
    "from one network per outer fold trained on the training sessions only. CPU only, no GPU.");
}

// 10. What didn't help
{
  const s = pres.addSlide({ masterName: "Content", sectionTitle: "Neural networks and alternatives" });
  s.addText("Learned trace embeddings and other models didn't help", { placeholder: "title" });
  const labels = ["Best other model family (3-booster average)", "XGBoost / CatBoost instead of LightGBM", "Normalized trace features added",
    "ROICaT ROInet embedding (6-session pilot)", "MiniRocket trace embedding added", "MiniRocket instead of hand-made trace features",
    "1D-CNN trace embedding added", "1D-CNN trace embedding instead of hand-made"];
  const vals = [0.003, -0.001, 0.001, 0.001, 0.000, -0.007, -0.010, -0.017];
  s.addChart(pres.charts.BAR, [{ name: "ΔF1", labels, values: vals }], Object.assign(chartBase(), {
    objectName: "did not help chart", x: M, y: 1.35, w: 8.3, h: 5.45, barDir: "bar",
    chartColors: [H.accent2], invertedColors: [H.accent5],
    valAxisMinVal: -0.02, valAxisMaxVal: 0.01, valAxisMajorUnit: 0.005, valAxisLabelFormatCode: "+0.000;-0.000;0",
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "+0.000;-0.000;0.000", showLegend: false,
    catAxisOrientation: "maxMin", catAxisLabelFontSize: 11, barGapWidthPct: 40, catAxisLabelPos: "low",
    showTitle: true, title: "ΔF1, paired, vs the same model without the change", titleFontSize: 13, titleColor: H.accent4,
  }));
  const x2 = M + 8.7, w2 = W - M - x2;
  card(s, "did not help card", x2, 1.35, w2, 5.45);
  s.addText([
    para("Take-aways", { bold: true, fontSize: 18, color: C.text2, paraSpaceAfter: 8 }),
    para("Your hand-made trace features beat both learned trace encoders", { bullet: true, fontSize: 15, paraSpaceAfter: 6 }),
    para("Model family is worth about ±0.003: keep LightGBM", { bullet: true, fontSize: 15, paraSpaceAfter: 6 }),
    { text: "A lightly tuned LightGBM (real row bagging) is a free +0.002", options: { bullet: true, fontSize: 15 } },
  ], { objectName: "did not help text", x: x2 + 0.3, y: 1.55, w: w2 - 0.6, h: 5.05, valign: "top", margin: 0, color: C.text1, isTextBox: true });
  s.addNotes("ROICaT result is from a 6-session leave-one-session-out pilot; it was stopped there. Logistic regression and an MLP " +
    "on the 44 features scored 0.829 and 0.846, below the tree boosters.");
}

// =====================================================================
pres.addSection({ title: "Labels" });

// 11. Labelling style
{
  const s = pres.addSlide({ masterName: "Content", sectionTitle: "Labels" });
  s.addText("Hardest sessions: mostly labelling strictness", { placeholder: "title" });
  const sess = ["Inbar3", "inbar1", "Stav14", "Inbar12", "Inbar9"];
  s.addChart(pres.charts.BAR, [
    { name: "F1 at the trained threshold", labels: sess, values: [0.748, 0.771, 0.774, 0.806, 0.799] },
    { name: "F1 at the session's own best threshold", labels: sess, values: [0.868, 0.880, 0.819, 0.819, 0.814] },
  ], Object.assign(chartBase(), {
    objectName: "threshold gap chart", x: M, y: 1.35, w: 7.4, h: 4.3, barDir: "col", barGrouping: "clustered",
    chartColors: [H.accent4, H.accent3], valAxisMinVal: 0.6, valAxisMaxVal: 0.95, valAxisMajorUnit: 0.05, valAxisLabelFormatCode: "0.00",
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", showLegend: true, legendPos: "t",
    legendFontSize: 12, legendColor: H.dk1, barGapWidthPct: 60,
  }));
  s.addText("Same model, only the cut-off changes. Inbar: trained on Stav + other Inbar; Stav14: trained on other Stav.",
    { objectName: "gap chart caption", x: M, y: 5.75, w: 7.4, h: 0.55, fontSize: 12, italic: true, color: C.accent4, margin: 0, isTextBox: true });
  const x2 = M + 7.8, w2 = W - M - x2;
  s.addTable([
    [{ text: "Trained on", options: { bold: true, color: C.background1, fill: { color: C.text2 } } },
     { text: "4 hard Inbar", options: { bold: true, color: C.background1, fill: { color: C.text2 } } },
     { text: "Stav14", options: { bold: true, color: C.background1, fill: { color: C.text2 } } }],
    ["Stav only", "0.741", { text: "0.774", options: { bold: true } }],
    ["Inbar only", { text: "0.782", options: { bold: true } }, "0.744"],
    ["Both labs", "0.767", "—"],
  ], { objectName: "lab hold-out table", x: x2, y: 1.45, w: w2, colW: [w2 * 0.4, w2 * 0.32, w2 * 0.28], fontSize: 14,
    color: C.text1, border: { type: "solid", pt: 0.5, color: "DDE5E4" }, fill: { color: C.background1 }, rowH: 0.42 });
  s.addText([
    para("Each lab's labels are best predicted by that lab's data", { bold: true, fontSize: 15, color: C.text2, paraSpaceAfter: 6 }),
    para("Inbar3: labelled 26.5% cells, models predict 15–17%", { bullet: true, fontSize: 14 }),
    para("inbar1: labelled 5.2%, models predict about 8%", { bullet: true, fontSize: 14 }),
    { text: "Stav14: labelled 23.5%, models predict 32–35%", options: { bullet: true, fontSize: 14 } },
  ], { objectName: "labelling text", x: x2, y: 3.45, w: w2, h: 2.7, valign: "top", margin: 0, color: C.text1, isTextBox: true });
  s.addNotes("Hold-out experiment: train on Stav only, Inbar only, or both, and test on Inbar3, Inbar12, inbar1 and Inbar9, with the " +
    "threshold picked by inner CV on the training sessions. Mirror check on Stav14. Inbar12 and Inbar9 are well calibrated; " +
    "Inbar3, inbar1 and Stav14 rank well but their labellers drew the line more or less strictly than the rest.");
}

// =====================================================================
pres.addSection({ title: "Next steps" });

// 12. Deployment
{
  const s = pres.addSlide({ masterName: "Content", sectionTitle: "Next steps" });
  s.addText("Ready to use: the image preset in apply_AI.py", { placeholder: "title" });
  card(s, "command card", M, 1.45, W - 2 * M, 1.25, C.text2);
  s.addText([
    para("python apply_AI.py /path/to/suite2p/plane0 image", { fontFace: "Courier New", fontSize: 20, color: C.background1, bold: true }),
    { text: "Without 'image', the current model runs exactly as before; nothing in models/ or any iscell file was changed", options: { fontSize: 14, color: C.accent6 } },
  ], { objectName: "command text", x: M + 0.35, y: 1.55, w: W - 2 * M - 0.7, h: 1.05, valign: "middle", margin: 0, isTextBox: true });
  const items = [
    ["Model", "models/image/suite2p_image_lgb.pkl: 27 + 17 features, 29 sessions, 466 trees"],
    ["Threshold", "0.65, chosen by inner CV and stored in the model's JSON instead of hard-coded"],
    ["Same code as training", "Inference features computed by the training code; all 44 verified identical"],
    ["Safe fallback", "If ops.npy is missing or belongs to another recording, the regular model runs"],
  ];
  const cw = (W - 2 * M - 0.3) / 2, ch = 1.75;
  items.forEach(([h, b], i) => {
    const x = M + (i % 2) * (cw + 0.3), y = 3.0 + Math.floor(i / 2) * (ch + 0.25);
    card(s, `deploy card ${i + 1}`, x, y, cw, ch);
    numberDot(s, `deploy ${i + 1}`, i + 1, x + 0.3, y + 0.3, 0.5);
    s.addText([para(h, { bold: true, fontSize: 17, color: C.text2 }), { text: b, options: { fontSize: 15, color: C.text1 } }],
      { objectName: `deploy card ${i + 1} text`, x: x + 1.05, y: y + 0.25, w: cw - 1.35, h: ch - 0.45, valign: "top", margin: 0, isTextBox: true });
  });
  s.addNotes("Tested on copies of Inbar2 and Stav11 (end to end) and Stav1 (fallback), with the original iscell files checksummed " +
    "before and after. Found along the way: the regular model computes some features differently at training and inference time.");
}

// 13. Recommendations
{
  const s = pres.addSlide({ masterName: "Closing dark", sectionTitle: "Next steps" });
  s.addText("Recommendations", { placeholder: "title" });
  const recs = [
    ["Ship the image preset", "+0.026 F1, cheap, explainable with SHAP"],
    ["Review labels in Inbar3, inbar1, Stav14", "Many 'errors' may be labelling calls; fixing them helps scoring and training"],
    ["Use a per-lab or per-session threshold", "Worth about +0.015–0.04 F1 on Inbar sessions; the dashboard slider already allows it"],
    ["Add the CNN only if 0.007 matters", "Best model is 0.861, but needs a CNN at inference"],
    ["Fix the regular model's feature mismatch", "Its cached quantile features are scaled differently at training and inference"],
  ];
  recs.forEach(([h, b], i) => {
    const y = 1.55 + i * 1.05;
    numberDot(s, `recommendation ${i + 1}`, i + 1, M, y + 0.05, 0.55, C.accent1);
    s.addText([para(h, { bold: true, fontSize: 19, color: C.background1 }), { text: b, options: { fontSize: 15, color: C.accent6 } }],
      { objectName: `recommendation ${i + 1} text`, x: M + 0.85, y, w: W - 2 * M - 0.85, h: 0.95, valign: "top", margin: 0, isTextBox: true });
  });
  s.addNotes("Code, results and this deck are on the branch feat/image-features in AddadyTom/suite2p-classification " +
    "(results/REPORT.md and results/*.md for the full tables).");
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
