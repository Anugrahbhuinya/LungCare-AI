/**
 * PDF Analysis Report Generator for LungCare AI.
 *
 * Generates structured, professional, multi-page client-side PDF documents
 * directly from browser-local IndexedDB HistoryAnalysisRecord metadata.
 * Does not transmit or store reports remotely.
 */

import { jsPDF } from "jspdf";
import type { HistoryAnalysisRecord } from "@/services/history/types";
import { MODEL_CLASSES, getDisplayClassName } from "@/types/prediction";

export interface GenerateReportOptions {
  generatedAt?: Date;
}

/**
 * Sanitizes an analysis record ID to create a safe, filesystem-friendly PDF filename.
 * Format: LungCare_Analysis_<record-id>.pdf
 */
export function getReportFileName(recordId: string): string {
  const safeId = (recordId || "unknown").replace(/[^a-zA-Z0-9_-]/g, "_");
  return `LungCare_Analysis_${safeId}.pdf`;
}

/**
 * Builds the complete jsPDF document instance from a HistoryAnalysisRecord.
 * Exported for testing, programmatic inspection, and direct downloading.
 */
export function buildAnalysisReportPDF(
  record: HistoryAnalysisRecord,
  options?: GenerateReportOptions,
): jsPDF {
  if (!record || !record.id) {
    throw new Error(
      "Invalid analysis record: cannot generate PDF for missing record data.",
    );
  }

  const doc = new jsPDF({
    orientation: "portrait",
    unit: "mm",
    format: "a4",
  });

  const pageWidth = 210;
  const pageHeight = 297;
  const marginX = 15;
  const contentWidth = pageWidth - marginX * 2; // 180mm
  const maxY = 270; // safe threshold before page overflow

  let currentY = 18;

  function checkOverflow(neededHeight: number) {
    if (currentY + neededHeight > maxY) {
      doc.addPage();
      currentY = 20;
    }
  }

  // ==========================================
  // 1. HEADER & BRANDING
  // ==========================================
  doc.setFont("helvetica", "bold");
  doc.setFontSize(9);
  doc.setTextColor(2, 132, 199); // Clinical primary blue (#0284c7)
  doc.text("LUNGCARE AI • PULMONARY RESEARCH PLATFORM", marginX, currentY);

  currentY += 6;
  doc.setFont("helvetica", "bold");
  doc.setFontSize(16);
  doc.setTextColor(15, 23, 42); // slate-900 (#0f172a)
  doc.text("CHEST RADIOGRAPH ANALYSIS REPORT", marginX, currentY);

  currentY += 5;
  doc.setFont("helvetica", "normal");
  doc.setFontSize(8.5);
  doc.setTextColor(100, 116, 139); // slate-500
  doc.text(
    "Automated Deep Learning Classification & Explainability Metadata Summary",
    marginX,
    currentY,
  );

  const genDate = options?.generatedAt || new Date();
  const genDateStr = genDate.toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "medium",
  });
  doc.text(`Generated: ${genDateStr}`, pageWidth - marginX, currentY, {
    align: "right",
  });

  currentY += 3;
  doc.setDrawColor(2, 132, 199); // Blue accent rule
  doc.setLineWidth(0.6);
  doc.line(marginX, currentY, marginX + contentWidth, currentY);

  currentY += 6;

  // ==========================================
  // 2. STUDY & ANALYSIS IDENTIFICATION BOX
  // ==========================================
  checkOverflow(28);
  const infoBoxY = currentY;
  const infoBoxHeight = 24;

  doc.setFillColor(248, 250, 252); // slate-50
  doc.setDrawColor(226, 232, 240); // slate-200
  doc.setLineWidth(0.3);
  doc.roundedRect(marginX, infoBoxY, contentWidth, infoBoxHeight, 2, 2, "FD");

  const analysisDate = record.timestamp
    ? new Date(record.timestamp).toLocaleString(undefined, {
        dateStyle: "medium",
        timeStyle: "medium",
      })
    : "Unknown";

  const formattedFileSize = record.fileSize
    ? record.fileSize < 1024 * 1024
      ? `${(record.fileSize / 1024).toFixed(1)} KB`
      : `${(record.fileSize / (1024 * 1024)).toFixed(2)} MB`
    : "Not Recorded";

  // Left Column
  doc.setFontSize(8);
  doc.setFont("helvetica", "bold");
  doc.setTextColor(71, 85, 105); // slate-600
  doc.text("Record Identifier:", marginX + 3, infoBoxY + 5.5);
  doc.setFont("helvetica", "normal");
  doc.setTextColor(15, 23, 42);
  const truncatedId = doc.splitTextToSize(record.id, 80)[0];
  doc.text(truncatedId, marginX + 30, infoBoxY + 5.5);

  doc.setFont("helvetica", "bold");
  doc.setTextColor(71, 85, 105);
  doc.text("Analysis Date:", marginX + 3, infoBoxY + 11.5);
  doc.setFont("helvetica", "normal");
  doc.setTextColor(15, 23, 42);
  doc.text(analysisDate, marginX + 30, infoBoxY + 11.5);

  doc.setFont("helvetica", "bold");
  doc.setTextColor(71, 85, 105);
  doc.text("Request Tracking:", marginX + 3, infoBoxY + 17.5);
  doc.setFont("helvetica", "normal");
  doc.setTextColor(15, 23, 42);
  const safeReqId = record.request_id || "N/A";
  doc.text(safeReqId, marginX + 30, infoBoxY + 17.5);

  // Right Column
  const rightColX = marginX + 95;
  doc.setFont("helvetica", "bold");
  doc.setTextColor(71, 85, 105);
  doc.text("Radiograph File:", rightColX, infoBoxY + 5.5);
  doc.setFont("helvetica", "normal");
  doc.setTextColor(15, 23, 42);
  const truncatedFile = doc.splitTextToSize(
    record.fileName || "Unknown",
    60,
  )[0];
  doc.text(truncatedFile, rightColX + 25, infoBoxY + 5.5);

  doc.setFont("helvetica", "bold");
  doc.setTextColor(71, 85, 105);
  doc.text("File Size / MIME:", rightColX, infoBoxY + 11.5);
  doc.setFont("helvetica", "normal");
  doc.setTextColor(15, 23, 42);
  doc.text(
    `${formattedFileSize} ${record.fileType ? `(${record.fileType})` : ""}`,
    rightColX + 25,
    infoBoxY + 11.5,
  );

  doc.setFont("helvetica", "bold");
  doc.setTextColor(71, 85, 105);
  doc.text("Schema Version:", rightColX, infoBoxY + 17.5);
  doc.setFont("helvetica", "normal");
  doc.setTextColor(15, 23, 42);
  doc.text(`v${record.schema_version || 1}`, rightColX + 25, infoBoxY + 17.5);

  currentY = infoBoxY + infoBoxHeight + 7;

  // ==========================================
  // 3. PRIMARY MODEL PREDICTION HIGHLIGHT
  // ==========================================
  checkOverflow(26);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(10);
  doc.setTextColor(15, 23, 42);
  doc.text("1. PRIMARY MODEL PREDICTION", marginX, currentY);

  currentY += 4;
  const predBoxY = currentY;
  const predBoxHeight = 18;

  doc.setFillColor(240, 249, 255); // sky-50
  doc.setDrawColor(186, 230, 253); // sky-200
  doc.setLineWidth(0.4);
  doc.roundedRect(marginX, predBoxY, contentWidth, predBoxHeight, 2, 2, "FD");

  const displayName =
    record.predicted_label ||
    getDisplayClassName(record.predicted_class || "Unknown");
  const confPercent = (
    typeof record.confidence === "number" ? record.confidence * 100 : 0
  ).toFixed(2);

  doc.setFont("helvetica", "bold");
  doc.setFontSize(13);
  doc.setTextColor(2, 132, 199); // Blue
  doc.text(displayName, marginX + 4, predBoxY + 7);

  doc.setFont("helvetica", "normal");
  doc.setFontSize(8);
  doc.setTextColor(100, 116, 139);
  doc.text(
    `Category Identifier: ${record.predicted_class || "Unknown"} • DenseNet121 Transfer Learning Output`,
    marginX + 4,
    predBoxY + 13,
  );

  // Confidence Score Banner on Right
  doc.setFont("helvetica", "bold");
  doc.setFontSize(16);
  doc.setTextColor(15, 23, 42);
  doc.text(`${confPercent}%`, marginX + contentWidth - 4, predBoxY + 9, {
    align: "right",
  });

  doc.setFont("helvetica", "normal");
  doc.setFontSize(7.5);
  doc.setTextColor(100, 116, 139);
  doc.text("Confidence Score", marginX + contentWidth - 4, predBoxY + 14, {
    align: "right",
  });

  currentY = predBoxY + predBoxHeight + 7;

  // ==========================================
  // 4. FOUR-CLASS SOFTMAX PROBABILITY DISTRIBUTION TABLE
  // ==========================================
  checkOverflow(44);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(10);
  doc.setTextColor(15, 23, 42);
  doc.text("2. FOUR-CLASS SOFTMAX PROBABILITY DISTRIBUTION", marginX, currentY);

  currentY += 4;
  const tableHeaderY = currentY;
  const rowHeight = 7;

  // Table Header
  doc.setFillColor(241, 245, 249); // slate-100
  doc.setDrawColor(203, 213, 225); // slate-300
  doc.setLineWidth(0.2);
  doc.rect(marginX, tableHeaderY, contentWidth, rowHeight, "FD");

  doc.setFont("helvetica", "bold");
  doc.setFontSize(8);
  doc.setTextColor(71, 85, 105);
  doc.text("Classification Category", marginX + 4, tableHeaderY + 4.8);
  doc.text("Model Status", marginX + 65, tableHeaderY + 4.8);
  doc.text("Probability Score", marginX + 115, tableHeaderY + 4.8);
  doc.text("Distribution Bar", marginX + 145, tableHeaderY + 4.8);

  currentY = tableHeaderY + rowHeight;

  MODEL_CLASSES.forEach((clsKey, index) => {
    checkOverflow(rowHeight + 1);
    const prob = record.probabilities?.[clsKey] ?? 0;
    const probPct = (prob * 100).toFixed(2);
    const isWinner =
      (record.predicted_class || "").toUpperCase() === clsKey.toUpperCase();
    const label = getDisplayClassName(clsKey);

    // Row alternating background
    if (index % 2 === 1) {
      doc.setFillColor(248, 250, 252);
      doc.rect(marginX, currentY, contentWidth, rowHeight, "F");
    }
    doc.setDrawColor(226, 232, 240);
    doc.setLineWidth(0.15);
    doc.line(
      marginX,
      currentY + rowHeight,
      marginX + contentWidth,
      currentY + rowHeight,
    );

    // Text
    doc.setFont("helvetica", isWinner ? "bold" : "normal");
    doc.setFontSize(8);
    doc.setTextColor(
      isWinner ? 2 : 51,
      isWinner ? 132 : 65,
      isWinner ? 199 : 85,
    );
    doc.text(label, marginX + 4, currentY + 4.8);

    doc.setFont("helvetica", "normal");
    doc.setTextColor(
      isWinner ? 2 : 100,
      isWinner ? 132 : 116,
      isWinner ? 199 : 139,
    );
    doc.text(
      isWinner ? "[PREDICTED CATEGORY]" : "Evaluated",
      marginX + 65,
      currentY + 4.8,
    );

    doc.setFont("helvetica", isWinner ? "bold" : "normal");
    doc.setTextColor(15, 23, 42);
    doc.text(`${probPct}%`, marginX + 115, currentY + 4.8);

    // Miniature Progress Bar
    const barX = marginX + 145;
    const maxBarW = 30;
    const barW = Math.max(0.5, prob * maxBarW);
    doc.setFillColor(226, 232, 240);
    doc.roundedRect(barX, currentY + 2, maxBarW, 3, 1, 1, "F");

    doc.setFillColor(
      isWinner ? 2 : 148,
      isWinner ? 132 : 163,
      isWinner ? 199 : 184,
    );
    doc.roundedRect(barX, currentY + 2, barW, 3, 1, 1, "F");

    currentY += rowHeight;
  });

  currentY += 6;

  // ==========================================
  // 5. VISUAL EXPLAINABILITY & EXECUTION TELEMETRY
  // ==========================================
  checkOverflow(32);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(10);
  doc.setTextColor(15, 23, 42);
  doc.text(
    "3. EXPLAINABILITY PARAMETERS & EXECUTION TELEMETRY",
    marginX,
    currentY,
  );

  currentY += 4;
  const telemY = currentY;
  const telemHeight = 27;

  doc.setFillColor(248, 250, 252);
  doc.setDrawColor(226, 232, 240);
  doc.setLineWidth(0.3);
  doc.roundedRect(marginX, telemY, contentWidth, telemHeight, 2, 2, "FD");

  const targetLayer = record.target_layer || "conv5_block16_2_conv";
  const targetClass =
    record.selected_target_label ||
    record.selected_target_class ||
    record.predicted_label ||
    record.predicted_class;
  const inferDuration = record.inference_time_ms
    ? `${record.inference_time_ms.toFixed(1)} ms`
    : "N/A";
  const totalDuration = record.total_time_ms
    ? `${record.total_time_ms.toFixed(1)} ms`
    : "N/A";

  doc.setFontSize(7.8);
  // Row 1
  doc.setFont("helvetica", "bold");
  doc.setTextColor(71, 85, 105);
  doc.text("Explainability Layer:", marginX + 4, telemY + 5.2);
  doc.setFont("helvetica", "normal");
  doc.setTextColor(15, 23, 42);
  doc.text(targetLayer, marginX + 38, telemY + 5.2);

  const col2X = marginX + 110;
  doc.setFont("helvetica", "bold");
  doc.setTextColor(71, 85, 105);
  doc.text("Inference Duration:", col2X, telemY + 5.2);
  doc.setFont("helvetica", "normal");
  doc.setTextColor(15, 23, 42);
  doc.text(inferDuration, col2X + 28, telemY + 5.2);

  // Row 2
  doc.setFont("helvetica", "bold");
  doc.setTextColor(71, 85, 105);
  doc.text("Grad-CAM Target:", marginX + 4, telemY + 10.6);
  doc.setFont("helvetica", "normal");
  doc.setTextColor(15, 23, 42);
  doc.text(targetClass, marginX + 38, telemY + 10.6);

  doc.setFont("helvetica", "bold");
  doc.setTextColor(71, 85, 105);
  doc.text("Total Processing:", col2X, telemY + 10.6);
  doc.setFont("helvetica", "normal");
  doc.setTextColor(15, 23, 42);
  doc.text(totalDuration, col2X + 28, telemY + 10.6);

  // Row 3
  doc.setFont("helvetica", "bold");
  doc.setTextColor(71, 85, 105);
  doc.text("Explainability Math:", marginX + 4, telemY + 16.0);
  doc.setFont("helvetica", "normal");
  doc.setTextColor(15, 23, 42);
  doc.text(
    "Gradient-weighted Class Activation Mapping (Grad-CAM)",
    marginX + 38,
    telemY + 16.0,
  );

  // Row 4 - Preprocessing with full-width fit/wrap
  doc.setFont("helvetica", "bold");
  doc.setTextColor(71, 85, 105);
  doc.text("Preprocessing:", marginX + 4, telemY + 21.4);
  doc.setFont("helvetica", "normal");
  doc.setTextColor(15, 23, 42);
  const prepText =
    "TensorFlow/Keras DenseNet121 preprocessing (tf.keras.applications.densenet.preprocess_input)";
  const splitPrep = doc.splitTextToSize(prepText, contentWidth - 42);
  doc.text(splitPrep, marginX + 38, telemY + 21.4);

  currentY = telemY + telemHeight + 5.5;

  // ==========================================
  // 6. MODEL ARCHITECTURE & EVALUATION BENCHMARKS
  // ==========================================
  checkOverflow(30);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(10);
  doc.setTextColor(15, 23, 42);
  doc.text(
    "4. VERIFIED MODEL ARCHITECTURE & RESEARCH BENCHMARK",
    marginX,
    currentY,
  );

  currentY += 4;
  const archBoxY = currentY;
  const archBoxHeight = 25;

  doc.setFillColor(248, 250, 252);
  doc.setDrawColor(226, 232, 240);
  doc.setLineWidth(0.3);
  doc.roundedRect(marginX, archBoxY, contentWidth, archBoxHeight, 2, 2, "FD");

  doc.setFontSize(7.8);
  doc.setFont("helvetica", "bold");
  doc.setTextColor(71, 85, 105);
  doc.text("Neural Architecture:", marginX + 4, archBoxY + 5);
  doc.setFont("helvetica", "normal");
  doc.setTextColor(15, 23, 42);
  doc.text(
    "DenseNet121 Backbone • GAP (1024-d) • BatchNorm • Dropout (0.4) • Dense (4 Softmax Units)",
    marginX + 38,
    archBoxY + 5,
  );

  doc.setFont("helvetica", "bold");
  doc.setTextColor(71, 85, 105);
  doc.text("Validation Benchmark:", marginX + 4, archBoxY + 10.5);
  doc.setFont("helvetica", "normal");
  doc.setTextColor(15, 23, 42);
  doc.text(
    "97.68% Validation Accuracy • 97.69% Weighted F1 • 97.34% Macro F1 (949 stratified samples)",
    marginX + 38,
    archBoxY + 10.5,
  );

  // Multi-line wrapped Artifact Provenance
  doc.setFont("helvetica", "bold");
  doc.setTextColor(71, 85, 105);
  doc.text("Artifact Provenance:", marginX + 4, archBoxY + 16);
  doc.setFont("helvetica", "normal");
  doc.setTextColor(100, 116, 139);
  const provText =
    "Source: lungcare_evaluation.json • Kaggle jtiptj/chest-xray-pneumoniacovid19tuberculosis. Test metrics omitted (unrecorded in artifact).";
  const splitProv = doc.splitTextToSize(provText, contentWidth - 42);
  doc.text(splitProv, marginX + 38, archBoxY + 16);

  currentY = archBoxY + archBoxHeight + 5.5;

  // ==========================================
  // 7. MANDATORY PRIVACY & MEDICAL DISCLAIMERS
  // ==========================================
  const disclaimerText1 =
    "ACADEMIC RESEARCH PROTOTYPE NOTICE: LungCare AI is an experimental computer-vision research system developed strictly for academic and educational evaluation. It is NOT an FDA-cleared, CE-marked, or clinically validated diagnostic medical device. Model probability predictions and layer attribution parameters must NEVER be used independently to guide clinical diagnoses, patient management, or clinical decision-making. Always consult licensed medical professionals for clinical diagnosis.";

  const disclaimerText2 =
    "IMAGE RETENTION & LOCAL PRIVACY POLICY: In accordance with client-side browser privacy protocols, original uploaded radiographs and Grad-CAM colorized heatmap images are not retained in browser IndexedDB history to conserve local storage and prevent persisting sensitive patient imagery. Consequently, image overlays are omitted from this historical report.";

  const split1 = doc.splitTextToSize(disclaimerText1, contentWidth - 8);
  const split2 = doc.splitTextToSize(disclaimerText2, contentWidth - 8);
  const neededDisclaimerH =
    10 + split1.length * 3.4 + 4 + split2.length * 3.4 + 4;

  checkOverflow(neededDisclaimerH);

  doc.setFont("helvetica", "bold");
  doc.setFontSize(9.5);
  doc.setTextColor(180, 83, 9); // amber-700
  doc.text("5. MANDATORY RESEARCH & PRIVACY DISCLAIMERS", marginX, currentY);

  currentY += 3.5;
  const discBoxY = currentY;

  doc.setFillColor(255, 251, 235); // amber-50
  doc.setDrawColor(253, 230, 138); // amber-200
  doc.setLineWidth(0.4);
  doc.roundedRect(
    marginX,
    discBoxY,
    contentWidth,
    neededDisclaimerH,
    2,
    2,
    "FD",
  );

  let textY = discBoxY + 5;
  doc.setFont("helvetica", "bold");
  doc.setFontSize(7.5);
  doc.setTextColor(146, 64, 14); // amber-800
  doc.text("RESEARCH DISCLAIMER:", marginX + 4, textY);

  textY += 3.5;
  doc.setFont("helvetica", "normal");
  doc.setTextColor(180, 83, 9);
  doc.text(split1, marginX + 4, textY);

  textY += split1.length * 3.4 + 2;
  doc.setFont("helvetica", "bold");
  doc.setTextColor(146, 64, 14);
  doc.text("PRIVACY & IMAGE RETENTION:", marginX + 4, textY);

  textY += 3.5;
  doc.setFont("helvetica", "normal");
  doc.setTextColor(180, 83, 9);
  doc.text(split2, marginX + 4, textY);

  currentY = discBoxY + neededDisclaimerH + 6;

  // ==========================================
  // 8. PAGINATION & CONFIDENTIAL FOOTERS
  // ==========================================
  const totalPages = doc.getNumberOfPages();
  for (let i = 1; i <= totalPages; i++) {
    doc.setPage(i);

    doc.setDrawColor(226, 232, 240); // slate-200
    doc.setLineWidth(0.2);
    doc.line(marginX, pageHeight - 12, marginX + contentWidth, pageHeight - 12);

    doc.setFont("helvetica", "normal");
    doc.setFontSize(7.5);
    doc.setTextColor(148, 163, 184); // slate-400
    doc.text(
      "LungCare AI • Academic Research Prototype • Confidential Research Evaluation Document",
      marginX,
      pageHeight - 8,
    );
    doc.text(
      `Page ${i} of ${totalPages}`,
      pageWidth - marginX,
      pageHeight - 8,
      { align: "right" },
    );
  }

  return doc;
}

/**
 * Generates and downloads the PDF report for the given analysis record.
 * Automatically saves the file locally in the browser with filename:
 * LungCare_Analysis_<record-id>.pdf
 */
export async function downloadAnalysisReportPDF(
  record: HistoryAnalysisRecord,
  options?: GenerateReportOptions,
): Promise<void> {
  const doc = buildAnalysisReportPDF(record, options);
  const filename = getReportFileName(record.id);
  doc.save(filename);
}
