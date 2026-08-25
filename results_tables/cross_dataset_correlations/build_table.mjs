import { createHash } from "node:crypto";
import { watch } from "node:fs";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Workbook } from "@oai/artifact-tool";

const artifactDir = path.dirname(fileURLToPath(import.meta.url));
const repositoryRoot = path.resolve(artifactDir, "../..");
const sourcePath = path.join(
  repositoryRoot,
  "focus_alt_exp_pipeline/results/big_table_development/big_table_correlations.csv",
);
const outputCsvPath = path.join(artifactDir, "cross_dataset_correlations.csv");
const outputPngPath = path.join(artifactDir, "cross_dataset_correlations.png");

const shortLabels = {
  hu_rx22: "RX22",
  hu_pvt21: "PVT21",
  hu_g18: "G18",
  hu_vt16: "VT16",
  rnx_esi: "ESI",
  rnx_eweak: "Eweak",
  rnx_estrong: "Estrong",
  rnx_eonly: "Eonly",
  rnx_eonlystrong: "Eonly+",
  novel_focus: "Focus",
};

const displayColumns = [
  { source: "analysis_dataset_id", label: "Dataset", color: null },
  { source: "N", label: "N", color: null },
  { source: "hu_original_expectedness_r", label: "Hu original", color: "#9A4F72" },
  { source: "qwen_x_but_not_y_r", label: "X-but-not-Y", color: "#C88925" },
  { source: "qwen_no_frame_r", label: "No frame", color: "#65788B" },
  { source: "qwen_set_r", label: "Set", color: "#3154D8" },
  { source: "qwen_ordering_r", label: "Ordering", color: "#D26751" },
  { source: "qwen_disjunction_r", label: "Disjunction", color: "#27837E" },
  { source: "qwen_conjunction_r", label: "Conjunction", color: "#7756A8" },
];

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function numericOrNull(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  if (!Number.isFinite(number)) throw new Error(`Expected a finite number, received ${value}`);
  return number;
}

async function build() {
  const sourceBytes = await fs.readFile(sourcePath);
  const sourceText = sourceBytes.toString("utf8");
  const sourceWorkbook = await Workbook.fromCSV(sourceText, { sheetName: "Source" });
  const sourceSheet = sourceWorkbook.worksheets.getItem("Source");
  const sourceValues = sourceSheet.getUsedRange(true).values;
  const [sourceHeader, ...sourceRows] = sourceValues;
  const columnIndex = Object.fromEntries(sourceHeader.map((label, index) => [String(label), index]));

  for (const column of displayColumns) {
    if (!(column.source in columnIndex)) {
      throw new Error(`Required source column is missing: ${column.source}`);
    }
  }
  if (sourceRows.length !== 10) {
    throw new Error(`Expected 10 result rows, found ${sourceRows.length}`);
  }

  const displayRows = sourceRows.map((sourceRow) =>
    displayColumns.map((column, columnNumber) => {
      const value = sourceRow[columnIndex[column.source]];
      if (columnNumber === 0) {
        const id = String(value);
        if (!(id in shortLabels)) throw new Error(`Unknown analysis dataset id: ${id}`);
        return shortLabels[id];
      }
      return numericOrNull(value);
    }),
  );

  const workbook = Workbook.create();
  const sheet = workbook.worksheets.add("Correlations");
  sheet.showGridLines = false;
  sheet.getRange("A1:I1").merge();
  sheet.getRange("A1").values = [["Cross-dataset correlations"]];
  sheet.getRange("A2:I2").merge();
  sheet.getRange("A2").values = [[
    "Pearson r at the source-grain analysis unit. Bars share a fixed −0.40 to +0.70 scale.",
  ]];
  sheet.getRange("A3:I3").values = [displayColumns.map((column) => column.label)];
  sheet.getRange(`A4:I${3 + displayRows.length}`).values = displayRows;

  sheet.getRange("A1:I1").format = {
    fill: "#15243A",
    font: { bold: true, color: "#FFFFFF", size: 16 },
    rowHeight: 34,
    verticalAlignment: "center",
  };
  sheet.getRange("A2:I2").format = {
    fill: "#E9EDF5",
    font: { color: "#41516A", size: 10 },
    rowHeight: 28,
    verticalAlignment: "center",
  };
  sheet.getRange("A3:I3").format = {
    fill: "#F3F1EB",
    font: { bold: true, color: "#536177", size: 10 },
    rowHeight: 26,
    horizontalAlignment: "center",
    verticalAlignment: "center",
    borders: { bottom: { style: "medium", color: "#BFC5CF" } },
  };
  sheet.getRange("A3").format.horizontalAlignment = "left";
  sheet.getRange("A4:I13").format = {
    fill: "#FFFEFA",
    font: { color: "#17243A", size: 10 },
    rowHeight: 31,
    verticalAlignment: "center",
    borders: { insideHorizontal: { style: "thin", color: "#DEDAD1" } },
  };
  sheet.getRange("A13:I13").format.fill = "#E8EEFF";
  sheet.getRange("A13").format.font = { bold: true, color: "#1646CC", size: 11 };
  sheet.getRange("B4:B13").format = { horizontalAlignment: "center", numberFormat: "0" };
  sheet.getRange("C4:I13").format = {
    horizontalAlignment: "right",
    numberFormat: "+0.000;-0.000;—",
  };

  sheet.getRange("A1:A13").format.columnWidthPx = 105;
  sheet.getRange("B1:B13").format.columnWidthPx = 45;
  sheet.getRange("C1:I13").format.columnWidthPx = 132;

  for (let column = 2; column < displayColumns.length; column += 1) {
    const letter = String.fromCharCode("A".charCodeAt(0) + column);
    sheet.getRange(`${letter}4:${letter}13`).conditionalFormats.add("dataBar", {
      color: displayColumns[column].color,
      thresholds: [
        { type: "num", value: -0.4 },
        { type: "num", value: 0.7 },
      ],
      gradient: false,
    });
  }

  const tableInspection = await workbook.inspect({
    kind: "table",
    range: "Correlations!A1:I13",
    include: "values,formulas",
    tableMaxRows: 13,
    tableMaxCols: 9,
    maxChars: 8000,
  });
  const errorInspection = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 50 },
    summary: "formula error scan",
  });

  const preview = await workbook.render({
    sheetName: "Correlations",
    range: "A1:I13",
    scale: 2,
    format: "png",
    headers: false,
  });
  const outputPngBytes = new Uint8Array(await preview.arrayBuffer());

  await Promise.all([
    fs.writeFile(outputCsvPath, sourceBytes),
    fs.writeFile(outputPngPath, outputPngBytes),
  ]);

  const outputCsvBytes = await fs.readFile(outputCsvPath);
  if (sha256(sourceBytes) !== sha256(outputCsvBytes)) {
    throw new Error("Output CSV is not byte-identical to its canonical source");
  }

  process.stdout.write(`${JSON.stringify({
    source: path.relative(repositoryRoot, sourcePath),
    csv: path.relative(repositoryRoot, outputCsvPath),
    png: path.relative(repositoryRoot, outputPngPath),
    rows: sourceRows.length,
    source_sha256: sha256(sourceBytes),
    output_csv_sha256: sha256(outputCsvBytes),
    table_inspection: tableInspection.ndjson,
    formula_error_scan: errorInspection.ndjson,
  }, null, 2)}\n`);
}

await build();

if (process.argv.includes("--watch")) {
  let timer = null;
  process.stdout.write(`Watching ${sourcePath}\n`);
  watch(sourcePath, () => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      build().catch((error) => process.stderr.write(`${error.stack ?? error}\n`));
    }, 200);
  });
}
