#!/usr/bin/env node

const fs = require("node:fs");
const path = require("node:path");

const lutPath = path.join(__dirname, "hydrolight_lut_all_platforms.json");
const lut = JSON.parse(fs.readFileSync(lutPath, "utf8"));
const rows = lut.rows;

if (!Array.isArray(rows) || rows.length === 0) {
  throw new Error("HydroLight LUT has no rows.");
}

const unique = (key) => [...new Set(rows.map((row) => row[key]))].sort((a, b) =>
  typeof a === "number" ? a - b : String(a).localeCompare(String(b))
);

const invalid = rows.filter((row) =>
  !Array.isArray(row.values) || row.values.length !== 3 ||
  row.values.some((value) => !Number.isFinite(value))
);

console.log(lut.meaning);
console.log(`Rows: ${rows.length}`);
console.log(`Platforms: ${unique("platform").join(", ")}`);
console.log(`Solar zenith angles: ${unique("sza").join(", ")} degrees`);
console.log(`Optical backgrounds: ${unique("background").join(", ")}`);
console.log(`TSM/NAP values: ${unique("tsm").join(", ")} g m^-3`);
console.log(`Invalid rows: ${invalid.length}`);

if (rows.length !== 675 || invalid.length !== 0) {
  process.exitCode = 1;
}
