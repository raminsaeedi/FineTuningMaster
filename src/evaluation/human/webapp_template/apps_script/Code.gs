/**
 * Collector for the blind human-evaluation web app.
 *
 * Deploy this as a Google Apps Script web app bound to a Google Sheet
 * ("Extensions > Apps Script" from the sheet), with access "Anyone".
 * Every POST appends one row per rating to the sheet named below.
 *
 * The app never sends personal data: only the rater ID assigned by the study
 * owner, the opaque unit token, the six scores, an optional comment and
 * timestamps.
 */

var SHEET_NAME = "ratings";
var DIMENSIONS = [
  "chart_appropriateness",
  "layout_quality",
  "styling_accessibility",
  "interaction_design",
  "rationale_quality",
  "overall_usefulness"
];
var HEADER = [
  "received_at",
  "study_id",
  "export_id",
  "rater_id",
  "rating_id",
  "unit_token",
  "position"
].concat(DIMENSIONS).concat([
  "comment",
  "scores_json",
  "client_timestamp",
  "duration_ms",
  "consent_at",
  "client_sent_at",
  "app_version"
]);

function getSheet_() {
  var spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  var sheet = spreadsheet.getSheetByName(SHEET_NAME);
  if (!sheet) {
    sheet = spreadsheet.insertSheet(SHEET_NAME);
  }
  if (sheet.getLastRow() === 0) {
    sheet.appendRow(HEADER);
    sheet.setFrozenRows(1);
  }
  return sheet;
}

function rowFor_(payload, rating, receivedAt) {
  var scores = rating.scores || {};
  var row = [
    receivedAt,
    payload.study_id || "",
    payload.export_id || "",
    payload.rater_id || "",
    rating.rating_id || "",
    rating.unit_token || "",
    rating.position || ""
  ];
  for (var i = 0; i < DIMENSIONS.length; i++) {
    var value = scores[DIMENSIONS[i]];
    row.push(value === undefined || value === null ? "" : value);
  }
  row.push(rating.comment || "");
  row.push(JSON.stringify(scores));
  row.push(rating.client_timestamp || "");
  row.push(rating.duration_ms === undefined ? "" : rating.duration_ms);
  row.push(payload.consent_at || "");
  row.push(payload.client_sent_at || "");
  row.push(payload.app_version || "");
  return row;
}

function doPost(e) {
  var lock = LockService.getScriptLock();
  lock.waitLock(30000);
  try {
    var payload = JSON.parse(e.postData.contents);
    var ratings = payload.ratings || [];
    if (!ratings.length && payload.unit_token) {
      ratings = [payload];
    }
    var sheet = getSheet_();
    var receivedAt = new Date().toISOString();
    var rows = [];
    for (var i = 0; i < ratings.length; i++) {
      rows.push(rowFor_(payload, ratings[i], receivedAt));
    }
    if (rows.length) {
      sheet.getRange(sheet.getLastRow() + 1, 1, rows.length, HEADER.length).setValues(rows);
    }
    return ContentService
      .createTextOutput(JSON.stringify({ ok: true, stored: rows.length }))
      .setMimeType(ContentService.MimeType.JSON);
  } catch (error) {
    return ContentService
      .createTextOutput(JSON.stringify({ ok: false, error: String(error) }))
      .setMimeType(ContentService.MimeType.JSON);
  } finally {
    lock.releaseLock();
  }
}

/**
 * Progress lookup used by the app to resume a rater who lost their browser
 * storage or continues on another device. Returns only the unit tokens that
 * are already stored for that rater, never any scores.
 */
function doGet(e) {
  var raterId = e && e.parameter ? e.parameter.rater_id : null;
  if (!raterId) {
    return ContentService
      .createTextOutput(JSON.stringify({ ok: true, service: "human-eval-collector" }))
      .setMimeType(ContentService.MimeType.JSON);
  }
  var sheet = getSheet_();
  var lastRow = sheet.getLastRow();
  var units = [];
  if (lastRow > 1) {
    var raterColumn = HEADER.indexOf("rater_id") + 1;
    var unitColumn = HEADER.indexOf("unit_token") + 1;
    var values = sheet.getRange(2, 1, lastRow - 1, HEADER.length).getValues();
    var seen = {};
    for (var i = 0; i < values.length; i++) {
      var row = values[i];
      if (String(row[raterColumn - 1]) !== String(raterId)) { continue; }
      var token = String(row[unitColumn - 1]);
      if (token && !seen[token]) {
        seen[token] = true;
        units.push(token);
      }
    }
  }
  return ContentService
    .createTextOutput(JSON.stringify({ ok: true, rater_id: raterId, units: units }))
    .setMimeType(ContentService.MimeType.JSON);
}
