/**
 * CAPTURE RATE TRACKER
 * Analyzes hearing test data and creates professional dashboards
 *
 * FILES:
 *   Code.gs (or any .gs)  — paste this file
 *   ImportSidebar.html     — create as HTML file in Apps Script editor
 */

// ====================
// MENU & SETUP
// ====================

function onOpen() {
  SpreadsheetApp.getUi()
    .createMenu('Capture Rate Tracker')
    .addItem('Import CSV / ZIP', 'showImportSidebar')
    .addSeparator()
    .addItem('Refresh Dashboard', 'refreshDashboard')
    .addItem('Update Charts', 'updateAllCharts')
    .addItem('Format All Sheets', 'formatAllSheets')
    .addSeparator()
    .addItem('Create New Period Sheet', 'createNewPeriodSheet')
    .addItem('Setup Dashboard', 'setupDashboard')
    .addToUi();
}

function setupDashboard() {
  createDashboardSheet();
  createStaffPerformanceSheet();
  createTrendsSheet();
  const allData = aggregateAllPeriods();
  updateDashboardSheet(allData);
  updateStaffPerformanceSheet(allData);
  updateTrendsSheet(allData);
  updateAllCharts();
  SpreadsheetApp.getUi().alert('Dashboard setup complete!');
}

function refreshDashboard() {
  const allData = aggregateAllPeriods();
  updateDashboardSheet(allData);
  updateStaffPerformanceSheet(allData);
  updateTrendsSheet(allData);
  updateAllCharts();
  SpreadsheetApp.getUi().alert('Dashboard refreshed!');
}

// ====================
// SIDEBAR IMPORT
// ====================

function showImportSidebar() {
  const html = HtmlService.createHtmlOutputFromFile('ImportSidebar')
    .setTitle('Import CSV Data')
    .setWidth(350);
  SpreadsheetApp.getUi().showSidebar(html);
}

function getExistingSheetNames() {
  return SpreadsheetApp.getActiveSpreadsheet().getSheets().map(function(s) { return s.getName(); });
}

function importCsvToSheet(csvString, sheetName) {
  if (!csvString || !sheetName) {
    return { success: false, message: 'Missing CSV data or sheet name' };
  }

  // Strip BOM if present
  if (csvString.charCodeAt(0) === 0xFEFF) {
    csvString = csvString.substring(1);
  }

  var data;
  try {
    data = Utilities.parseCsv(csvString);
  } catch (err) {
    return { success: false, message: 'Failed to parse CSV: ' + err.message };
  }

  if (data.length < 2) {
    return { success: false, message: 'CSV has no data rows' };
  }

  var expectedHeaders = ['Staff', 'Total Tests', 'Purchases', 'Capture Rate',
    'Patient ID', 'Appointment Type', 'Appointment Date', 'Period Week Day',
    'Notes', 'Outcome Notes', 'Has Purchase', 'Appointment Link'];

  var headers = data[0].map(function(h) { return h.trim(); });
  var missing = expectedHeaders.filter(function(h) { return headers.indexOf(h) === -1; });

  if (missing.length > 0) {
    return { success: false, message: 'Missing columns: ' + missing.join(', ') };
  }

  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var sheet = ss.getSheetByName(sheetName);

  if (sheet) {
    sheet.clear();
    sheet.clearFormats();
  } else {
    sheet = ss.insertSheet(sheetName);
  }

  // Normalize column count (some rows may have fewer columns)
  var colCount = headers.length;
  var normalized = data.map(function(row) {
    while (row.length < colCount) row.push('');
    return row.slice(0, colCount);
  });

  sheet.getRange(1, 1, normalized.length, colCount).setValues(normalized);
  sheet.getRange(1, 1, 1, colCount)
    .setBackground('#4285f4').setFontColor('#ffffff').setFontWeight('bold');
  sheet.setFrozenRows(1);

  return { success: true, message: 'Imported ' + (data.length - 1) + ' rows to "' + sheetName + '"' };
}

// ====================
// DATA COLLECTION
// ====================

function getAllPeriodSheets() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var sheets = ss.getSheets();
  var periodSheets = [];

  sheets.forEach(function(sheet) {
    var match = sheet.getName().trim().match(/^period\s+(\d+)$/i);
    if (match) {
      periodSheets.push({
        sheet: sheet,
        name: sheet.getName(),
        period: parseInt(match[1])
      });
    }
  });

  periodSheets.sort(function(a, b) { return a.period - b.period; });
  return periodSheets;
}

function getPeriodData(sheet) {
  var data = sheet.getDataRange().getValues();
  if (data.length < 2) return null;

  var headers = data[0];
  var rows = data.slice(1);

  var colMap = {};
  headers.forEach(function(header, idx) {
    colMap[String(header).trim()] = idx;
  });

  var staffStats = {};
  var apptTypes = {};
  var currentStaff = '';

  rows.forEach(function(row) {
    var staff = row[colMap['Staff']];
    var totalTests = row[colMap['Total Tests']];
    var purchases = row[colMap['Purchases']];
    var apptType = row[colMap['Appointment Type']];
    var hasPurchase = row[colMap['Has Purchase']];

    if (staff && staff !== '') {
      currentStaff = staff;
      if (!staffStats[currentStaff]) {
        staffStats[currentStaff] = { totalTests: 0, purchases: 0, appointments: [] };
      }
      if (totalTests) {
        staffStats[currentStaff].totalTests += parseInt(totalTests) || 0;
        staffStats[currentStaff].purchases += parseInt(purchases) || 0;
      }
    }

    if (apptType && apptType !== '') {
      if (!apptTypes[apptType]) {
        apptTypes[apptType] = { total: 0, purchased: 0 };
      }
      apptTypes[apptType].total++;
      if (hasPurchase === 'Yes') {
        apptTypes[apptType].purchased++;
      }
      if (currentStaff && staffStats[currentStaff]) {
        staffStats[currentStaff].appointments.push({
          type: apptType,
          purchased: hasPurchase === 'Yes'
        });
      }
    }
  });

  return { staffStats: staffStats, apptTypes: apptTypes };
}

function aggregateAllPeriods() {
  var periodSheets = getAllPeriodSheets();
  var allData = [];

  periodSheets.forEach(function(ps) {
    var periodData = getPeriodData(ps.sheet);
    if (periodData) {
      allData.push({ period: ps.period, name: ps.name, data: periodData });
    }
  });

  return allData;
}

// ====================
// DASHBOARD SHEET
// ====================

function createDashboardSheet() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var dashboard = ss.getSheetByName('Dashboard');
  if (!dashboard) {
    dashboard = ss.insertSheet('Dashboard', 0);
  } else {
    dashboard.clear();
    dashboard.clearFormats();
  }
  return dashboard;
}

function updateDashboardSheet(allData) {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var dashboard = ss.getSheetByName('Dashboard');
  if (!dashboard) return;

  dashboard.clear();
  dashboard.clearFormats();
  dashboard.getCharts().forEach(function(c) { dashboard.removeChart(c); });

  // Header
  dashboard.getRange('A1').setValue('CAPTURE RATE TRACKER');
  dashboard.getRange('A1:G1').merge().setBackground('#1a73e8').setFontColor('#ffffff')
    .setFontSize(16).setFontWeight('bold').setHorizontalAlignment('center');

  dashboard.getRange('A2').setValue('Last Updated: ' + new Date().toLocaleString());
  dashboard.getRange('A2:G2').merge().setFontSize(10).setFontStyle('italic');

  if (allData.length === 0) {
    dashboard.getRange('A4').setValue('No period data found. Use "Import CSV / ZIP" to add data.');
    return;
  }

  var row = 4;

  // === OVERALL SUMMARY ===
  dashboard.getRange('A' + row).setValue('OVERALL PERFORMANCE');
  dashboard.getRange('A' + row + ':G' + row).merge().setBackground('#f1f3f4')
    .setFontWeight('bold').setFontSize(12);
  row++;

  var totalTests = 0;
  var totalPurchases = 0;
  var allStaff = {};

  allData.forEach(function(pd) {
    Object.keys(pd.data.staffStats).forEach(function(staff) {
      if (!allStaff[staff]) allStaff[staff] = { tests: 0, purchases: 0 };
      allStaff[staff].tests += pd.data.staffStats[staff].totalTests;
      allStaff[staff].purchases += pd.data.staffStats[staff].purchases;
      totalTests += pd.data.staffStats[staff].totalTests;
      totalPurchases += pd.data.staffStats[staff].purchases;
    });
  });

  var overallRate = totalTests > 0 ? totalPurchases / totalTests : 0;

  dashboard.getRange('A' + row + ':G' + (row + 2)).setBackground('#ffffff')
    .setBorder(true, true, true, true, true, true);

  dashboard.getRange('A' + row).setValue('Total Hearing Tests:');
  dashboard.getRange('B' + row).setValue(totalTests).setFontWeight('bold').setFontSize(14);
  dashboard.getRange('D' + row).setValue('Total Purchases:');
  dashboard.getRange('E' + row).setValue(totalPurchases).setFontWeight('bold').setFontSize(14);
  row++;

  dashboard.getRange('A' + row).setValue('Overall Capture Rate:');
  dashboard.getRange('B' + row).setValue(overallRate).setNumberFormat('0.0%')
    .setFontWeight('bold').setFontSize(16)
    .setFontColor(overallRate >= 0.5 ? '#0f9d58' : '#ea4335');
  dashboard.getRange('D' + row).setValue('Active Staff:');
  dashboard.getRange('E' + row).setValue(Object.keys(allStaff).length).setFontWeight('bold').setFontSize(14);
  row++;

  dashboard.getRange('A' + row).setValue('Periods Tracked:');
  dashboard.getRange('B' + row).setValue(allData.length).setFontWeight('bold').setFontSize(14);
  row += 2;

  // === PERIOD BREAKDOWN ===
  dashboard.getRange('A' + row).setValue('PERIOD BREAKDOWN');
  dashboard.getRange('A' + row + ':G' + row).merge().setBackground('#f1f3f4')
    .setFontWeight('bold').setFontSize(12);
  row++;

  var periodHeaders = ['Period', 'Tests', 'Purchases', 'Capture Rate', 'Change vs Prev', 'Top Performer', 'Top Rate'];
  dashboard.getRange('A' + row + ':G' + row).setValues([periodHeaders])
    .setBackground('#e8eaed').setFontWeight('bold').setHorizontalAlignment('center');
  row++;

  var periodSummaryStart = row;
  var prevCaptureRate = null;
  var periodRows = [];

  allData.forEach(function(pd) {
    var periodTests = 0;
    var periodPurchases = 0;
    var topStaff = { name: '', rate: 0 };

    Object.keys(pd.data.staffStats).forEach(function(staff) {
      var stats = pd.data.staffStats[staff];
      periodTests += stats.totalTests;
      periodPurchases += stats.purchases;
      var staffRate = stats.totalTests > 0 ? (stats.purchases / stats.totalTests) : 0;
      if (staffRate > topStaff.rate) {
        topStaff = { name: staff, rate: staffRate };
      }
    });

    var captureRate = periodTests > 0 ? (periodPurchases / periodTests) : 0;
    var change = prevCaptureRate !== null ? captureRate - prevCaptureRate : null;
    periodRows.push([pd.name, periodTests, periodPurchases, captureRate, change !== null ? change : 'N/A', topStaff.name, topStaff.rate]);
    prevCaptureRate = captureRate;
  });

  if (periodRows.length > 0) {
    var pRange = dashboard.getRange(row, 1, periodRows.length, 7);
    pRange.setValues(periodRows);
    dashboard.getRange(row, 4, periodRows.length, 1).setNumberFormat('0.0%');
    dashboard.getRange(row, 7, periodRows.length, 1).setNumberFormat('0.0%');
    for (var pi = 0; pi < periodRows.length; pi++) {
      var changeVal = periodRows[pi][4];
      if (changeVal !== 'N/A') {
        var cCell = dashboard.getRange(row + pi, 5);
        cCell.setNumberFormat('+0.0%;-0.0%');
        cCell.setFontColor(changeVal >= 0 ? '#0f9d58' : '#ea4335');
      }
    }
    row += periodRows.length;
  }

  dashboard.getRange('A' + periodSummaryStart + ':G' + (row - 1))
    .setBorder(true, true, true, true, true, true);
  row += 2;

  // === STAFF RANKINGS ===
  dashboard.getRange('A' + row).setValue('STAFF RANKINGS');
  dashboard.getRange('A' + row + ':G' + row).merge().setBackground('#f1f3f4')
    .setFontWeight('bold').setFontSize(12);
  row++;

  var staffArray = Object.keys(allStaff).map(function(name) {
    return {
      name: name,
      tests: allStaff[name].tests,
      purchases: allStaff[name].purchases,
      rate: allStaff[name].tests > 0 ? (allStaff[name].purchases / allStaff[name].tests) : 0
    };
  });
  staffArray.sort(function(a, b) { return b.rate - a.rate; });

  var staffHeaders = ['Rank', 'Staff', 'Tests', 'Purchases', 'Capture Rate', '', ''];
  dashboard.getRange('A' + row + ':G' + row).setValues([staffHeaders])
    .setBackground('#e8eaed').setFontWeight('bold').setHorizontalAlignment('center');
  row++;

  var staffChartStart = row;
  if (staffArray.length > 0) {
    var staffRows = staffArray.map(function(staff, idx) {
      return [idx + 1, staff.name, staff.tests, staff.purchases, staff.rate, '', ''];
    });
    dashboard.getRange(row, 1, staffRows.length, 7).setValues(staffRows);
    dashboard.getRange(row, 5, staffRows.length, 1).setNumberFormat('0.0%');
    var topCount = Math.min(3, staffArray.length);
    if (topCount > 0) {
      dashboard.getRange(row, 1, topCount, 7).setBackground('#e6f4ea');
    }
    row += staffRows.length;
  }
  var staffChartEnd = row - 1;

  dashboard.getRange('A' + staffChartStart + ':G' + staffChartEnd)
    .setBorder(true, true, true, true, true, true);

  // Column widths
  dashboard.setColumnWidth(1, 150);
  dashboard.setColumnWidth(2, 120);
  dashboard.setColumnWidth(3, 100);
  dashboard.setColumnWidth(4, 120);
  dashboard.setColumnWidth(5, 120);
  dashboard.setColumnWidth(6, 150);
  dashboard.setColumnWidth(7, 100);
  dashboard.setFrozenRows(3);

  // Store chart range for later use
  var props = PropertiesService.getScriptProperties();
  props.setProperty('staffChartStart', String(staffChartStart));
  props.setProperty('staffChartEnd', String(staffChartEnd));
}

// ====================
// STAFF PERFORMANCE SHEET
// ====================

function createStaffPerformanceSheet() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var sheet = ss.getSheetByName('Staff Performance');
  if (!sheet) {
    sheet = ss.insertSheet('Staff Performance');
  } else {
    sheet.clear();
    sheet.clearFormats();
  }
  return sheet;
}

function updateStaffPerformanceSheet(allData) {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var sheet = ss.getSheetByName('Staff Performance');
  if (!sheet) return;

  sheet.clear();
  sheet.clearFormats();

  sheet.getRange('A1').setValue('STAFF PERFORMANCE ANALYSIS');
  sheet.getRange('A1:H1').merge().setBackground('#1a73e8').setFontColor('#ffffff')
    .setFontSize(16).setFontWeight('bold').setHorizontalAlignment('center');

  if (allData.length === 0) return;

  var row = 3;
  var staffByPeriod = {};

  allData.forEach(function(pd) {
    Object.keys(pd.data.staffStats).forEach(function(staff) {
      if (!staffByPeriod[staff]) staffByPeriod[staff] = [];
      var stats = pd.data.staffStats[staff];
      staffByPeriod[staff].push({
        period: pd.period,
        tests: stats.totalTests,
        purchases: stats.purchases,
        rate: stats.totalTests > 0 ? (stats.purchases / stats.totalTests) : 0
      });
    });
  });

  var headers = ['Staff', 'Total Tests', 'Total Purchases', 'Capture Rate', 'Avg Tests/Period', 'Best Period', 'Worst Period', 'Trend'];
  sheet.getRange('A' + row + ':H' + row).setValues([headers])
    .setBackground('#e8eaed').setFontWeight('bold').setHorizontalAlignment('center');
  row++;

  var dataStart = row;

  // Sort staff by capture rate descending
  var staffNames = Object.keys(staffByPeriod);
  staffNames.sort(function(a, b) {
    var aTests = staffByPeriod[a].reduce(function(s, p) { return s + p.tests; }, 0);
    var aPurch = staffByPeriod[a].reduce(function(s, p) { return s + p.purchases; }, 0);
    var bTests = staffByPeriod[b].reduce(function(s, p) { return s + p.tests; }, 0);
    var bPurch = staffByPeriod[b].reduce(function(s, p) { return s + p.purchases; }, 0);
    var aRate = aTests > 0 ? aPurch / aTests : 0;
    var bRate = bTests > 0 ? bPurch / bTests : 0;
    return bRate - aRate;
  });

  var staffDataRows = [];
  var trendMeta = [];

  staffNames.forEach(function(staff) {
    var periods = staffByPeriod[staff];
    var totalTests = periods.reduce(function(s, p) { return s + p.tests; }, 0);
    var totalPurchases = periods.reduce(function(s, p) { return s + p.purchases; }, 0);
    var captureRate = totalTests > 0 ? (totalPurchases / totalTests) : 0;
    var avgPerPeriod = totalTests / periods.length;

    var bestPeriod = periods.reduce(function(best, p) { return p.rate > best.rate ? p : best; }, periods[0]);
    var worstPeriod = periods.reduce(function(worst, p) { return p.rate < worst.rate ? p : worst; }, periods[0]);

    var trendVal, trendColor;
    if (periods.length < 2) {
      trendVal = '—';
      trendColor = '#5f6368';
    } else {
      var midpoint = Math.floor(periods.length / 2);
      var firstHalf = periods.slice(0, midpoint);
      var secondHalf = periods.slice(midpoint);
      var firstAvg = firstHalf.reduce(function(s, p) { return s + p.rate; }, 0) / firstHalf.length;
      var secondAvg = secondHalf.reduce(function(s, p) { return s + p.rate; }, 0) / secondHalf.length;
      var trend = secondAvg - firstAvg;
      if (trend > 0.02) { trendVal = 'Improving'; trendColor = '#0f9d58'; }
      else if (trend < -0.02) { trendVal = 'Declining'; trendColor = '#ea4335'; }
      else { trendVal = 'Stable'; trendColor = '#5f6368'; }
    }

    staffDataRows.push([
      staff, totalTests, totalPurchases, captureRate, avgPerPeriod,
      'P' + bestPeriod.period + ': ' + (bestPeriod.rate * 100).toFixed(1) + '%',
      'P' + worstPeriod.period + ': ' + (worstPeriod.rate * 100).toFixed(1) + '%',
      trendVal
    ]);
    trendMeta.push(trendColor);
  });

  if (staffDataRows.length > 0) {
    sheet.getRange(row, 1, staffDataRows.length, 8).setValues(staffDataRows);
    sheet.getRange(row, 4, staffDataRows.length, 1).setNumberFormat('0.0%');
    sheet.getRange(row, 5, staffDataRows.length, 1).setNumberFormat('0.0');
    for (var ti = 0; ti < trendMeta.length; ti++) {
      sheet.getRange(row + ti, 8).setFontColor(trendMeta[ti]);
    }
    row += staffDataRows.length;
  }

  if (row > dataStart) {
    sheet.getRange('A' + dataStart + ':H' + (row - 1)).setBorder(true, true, true, true, true, true);
  }

  sheet.setColumnWidth(1, 150);
  sheet.setColumnWidth(2, 100);
  sheet.setColumnWidth(3, 120);
  sheet.setColumnWidth(4, 110);
  sheet.setColumnWidth(5, 120);
  sheet.setColumnWidth(6, 130);
  sheet.setColumnWidth(7, 130);
  sheet.setColumnWidth(8, 120);
  sheet.setFrozenRows(3);
}

// ====================
// TRENDS SHEET
// ====================

function createTrendsSheet() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var sheet = ss.getSheetByName('Trends');
  if (!sheet) {
    sheet = ss.insertSheet('Trends');
  } else {
    sheet.clear();
    sheet.clearFormats();
  }
  return sheet;
}

function updateTrendsSheet(allData) {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var sheet = ss.getSheetByName('Trends');
  if (!sheet) return;

  sheet.clear();
  sheet.clearFormats();
  sheet.getCharts().forEach(function(c) { sheet.removeChart(c); });

  sheet.getRange('A1').setValue('TRENDS & ANALYSIS');
  sheet.getRange('A1:F1').merge().setBackground('#1a73e8').setFontColor('#ffffff')
    .setFontSize(16).setFontWeight('bold').setHorizontalAlignment('center');

  if (allData.length === 0) return;

  var row = 3;

  // === CAPTURE RATE OVER TIME ===
  sheet.getRange('A' + row).setValue('Capture Rate Trend');
  sheet.getRange('A' + row + ':F' + row).merge().setBackground('#f1f3f4')
    .setFontWeight('bold').setFontSize(12);
  row++;

  var trendHeaders = ['Period', 'Tests', 'Purchases', 'Capture Rate %'];
  sheet.getRange('A' + row + ':D' + row).setValues([trendHeaders])
    .setBackground('#e8eaed').setFontWeight('bold');
  var trendHeaderRow = row;
  row++;

  var trendRows = [];
  allData.forEach(function(pd) {
    var periodTests = 0;
    var periodPurchases = 0;
    Object.keys(pd.data.staffStats).forEach(function(staff) {
      periodTests += pd.data.staffStats[staff].totalTests;
      periodPurchases += pd.data.staffStats[staff].purchases;
    });
    var captureRate = periodTests > 0 ? (periodPurchases / periodTests * 100) : 0;
    trendRows.push([pd.name, periodTests, periodPurchases, parseFloat(captureRate.toFixed(1))]);
  });

  if (trendRows.length > 0) {
    sheet.getRange(row, 1, trendRows.length, 4).setValues(trendRows);
    row += trendRows.length;
  }

  var trendDataEnd = row - 1;

  // Store range info for chart creation
  var props = PropertiesService.getScriptProperties();
  props.setProperty('trendHeaderRow', String(trendHeaderRow));
  props.setProperty('trendDataEnd', String(trendDataEnd));

  row += 2;

  // === APPOINTMENT TYPE BREAKDOWN ===
  sheet.getRange('A' + row).setValue('Appointment Type Breakdown');
  sheet.getRange('A' + row + ':F' + row).merge().setBackground('#f1f3f4')
    .setFontWeight('bold').setFontSize(12);
  row++;

  var apptHeaders = ['Appointment Type', 'Total', 'Purchased', 'Capture Rate %', 'Conversion'];
  sheet.getRange('A' + row + ':E' + row).setValues([apptHeaders])
    .setBackground('#e8eaed').setFontWeight('bold');
  var apptHeaderRow = row;
  row++;

  var allApptTypes = {};
  allData.forEach(function(pd) {
    Object.keys(pd.data.apptTypes).forEach(function(type) {
      if (!allApptTypes[type]) allApptTypes[type] = { total: 0, purchased: 0 };
      allApptTypes[type].total += pd.data.apptTypes[type].total;
      allApptTypes[type].purchased += pd.data.apptTypes[type].purchased;
    });
  });

  // Sort by total descending
  var sortedTypes = Object.keys(allApptTypes).sort(function(a, b) {
    return allApptTypes[b].total - allApptTypes[a].total;
  });

  var apptRows = [];
  sortedTypes.forEach(function(type) {
    var stats = allApptTypes[type];
    var rate = stats.total > 0 ? (stats.purchased / stats.total * 100) : 0;
    apptRows.push([type || 'Unknown', stats.total, stats.purchased, parseFloat(rate.toFixed(1)), stats.purchased + '/' + stats.total]);
  });

  if (apptRows.length > 0) {
    sheet.getRange(row, 1, apptRows.length, 5).setValues(apptRows);
    row += apptRows.length;
  }

  var apptDataEnd = row - 1;
  props.setProperty('apptHeaderRow', String(apptHeaderRow));
  props.setProperty('apptDataEnd', String(apptDataEnd));

  sheet.setColumnWidth(1, 180);
  sheet.setColumnWidth(2, 100);
  sheet.setColumnWidth(3, 100);
  sheet.setColumnWidth(4, 120);
  sheet.setColumnWidth(5, 120);
}

// ====================
// CHARTS
// ====================

function updateAllCharts() {
  createCaptureRateTrendChart();
  createAppointmentTypeChart();
  createStaffComparisonChart();
  SpreadsheetApp.flush();
}

function createCaptureRateTrendChart() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var sheet = ss.getSheetByName('Trends');
  if (!sheet) return;

  sheet.getCharts().forEach(function(c) { sheet.removeChart(c); });

  var props = PropertiesService.getScriptProperties();
  var headerRow = parseInt(props.getProperty('trendHeaderRow'));
  var dataEnd = parseInt(props.getProperty('trendDataEnd'));
  if (!headerRow || !dataEnd || dataEnd <= headerRow) return;

  var dataRange = sheet.getRange('A' + headerRow + ':D' + dataEnd);

  var chart = sheet.newChart()
    .setChartType(Charts.ChartType.LINE)
    .addRange(dataRange)
    .setNumHeaders(1)
    .setPosition(3, 7, 0, 0)
    .setOption('title', 'Capture Rate Trend Over Time')
    .setOption('width', 600)
    .setOption('height', 400)
    .setOption('legend', { position: 'bottom' })
    .setOption('series', {
      0: { labelInLegend: 'Tests', targetAxisIndex: 0, color: '#4285f4' },
      1: { labelInLegend: 'Purchases', targetAxisIndex: 0, color: '#34a853' },
      2: { labelInLegend: 'Capture Rate %', targetAxisIndex: 1, color: '#ea4335', lineWidth: 3 }
    })
    .setOption('vAxes', {
      0: { title: 'Count' },
      1: { title: 'Capture Rate %' }
    })
    .setOption('hAxis', { title: 'Period' })
    .build();

  sheet.insertChart(chart);
}

function createAppointmentTypeChart() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var sheet = ss.getSheetByName('Trends');
  if (!sheet) return;

  var props = PropertiesService.getScriptProperties();
  var headerRow = parseInt(props.getProperty('apptHeaderRow'));
  var dataEnd = parseInt(props.getProperty('apptDataEnd'));
  if (!headerRow || !dataEnd || dataEnd <= headerRow) return;

  // Position below trend chart area
  var chartRow = parseInt(props.getProperty('trendDataEnd')) + 4;

  var dataRange = sheet.getRange('A' + headerRow + ':D' + dataEnd);

  var chart = sheet.newChart()
    .setChartType(Charts.ChartType.COLUMN)
    .addRange(dataRange)
    .setNumHeaders(1)
    .setPosition(chartRow, 7, 0, 0)
    .setOption('title', 'Capture Rate by Appointment Type')
    .setOption('width', 600)
    .setOption('height', 400)
    .setOption('legend', { position: 'bottom' })
    .setOption('colors', ['#4285f4', '#34a853', '#fbbc04'])
    .build();

  sheet.insertChart(chart);
}

function createStaffComparisonChart() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var dashboard = ss.getSheetByName('Dashboard');
  if (!dashboard) return;

  dashboard.getCharts().forEach(function(c) { dashboard.removeChart(c); });

  var props = PropertiesService.getScriptProperties();
  var startRow = parseInt(props.getProperty('staffChartStart'));
  var endRow = parseInt(props.getProperty('staffChartEnd'));
  if (!startRow || !endRow || endRow < startRow) return;

  // Use Staff name (col B) and Capture Rate (col E)
  var nameRange = dashboard.getRange('B' + (startRow - 1) + ':B' + endRow);
  var rateRange = dashboard.getRange('E' + (startRow - 1) + ':E' + endRow);

  var chart = dashboard.newChart()
    .setChartType(Charts.ChartType.BAR)
    .addRange(nameRange)
    .addRange(rateRange)
    .setNumHeaders(1)
    .setPosition(startRow, 6, 0, 0)
    .setOption('title', 'Staff Capture Rates')
    .setOption('width', 400)
    .setOption('height', Math.max(250, (endRow - startRow + 1) * 30 + 80))
    .setOption('legend', { position: 'none' })
    .setOption('colors', ['#4285f4'])
    .setOption('hAxis', { title: 'Capture Rate', format: '#%' })
    .setOption('vAxis', { title: '' })
    .build();

  dashboard.insertChart(chart);
}

// ====================
// UTILITY FUNCTIONS
// ====================

function createNewPeriodSheet() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var periodSheets = getAllPeriodSheets();
  var nextPeriod = periodSheets.length > 0 ?
    Math.max.apply(null, periodSheets.map(function(ps) { return ps.period; })) + 1 : 1;

  var newSheet = ss.insertSheet('Period ' + nextPeriod);

  var headers = ['Staff', 'Total Tests', 'Purchases', 'Capture Rate', 'Patient ID',
    'Appointment Type', 'Appointment Date', 'Period Week Day', 'Notes',
    'Outcome Notes', 'Has Purchase', 'Appointment Link'];
  newSheet.getRange('A1:L1').setValues([headers])
    .setBackground('#4285f4').setFontColor('#ffffff').setFontWeight('bold');
  newSheet.setFrozenRows(1);

  SpreadsheetApp.getUi().alert('Created "Period ' + nextPeriod + '" sheet.');
}

function formatAllSheets() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var periodSheets = getAllPeriodSheets();

  periodSheets.forEach(function(ps) {
    var sheet = ps.sheet;
    var lastCol = sheet.getLastColumn();
    if (lastCol > 0) {
      sheet.setFrozenRows(1);
      sheet.getRange(1, 1, 1, lastCol)
        .setBackground('#4285f4').setFontColor('#ffffff').setFontWeight('bold');
      sheet.autoResizeColumns(1, lastCol);
    }
  });

  SpreadsheetApp.getUi().alert('Formatted all period sheets!');
}

// ====================
// INSTRUCTIONS SHEET
// ====================

function createInstructionsSheet() {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var sheet = ss.getSheetByName('Instructions');

  if (sheet) {
    sheet.clear();
    sheet.clearFormats();
  } else {
    sheet = ss.insertSheet('Instructions', 0);
  }

  sheet.setTabColor('#FFA500');
  var row = 1;

  sheet.getRange('A' + row + ':H' + row).merge()
    .setValue('CAPTURE RATE TRACKER - USER GUIDE')
    .setBackground('#1a73e8').setFontColor('#ffffff')
    .setFontSize(18).setFontWeight('bold')
    .setHorizontalAlignment('center').setVerticalAlignment('middle');
  sheet.setRowHeight(row, 50);
  row += 2;

  sheet.getRange('A' + row).setValue('Welcome to the Capture Rate Tracker!');
  sheet.getRange('A' + row + ':H' + row).merge()
    .setBackground('#e8f0fe').setFontSize(14).setFontWeight('bold');
  row++;

  sheet.getRange('A' + row).setValue(
    'This tool helps you analyze hearing test performance, track capture rates over time, and identify top performers.');
  sheet.getRange('A' + row + ':H' + row).merge().setWrap(true);
  row += 2;

  addSectionHeader(sheet, row, 'GETTING STARTED');
  row++;

  var steps = [
    ['Step 1:', 'Run your bookmarklet on costco.sycle.net to generate hearing test data'],
    ['Step 2:', 'Export CSV file(s) — single or batch (ZIP)'],
    ['Step 3:', 'Open "Capture Rate Tracker" menu → "Import CSV / ZIP"'],
    ['Step 4:', 'Drop your files into the sidebar — sheets are created automatically'],
    ['Step 5:', 'Click "Refresh Dashboard" to update all analytics'],
    ['', 'Your dashboard, charts, and analytics will automatically update!']
  ];

  sheet.getRange('A' + row + ':B' + (row + steps.length - 1))
    .setValues(steps).setBorder(true, true, true, true, true, true).setWrap(true);
  sheet.getRange('A' + row + ':A' + (row + steps.length - 1))
    .setFontWeight('bold').setBackground('#fff3e0');
  row += steps.length + 2;

  addSectionHeader(sheet, row, 'SHEET STRUCTURE');
  row++;

  var structure = [
    ['Sheet Name', 'Purpose', 'How to Use'],
    ['Period 1, Period 2, etc.', 'Raw CSV data for each period', 'Created automatically during import'],
    ['Dashboard', 'Executive summary view', 'Overall performance, period trends, staff rankings'],
    ['Staff Performance', 'Individual staff analysis', 'Detailed metrics with trends for each staff member'],
    ['Trends', 'Historical analysis', 'Capture rate trends and appointment type breakdowns'],
    ['Instructions', 'This guide', 'Reference whenever you need help']
  ];

  sheet.getRange('A' + row + ':C' + (row + structure.length - 1))
    .setValues(structure).setBorder(true, true, true, true, true, true);
  sheet.getRange('A' + row + ':C' + row)
    .setBackground('#e8eaed').setFontWeight('bold').setHorizontalAlignment('center');

  sheet.setColumnWidth(1, 200);
  sheet.setColumnWidth(2, 350);
  sheet.setColumnWidth(3, 250);
  sheet.setFrozenRows(1);

  SpreadsheetApp.getUi().alert('Instructions sheet created!');
}

function addSectionHeader(sheet, row, text) {
  sheet.getRange('A' + row + ':H' + row).merge()
    .setValue(text).setBackground('#f1f3f4')
    .setFontWeight('bold').setFontSize(12).setHorizontalAlignment('left');
  sheet.setRowHeight(row, 30);
}
