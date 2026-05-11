const http = require('http');
const fs = require('fs');
const path = require('path');

// resourcesUrl = "http://88.200.63.148/~domen/braid/samples/";
resourcesUrl = "http://osebje.famnit.upr.si/~domen.soberl/braid/samples/";
serverPort = 8989;

let csvData = null;
let csvPulses = null;
let savedContent = null;
let savedFilesProcessing = 0;

function addZero(i) {
  if (i < 10) {i = "0" + i}
  return i;
}

function log(msg) {
    let now = new Date();

    let year = now.getFullYear();
    let month = now.getMonth() + 1;
    let day = now.getDate();
    let hours = addZero(now.getHours());
    let minutes = addZero(now.getMinutes());
    let seconds = addZero(now.getSeconds());

    console.log(`[${day}.${month}.${year} ${hours}:${minutes}:${seconds}] ${msg}`);

}

function processCSV(data, dbName) {
    log(`Processing CSV ${dbName}.`)

    if (csvData == null)
        csvData = {};

    if (csvPulses == null)
        csvPulses = {};

    csvData[dbName] = data.trim().split('\n').map(row => row.split(','));

    for (let row in csvData[dbName]) {
        let photo_id = csvData[dbName][row][0].toString();

        csvPulses[photo_id] = [];
        for (let i = 1304; i < 2604; i++) {
            if (csvData[dbName][row][i] > 0)
                csvPulses[photo_id].push(i - 1304);
        }
    }

    log(`CSV ${dbName} processed.`);
}

function loadCSV(location, dbName) {
    url = `${location}${dbName}.csv`;
    log(`Fetching ${url}`);

    http.get(url,
        csv_response => {
            let data = '';
            log(`Loading CSV ${dbName}.`);
            
            let nextSize = 1024 * 1024;
            csv_response.on('data', chunk => { 
                data += chunk;
                if (data.length >= nextSize) {
                    log(`Received for ${dbName}: ${data.length / 1024} KB.`);
                    nextSize += 1024 * 1024;
                }
            });

            csv_response.on('end', () => {
                log(`Received for ${dbName}: ${data.length / 1024} KB.`);
                log(`CSV ${dbName} loaded.`)
                processCSV(data, dbName);
            })
        }
    );
}

function loadSavedContent() {
    log("Loading saved content.");
    savedContent = {};
    savedFilesProcessing = 0;

    const files = fs.readdirSync(".");
    const csvFiles = files.filter(file => path.extname(file).toLowerCase() === '.csv');

    for (let i in csvFiles) {
        filename = csvFiles[i];
        if (!filename.startsWith("saved"))
            continue;

        let dbName = filename.split("_")[1];

        fs.readFile(csvFiles[i], "utf8", (err, data) => {
            if (err) throw err;

            if (!(dbName in savedContent))
                savedContent[dbName] = {};
            savedContent[dbName][csvFiles[i]] = data.split("|");
            
            savedFilesProcessing--;
            if (savedFilesProcessing == 0) {
                log('Saved content loaded.');
            }
        });
        
        savedFilesProcessing++;
    }
}

function exportData() {
    let export_json = {};

    for (let dbname in csvData) {
        console.log(`Exporting database ${dbname}.`);
        cnt_adjusted = 0;

        for (let i in csvData[dbname]) {
            record = csvData[dbname][i];
            
            let photo_id = record[0];
            let groups_detected = record[1];
            let groups_weighed = record[2];
            let groups_final = record[3];

            let record_json = {
                'database': dbname,
                'detected_groups': groups_detected,
                'weighed_groups': groups_weighed,
                'final_groups': groups_final,
                'keep': false,
                'adjusted': false,
                'comment': '',
                'siwim_pulses': [],
                'ai_pulses': [],
                'expert_pulses': [],
                'adjusted_pulses': []
            };

            for (let i = 1304; i < 2604; i++) {
                if (record[i] > 0)
                    record_json['siwim_pulses'].push(i - 1304);
            }

            for (let i = 2604; i < 3904; i++) {
                if (record[i] >= 0.2)
                    record_json['ai_pulses'].push(i - 2604);
            }

            let filename = `saved_${dbname}_${photo_id}.csv`;
            let saved = savedContent[dbname][filename];

            record_json['keep'] = (saved[0] == 'true');
            record_json['comment'] = saved[1];

            for (let i = 2; i < saved.length; i++) {
                record_json['expert_pulses'].push(parseInt(saved[i]));
            }

            /* Janov algoritem. */
            record_json['adjusted_pulses'] = Array.from(record_json['expert_pulses']);
            if ((groups_detected == '1111' && groups_final == '1211') || (groups_detected == '112' && groups_final == '122')) {
                let distance = record_json['siwim_pulses'][2] - record_json['siwim_pulses'][1];
                if (distance >= 26 && distance <= 28) {
                    /* Have the axles been changed manually? */
                    let changed = record_json['siwim_pulses'].join() != record_json['expert_pulses'].join();
                    for (let i = 0; i < record_json['siwim_pulses'].length && !changed; i++) {
                        if (record_json['siwim_pulses'][i] != record_json['expert_pulses'][i])
                            changed = true;
                    }

                    if (!changed && record_json['siwim_pulses'].length == record_json['ai_pulses'].length) {
                        record_json['adjusted'] = true;
                        record_json['adjusted_pulses'][2] = record_json['ai_pulses'][2];
                        cnt_adjusted++;
                    }
                }
            }
            export_json[photo_id] = record_json;
        }
        console.log(`Adjusted ${cnt_adjusted} cases.`);
    }

    console.log("Exporting done.")

    const file_data = JSON.stringify(export_json, null, 2);

    fs.writeFile("ground_truth.json", file_data, 'utf8', (err) => {
        if (err) {
            console.error('Error writing to file', err);
        } else {
            console.log(`File ground_truth.json saved.`);
        }
    });
}

function sendPlotPage(response, dbName, number = null) {
    if (csvData == null) {
        response.writeHead(200, {'Content-Type': 'text/html'});
        response.write("No data");
        response.end();
        return;
    }

    /* Find first that is not checked. */
    if (number == null) {
        if (dbName in savedContent) {
            let row = 0;
            let found = false;
            while (!found && row < csvData[dbName].length) {
                let filename = `saved_${dbName}_${csvData[dbName][row][0]}.csv`;
                if (!(filename in savedContent[dbName])) {
                    found = true;
                    continue;
                }
                row += 1;
            }

            if (found)
                number = row + 1;
            else
                number = csvData[dbName].length;
        }
        else {
            number = 1;
        }
    }

    /* Display the plot page. */
    number = parseInt(number);
    if (isNaN(number)) number = 1;

    if (number < 1) number = 1;
    if (number > csvData[dbName].length) number = csvData[dbName].length;

    row = number - 1;

    response.writeHead(200, {'Content-Type': 'text/html'});

    response.write("<!DOCTYPE html>\n");
    response.write("<html>\n");
    response.write("<head>\n");
    response.write("<meta charset=\"UTF-8\">\n");
    response.write("<script src=\"plots.js\"></script>\n");
    response.write("<title>BRAID time series check</title>\n");
    response.write("</head>\n");
    response.write("<body>\n");
    
    response.write("<style>\n");
    response.write("a { color: white; }\n");
    response.write(".topbar { position: fixed; top: 0; left: 0; width: 100%; height: 24px; font-weight:bold; font-size: 20px; color:white; background: black; padding: 10px; }\n");
    response.write(".content { margin-top: 40px; }\n");
    response.write(".legend { display:inline-block; width:12px; height:12px; background-color:blue; margin:0 4px; }\n");
    response.write("</style>\n");

    response.write(`<div class="topbar">\n`);
    response.write(`<span><a href="/index.html?db=${dbName}">Indeks</a> &nbsp; | &nbsp; Baza: ${dbName} &nbsp; | &nbsp; Primer:\n`);

    if (number > 1)
        response.write(`<button style="font-weight: bold; width:40px;" onclick="location.href='/plot.html?db=${dbName}&number=${number - 1}'">&#x25C0;</button>\n`);
    else
        response.write(`<button style="font-weight: bold; width:40px;" disabled>&#x25C0;</button>\n`);
    
    response.write(`<td><input id="rownum" value="${number}" onchange="location.href='/plot.html?db=${dbName}&number='+this.value" style="width:30px; text-align:center;"></td>\n`);
    
    if (number < csvData[dbName].length)
        response.write(`<button style="font-weight: bold; width:40px;" onclick="location.href='/plot.html?db=${dbName}&number=${number + 1}'">&#x25B6;</button>\n`);
    else
        response.write(`<button style="font-weight: bold; width:40px;" disabled>&#x25B6;</button>\n`);
    
    response.write(`</span>&nbsp;|&nbsp;\n`);
    response.write(`<span id="statusbar" style="display:inline-block;">Nalaganje strani ... počakajte.</span>\n`);
    response.write("</div>\n");

    response.write(`<div class="content">\n`);

    let photo_id = csvData[dbName][row][0];
    let groups_detected = csvData[dbName][row][1];
    let groups_weighed = csvData[dbName][row][2];
    let groups_final = csvData[dbName][row][3];
    let pulses = csvPulses[photo_id];

    response.write(`<div id="area">\n`);
    response.write(`<input type="hidden" id="checked" value="0">\n`);
    response.write("<hr>\n");

    response.write("<script>\n");
    
    response.write(`let points = [`);
    for (let i = 4; i < 1304; i++) {
        point = csvData[dbName][row][i];
        if (i > 4)
            response.write(`,${point}`);
        else
            response.write(`${point}`);
    }
    response.write("];\n");

    response.write(`let predictions = [`);
    for (let i = 2604; i < 3904; i++) {
        point = csvData[dbName][row][i];
        if (i > 2604)
            response.write(`,${point}`);
        else
            response.write(`${point}`);
    }
    response.write("];\n");

    response.write("</script>\n");

    response.write("<table>\n");
    response.write("<tr>\n");
    response.write(`<td><img src='${resourcesUrl}${photo_id}.png'></td>\n`);
    response.write(`<td style="vertical-align: bottom;">\n`);
    response.write(`<b>Photo ID: ${photo_id}</b><br>\n`);
    response.write(`Detected groups: ${groups_detected}<br>\n`);
    response.write(`Weighed groups: ${groups_weighed}<br>\n`);
    response.write(`Final groups: ${groups_final}<br><br>\n`);
    response.write(`<div id="status" style="font-weight: bold; color: black;">SE NALAGA</div><br>\n`);
    response.write(`<input id="discard" type="checkbox" onchange="discard_changed();"><label for="discard" style="font-weight: bold;">Osi ni možno določiti / okvarjen signal.</label><br><br>`);
    response.write(`<span class="legend" style="background-color:blue"></span> &mdash; signal (11admp) &nbsp; &nbsp; &nbsp;`);
    response.write(`<span class="legend" style="background-color:green"></span> &mdash; Ground truth (potrdi oz. popravi) &nbsp; &nbsp; &nbsp;`);
    response.write(`<span class="legend" style="background-color:red"></span> &mdash; AI osi (informativno)`);
    response.write("</td>\n");
    response.write("</tr>\n");
    response.write("</table>\n");
    
    response.write(`<canvas id="canvas" width="1300" height="200"></canvas>\n`);
    
    response.write("<table>\n");
    response.write("<tr>\n");

    let input_cnt = 0;
    for (let i in pulses) {
        let pulse = pulses[i];
        response.write(`<td><input id="pulse_${i}" type="number" value="${pulse}" readonly disabled style="width:50px;"></td>\n`);
        input_cnt++;
    }
    for (let i = pulses.length; i < pulses.length + 3; i++) {
        response.write(`<td><input id="pulse_${i}" type="number" value="0" readonly disabled style="width:50px;"></td>\n`);
        input_cnt++;
    }
    response.write(`<td></td>\n`);
    response.write("</tr>\n");

    response.write("<tr>\n");
    for (let i = 0; i < input_cnt; i++) {
        response.write(`<td><input id="axle_${i}" disabled type="number" value="0" min="0" max="1300" step="1" onchange="axles_changed(points);" style="width:50px;"></td>\n`);
    }
    
    response.write(`<td><button id="delete" onclick="delete_axles('${dbName}', ${photo_id});" style="width:100px;">Ponastavi</button></td>\n`);
    response.write("</tr>\n");

    response.write("<tr>\n");
    response.write(`<td colspan="${input_cnt + 1}"><input id="comment" placeholder="Opombe" oninput="comment_changed();" style="width:100%; box-sizing: border-box;"></td>\n`);
    response.write("</tr>\n");

    response.write("<tr>\n");
    response.write(`<td colspan="${input_cnt + 1}"></td>\n`);
    response.write("</tr>\n");

    response.write("<tr>\n");
    response.write(`<td colspan="${input_cnt + 1}"><button id="save" onclick="save_axles('${dbName}', ${photo_id}, null);" disabled style="width:100px;">Potrdi</button>&nbsp;`);
    response.write(`<button id="save_next" onclick="save_axles('${dbName}', ${photo_id}, ${number + 1});" disabled style="width:150px;">Potrdi in naprej</button>&nbsp;`);
    response.write(`<button id="correct_last_next" onclick="correct_last_axle('${dbName}', ${photo_id}, ${number + 1});" style="width:150px;">Popravi zadnjo</button></td>\n`);
    response.write("<tr>\n");
    response.write("</table>\n");
    response.write("</div>\n");

    response.write(`</div>\n`);

    response.write("<script>\n");

    response.write(`let grabx = null;`);
    response.write(`let graby = null;`);
    response.write(`let heldAxle = null;`);
    response.write(`canvas.addEventListener("mousedown", plot_mousedown);`);
    response.write(`canvas.addEventListener("mouseup", plot_mouseup);`);
    response.write(`canvas.addEventListener("mousemove", plot_mousemove);`);
    response.write(`canvas.addEventListener("mouseleave", plot_mouseup);`);

    response.write("window.addEventListener('load', () => {\n");
    response.write(`update_axles('${dbName}', ${photo_id});\n`);
    response.write("});\n");
    response.write("</script>\n");

    response.write("<hr>\n");
    response.write(`<div style="font-size:12px"><b>Brisanje osi:</b> položaj osi nastavimo na 0. &nbsp;`);
    response.write(`<b>Dodajanje osi:</b> prepišemo eno od ničel, osi se razvrstijo samodejno.</div>\n`);

    response.write("</body>\n");
    response.write("</html>\n");

    response.end();
    log(`Sent plot page for ${dbName}/${number}.`);
}

function sendIndexPage(response, dbName) {
    if (csvData == null) {
        response.writeHead(200, {'Content-Type': 'text/html'});
        response.write("No data");
        response.end();
        return;
    }

    let cnt_saved = 0;

    if (savedContent != null && dbName in savedContent)
        cnt_saved = Object.keys(savedContent[dbName]).length;
    let cnt_all = csvData[dbName].length;

    response.writeHead(200, {'Content-Type': 'text/html'});

    response.write("<!DOCTYPE html>\n");
    response.write("<html>\n");
    response.write("<head>\n");
    response.write("<meta charset=\"UTF-8\">\n");
    response.write("<script src=\"plots.js\"></script>");
    response.write("<title>BRAID time series check</title>\n");
    response.write("</head>\n");
    response.write("<body>\n");
    
    response.write("<style>");
    response.write(".topbar { position: fixed; top: 0; left: 0; width: 100%; height: 24px; font-weight:bold; font-size: 20px; color:white; background: black; padding: 10px; }");
    response.write(".content { margin-top: 40px; padding: 30px; }");
    response.write("table { border-collapse: collapse; }");
    response.write("table, th, td { border: 1px solid black; }");
    response.write("th, td { padding: 5px; }");
    response.write("th { color: white; background-color: black; }");
    response.write("</style>");

    response.write(`<div class="topbar">\n`);
    response.write(`<span id="statusbar" style="display:inline-block;">Baza: &nbsp;`);
    response.write(`<select id="dbselect" style="width:150px" onchange="change_db();">\n`);
    
    if (dbName == 'validation')
        response.write(`<option selected>BAZA 1</option>\n`);
    else
        response.write(`<option>BAZA 1</option>\n`);
    
    if (dbName == 'correct')
        response.write(`<option selected>BAZA 2</option>\n`);
    else
        response.write(`<option>BAZA 2</option>\n`);

    if (dbName == 'fixed')
        response.write(`<option selected>BAZA 3</option>\n`);
    else
        response.write(`<option>BAZA 3</option>\n`);

    response.write(`</select>\n`);
    response.write(`&nbsp; | Pregledanih primerov: ${cnt_saved} / ${cnt_all}</span>\n`);
    response.write("</div>\n");

    response.write(`<div class="content" style="display: flex; justify-content: center;">\n`);
    
    response.write("<table>\n");
    response.write("<tr>\n");
    response.write('<th style="width: 50px">#</th>\n');
    response.write('<th style="width: 120px">Photo ID</th>\n');
    response.write('<th style="width: 150px">Status</th>\n');
    response.write('<th style="width: 600px">Opombe</th>\n');
    response.write("</tr>\n");

    for (let row in csvData[dbName]) {
        let rowData = csvData[dbName][row];
        let number = parseInt(row) + 1;
        
        let filename = `saved_${dbName}_${rowData[0]}.csv`;
        
        let savedData = null;
        if (dbName in savedContent && filename in savedContent[dbName]) {
            savedData = savedContent[dbName][filename];
        }

        let checked = (savedData != null && savedData[0] == 'true');
        let removed = (savedData != null && savedData[0] == 'false');

        if (checked)
            response.write(`<tr style="background-color: #abdbba">\n`);
        else if (removed)
            response.write(`<tr style="background-color: #dbabab">\n`);
        else
            response.write(`<tr>\n`);

        response.write(`<td style="text-align: center">${number}</td>\n`);
        response.write(`<td style="text-align: center"><a href="/plot.html?db=${dbName}&number=${number}">${rowData[0]}</a></td>\n`);
        
        if (checked)
            response.write(`<td style="text-align: center">pregledano</td>\n`);
        else if (removed)
            response.write(`<td style="text-align: center">izločeno</td>\n`);
        else
            response.write(`<td style="text-align: center">ni pregledano</td>\n`);

        if (savedData != null)
            response.write(`<td>${savedData[1]}</td>\n`);
        else
            response.write(`<td></td>\n`);

        response.write("</tr>\n");
    }
    
    response.write("</table>\n");

    response.write(`</div>\n`);
    response.write("</body>\n");
    response.write("</html>\n");

    response.end();
    log(`Sent index page for ${dbName}.`);
}

function sendIntroPage(response) {
    response.writeHead(200, {'Content-Type': 'text/html'});
    response.write("<!DOCTYPE html>\n");
    response.write("<html>\n");
    response.write("<head>\n");
    response.write("<meta charset=\"UTF-8\">\n");
    response.write("</head>\n");
    response.write("<body>\n");

    response.write(`<div>\n`);
    response.write(`<h1>BRAID preverjanje časovnih vrst</h1>\n`);

    response.write(`<p>\n`);
    response.write(`<a href="/index.html?db=validation">BAZA 1 (306 primerov)</a>\n`);
    response.write(`</p>\n`);

    response.write(`<a href="/index.html?db=correct">BAZA 2 (188 primerov)</a>\n`);
    response.write(`</p>\n`);

    response.write(`<a href="/index.html?db=fixed">BAZA 3 (1685 primerov)</a>\n`);
    response.write(`</p>\n`);

    response.write(`</div>\n`);

    response.write("</body>\n");
    response.write("</html>\n");
    response.end();

    log(`Sent intro page.`);
}

function redirectBrowser(response) {
    response.writeHead(200, {'Content-Type': 'text/html'});

    response.write("<!DOCTYPE html>\n");
    response.write("<html>\n");
    response.write("<head>\n");
    response.write(`<script>\n`);
    response.write(`window.location.href = "/index.html";\n`);
    response.write(`</script>\n`);
    response.write("</head>\n");
    response.write("<body>\n");
    response.write("</body>\n");
    response.write("</html>\n");
    response.end();

    log(`Redirected to index.html`);
}

function handleRequest(request, response) {
    const urlData = new URL(request.url, `http://${request.headers.host}`);   
    let parameters = urlData.searchParams;

    if (urlData.pathname == "/plots.js") {
        fs.readFile('plots.js',
            function(error, data) {
                if (error) {
                    response.writeHead(404, {'Content-Type': 'text/plain'});
                    response.write("Error 404: " + urlData.pathname + " not found!");
                    response.end();
                }
                else {
                    response.writeHead(200, {'Content-Type': 'text/javascript'});
                    response.write(data);
                    response.end();
                }
            }
        );
    }

    else if (urlData.pathname == "/load") {
        let dbName = parameters.get('db');
        let photo_id = parameters.get('id');

        let filename = `saved_${dbName}_${photo_id}.csv`;
        fs.access(filename, fs.constants.F_OK, (err) => {
            if (err) {
                /* No save file, send original data. */
                response.writeHead(200, {'Content-Type': 'text/plain'});
                response.write(`|${photo_id}|false|true|`);
                for (i in csvPulses[photo_id.toString()]) {
                    response.write(`|${csvPulses[photo_id.toString()][i]}`);
                }
                response.end();

                if (dbName in savedContent && filename in savedContent[dbName])
                    delete savedContent[dbName][filename];

                return;
            }

            fs.stat(filename, (err, stats) => {
                if (err) throw err;

                fs.readFile(filename, "utf8", (err, data) => {
                    if (err) throw err;
                    let values = data.split("|");
                    response.writeHead(200, {'Content-Type': 'text/plain'});
                    response.write(stats.mtime.toLocaleString("sl-SI"));
                    response.write(`|${photo_id}|true|${values[0]}|${values[1]}`);
                    for (let i = 2; i < values.length; i++) {
                        response.write(`|${values[i]}`);
                    }
                    response.end();

                    if (!(dbName in savedContent))
                        savedContent[dbName] = {};
                    savedContent[dbName][filename] = values;
                });
            });
        });
    }

    else if (urlData.pathname == "/save") {
        let dbName = parameters.get('db');
        let photo_id = parameters.get('id');
        let keep = parameters.get('keep') == 'true';
        let comment = parameters.get('comment');

        let axleNum = 0;
        let axleFound = true;

        let axles = []
        while (axleFound) {
            if (!parameters.has('axle' + axleNum)) {
                axleFound = false;
                continue;
            }
            
            let value = parseInt(parameters.get('axle' + axleNum));
            if (value > 0) axles.push(value);
            axleNum++;
        }
        axles.sort(function(a, b){return a - b});

        let output = "";
        if (keep)
            output += `true|`;
        else
            output += `false|`;

        output += `${comment}|`;

        for (let i in axles) {
            if (i > 0) output += "|";
            output += axles[i].toString();
        }
        
        log(`Saving axles for ${dbName}/${photo_id}.`);

        let filename = `saved_${dbName}_${photo_id}.csv`;
        fs.writeFile(filename, output, (err) => {
            if (err) {
                console.error("Error writing file:", err);
                return;
            }

            if (!(dbName in savedContent))
                savedContent[dbName] = {};
            savedContent[dbName][filename] = output.split("|");
        });

        response.writeHead(200, {'Content-Type': 'text/plain'});
        response.write(`${photo_id}`);
        response.end();
    }

    else if (urlData.pathname == "/delete") {
        let dbName = parameters.get('db');
        let photo_id = parameters.get('id');
        
        let filename = `saved_${dbName}_${photo_id}.csv`;
        fs.rm(filename, (err) => {
            if (err) return;
            log(`Removed saved axles for ${dbName}/${photo_id}.`);
        });

        if (dbName in savedContent && filename in savedContent[dbName])
            delete savedContent[dbName][filename];

        response.writeHead(200, {'Content-Type': 'text/plain'});
        response.write(`${photo_id}`);
        response.end();
    }

    else if (urlData.pathname == "/status") {
        let dbName = parameters.get('db');

        let cnt_saved = 0;
        if (dbName in savedContent)
            cnt_saved = Object.keys(savedContent[dbName]).length;
        let cnt_all = csvData[dbName].length;

        response.writeHead(200, {'Content-Type': 'text/plain'});
        response.write(`Pregledanih primerov: ${cnt_saved} / ${cnt_all}`);
        response.end();
    }

    else if (urlData.pathname == "/plot.html") {
        if (!parameters.has("db")) {
            response.writeHead(200, {'Content-Type': 'text/plain'});
            response.write("No database specified.");
            response.end();
            return;
        }

        let dbName = parameters.get("db");

        if (parameters.has("number")) {
            sendPlotPage(response, dbName, parseInt(parameters.get("number")));
        }
        else {
            sendPlotPage(response, dbName);
        }
    }

    else if (urlData.pathname == "/index.html") {
        if (!parameters.has("db")) {
            sendIntroPage(response);
            return;
        }

        let dbName = parameters.get("db");
        sendIndexPage(response, dbName);
    }

    else if (urlData.pathname == "/export") {
        response.writeHead(200, {'Content-Type': 'text/plain'});
        response.write("Exporting database.");
        response.end();

        exportData();
    }

    else if (urlData.pathname == "/") {
        redirectBrowser(response);
    }

    else {
        response.writeHead(404);
        response.end();
    }
}

loadCSV(resourcesUrl, 'validation');
loadCSV(resourcesUrl, 'correct');
loadCSV(resourcesUrl, 'fixed');
loadSavedContent();

let server = http.createServer(handleRequest);
server.listen(serverPort);
log(`Listening on port ${serverPort}.`);
