const http = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');

resources_url = "https://osebje.famnit.upr.si/~domen.soberl/braid/validation/";

let csv_rows = null;
let original_pulses = {};

function loadCSV(url) {
    console.log(`Fetching ${url}`)

    https.get(url,
        csv_response => {
            data = '';
            console.log('Loading CSV.');
            
            let nextSize = 1024 * 1024;
            csv_response.on('data', chunk => { 
                data += chunk;
                if (data.length >= nextSize) {
                    console.log(`Received ${data.length / 1024} KB.`);
                    nextSize += 1024 * 1024;
                }
            });

            csv_response.on('end', () => {
                console.log(`Received ${data.length / 1024} KB.`);
                console.log("CSV loaded.")
                csv_rows = data.trim().split('\n').map(row => row.split(','));
                console.log("Data ready.")
            })
        }
    );
}

function build_page(response, number = null) {
    if (csv_rows == null) {
        response.writeHead(200, {'Content-Type': 'text/html'});
        response.write("No data");
        response.end();
        return;
    }

    /* Find first that is not checked. */
    if (number == null) {
        const files = fs.readdirSync(".");
        const csvFiles = files.filter(file => path.extname(file).toLowerCase() === '.csv');

        let row = 0;
        let found = false;
        while (!found && row < csv_rows.length) {
            let filename = `saved_${csv_rows[row][0]}.csv`;
            if (!csvFiles.includes(filename)) {
                found = true;
                continue;
            }
            row += 1;
        }

        if (found)
            number = row + 1;
        else
            number = csv_rows.length;
    }

    number = parseInt(number);
    if (isNaN(number)) number = 1;

    if (number < 1) number = 1;
    if (number > csv_rows.length) number = csv_rows.length;

    row = number - 1;

    response.writeHead(200, {'Content-Type': 'text/html'});

    response.write("<!DOCTYPE html>\n");
    response.write("<html>\n");
    response.write("<head>\n");
    response.write("<meta charset=\"UTF-8\">\n");
    response.write("<script src=\"plots.js\"></script>");
    response.write("<title>BRAID validation set</title>\n");
    response.write("</head>\n");
    response.write("<body>\n");
    
    response.write("<style>");
    response.write(".topbar { position: fixed; top: 0; left: 0; width: 100%; height: 24px; font-weight:bold; font-size: 20px; color:white; background: black; padding: 10px; }");
    response.write(".content { margin-top: 40px; }");
    response.write("</style>");

    response.write(`<div class="topbar">\n`);
    response.write(`<span>Primer:\n`);

    if (number > 1)
        response.write(`<button style="font-weight: bold; width:40px;" onclick="location.href='/index?number=${number - 1}'">&#x25C0;</button>\n`);
    else
        response.write(`<button style="font-weight: bold; width:40px;" disabled>&#x25C0;</button>\n`);
    
    response.write(`<td><input id="rownum" value="${number}" onchange="location.href='/index?number='+this.value" style="width:30px; text-align:center;"></td>\n`);
    
    if (number < csv_rows.length)
        response.write(`<button style="font-weight: bold; width:40px;" onclick="location.href='/index?number=${number + 1}'">&#x25B6;</button>\n`);
    else
        response.write(`<button style="font-weight: bold; width:40px;" disabled>&#x25B6;</button>\n`);
    
    response.write(`</span>&nbsp;|&nbsp;\n`);
    response.write(`<span id="statusbar" style="display:inline-block;">Nalaganje strani ... počakajte.</span>\n`);
    response.write("</div>\n");

    response.write("<script>\n");
    response.write("let points = {};");
    response.write("</script>\n");

    response.write(`<div class="content">\n`);

    /* Start row */

    let photo_id = csv_rows[row][0];

    groups_detected = csv_rows[row][1];
    groups_weighed = csv_rows[row][2];
    groups_final = csv_rows[row][3];

    let pulses = [];
    for (let i = 1304; i < 2604; i++) {
        if (csv_rows[row][i] > 0)
            pulses.push(i - 1304);
    }

    /* Save the pulses globally. */
    original_pulses[photo_id.toString()] = pulses;

    response.write(`<div id="area_${photo_id}">\n`);
    response.write(`<input type="hidden" id="checked_${photo_id}" value="0">\n`);
    response.write("<hr>\n");

    response.write("<script>\n");
    response.write(`let points_${photo_id} = [`);
    for (let i = 4; i < 1304; i++) {
        point = csv_rows[row][i];
        if (i > 4)
            response.write(`,${point}`);
        else
            response.write(`${point}`);
    }
    response.write("]\n");
    response.write(`points['${photo_id}'] = points_${photo_id}\n`);
    response.write("</script>\n");

    response.write("<table>\n");
    response.write("<tr>\n");
    response.write(`<td><img src='${resources_url}${photo_id}.png'></td>\n`);
    response.write(`<td style="vertical-align: bottom;">\n`);
    response.write(`<b>Photo ID: ${photo_id}</b><br>\n`);
    response.write(`Detected groups: ${groups_detected}<br>\n`);
    response.write(`Weighed groups: ${groups_weighed}<br>\n`);
    response.write(`Final groups: ${groups_final}<br><br>\n`);
    response.write(`<div id="status_${photo_id}" style="font-weight: bold; color: black;">SE NALAGA</div><br>\n`);
    response.write(`<button id="discard_${photo_id}" onclick="save_axles(${photo_id}, false);" style="width:100px; margin-bottom: 5px;">Izloči</button>&nbsp;`);
    response.write(`<button id="delete_${photo_id}" onclick="delete_axles(${photo_id});" disabled style="width:100px;">Ponastavi</button>`);
    response.write("</td>\n");
    response.write("</tr>\n");
    response.write("</table>\n");
    
    response.write(`<canvas id="canvas_${photo_id}" width="1300" height="200"></canvas>\n`);
    
    response.write("<table>\n");
    response.write("<tr>\n");

    let input_cnt = 0;
    for (let i in pulses) {
        let pulse = pulses[i];
        response.write(`<td><input id="pulse_${photo_id}_${i}" type="number" value="${pulse}" readonly disabled style="width:50px;"></td>\n`);
        input_cnt++;
    }
    for (let i = pulses.length; i < pulses.length + 3; i++) {
        response.write(`<td><input id="pulse_${photo_id}_${i}" type="number" value="0" readonly disabled style="width:50px;"></td>\n`);
        input_cnt++;
    }
    response.write("</tr>\n");

    response.write("<tr>\n");
    for (let i = 0; i < input_cnt; i++) {
        response.write(`<td><input id="axle_${photo_id}_${i}" disabled type="number" value="0" min="0" max="1300" step="1" onchange="axles_changed(${photo_id}, points_${photo_id});" style="width:50px;"></td>\n`);
    }

    response.write("</tr>\n");
    response.write("<tr>\n");
    response.write(`<td colspan="${input_cnt}"><input id="comment_${photo_id}" placeholder="Opombe" oninput="comment_changed(${photo_id})" style="width:100%; box-sizing: border-box;"></td>\n`);
    response.write("</tr>\n");
    response.write("<tr>\n");
    response.write(`<td colspan="${input_cnt}"><button id="save_${photo_id}" onclick="save_axles(${photo_id}, true, null);" disabled style="width:100px;">Potrdi</button>&nbsp;`);
    response.write(`<button id="save_next_${photo_id}" onclick="save_axles(${photo_id}, true, ${number + 1});" disabled style="width:150px;">Potrdi in naprej</button></td>\n`);
    response.write("<tr>\n");
    response.write("</table>\n");
    response.write("</div>\n");
    
    /* End row */

    response.write(`</div>\n`);

    response.write("<script>\n");
    response.write("window.addEventListener('load', () => {\n");
    response.write("update_all_axles();\n");
    response.write("});\n");
    response.write("</script>\n");

    response.write("<hr>\n");
    response.write(`<div style="font-size:12px"><b>Brisanje osi:</b> položaj osi nastavimo na 0. &nbsp;`);
    response.write(`<b>Dodajanje osi:</b> prepišemo eno od ničel, osi se razvrstijo samodejno.</div>\n`);

    response.write("</body>\n");
    response.write("</html>\n");

    response.end();
    console.log(`Sent page for number ${number}`);
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
        let photo_id = parameters.get('id');

        let filename = `saved_${photo_id}.csv`;
        fs.access(filename, fs.constants.F_OK, (err) => {
            if (err) {
                /* No save file, send original data. */
                response.writeHead(200, {'Content-Type': 'text/plain'});
                response.write(`${photo_id},false,true,`);
                for (i in original_pulses[photo_id.toString()]) {
                    response.write(`,${original_pulses[photo_id.toString()][i]}`);
                }
                response.end();
                return;
            }

            fs.readFile(filename, "utf8", (err, data) => {
                if (err) throw err;
                let values = data.split(",");
                response.writeHead(200, {'Content-Type': 'text/plain'});
                response.write(`${photo_id},true,${values[0]},${values[1]}`);
                for (let i = 2; i < values.length; i++) {
                    response.write(`,${values[i]}`);
                }
                response.end();
            });
        });
    }
    else if (urlData.pathname == "/save") {
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
            output += `true,`;
        else
            output += `false,`;

        output += `${comment},`;

        for (let i in axles) {
            if (i > 0) output += ",";
            output += axles[i].toString();
        }
        
        console.log(`Saving axles for ${photo_id}.`);

        fs.writeFile(`saved_${photo_id}.csv`, output, (err) => {
            if (err) {
                console.error("Error writing file:", err);
            }
        });

        response.writeHead(200, {'Content-Type': 'text/plain'});
        response.write(`${photo_id}`);
        response.end();
    }
    else if (urlData.pathname == "/delete") {
        let photo_id = parameters.get('id');
        
        fs.rm(`saved_${photo_id}.csv`, (err) => {
            if (err) return;
            console.log(`Removed saved axles for ${photo_id}.`);
        });

        response.writeHead(200, {'Content-Type': 'text/plain'});
        response.write(`${photo_id}`);
        response.end();
    }
    else if (urlData.pathname == "/status") {
        const files = fs.readdirSync(".");
        const csvFiles = files.filter(file => path.extname(file).toLowerCase() === '.csv');
        let cnt_saved = csvFiles.length;
        let cnt_all = csv_rows.length;

        response.writeHead(200, {'Content-Type': 'text/plain'});
        response.write(`Pregledanih primerov: ${cnt_saved} / ${cnt_all}`);
        response.end();
    }
    else if (urlData.pathname == "/favicon.ico") {
        response.writeHead(200, {'Content-Type': 'text/plain'});
        response.end();
    }
    else {
        if (parameters.has("number")) {
            build_page(response, parseInt(parameters.get("number")));
        }
        else {
            build_page(response);
        }
    }
}

loadCSV(`${resources_url}samples.csv`);

let server = http.createServer(handleRequest);
server.listen(8989);
console.log("Listening on port 8989.");