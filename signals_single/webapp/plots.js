function get_pulses() {
    let pulses = [];
    let i = 0;
    let axleFound = true;

    while (axleFound) {
        let axleInput = document.getElementById(`axle_${i}`);
        if (axleInput == null) {
            axleFound = false;
            continue;
        }
        pulses.push(parseInt(axleInput.value));
        i++;
    }

    return pulses;
}

function plot_signal(points) {
    let pulses = get_pulses();
    
    let canvas = document.getElementById("canvas");
    let ctx = canvas.getContext("2d");

    ctx.fillStyle = '#fff8c6';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = '#f0e380';
    ctx.fillRect(0, 130, canvas.width, 20);

    ctx.lineWidth = 1;
    ctx.strokeStyle = "black";
    ctx.beginPath();
    ctx.moveTo(0, 150);
    ctx.lineTo(1300, 150);
    ctx.stroke();

    for (i in pulses) {
        let pulse = pulses[i];
        if (pulse == 0) continue;

        if (heldAxle != null && heldAxle == i)
            ctx.strokeStyle = "lime";
        else
            ctx.strokeStyle = "green";

        ctx.beginPath();
        ctx.moveTo(pulse, 0);
        ctx.lineTo(pulse, 200);
        ctx.stroke();
    }

    ctx.strokeStyle = "red";
    for (i in predictions) {
        let value = predictions[i];
        if (value == 0) continue;

        ctx.beginPath();
        ctx.moveTo(i, 150);
        ctx.lineTo(i, 150 - 100 * value);
        ctx.stroke();
    }

    let x = 0;
    let y = 150;
    let y0 = 150;
    let first = true;
    
    ctx.strokeStyle = "blue";
    ctx.beginPath();
    for (i in points) {
        value = points[i];

        if (first) {
            first = false;
            continue;
        }

        ctx.moveTo(x, y0);

        x = x + 1;
        y = -Math.round(100 * value) + 150;

        ctx.lineTo(x, y);

        y0 = y;
    }
    ctx.stroke();
}

function plot_mousedown(e) {
    const rect = document.getElementById("canvas").getBoundingClientRect();
    grabx = e.x - rect.left;
    graby = e.y - rect.top;
    
    let pulses = get_pulses();
    
    heldAxle = null;
    let i = 0;

    while (heldAxle == null && i < pulses.length) {
        if (grabx >= pulses[i] - 5 && grabx <= pulses[i] + 5){
            heldAxle = i;
            continue;
        }

        i++;
    }

    if (heldAxle == null) return;

    plot_signal(points);
    document.getElementById("save").disabled = false;
    document.getElementById("save_next").disabled = false;
    document.getElementById("correct_last_next").disabled = false;
}

function plot_mouseup(e) {
    if (grabx == null && graby == null)
        return;

    const rect = document.getElementById("canvas").getBoundingClientRect();
    let mousex = e.x - rect.left;
    let mousey = e.y - rect.top;

    if (mousex == grabx && mousey == graby) {
        if (heldAxle == null) {
            let i = 0;
            let axlesDone = false;
            while (!axlesDone && heldAxle == null) {
                let axle = document.getElementById(`axle_${i}`);
                if (axle == null) {
                    axlesDone = true;
                    continue;
                }
                if (axle.value == 0) {
                    axlesDone = true;
                    axle.value = Math.round(mousex);
                }
                i++;
            }
        }
        else {
            document.getElementById(`axle_${heldAxle}`).value = 0;
        }
    }

    grabx = null;
    graby = null;
    heldAxle = null;
    
    plot_signal(points);
    document.getElementById("save").disabled = false;
    document.getElementById("save_next").disabled = false;
    document.getElementById("correct_last_next").disabled = false;
}

function plot_mousemove(e) {
    if (heldAxle == null) return;

    const rect = document.getElementById("canvas").getBoundingClientRect();
    let mousex = e.x - rect.left;
    let mousey = e.y - rect.top;

    document.getElementById(`axle_${heldAxle}`).value = Math.round(mousex);

    plot_signal(points);
    document.getElementById("save").disabled = false;
    document.getElementById("save_next").disabled = false;
    document.getElementById("correct_last_next").disabled = false;
}

function update_status(dbName) {
    let xhttp = new XMLHttpRequest(); // AJAX object
    
    xhttp.onreadystatechange = () => {
        if (xhttp.readyState == 4) {
            if (xhttp.status == 200) {
                let statusBar = document.getElementById("statusbar");
                statusBar.innerHTML = xhttp.responseText;
            }
        }
    }

    xhttp.open("GET", `status?db=${dbName}`, true);
    xhttp.send();
}

function update_axles(dbName, photo_id) {
    let xhttp = new XMLHttpRequest(); // AJAX object
    
    xhttp.onreadystatechange = () => {
        if (xhttp.readyState == 4) {
            if (xhttp.status == 200) {
                values = xhttp.responseText.split("|");

                let time = values[0];
                let photo_id = parseInt(values[1]);
                let saved = (values[2] == 'true');
                let keep = (values[3] == 'true');
                let comment = values[4];

                let area = document.getElementById("area");
                let status = document.getElementById("status");
                if (keep) {
                    if (saved) {
                        area.style.backgroundColor = "#abdbba";
                        status.innerText = `PREGLEDANO (${time})`;
                    }
                    else {
                        area.style.backgroundColor = "white";
                        status.innerText = "NI PREGLEDANO";
                    }

                    document.getElementById("discard").checked = false;
                }
                else {
                    area.style.backgroundColor = "#dbabab";
                    status.innerText = `IZLOČENO (${time})`;
                    
                    document.getElementById("discard").checked = true;
                }

                let inputChecked = document.getElementById("checked");
                if (saved)
                    inputChecked.value = "1";
                else
                    inputChecked.value = "0";

                let btnSave = document.getElementById("save");
                let btnSaveNext = document.getElementById("save_next");
                let btnCorrectLast = document.getElementById("correct_last_next");

                if (saved) {
                    btnSave.disabled = true;
                    btnSaveNext.disabled = true;
                }
                else {
                    btnSave.disabled = false;
                    btnSaveNext.disabled = false;
                    btnCorrectLast.disabled = false;
                }

                cnt = 0;
                for (let i = 5; i < values.length; i++) {
                    let axleInput = document.getElementById(`axle_${i-5}`);
                    axleInput.value = parseInt(values[i]);
                    axleInput.disabled = false;
                    cnt++;
                }
                
                axleFound = true;
                while (axleFound) {
                    let axleInput = document.getElementById(`axle_${cnt}`);
                    if (axleInput == null) {
                        axleFound = false;
                        continue;
                    }
                    axleInput.value = 0;
                    axleInput.disabled = false;
                    cnt++;
                }

                /* Mark differences. */
                axleFound = true;
                let i = 0;
                while (axleFound) {
                    let pulseInput = document.getElementById(`pulse_${i}`);
                    let axleInput = document.getElementById(`axle_${i}`);
                    if (pulseInput == null || axleInput == null) {
                        axleFound = false;
                        continue;
                    }

                    color = "white";
                    if (pulseInput.value != axleInput.value)
                        color = "red";
                    
                    axleInput.style.backgroundColor = color;
                    i++;
                }

                /* Comment */
                let commentInput = document.getElementById("comment");
                commentInput.value = comment;

                plot_signal(points);
                update_status(dbName);
            }
        }
    };
    
    xhttp.open("GET", `load?db=${dbName}&id=${photo_id}`, true);
    xhttp.send();
}

function axles_changed(points) {
    plot_signal(points);

    document.getElementById("save").disabled = false;
    document.getElementById("save_next").disabled = false;
    document.getElementById("correct_last_next").disabled = false;
}

function discard_changed() {
    document.getElementById("save").disabled = false;
    document.getElementById("save_next").disabled = false;
    document.getElementById("correct_last_next").disabled = false;
}

function comment_changed() {
    document.getElementById("save").disabled = false;
    document.getElementById("save_next").disabled = false;
    document.getElementById("correct_last_next").disabled = false;
}

function save_axles(dbName, photo_id, number = null) {
    let pulses = get_pulses();
    let comment = document.getElementById("comment").value;

    let parms = `db=${dbName}&id=${photo_id}`;

    btnDiscard = document.getElementById("discard");
    if (btnDiscard.checked)
        parms += `&keep=false`;
    else
        parms += `&keep=true`;

    parms += `&comment=${comment}`;
    for (i in pulses) {
        parms += `&axle${i}=${pulses[i]}`;
    }

    document.getElementById("save").disabled = true;
    document.getElementById("save_next").disabled = true;
    document.getElementById("correct_last_next").disabled = true;

    let xhttp = new XMLHttpRequest(); // AJAX object
    
    xhttp.onreadystatechange = () => {
        if (xhttp.readyState == 4) {
            if (xhttp.status == 200) {
                photo_id = parseInt(xhttp.responseText);
                console.log(`Axles for ${photo_id} saved.`);
                update_axles(dbName, photo_id);

                if (number != null) {
                    location.href=`/plot.html?db=${dbName}&number=${number}`;
                }
            }
            else {
                window.alert("Server communication error!");

                document.getElementById("save").disable= false;
                document.getElementById("save_next").disable= false;
                document.getElementById("correct_last_next").disabled = false;
            }
        }
    };
    
    xhttp.open("GET", "save?" + parms, true);
    xhttp.send();
}

function correct_last_axle(dbName, photo_id, number = null) {
    let pulses = get_pulses();

    let cnt = 0;
    let last_nonzero = 0;
    let axleFound = true;
    while (axleFound) {
        let axleInput = document.getElementById(`axle_${cnt}`);
        if (axleInput == null) {
            axleFound = false;
            continue;
        }

        if (parseInt(axleInput.value) > 0)
            last_nonzero = cnt;

        cnt++;
    }

    let i = 1300;
    let last_prediction = null;
    while (i > 0 && last_prediction == null) {
        i--;
        if (predictions[i] > 0) {
            last_prediction = i;
        }
    }

    let axleInput = document.getElementById(`axle_${last_nonzero}`);
    axleInput.value = i;

    plot_signal(points);
    save_axles(dbName, photo_id, number);
}

function delete_axles(dbName, photo_id) {
    let xhttp = new XMLHttpRequest(); // AJAX object
    
    xhttp.onreadystatechange = () => {
        if (xhttp.readyState == 4) {
            if (xhttp.status == 200) {
                photo_id = parseInt(xhttp.responseText);
                console.log(`Axles for ${photo_id} deleted.`);
                update_axles(dbName, photo_id);
            }
            else {
                window.alert("Server communication error!");

                document.getElementById("save").disabled = false;
                document.getElementById("save_next").disabled = false;
                document.getElementById("correct_last_next").disabled = false;
            }
        }
    };
    
    xhttp.open("GET", `delete?db=${dbName}&id=${photo_id}`, true);
    xhttp.send();
}

function change_db() {
    let dbselect = document.getElementById("dbselect");
    
    switch (dbselect.value) {
        case 'BAZA 1':
            location.href=`/index.html?db=validation`;
            break;
        
        case 'BAZA 2':
            location.href=`/index.html?db=correct`;
            break;

        case 'BAZA 3':
            location.href=`/index.html?db=fixed`;
            break;    
    }

}