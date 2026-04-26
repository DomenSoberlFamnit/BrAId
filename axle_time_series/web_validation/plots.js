function get_pulses(photo_id) {
    let pulses = [];
    let i = 0;
    let axleFound = true;

    while (axleFound) {
        let axleInput = document.getElementById(`axle_${photo_id}_${i}`);
        if (axleInput == null) {
            axleFound = false;
            continue;
        }
        pulses.push(parseInt(axleInput.value));
        i++;
    }

    return pulses;
}

function plot_signal(photo_id, points) {
    let pulses = get_pulses(photo_id);
    
    let canvas = document.getElementById(`canvas_${photo_id}`);
    let ctx = canvas.getContext("2d");

    ctx.fillStyle = '#fff8c6';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.lineWidth = 1;
    ctx.strokeStyle = "black";
    ctx.beginPath();
    ctx.moveTo(0, 150);
    ctx.lineTo(1300, 150);
    ctx.stroke();

    ctx.strokeStyle = "red";
    for (i in pulses) {
        let pulse = pulses[i];
        if (pulse == 0) continue;

        ctx.beginPath();
        ctx.moveTo(pulse, 0);
        ctx.lineTo(pulse, 200);
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

function update_status() {
    let xhttp = new XMLHttpRequest(); // AJAX object
    
    xhttp.onreadystatechange = () => {
        if (xhttp.readyState == 4) {
            if (xhttp.status == 200) {
                let statusBar = document.getElementById("statusbar");
                statusBar.innerHTML = xhttp.responseText;
            }
        }
    }

    xhttp.open("GET", "status", true);
    xhttp.send();
}

function update_axles(photo_id, update=true) {
    let xhttp = new XMLHttpRequest(); // AJAX object
    
    xhttp.onreadystatechange = () => {
        if (xhttp.readyState == 4) {
            if (xhttp.status == 200) {
                values = xhttp.responseText.split(",");

                let photo_id = parseInt(values[0]);
                let saved = (values[1] == 'true');
                let keep = (values[2] == 'true');
                let comment = values[3];

                let area = document.getElementById(`area_${photo_id}`);
                let status = document.getElementById(`status_${photo_id}`);
                if (keep) {
                    if (saved) {
                        area.style.backgroundColor = "#abdbba";
                        status.innerText = "PREGLEDANO";
                    }
                    else {
                        area.style.backgroundColor = "white";
                        status.innerText = "NI PREGLEDANO";
                    }
                }
                else {
                    area.style.backgroundColor = "#dbabab";
                    status.innerText = "IZLOČENO";
                }

                let inputChecked = document.getElementById(`checked_${photo_id}`);
                if (saved)
                    inputChecked.value = "1";
                else
                    inputChecked.value = "0";

                let btnSave = document.getElementById(`save_${photo_id}`);
                let btnSaveNext = document.getElementById(`save_next_${photo_id}`);
                let btnDelete = document.getElementById(`delete_${photo_id}`);
                let btnDiscard = document.getElementById(`discard_${photo_id}`);

                if (saved) {
                    btnSave.disabled = true;
                    btnSaveNext.disabled = true;
                    btnDiscard.disabled = true;
                    btnDelete.disabled = false;
                }
                else {
                    btnSave.disabled = false;
                    btnSaveNext.disabled = false;
                    btnDiscard.disabled = false;
                    btnDelete.disabled = true;
                }

                cnt = 0;
                for (let i = 4; i < values.length; i++) {
                    let axleInput = document.getElementById(`axle_${photo_id}_${i-4}`);
                    axleInput.value = parseInt(values[i]);
                    axleInput.disabled = false;
                    cnt++;
                }
                
                axleFound = true;
                while (axleFound) {
                    let axleInput = document.getElementById(`axle_${photo_id}_${cnt}`);
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
                    let pulseInput = document.getElementById(`pulse_${photo_id}_${i}`);
                    let axleInput = document.getElementById(`axle_${photo_id}_${i}`);
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
                let commentInput = document.getElementById(`comment_${photo_id}`);
                commentInput.value = comment;

                plot_signal(photo_id, points[photo_id]);

                if (update) update_status();
            }
        }
    };
    
    xhttp.open("GET", `load?id=${photo_id}`, true);
    xhttp.send();
}

function update_all_axles() {
    for (photo_id in points) {
        update_axles(photo_id, false);
    }
    update_status();
}

function axles_changed(photo_id, points) {
    plot_signal(photo_id, points);

    document.getElementById(`save_${photo_id}`).disabled = false;
    document.getElementById(`save_next_${photo_id}`).disabled = false;
}

function save_axles(photo_id, keep = true, number = null) {
    let pulses = get_pulses(photo_id);
    let comment = document.getElementById('comment_' + photo_id.toString()).value;

    let parms = `id=${photo_id}`;
    if (keep)
        parms += `&keep=true`;
    else
        parms += `&keep=false`;
    parms += `&comment=${comment}`;
    for (i in pulses) {
        parms += `&axle${i}=${pulses[i]}`;
    }

    document.getElementById(`save_${photo_id}`).disabled = true;
    document.getElementById(`save_next_${photo_id}`).disabled = true;

    let xhttp = new XMLHttpRequest(); // AJAX object
    
    xhttp.onreadystatechange = () => {
        if (xhttp.readyState == 4) {
            if (xhttp.status == 200) {
                photo_id = parseInt(xhttp.responseText);
                console.log(`Axles for ${photo_id} saved.`);
                update_axles(photo_id);

                if (number != null) {
                    location.href=`/index?number=${number}`;
                }
            }
            else {
                window.alert("Server communication error!");

                document.getElementById(`save_${photo_id}`).disable= false;
                document.getElementById(`save_next_${photo_id}`).disable= false;
            }
        }
    };
    
    xhttp.open("GET", "save?" + parms, true);
    xhttp.send();
}

function delete_axles(photo_id) {
    let xhttp = new XMLHttpRequest(); // AJAX object
    
    xhttp.onreadystatechange = () => {
        if (xhttp.readyState == 4) {
            if (xhttp.status == 200) {
                photo_id = parseInt(xhttp.responseText);
                console.log(`Axles for ${photo_id} deleted.`);
                update_axles(photo_id);
            }
            else {
                window.alert("Server communication error!");

                var button = document.getElementById(`save_${photo_id}`).disabled = false;
                var button = document.getElementById(`save_next_${photo_id}`).disabled = false;
            }
        }
    };
    
    xhttp.open("GET", `delete?id=${photo_id}`, true);
    xhttp.send();
}

function comment_changed(photo_id) {
    document.getElementById(`save_${photo_id}`).disabled = false;
    document.getElementById(`save_next_${photo_id}`).disabled = false;
}