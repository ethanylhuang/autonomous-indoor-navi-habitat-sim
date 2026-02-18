/**
 * WebSocket client for the Autonomous Vehicle Viewer.
 *
 * Handles keyboard input, WebSocket communication, and canvas/image updates.
 * Uses a waitingForResponse flag to prevent action queuing.
 */

(function () {
    "use strict";

    // -- DOM references ---------------------------------------------------
    const fwdImg = document.getElementById("fwd");
    const rearImg = document.getElementById("rear");
    const depthImg = document.getElementById("depthImg");
    const topdownImg = document.getElementById("topdown");
    const stepCountEl = document.getElementById("stepCount");
    const collidedEl = document.getElementById("collided");
    const posX = document.getElementById("posX");
    const posY = document.getElementById("posY");
    const posZ = document.getElementById("posZ");
    const rotW = document.getElementById("rotW");
    const rotX = document.getElementById("rotX");
    const rotY = document.getElementById("rotY");
    const rotZ = document.getElementById("rotZ");
    const imuAccel = document.getElementById("imuAccel");
    const imuAngVel = document.getElementById("imuAngVel");
    const imuStep = document.getElementById("imuStep");
    const resetBtn = document.getElementById("resetBtn");
    const statusEl = document.getElementById("connection-status");

    // -- Key mappings -----------------------------------------------------
    const KEY_MAP = {
        w: "move_forward",
        W: "move_forward",
        a: "turn_left",
        A: "turn_left",
        d: "turn_right",
        D: "turn_right",
    };

    // -- State ------------------------------------------------------------
    let ws = null;
    let waitingForResponse = false;

    // -- Formatting helper ------------------------------------------------
    function fmt(val) {
        return typeof val === "number" ? val.toFixed(4) : String(val);
    }

    function fmtArr(arr) {
        if (!Array.isArray(arr)) return "--";
        return "[" + arr.map(function (v) { return v.toFixed(4); }).join(", ") + "]";
    }

    // -- Update UI --------------------------------------------------------
    function updateFrame(data) {
        // Update images (base64 -> data URL)
        fwdImg.src = "data:image/jpeg;base64," + data.forward_rgb;
        rearImg.src = "data:image/jpeg;base64," + data.rear_rgb;
        depthImg.src = "data:image/jpeg;base64," + data.depth;
        topdownImg.src = "data:image/png;base64," + data.topdown;

        // Update state display
        var st = data.state;
        stepCountEl.textContent = st.step_count;
        collidedEl.textContent = st.collided ? "Yes" : "No";
        posX.textContent = fmt(st.position[0]);
        posY.textContent = fmt(st.position[1]);
        posZ.textContent = fmt(st.position[2]);
        rotW.textContent = fmt(st.rotation[0]);
        rotX.textContent = fmt(st.rotation[1]);
        rotY.textContent = fmt(st.rotation[2]);
        rotZ.textContent = fmt(st.rotation[3]);

        // Update IMU display
        var imu = data.imu;
        imuAccel.textContent = fmtArr(imu.linear_acceleration);
        imuAngVel.textContent = fmtArr(imu.angular_velocity);
        imuStep.textContent = imu.timestamp_step;

        waitingForResponse = false;
    }

    // -- WebSocket --------------------------------------------------------
    function connect() {
        var protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        var url = protocol + "//" + window.location.host + "/ws";
        ws = new WebSocket(url);

        ws.onopen = function () {
            statusEl.textContent = "Connected";
            statusEl.className = "connected";
            waitingForResponse = false;
        };

        ws.onmessage = function (event) {
            var data = JSON.parse(event.data);
            updateFrame(data);
        };

        ws.onclose = function () {
            statusEl.textContent = "Disconnected";
            statusEl.className = "disconnected";
            waitingForResponse = false;
            // Reconnect after 2 seconds
            setTimeout(connect, 2000);
        };

        ws.onerror = function () {
            statusEl.textContent = "Error";
            statusEl.className = "disconnected";
        };
    }

    // -- Send action ------------------------------------------------------
    function sendAction(action) {
        if (!ws || ws.readyState !== WebSocket.OPEN || waitingForResponse) {
            return;
        }
        waitingForResponse = true;
        ws.send(JSON.stringify({ action: action }));
    }

    function sendReset() {
        if (!ws || ws.readyState !== WebSocket.OPEN || waitingForResponse) {
            return;
        }
        waitingForResponse = true;
        ws.send(JSON.stringify({ type: "reset" }));
    }

    // -- Keyboard handler -------------------------------------------------
    document.addEventListener("keydown", function (e) {
        // Ignore if typing in an input field
        if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") {
            return;
        }

        if (e.key === "r" || e.key === "R") {
            e.preventDefault();
            sendReset();
            return;
        }

        var action = KEY_MAP[e.key];
        if (action) {
            e.preventDefault();
            sendAction(action);
        }
    });

    // -- Reset button handler ---------------------------------------------
    resetBtn.addEventListener("click", function () {
        sendReset();
    });

    // -- Initialize -------------------------------------------------------
    connect();
})();
