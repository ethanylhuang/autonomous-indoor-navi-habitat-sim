/**
 * WebSocket client for the Autonomous Vehicle Viewer.
 *
 * Handles keyboard input, WebSocket communication, and canvas/image updates.
 * Uses a waitingForResponse flag to prevent action queuing.
 *
 * M3 additions: autonomous navigation mode with start/stop and tick-based
 * frame requests.
 */

(function () {
    "use strict";

    // -- DOM references ---------------------------------------------------
    var fwdImg = document.getElementById("fwd");
    var rearImg = document.getElementById("rear");
    var depthImg = document.getElementById("depthImg");
    var topdownImg = document.getElementById("topdown");
    var stepCountEl = document.getElementById("stepCount");
    var collidedEl = document.getElementById("collided");
    var posX = document.getElementById("posX");
    var posY = document.getElementById("posY");
    var posZ = document.getElementById("posZ");
    var rotW = document.getElementById("rotW");
    var rotX = document.getElementById("rotX");
    var rotY = document.getElementById("rotY");
    var rotZ = document.getElementById("rotZ");
    var imuAccel = document.getElementById("imuAccel");
    var imuAngVel = document.getElementById("imuAngVel");
    var imuStep = document.getElementById("imuStep");
    var resetBtn = document.getElementById("resetBtn");
    var statusEl = document.getElementById("connection-status");
    // M2 DOM references
    var pointCloudBevImg = document.getElementById("pointCloudBev");
    var occGridImg = document.getElementById("occGrid");
    var semanticFwdImg = document.getElementById("semanticFwd");
    var voTrajectoryImg = document.getElementById("voTrajectory");
    var voInliersEl = document.getElementById("voInliers");
    var voValidEl = document.getElementById("voValid");
    var occupiedCellsEl = document.getElementById("occupiedCells");
    var freeCellsEl = document.getElementById("freeCells");
    var fwdObstaclePxEl = document.getElementById("fwdObstaclePx");
    var rearObstaclePxEl = document.getElementById("rearObstaclePx");
    // M3 DOM references
    var startNavBtn = document.getElementById("startNavBtn");
    var startNavToBtn = document.getElementById("startNavToBtn");
    var stopNavBtn = document.getElementById("stopNavBtn");
    var navModeValue = document.getElementById("navModeValue");
    var navResult = document.getElementById("navResult");
    var navGoal = document.getElementById("navGoal");
    var navDistance = document.getElementById("navDistance");
    var navHeading = document.getElementById("navHeading");
    var navSteps = document.getElementById("navSteps");
    var navCollisions = document.getElementById("navCollisions");
    var navPathLength = document.getElementById("navPathLength");
    var navSPL = document.getElementById("navSPL");
    var navAction = document.getElementById("navAction");
    var controlsDisabledHint = document.getElementById("controlsDisabledHint");
    var goalPin = document.getElementById("goalPin");
    var pinnedGoalLabel = document.getElementById("pinnedGoalLabel");
    var pinnedGoalCoords = document.getElementById("pinnedGoalCoords");

    // -- Key mappings -----------------------------------------------------
    var KEY_MAP = {
        w: "move_forward",
        W: "move_forward",
        a: "turn_left",
        A: "turn_left",
        d: "turn_right",
        D: "turn_right",
    };

    // -- State ------------------------------------------------------------
    var ws = null;
    var waitingForResponse = false;
    var navMode = "manual"; // "manual" or "autonomous"
    var tickTimer = null;
    var navmeshBounds = null; // {lower: [x,y,z], upper: [x,y,z]}
    var pinnedGoal = null; // [x, y, z] world coords, or null
    var agentY = 0; // current agent Y for goal height

    // -- Formatting helper ------------------------------------------------
    function fmt(val) {
        return typeof val === "number" ? val.toFixed(4) : String(val);
    }

    function fmtArr(arr) {
        if (!Array.isArray(arr)) return "--";
        return "[" + arr.map(function (v) { return v.toFixed(4); }).join(", ") + "]";
    }

    function fmtGoal(arr) {
        if (!Array.isArray(arr)) return "--";
        return "[" + arr.map(function (v) { return v.toFixed(2); }).join(", ") + "]";
    }

    // -- Update UI --------------------------------------------------------
    function updateFrame(data) {
        // Update images (base64 -> data URL)
        fwdImg.src = "data:image/jpeg;base64," + data.forward_rgb;
        rearImg.src = "data:image/jpeg;base64," + data.rear_rgb;
        depthImg.src = "data:image/jpeg;base64," + data.depth;
        topdownImg.src = "data:image/png;base64," + data.topdown;

        // Track navmesh bounds for click-to-goal
        if (data.navmesh_bounds) {
            navmeshBounds = data.navmesh_bounds;
        }

        // Update state display
        var st = data.state;
        agentY = st.position[1];
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

        // M2 perception images
        if (data.point_cloud_bev) {
            pointCloudBevImg.src = "data:image/png;base64," + data.point_cloud_bev;
        }
        if (data.occupancy_grid) {
            occGridImg.src = "data:image/png;base64," + data.occupancy_grid;
        }
        if (data.semantic_fwd) {
            semanticFwdImg.src = "data:image/jpeg;base64," + data.semantic_fwd;
        }
        if (data.vo_trajectory) {
            voTrajectoryImg.src = "data:image/png;base64," + data.vo_trajectory;
        }

        // M2 stats
        if (data.m2_stats) {
            var m2 = data.m2_stats;
            voInliersEl.textContent = m2.vo_inliers;
            voValidEl.textContent = m2.vo_valid ? "Yes" : "No";
            occupiedCellsEl.textContent = m2.occupied_cells;
            freeCellsEl.textContent = m2.free_cells;
            fwdObstaclePxEl.textContent = m2.fwd_obstacle_pixels;
            rearObstaclePxEl.textContent = m2.rear_obstacle_pixels;
        }

        // M3 navigation status
        if (data.nav_status) {
            var ns = data.nav_status;
            var mode = ns.mode || "manual";

            navModeValue.textContent = mode === "autonomous" ? "Autonomous" : "Manual";

            if (ns.goal) {
                navGoal.textContent = fmtGoal(ns.goal);
            } else {
                navGoal.textContent = "--";
            }

            navDistance.textContent = ns.distance_to_goal != null ? ns.distance_to_goal + "m" : "--";
            navHeading.textContent = ns.heading_error != null ? ns.heading_error + " rad" : "--";
            navSteps.textContent = ns.steps_taken || 0;
            navCollisions.textContent = ns.collisions || 0;
            navPathLength.textContent = ns.path_length != null ? ns.path_length + "m" : "--";
            navSPL.textContent = ns.spl != null ? ns.spl : "--";
            navAction.textContent = ns.action || "--";

            // Check for episode completion
            if (ns.goal_reached) {
                showNavResult(true, "Goal reached! SPL: " + (ns.spl || 0));
                setNavMode("manual");
            } else if (ns.termination_reason && ns.termination_reason !== "goal_reached") {
                showNavResult(false, "Failed: " + ns.termination_reason + " (SPL: " + (ns.spl || 0) + ")");
                setNavMode("manual");
            }

            // Update nav mode from server
            if (mode !== navMode) {
                setNavMode(mode);
            }
        }

        waitingForResponse = false;

        // If in autonomous mode, send next tick
        if (navMode === "autonomous") {
            scheduleNextTick();
        }
    }

    function showNavResult(success, message) {
        navResult.style.display = "block";
        navResult.textContent = message;
        if (success) {
            navResult.style.background = "#1b5e20";
            navResult.style.color = "#a5d6a7";
        } else {
            navResult.style.background = "#b71c1c";
            navResult.style.color = "#ef9a9a";
        }
        // Auto-hide after 10 seconds
        setTimeout(function () {
            navResult.style.display = "none";
        }, 10000);
    }

    function setNavMode(mode) {
        navMode = mode;
        if (mode === "autonomous") {
            startNavBtn.disabled = true;
            startNavToBtn.disabled = true;
            stopNavBtn.disabled = false;
            controlsDisabledHint.style.display = "inline";
            navModeValue.textContent = "Autonomous";
        } else {
            startNavBtn.disabled = false;
            startNavToBtn.disabled = !pinnedGoal;
            stopNavBtn.disabled = true;
            controlsDisabledHint.style.display = "none";
            navModeValue.textContent = "Manual";
            clearTickTimer();
        }
    }

    function scheduleNextTick() {
        // Prevent multiple timers
        if (tickTimer !== null) return;
        tickTimer = setTimeout(function () {
            tickTimer = null;
            sendTick();
        }, 50); // 50ms delay for ~20fps
    }

    function clearTickTimer() {
        if (tickTimer !== null) {
            clearTimeout(tickTimer);
            tickTimer = null;
        }
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
            setNavMode("manual");
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
        setNavMode("manual");
        // Clear pin
        pinnedGoal = null;
        goalPin.style.display = "none";
        pinnedGoalLabel.style.display = "none";
        startNavToBtn.disabled = true;
        ws.send(JSON.stringify({ type: "reset" }));
    }

    function sendStartNav() {
        if (!ws || ws.readyState !== WebSocket.OPEN || waitingForResponse) {
            return;
        }
        waitingForResponse = true;
        navResult.style.display = "none";
        setNavMode("autonomous");
        ws.send(JSON.stringify({ type: "start_nav" }));
    }

    function sendStopNav() {
        if (!ws || ws.readyState !== WebSocket.OPEN || waitingForResponse) {
            return;
        }
        waitingForResponse = true;
        ws.send(JSON.stringify({ type: "stop_nav" }));
        setNavMode("manual");
    }

    function sendTick() {
        if (!ws || ws.readyState !== WebSocket.OPEN || waitingForResponse) {
            return;
        }
        if (navMode !== "autonomous") return;
        waitingForResponse = true;
        ws.send(JSON.stringify({ type: "tick" }));
    }

    function sendStartNavTo(goal) {
        if (!ws || ws.readyState !== WebSocket.OPEN || waitingForResponse) {
            return;
        }
        waitingForResponse = true;
        navResult.style.display = "none";
        setNavMode("autonomous");
        ws.send(JSON.stringify({ type: "start_nav_to", goal: goal }));
    }

    // -- NavMesh click-to-pin handler ------------------------------------
    topdownImg.addEventListener("click", function (e) {
        if (navMode === "autonomous") return;
        if (!navmeshBounds) return;

        var rect = topdownImg.getBoundingClientRect();
        var fracX = (e.clientX - rect.left) / rect.width;
        var fracY = (e.clientY - rect.top) / rect.height;

        // Convert pixel fraction to world XZ
        var lower = navmeshBounds.lower;
        var upper = navmeshBounds.upper;
        var worldX = lower[0] + fracX * (upper[0] - lower[0]);
        var worldZ = lower[2] + fracY * (upper[2] - lower[2]);

        pinnedGoal = [worldX, agentY, worldZ];

        // Show pin overlay
        goalPin.style.display = "block";
        goalPin.style.left = (fracX * 100) + "%";
        goalPin.style.top = (fracY * 100) + "%";

        // Update label
        pinnedGoalLabel.style.display = "block";
        pinnedGoalCoords.textContent = "[" + worldX.toFixed(2) + ", " + agentY.toFixed(2) + ", " + worldZ.toFixed(2) + "]";

        // Enable "Nav to Pin" button
        startNavToBtn.disabled = false;
    });

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

        // Ignore WASD during autonomous mode
        if (navMode === "autonomous") {
            return;
        }

        var action = KEY_MAP[e.key];
        if (action) {
            e.preventDefault();
            sendAction(action);
        }
    });

    // -- Button handlers --------------------------------------------------
    resetBtn.addEventListener("click", function () {
        sendReset();
    });

    startNavBtn.addEventListener("click", function () {
        sendStartNav();
    });

    startNavToBtn.addEventListener("click", function () {
        if (pinnedGoal) {
            sendStartNavTo(pinnedGoal);
        }
    });

    stopNavBtn.addEventListener("click", function () {
        sendStopNav();
    });

    // -- Initialize -------------------------------------------------------
    connect();
})();
