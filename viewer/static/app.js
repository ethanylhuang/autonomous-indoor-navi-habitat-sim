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

    console.log("app.js loaded!");

    // -- DOM references ---------------------------------------------------
    var sceneSelect = document.getElementById("sceneSelect");
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
    // Projection DOM references
    var fwdContainer = document.getElementById("fwdContainer");
    var fwdClickMarker = document.getElementById("fwdClickMarker");
    var projectedPin = document.getElementById("projectedPin");
    var projectedLabel = document.getElementById("projectedLabel");
    var projectedCoords = document.getElementById("projectedCoords");
    console.log("fwdContainer element:", fwdContainer);
    // M5 VLM DOM references
    var vlmInstructionInput = document.getElementById("vlmInstructionInput");
    var startVlmNavBtn = document.getElementById("startVlmNavBtn");
    var stopVlmNavBtn = document.getElementById("stopVlmNavBtn");
    var vlmStatusPanel = document.getElementById("vlmStatusPanel");
    var vlmInstruction = document.getElementById("vlmInstruction");
    var vlmSubgoal = document.getElementById("vlmSubgoal");
    var vlmReasoning = document.getElementById("vlmReasoning");
    var vlmConfidence = document.getElementById("vlmConfidence");
    var vlmStepsCalls = document.getElementById("vlmStepsCalls");
    var vlmResult = document.getElementById("vlmResult");
    // Semantic object navigation DOM references
    var semanticObjectSelect = document.getElementById("semanticObjectSelect");
    var navToObjectBtn = document.getElementById("navToObjectBtn");
    var objectDetail = document.getElementById("objectDetail");
    var objectLabel = document.getElementById("objectLabel");
    var objectPos = document.getElementById("objectPos");

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
    var navMode = "manual"; // "manual", "autonomous", or "vlm_nav"
    var tickTimer = null;
    var navmeshBounds = null; // {lower: [x,y,z], upper: [x,y,z]}
    var pinnedGoal = null; // [x, y, z] world coords, or null
    var semanticObjects = {}; // object_id -> object data from API
    var projectedPoint = null; // [x, y, z] world coords from image click, or null
    var agentY = 0; // current agent Y for goal height
    var vlmTickTimer = null; // Separate timer for VLM nav
    var changingScene = false; // True while scene change is in progress

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
        // Handle error responses
        if (data.error) {
            console.error("Server error:", data.error);
            showVlmResult(false, "Error: " + data.error);
            setVlmMode("manual");
            waitingForResponse = false;
            changingScene = false;
            sceneSelect.disabled = false;
            // Restore dropdown to current scene if scene change failed
            if (data.current_scene) {
                sceneSelect.value = data.current_scene;
            }
            return;
        }

        // Handle scene change completion
        if (data.scene_changed !== undefined) {
            changingScene = false;
            sceneSelect.disabled = false;
            if (data.current_scene) {
                sceneSelect.value = data.current_scene;
            }
            // Refresh semantic objects for new scene
            fetchSemanticObjects();
        }

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

            // Update nav mode from server (but not if VLM mode is active)
            if (mode !== navMode && navMode !== "vlm_nav") {
                setNavMode(mode);
            }
        }

        // M5 VLM status - process AFTER nav_status so VLM mode takes precedence
        if (data.vlm_status) {
            var vs = data.vlm_status;

            // If server says VLM is active, ensure client is in VLM mode
            if (vs.mode === "vlm_nav" && navMode !== "vlm_nav") {
                setVlmMode("vlm_nav");
            }

            updateVlmStatus(vs);

            // Check for VLM episode completion
            if (vs.goal_reached) {
                showVlmResult(true, "Goal reached! Steps: " + vs.steps_taken + ", VLM calls: " + vs.vlm_calls);
                setVlmMode("manual");
            } else if (vs.termination_reason && vs.termination_reason !== "goal_reached") {
                showVlmResult(false, "Ended: " + vs.termination_reason + " (steps: " + vs.steps_taken + ")");
                setVlmMode("manual");
            }
        }

        // Handle projection result
        if (data.projection_result) {
            updateProjection(data.projection_result);
        }

        waitingForResponse = false;

        // If in autonomous mode, send next tick
        if (navMode === "autonomous") {
            scheduleNextTick();
        }
        // If in VLM nav mode, send next VLM tick
        if (navMode === "vlm_nav") {
            scheduleNextVlmTick();
        }
    }

    function updateProjection(proj) {
        console.log("updateProjection:", proj);
        if (proj.is_valid && proj.navmesh_point && navmeshBounds) {
            projectedPoint = proj.navmesh_point;

            // Calculate position on topdown view
            var lower = navmeshBounds.lower;
            var upper = navmeshBounds.upper;
            var fracX = (proj.navmesh_point[0] - lower[0]) / (upper[0] - lower[0]);
            var fracY = (proj.navmesh_point[2] - lower[2]) / (upper[2] - lower[2]);

            // Show projected pin on navmesh
            projectedPin.style.display = "block";
            projectedPin.style.left = (fracX * 100) + "%";
            projectedPin.style.top = (fracY * 100) + "%";

            // Update label
            projectedLabel.style.display = "block";
            var coordStr = "[" + proj.navmesh_point[0].toFixed(2) + ", " +
                           proj.navmesh_point[1].toFixed(2) + ", " +
                           proj.navmesh_point[2].toFixed(2) + "]";
            if (proj.depth_value) {
                coordStr += " (d=" + proj.depth_value.toFixed(2) + "m)";
            }
            projectedCoords.textContent = coordStr;
        } else {
            // Invalid projection - show error
            projectedLabel.style.display = "block";
            projectedCoords.textContent = proj.failure_reason || "invalid";
            projectedCoords.style.color = "#ff6666";
            setTimeout(function() {
                projectedCoords.style.color = "#00ffff";
            }, 2000);
        }
    }

    function updateVlmStatus(vs) {
        if (vs.mode === "vlm_nav") {
            vlmStatusPanel.style.display = "block";
            vlmInstruction.textContent = vs.instruction || "--";
            vlmSubgoal.textContent = vs.subgoal || "--";
            vlmReasoning.textContent = vs.last_vlm_reasoning || "--";
            vlmConfidence.textContent = vs.confidence != null ? (vs.confidence * 100).toFixed(0) + "%" : "--";
            vlmStepsCalls.textContent = (vs.steps_taken || 0) + " / " + (vs.vlm_calls || 0);
        } else {
            vlmStatusPanel.style.display = "none";
        }
    }

    function showVlmResult(success, message) {
        vlmResult.style.display = "block";
        vlmResult.textContent = message;
        if (success) {
            vlmResult.style.background = "#1b5e20";
            vlmResult.style.color = "#a5d6a7";
        } else {
            vlmResult.style.background = "#b71c1c";
            vlmResult.style.color = "#ef9a9a";
        }
        setTimeout(function () {
            vlmResult.style.display = "none";
        }, 10000);
    }

    function setVlmMode(mode) {
        if (mode === "vlm_nav") {
            navMode = "vlm_nav";
            startVlmNavBtn.disabled = true;
            stopVlmNavBtn.disabled = false;
            vlmInstructionInput.disabled = true;
            // Also disable M3 nav buttons
            startNavBtn.disabled = true;
            startNavToBtn.disabled = true;
            controlsDisabledHint.style.display = "inline";
        } else {
            navMode = "manual";
            startVlmNavBtn.disabled = false;
            stopVlmNavBtn.disabled = true;
            vlmInstructionInput.disabled = false;
            // Re-enable M3 nav buttons
            startNavBtn.disabled = false;
            startNavToBtn.disabled = !pinnedGoal;
            controlsDisabledHint.style.display = "none";
            clearVlmTickTimer();
            vlmStatusPanel.style.display = "none";
        }
    }

    function scheduleNextVlmTick() {
        if (vlmTickTimer !== null) return;
        vlmTickTimer = setTimeout(function () {
            vlmTickTimer = null;
            sendVlmTick();
        }, 100); // 100ms delay - slower than M3 nav due to potential VLM latency
    }

    function clearVlmTickTimer() {
        if (vlmTickTimer !== null) {
            clearTimeout(vlmTickTimer);
            vlmTickTimer = null;
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
        // Clear pins
        pinnedGoal = null;
        goalPin.style.display = "none";
        pinnedGoalLabel.style.display = "none";
        startNavToBtn.disabled = true;
        // Clear projected point
        projectedPoint = null;
        projectedPin.style.display = "none";
        projectedLabel.style.display = "none";
        fwdClickMarker.style.display = "none";
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

    // -- M5 VLM Navigation send functions ---------------------------------
    function sendStartVlmNav(instruction) {
        if (!ws || ws.readyState !== WebSocket.OPEN || waitingForResponse) {
            return;
        }
        if (!instruction || instruction.trim() === "") {
            showVlmResult(false, "Please enter an instruction (e.g., 'go to the bedroom')");
            return;
        }
        waitingForResponse = true;
        vlmResult.style.display = "none";
        setVlmMode("vlm_nav");
        ws.send(JSON.stringify({ type: "start_semantic_nav", instruction: instruction.trim() }));
    }

    function sendStopVlmNav() {
        if (!ws || ws.readyState !== WebSocket.OPEN || waitingForResponse) {
            return;
        }
        waitingForResponse = true;
        ws.send(JSON.stringify({ type: "stop_semantic_nav" }));
        setVlmMode("manual");
    }

    function sendVlmTick() {
        if (!ws || ws.readyState !== WebSocket.OPEN || waitingForResponse) {
            return;
        }
        if (navMode !== "vlm_nav") return;
        waitingForResponse = true;
        ws.send(JSON.stringify({ type: "semantic_nav_tick" }));
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

    // -- Forward image click-to-project handler ---------------------------
    fwdContainer.addEventListener("click", function (e) {
        console.log("Forward image clicked, navMode=" + navMode + ", waiting=" + waitingForResponse);
        if (navMode === "autonomous" || navMode === "vlm_nav") return;
        if (waitingForResponse) return;

        var rect = fwdImg.getBoundingClientRect();
        var pixelX = Math.round((e.clientX - rect.left) * (640 / rect.width));
        var pixelY = Math.round((e.clientY - rect.top) * (480 / rect.height));

        // Clamp to image bounds
        pixelX = Math.max(0, Math.min(639, pixelX));
        pixelY = Math.max(0, Math.min(479, pixelY));

        // Show click marker on forward image
        var fracX = (e.clientX - rect.left) / rect.width;
        var fracY = (e.clientY - rect.top) / rect.height;
        fwdClickMarker.style.display = "block";
        fwdClickMarker.style.left = (fracX * 100) + "%";
        fwdClickMarker.style.top = (fracY * 100) + "%";

        // Send projection request
        sendProjectPixel(pixelX, pixelY);
    });

    function sendProjectPixel(u, v) {
        if (!ws || ws.readyState !== WebSocket.OPEN || waitingForResponse) {
            console.log("sendProjectPixel blocked: ws=" + (ws ? ws.readyState : "null") + ", waiting=" + waitingForResponse);
            return;
        }
        console.log("sendProjectPixel: u=" + u + ", v=" + v);
        waitingForResponse = true;
        ws.send(JSON.stringify({ type: "project_pixel", u: u, v: v }));
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

    // M5 VLM button handlers
    startVlmNavBtn.addEventListener("click", function () {
        sendStartVlmNav(vlmInstructionInput.value);
    });

    stopVlmNavBtn.addEventListener("click", function () {
        sendStopVlmNav();
    });

    // Allow Enter key to start VLM nav
    vlmInstructionInput.addEventListener("keydown", function (e) {
        if (e.key === "Enter" && navMode === "manual") {
            e.preventDefault();
            sendStartVlmNav(vlmInstructionInput.value);
        }
    });

    // -- Scene selection --------------------------------------------------
    function fetchScenes() {
        fetch("/api/scenes")
            .then(function (res) { return res.json(); })
            .then(function (data) {
                sceneSelect.innerHTML = "";
                if (!data.scenes || data.scenes.length === 0) {
                    var opt = document.createElement("option");
                    opt.value = "";
                    opt.textContent = "No scenes found";
                    sceneSelect.appendChild(opt);
                    return;
                }
                data.scenes.forEach(function (scene) {
                    var opt = document.createElement("option");
                    opt.value = scene.id;
                    opt.textContent = scene.name;
                    if (scene.id === data.current) {
                        opt.selected = true;
                    }
                    sceneSelect.appendChild(opt);
                });
                // Fetch semantic objects for current scene
                fetchSemanticObjects();
            })
            .catch(function (err) {
                console.error("Failed to fetch scenes:", err);
                sceneSelect.innerHTML = "<option value=''>Error loading scenes</option>";
            });
    }

    function fetchSemanticObjects() {
        fetch("/api/semantic_objects")
            .then(function (res) { return res.json(); })
            .then(function (data) {
                semanticObjectSelect.innerHTML = "";
                semanticObjects = {};

                if (!data.has_semantic || !data.objects || data.objects.length === 0) {
                    var opt = document.createElement("option");
                    opt.value = "";
                    opt.textContent = "No objects available";
                    semanticObjectSelect.appendChild(opt);
                    navToObjectBtn.disabled = true;
                    objectDetail.style.display = "none";
                    return;
                }

                // Add placeholder option
                var placeholder = document.createElement("option");
                placeholder.value = "";
                placeholder.textContent = "Select object (" + data.count + " available)";
                semanticObjectSelect.appendChild(placeholder);

                // Add each object
                data.objects.forEach(function (obj) {
                    semanticObjects[obj.object_id] = obj;
                    var opt = document.createElement("option");
                    opt.value = obj.object_id;
                    opt.textContent = obj.instance_name;
                    semanticObjectSelect.appendChild(opt);
                });

                navToObjectBtn.disabled = true; // Enable when selection made
            })
            .catch(function (err) {
                console.error("Failed to fetch semantic objects:", err);
                semanticObjectSelect.innerHTML = "<option value=''>Error loading objects</option>";
                navToObjectBtn.disabled = true;
            });
    }

    function sendChangeScene(sceneId) {
        if (!ws || ws.readyState !== WebSocket.OPEN || waitingForResponse || changingScene) {
            return;
        }
        if (!sceneId) return;

        changingScene = true;
        waitingForResponse = true;
        sceneSelect.disabled = true;

        // Stop any active navigation
        if (navMode === "autonomous") {
            setNavMode("manual");
        }
        if (navMode === "vlm_nav") {
            setVlmMode("manual");
        }

        // Clear pins
        pinnedGoal = null;
        goalPin.style.display = "none";
        pinnedGoalLabel.style.display = "none";
        startNavToBtn.disabled = true;
        // Clear projected point
        projectedPoint = null;
        projectedPin.style.display = "none";
        projectedLabel.style.display = "none";
        fwdClickMarker.style.display = "none";

        ws.send(JSON.stringify({ type: "change_scene", scene_id: sceneId }));
    }

    sceneSelect.addEventListener("change", function () {
        sendChangeScene(sceneSelect.value);
    });

    // Semantic object selection handler
    semanticObjectSelect.addEventListener("change", function () {
        var objId = parseInt(semanticObjectSelect.value);
        if (!objId || !semanticObjects[objId]) {
            objectDetail.style.display = "none";
            navToObjectBtn.disabled = true;
            return;
        }

        var obj = semanticObjects[objId];
        objectLabel.textContent = obj.label;
        if (obj.navmesh_position) {
            objectPos.textContent = "[" + obj.navmesh_position.map(function(v) { return v.toFixed(2); }).join(", ") + "]";
        } else {
            objectPos.textContent = "--";
        }
        objectDetail.style.display = "block";
        navToObjectBtn.disabled = false;
    });

    // Navigate to object button
    navToObjectBtn.addEventListener("click", function () {
        var objId = parseInt(semanticObjectSelect.value);
        if (!objId || !ws || ws.readyState !== WebSocket.OPEN || waitingForResponse) {
            return;
        }
        waitingForResponse = true;
        navResult.style.display = "none";
        setNavMode("autonomous");
        ws.send(JSON.stringify({ type: "start_nav_to_object", object_id: objId }));
    });

    // -- Initialize -------------------------------------------------------
    fetchScenes();
    connect();
})();
